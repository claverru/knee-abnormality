import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytorch_lightning as L
import torch
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from knee_abnormality.config import (
    EMAConfig,
    ExperimentConfig,
    OptimizerConfig,
    SWAConfig,
)
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader
from knee_abnormality.data.labels import FoldLabels, load_fold_labels
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.report_attributes import load_report_attribute_targets
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.evaluation.metrics import COMPARTMENT_PAIRS, validation_metrics
from knee_abnormality.evaluation.predictions import make_prediction_frame
from knee_abnormality.identity import preprocessing_identity
from knee_abnormality.modeling.model import (
    COMPARTMENT_MIDLINE_BAND_MM,
    KneeAbnormalityModel,
)
from knee_abnormality.training.checkpoints import checkpoint_sampling_epoch
from knee_abnormality.training.report_attributes import (
    ReportAttributeHeads,
    report_attribute_loss,
)

PRECISION = "bf16-mixed"
HALF_KNEE_TEMPERATURE = 1.0


def build_optimizer(
    model: KneeAbnormalityModel,
    settings: OptimizerConfig,
    heads: tuple[torch.nn.Module, ...],
) -> AdamW:
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    named = [
        *model.named_parameters(),
        *(
            (f"heads.{name}", parameter)
            for head in heads
            for name, parameter in head.named_parameters()
        ),
    ]
    for name, parameter in named:
        if not parameter.requires_grad:
            continue
        learning_rate = (
            settings.backbone_lr if name.startswith("backbone.") else settings.head_lr
        )
        weight_decay = 0.0 if parameter.ndim < 2 else settings.weight_decay
        groups.setdefault((learning_rate, weight_decay), []).append(parameter)
    return AdamW(
        [
            {"params": parameters, "lr": lr, "weight_decay": decay}
            for (lr, decay), parameters in groups.items()
        ],
        betas=(settings.beta1, settings.beta2),
        eps=settings.epsilon,
    )


def fixed_class_balance(
    training: pl.DataFrame, targets: tuple[str, ...], exponent: float
) -> dict:
    rows = []
    for target in targets:
        observed = training.filter(pl.col(f"observed_{target}"))
        positive = observed.filter(pl.col(f"target_{target}") >= 0.5).height
        negative = observed.height - positive
        if positive == 0 or negative == 0:
            raise ValueError(
                f"Fixed class-balanced BCE requires both target groups for {target}"
            )
        fraction = positive / observed.height
        rho = ((1 - fraction) / fraction) ** exponent
        negative_weight = 1 / (fraction * rho + 1 - fraction)
        positive_weight = rho * negative_weight
        rows.append(
            {
                "target": target,
                "observed": observed.height,
                "positive": positive,
                "negative": negative,
                "positive_fraction": fraction,
                "positive_weight": positive_weight,
                "negative_weight": negative_weight,
            }
        )
    return {"version": 1, "exponent": exponent, "targets": rows}


def class_balance_weight_tensor(class_balance: dict) -> torch.Tensor:
    return torch.tensor(
        [
            [row["positive_weight"] for row in class_balance["targets"]],
            [row["negative_weight"] for row in class_balance["targets"]],
        ],
        dtype=torch.float32,
    )


def validated_class_balance_weights(
    config: ExperimentConfig, targets: tuple[str, ...], class_balance: dict
) -> torch.Tensor:
    if class_balance.get("exponent") != config.trainer.class_balance_exponent:
        raise ValueError("Class-balanced BCE settings and statistics disagree")
    if class_balance.get("version") != 1 or [
        row.get("target") for row in class_balance.get("targets", [])
    ] != list(targets):
        raise ValueError("Class-balanced BCE statistics targets differ from the model")
    return class_balance_weight_tensor(class_balance)


def masked_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observed: torch.Tensor,
    class_balance_weights: torch.Tensor,
) -> torch.Tensor:
    if not observed.any():
        raise RuntimeError("Training batch has no observed targets")
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(
        targets >= 0.5, class_balance_weights[0], class_balance_weights[1]
    )
    return (losses * weights)[observed].mean()


def half_knee_mil_loss(
    model: KneeAbnormalityModel,
    encoded: dict,
    batch: dict,
    meniscus_indices: tuple[int, int],
    class_balance_weights: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    axis = encoded["medial_axis"].float()
    selected = encoded["visual_valid"] & (axis.abs() > COMPARTMENT_MIDLINE_BAND_MM)
    token_index = selected.flatten().nonzero(as_tuple=True)[0]
    if token_index.numel() == 0:
        return encoded["features"].sum() * 0, 0
    study = encoded["study"][:, None].expand_as(selected).flatten()[token_index]
    half = (axis.flatten()[token_index] < 0).long()
    group = study * 2 + half
    group_count = 2 * encoded["batch_size"]
    tokens = model.half_knee_tokens(encoded)[token_index]
    with torch.autocast(device_type=axis.device.type, enabled=False):
        logits = model.half_knee_detector(tokens.float())
        values = logits.squeeze(-1).float()
        peak = torch.full((group_count,), -torch.inf, device=axis.device)
        peak.scatter_reduce_(0, group, values, reduce="amax", include_self=True)
        counts = torch.zeros(group_count, device=axis.device)
        counts.scatter_add_(0, group, torch.ones_like(values))
        present = counts > 0
        safe_peak = torch.where(present, peak, 0)
        total = torch.zeros(group_count, device=axis.device)
        total.scatter_add_(
            0, group, ((values - safe_peak[group]) / HALF_KNEE_TEMPERATURE).exp()
        )
        scores = (
            safe_peak
            + HALF_KNEE_TEMPERATURE
            * (total.clamp_min(1e-20) / counts.clamp_min(1)).log()
        )
        scores = scores.view(-1, 2)
        observed = batch["target_observed"][:, list(meniscus_indices)]
        used = present.view(-1, 2).all(-1) & observed.all(-1)
        if not used.any():
            return values.sum() * 0, 0
        targets = batch["target"][:, list(meniscus_indices)][used].float()
        losses = F.binary_cross_entropy_with_logits(
            scores[used], targets, reduction="none"
        )
        weights = class_balance_weights[:, list(meniscus_indices)]
        losses = losses * torch.where(targets >= 0.5, weights[0], weights[1])
        return losses.mean(), int(used.sum())


class KneeDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: ExperimentConfig,
        fold: int,
        limit_train: int | None = None,
        limit_validation: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.fold = fold
        self.limit_train = limit_train
        self.limit_validation = limit_validation
        self.labels: FoldLabels | None = None
        self.train_dataset: KneeStudyDataset | None = None
        self.validation_dataset: KneeStudyDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.labels is not None:
            return
        self.labels = load_fold_labels(self.config, self.fold)
        index = build_series_index(
            self.config.paths.data_root, self.config.data, "train"
        )
        training = (
            self.labels.training.head(self.limit_train)
            if self.limit_train is not None
            else self.labels.training
        )
        validation = (
            self.labels.validation.head(self.limit_validation)
            if self.limit_validation is not None
            else self.labels.validation
        )
        self.train_dataset = KneeStudyDataset(
            training,
            index,
            self.config.data,
            self.labels.targets,
            self.config.seed,
        )
        self.validation_dataset = KneeStudyDataset(
            validation,
            index,
            self.config.data,
            self.labels.targets,
            None,
            self.config.data.validation_image_multiplier,
        )

    def train_dataloader(self) -> DataLoader:
        module = getattr(self.trainer, "lightning_module", None)
        if module is not None:
            self.train_dataset.set_epoch(
                max(module.sampling_epoch, self.trainer.current_epoch)
            )
        return build_data_loader(
            self.train_dataset,
            self.config.trainer,
            self.config.trainer.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return build_data_loader(
            self.validation_dataset,
            self.config.trainer,
            self.config.trainer.batch_size,
        )

    def batch_norm_loader(self) -> DataLoader:
        return build_data_loader(
            self.train_dataset,
            self.config.trainer,
            self.config.trainer.batch_size,
            shuffle=False,
            persistent_workers=False,
        )

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        return with_images(batch)


class KneeModule(L.LightningModule):
    def __init__(
        self,
        config: ExperimentConfig,
        targets: tuple[str, ...],
        fold: int,
        training_studies: int,
        validation_studies: int,
        prediction_dir: Path,
        class_balance: dict,
        load_pretrained: bool | None = None,
        input_identity: dict | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.sampling_epoch = 0
        self.targets = targets
        self.validation_studies = validation_studies
        self.prediction_dir = prediction_dir
        self.class_balance = class_balance
        self.register_buffer(
            "class_balance_weights",
            validated_class_balance_weights(config, targets, class_balance),
            persistent=False,
        )
        self.model = KneeAbnormalityModel(
            config.model, config.data, targets, load_pretrained=load_pretrained
        )
        self.meniscus_indices = tuple(
            targets.index(name) for name in COMPARTMENT_PAIRS["meniscus"]
        )
        attributes = load_report_attribute_targets(
            config.paths.report_attribute_manifest, config.paths.data_root
        )
        with torch.random.fork_rng(devices=[]):
            self.report_attribute_heads = ReportAttributeHeads(attributes)
        self.weight_averaging = config.optimizer.weight_averaging
        self._last_averaged_step = -1
        self.batch_norm_refresh_seconds = 0.0
        if self.weight_averaging is not None:
            self.averaged = copy.deepcopy(self.model).requires_grad_(False)
        self.validation_rows: list[
            tuple[list[str], np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self.live_validation_rows: list[
            tuple[list[str], np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self.save_hyperparameters(
            {
                "config": config.model_dump(mode="json"),
                "targets": list(targets),
                "fold": fold,
                "training_studies": training_studies,
                "validation_studies": validation_studies,
                "class_balance": class_balance,
                "input_identity": input_identity,
                "preprocessing_identity": preprocessing_identity(config.model.backbone),
                "report_attributes": {
                    **attributes.provenance,
                    "questions": [
                        question.model_dump(mode="json")
                        for question in attributes.questions
                    ],
                },
            }
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.sampling_epoch = checkpoint_sampling_epoch(checkpoint)
        if self.weight_averaging is not None:
            state = checkpoint["state_dict"]
            live = checkpoint.pop("live_state_dict")
            checkpoint["state_dict"] = {
                **{
                    f"averaged.{key.removeprefix('model.')}": value
                    for key, value in state.items()
                    if key.startswith("model.")
                },
                **{f"model.{key}": value for key, value in live.items()},
            }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.weight_averaging is None:
            return
        state = checkpoint["state_dict"]
        live = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
        averaged = {
            key.removeprefix("averaged."): value
            for key, value in state.items()
            if key.startswith("averaged.")
        }
        checkpoint["state_dict"] = {
            key: value
            for key, value in state.items()
            if not key.startswith("model.") and not key.startswith("averaged.")
        }
        checkpoint["state_dict"].update(
            {f"model.{key}": value for key, value in averaged.items()}
        )
        checkpoint["live_state_dict"] = live

    def on_train_epoch_start(self) -> None:
        self.sampling_epoch = self.current_epoch
        dataset = getattr(self.trainer.train_dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    def on_train_epoch_end(self) -> None:
        strengths = self.model.anatomical_log_strength.detach().float().exp()
        for family, strength in zip(("meniscus", "oa"), strengths, strict=True):
            self.log(
                f"anatomical_strength_{family}",
                strength,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
        self.log(
            "half_knee_detector_norm",
            self.model.half_knee_detector.weight.detach().float().norm(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        for region, scale in zip(
            ("medial", "lateral", "pf"),
            self.model.oa_detector_scale.detach().float(),
            strict=True,
        ):
            self.log(
                f"oa_detector_scale_{region}",
                scale,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
        self.log(
            "oa_detector_weight_norm",
            self.model.oa_detector.weight.detach().float().norm(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

    def training_step(self, batch: dict, batch_index: int) -> torch.Tensor:
        encoded = self.model.encode_batch(batch)
        logits = self.model.pool_encoded(encoded).to(torch.float32)
        counts, masked, absolute_corrections = self.model.oa_detector_activity
        for region, region_counts, absolute_correction in zip(
            ("medial", "lateral", "pf"),
            counts.unbind(1),
            absolute_corrections,
            strict=True,
        ):
            self.log(
                f"oa_detector_absolute_correction_{region}",
                absolute_correction,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch["study_uid"]),
            )
            self.log(
                f"oa_detector_tokens_{region}",
                region_counts.sum().float(),
                on_step=False,
                on_epoch=True,
                reduce_fx="sum",
                batch_size=1,
            )
            self.log(
                f"oa_detector_studies_{region}",
                (region_counts > 0).sum().float(),
                on_step=False,
                on_epoch=True,
                reduce_fx="sum",
                batch_size=1,
            )
            self.log(
                f"oa_detector_empty_studies_{region}",
                (region_counts == 0).sum().float(),
                on_step=False,
                on_epoch=True,
                reduce_fx="sum",
                batch_size=1,
            )
        self.log(
            "oa_detector_masked_tokens",
            masked.float(),
            on_step=False,
            on_epoch=True,
            reduce_fx="sum",
            batch_size=1,
        )
        loss = masked_binary_cross_entropy_with_logits(
            logits,
            batch["target"],
            batch["target_observed"],
            self.class_balance_weights,
        )
        auxiliary, used_studies = half_knee_mil_loss(
            self.model,
            encoded,
            batch,
            self.meniscus_indices,
            self.class_balance_weights,
        )
        loss = loss + self.config.trainer.half_knee_loss_weight * auxiliary
        loss = loss + self._report_attribute_step(encoded, batch)
        self.log(
            "half_knee_loss",
            auxiliary,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["study_uid"]),
        )
        self.log(
            "half_knee_used_studies",
            used_studies,
            on_step=False,
            on_epoch=True,
            reduce_fx="sum",
            batch_size=1,
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["study_uid"]),
        )
        return loss

    def _report_attribute_step(self, encoded: dict, batch: dict) -> torch.Tensor:
        heads = self.report_attribute_heads
        batch_size = len(batch["study_uid"])
        logits = heads(
            self.model.half_knee_tokens(encoded),
            encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            encoded["visual_valid"].flatten(),
            encoded["batch_size"],
        )
        targets, weights, observed = heads.targets(batch["study_uid"])
        loss, divergence = report_attribute_loss(
            logits, targets, weights, observed, heads.option_counts
        )
        self.log(
            "report_attribute_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "report_attribute_studies",
            observed.sum().float(),
            on_step=False,
            on_epoch=True,
            reduce_fx="sum",
            batch_size=1,
        )
        for question, value in zip(heads.question_ids, divergence, strict=True):
            self.log(
                f"report_attribute_kl_{question}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return self.config.trainer.report_attribute_loss_weight * loss

    def on_before_optimizer_step(self, optimizer) -> None:
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if gradients:
            self.log(
                "grad_norm",
                torch.nn.utils.get_total_norm(gradients),
                on_step=True,
                on_epoch=False,
                batch_size=1,
            )

    def _averaging_weight(self, step: int) -> float:
        averaging = self.weight_averaging
        if isinstance(averaging, EMAConfig):
            return 1.0 if step <= self.warmup_steps else 1.0 - averaging.decay
        return (
            1.0 if step <= self.swa_start_step else 1.0 / (step - self.swa_start_step)
        )

    def _refresh_batch_norm(self) -> None:
        self.averaged.train()
        modules = [
            module
            for module in self.averaged.modules()
            if isinstance(module, _BatchNorm) and module.training
        ]
        if not modules:
            return
        momentums = [module.momentum for module in modules]
        for module in modules:
            module.reset_running_stats()
            module.momentum = None
        device = self.device
        devices = [device] if device.type == "cuda" else []
        started = time.perf_counter()
        with (
            torch.random.fork_rng(devices=devices),
            torch.no_grad(),
            self.trainer.precision_plugin.forward_context(),
        ):
            for batch in self.trainer.datamodule.batch_norm_loader():
                batch = with_images(self.transfer_batch_to_device(batch, device, 0))
                self.averaged(batch)
        self.batch_norm_refresh_seconds += time.perf_counter() - started
        for module, momentum in zip(modules, momentums):
            module.momentum = momentum

    def on_train_batch_end(self, outputs, batch, batch_index: int) -> None:
        if self.weight_averaging is None:
            return
        step = self.trainer.global_step
        if step == self._last_averaged_step:
            return
        self._last_averaged_step = step
        weight = self._averaging_weight(step)
        with torch.no_grad():
            live_state = self.model.state_dict()
            averaged_state = self.averaged.state_dict()
            live_floats, averaged_floats = [], []
            for name, live_tensor in live_state.items():
                averaged_tensor = averaged_state[name]
                if torch.is_floating_point(live_tensor):
                    if weight >= 1.0:
                        averaged_tensor.copy_(live_tensor)
                    else:
                        live_floats.append(live_tensor)
                        averaged_floats.append(averaged_tensor)
                else:
                    averaged_tensor.copy_(live_tensor)
            if averaged_floats:
                torch._foreach_lerp_(averaged_floats, live_floats, weight)
        if (
            isinstance(self.weight_averaging, SWAConfig)
            and step > self.swa_start_step
            and step % self.steps_per_epoch == 0
        ):
            self._refresh_batch_norm()

    def on_validation_epoch_start(self) -> None:
        self.validation_rows.clear()
        self.live_validation_rows.clear()

    def validation_step(self, batch: dict, batch_index: int) -> None:
        saved_model = self.averaged if self.weight_averaging is not None else self.model
        prediction = torch.sigmoid(saved_model(batch).to(torch.float32))
        self.validation_rows.append(
            (
                batch["study_uid"],
                batch["target"].cpu().numpy(),
                prediction.cpu().numpy(),
                batch["is_gold"].cpu().numpy(),
            )
        )
        if self.weight_averaging is not None:
            live_prediction = torch.sigmoid(self.model(batch).to(torch.float32))
            self.live_validation_rows.append(
                (
                    batch["study_uid"],
                    batch["target"].cpu().numpy(),
                    live_prediction.cpu().numpy(),
                    batch["is_gold"].cpu().numpy(),
                )
            )

    def _log_and_write_validation(
        self,
        rows: list[tuple[list[str], np.ndarray, np.ndarray, np.ndarray]],
        metric_prefix: str,
        file_prefix: str,
    ) -> None:
        study_ids = [study for ids, _, _, _ in rows for study in ids]
        targets = np.concatenate([values for _, values, _, _ in rows])
        predictions = np.concatenate([values for _, _, values, _ in rows])
        is_gold = np.concatenate([values for _, _, _, values in rows])
        if len(study_ids) != self.validation_studies or len(set(study_ids)) != len(
            study_ids
        ):
            raise RuntimeError(
                "Validation epoch did not gather every study exactly once"
            )
        result = validation_metrics(targets, predictions, self.targets, is_gold)
        self.log(
            f"{metric_prefix}macro_auc",
            result["macro_auc"],
            prog_bar=True,
            sync_dist=False,
        )
        self.log(
            f"{metric_prefix}soft_macro_auc",
            result["soft_macro_auc"],
            sync_dist=False,
        )
        self.log(
            f"{metric_prefix}soft_target_bce",
            result["soft_target_bce"],
            sync_dist=False,
        )
        if result["gold"]["macro_auc"] is not None:
            self.log(
                f"{metric_prefix}gold_macro_auc",
                result["gold"]["macro_auc"],
                sync_dist=False,
            )
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        make_prediction_frame(study_ids, predictions, self.targets).write_csv(
            self.prediction_dir
            / f"{file_prefix}validation_epoch_{self.current_epoch:03d}.csv"
        )
        (
            self.prediction_dir
            / f"{file_prefix}metrics_epoch_{self.current_epoch:03d}.json"
        ).write_text(json.dumps(result, indent=2) + "\n")

    def on_validation_epoch_end(self) -> None:
        self._log_and_write_validation(self.validation_rows, "val_", "")
        if self.weight_averaging is not None:
            self._log_and_write_validation(
                self.live_validation_rows, "val_live_", "live_"
            )

    def configure_optimizers(self):
        settings = self.config.optimizer
        optimizer = build_optimizer(
            self.model, settings, (self.report_attribute_heads,)
        )
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = max(
            1, math.ceil(total_steps / self.config.trainer.max_epochs)
        )
        warmup_steps = settings.warmup_epochs * steps_per_epoch
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_steps
        self.swa_start_step = (
            total_steps - settings.weight_averaging.epochs * steps_per_epoch
            if isinstance(settings.weight_averaging, SWAConfig)
            else None
        )

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                start = settings.warmup_start_ratio
                return start + (1 - start) * step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
            return settings.minimum_lr_ratio + (1 - settings.minimum_lr_ratio) * cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, schedule),
                "interval": "step",
            },
        }
