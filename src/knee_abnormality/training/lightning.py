import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytorch_lightning as L
import torch
from timm.models import group_parameters
from torch.func import functional_call
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from knee_abnormality.config import (
    STUDY_HEADERS,
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
from knee_abnormality.modeling.augmentation import AugmentationActivity
from knee_abnormality.modeling.model import (
    COMPARTMENT_MIDLINE_BAND_MM,
    KneeAbnormalityModel,
)
from knee_abnormality.training.auxiliary.scanner_confusion import (
    SCANNER_GROUPS,
    SCANNER_TABLE_SHA256,
    ScannerConfusionHead,
    confusion_ramp,
    load_scanner_groups,
    scanner_group_labels,
    scanner_group_prior,
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
    anchored_backbone_names: frozenset[str],
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
        weight_decay = (
            0.0
            if parameter.ndim < 2
            or name in anchored_backbone_names
            or name == "view_label_bias"
            else settings.weight_decay
        )
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


def _gradient_rms(gradients: list[torch.Tensor]) -> torch.Tensor:
    squares = torch.stack([gradient.float().square().sum() for gradient in gradients])
    return (squares.sum() / sum(gradient.numel() for gradient in gradients)).sqrt()


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
        self.scanner_confusion_head = None
        self.scanner_group_counts = None
        if config.trainer.scanner_confusion_weight > 0:
            self.scanner_targets = load_scanner_groups(
                config.paths.data_root.parent / STUDY_HEADERS
            )
            prior, counts = scanner_group_prior(
                config.split_dir, fold, self.scanner_targets
            )
            self.register_buffer("scanner_prior", prior, persistent=False)
            self.scanner_group_counts = counts
            with torch.random.fork_rng(devices=[]):
                self.scanner_confusion_head = ScannerConfusionHead()
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
        self.view_names = config.data.view_names
        self.view_validation_rows = {name: [] for name in self.view_names}
        self.view_missing_studies = dict.fromkeys(self.view_names, 0)
        self.train_window_rows = []
        self.validation_window_rows = []
        self.validation_window_studies = []
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
                "scanner_groups": {
                    "table_sha256": SCANNER_TABLE_SHA256,
                    "groups": list(SCANNER_GROUPS),
                    "training_group_counts": self.scanner_group_counts,
                }
                if self.scanner_confusion_head is not None
                else None,
            }
        )
        backbone = self.model.backbone
        self.backbone_stages = {
            f"stage{index}": tuple(names)
            for index, names in group_parameters(
                backbone, backbone.group_matcher(coarse=True)
            ).items()
        }
        settings = config.optimizer
        self.anchor_parameters: list[tuple[str, torch.nn.Parameter, str, float]] = []
        self.backbone_origin = {}
        early_anchors = 0
        for index, (name, parameter) in enumerate(backbone.named_parameters()):
            if parameter.ndim < 2:
                continue
            early = name.startswith(("conv1.", "bn1.", "layer1."))
            early_anchors += early
            alpha = settings.anchor_alpha_early if early else settings.anchor_alpha_late
            if alpha > 0:
                buffer_name = f"anchor_weight_{index}"
                self.register_buffer(buffer_name, parameter.detach().float().clone())
                self.anchor_parameters.append((name, parameter, buffer_name, alpha))
            else:
                self.backbone_origin[name] = parameter.detach().float().cpu().clone()
        if settings.anchor_alpha_early > 0 and not early_anchors:
            raise ValueError(
                "optimizer.anchor_alpha_early matches no backbone layer; the early "
                "anchor is defined by ResNet stem and layer1 names"
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
        self.train_window_rows.clear()
        self.sampling_epoch = self.current_epoch
        dataset = getattr(self.trainer.train_dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    def on_train_epoch_end(self) -> None:
        self._log_window_stats("", self.train_window_rows)
        if self.model.view_label_bias is not None:
            for name, bias in zip(
                self.view_names,
                self.model.view_label_bias.detach().float(),
                strict=True,
            ):
                for label, value in zip(self.targets, bias, strict=True):
                    self._log_epoch_metric(
                        f"view_label_bias_{name}_{label.lower().replace(' ', '_')}",
                        value,
                        1,
                    )
        if self.config.model.plane_readout:
            self._log_epoch_metric(
                "plane_readout_drift", self.model.head.divergence().mean(), 1
            )
        if self.model.anatomical_log_strength is not None:
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
        parameters = dict(self.model.backbone.named_parameters())
        anchor_buffers = {
            name: buffer_name for name, _, buffer_name, _ in self.anchor_parameters
        }
        for stage, names in self.backbone_stages.items():
            moved = origin = 0.0
            for name in names:
                if name in self.backbone_origin or name in anchor_buffers:
                    start = (
                        getattr(self, anchor_buffers[name]).detach().float().cpu()
                        if name in anchor_buffers
                        else self.backbone_origin[name]
                    )
                    current = parameters[name].detach().float().cpu()
                    moved += float((current - start).square().sum())
                    origin += float(start.square().sum())
            if origin:
                self.log(
                    f"backbone_drift_{stage}",
                    math.sqrt(moved / origin),
                    on_step=False,
                    on_epoch=True,
                    batch_size=1,
                )

    def _log_epoch_metric(
        self,
        name: str,
        value: torch.Tensor | float,
        batch_size: int,
        reduce_fx: str = "mean",
    ) -> None:
        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            reduce_fx=reduce_fx,
            batch_size=batch_size,
        )

    def _log_family_activity(
        self, prefix: str, activity: AugmentationActivity, change_name: str
    ) -> None:
        affected = activity.affected_acquisitions.float()
        total = activity.total_acquisitions
        self._log_epoch_metric(f"{prefix}_affected_acquisitions", affected, 1, "sum")
        self._log_epoch_metric(f"{prefix}_total_acquisitions", float(total), 1, "sum")
        self._log_epoch_metric(
            f"{prefix}_effective_probability", affected / total, total
        )
        changed = int(activity.changed_elements.item())
        if changed > 0:
            self._log_epoch_metric(
                f"{prefix}_{change_name}", activity.mean_absolute_change, changed
            )

    def _log_augmentation_activity(self, study_count: int) -> None:
        data = self.config.data
        image_activity = self.model.image_augmentation.activity
        for prefix, settings in (
            ("smoothing", data.smoothing_augmentation),
            ("resolution", data.resolution_augmentation),
            ("gin", data.gin_augmentation),
        ):
            if settings.probability > 0:
                self._log_family_activity(
                    prefix, image_activity[prefix], "mean_absolute_pixel_change"
                )
        if self.config.model.feature_statistics_augmentation.probability > 0:
            self._log_family_activity(
                "feature_statistics",
                self.model.feature_statistics_activity,
                "mean_absolute_change",
            )
        if data.contrast_mixing_augmentation.probability > 0:
            mixing = self.model.contrast_mixing_activity
            for name, value in (
                ("acquisitions", mixing.acquisitions),
                ("studies", mixing.studies),
            ):
                self._log_epoch_metric(f"contrast_mixing_{name}", value, 1, "sum")
            for name, value in (
                ("lambda", mixing.mean_lambda),
                ("pixel_change", mixing.mean_pixel_change),
                ("effective_probability", mixing.effective_probability),
            ):
                self._log_epoch_metric(f"contrast_mixing_{name}", value, study_count)

    def _log_views(self, batch: dict, encoded: dict, prefix: str) -> None:
        valid = encoded["visual_valid"].any(-1)
        count = int(valid.sum())
        mass = self.model.attention_mass_view.mean(1)
        for view, (name, values) in enumerate(zip(self.view_names, mass, strict=True)):
            if count:
                self._log_epoch_metric(
                    f"{prefix}view_share_{name}",
                    ((encoded["view_index"] == view) & valid).sum().float() / count,
                    count,
                )
            for label, value in zip(self.targets, values, strict=True):
                self._log_epoch_metric(
                    f"{prefix}attention_mass_{name}_{label.lower().replace(' ', '_')}",
                    value,
                    len(batch["study_uid"]),
                )
            self._log_epoch_metric(
                f"{prefix}attention_mass_{name}_mean",
                values.mean(),
                len(batch["study_uid"]),
            )
        selected = self.model.detail_view[batch["view_index"]] & batch[
            "slice_mask"
        ].any(-1)
        views = batch["view_index"][selected]
        rows = (
            torch.cat(
                (
                    views[:, None].float(),
                    batch["window_center_mm"][selected],
                    batch["window_rule_distance_mm"][selected, None],
                ),
                -1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        destination = (
            self.train_window_rows if not prefix else self.validation_window_rows
        )
        destination.append(rows)
        if prefix:
            count = len(self.view_names)
            group = batch["series_study_index"][selected] * count + views
            total = torch.zeros(
                (len(batch["study_uid"]) * count, 2), device=self.device
            )
            number = torch.zeros(len(total), device=self.device)
            total.index_add_(0, group, batch["window_center_mm"][selected])
            number.index_add_(0, group, torch.ones_like(group, dtype=torch.float32))
            means = (total / number.clamp_min(1)[:, None]).reshape(-1, count, 2)
            self.validation_window_studies.append(
                (
                    means.cpu().numpy(),
                    (number > 0).reshape(-1, count).cpu().numpy(),
                    batch["target"].cpu().numpy(),
                )
            )

    def _log_window_stats(self, prefix: str, rows: list) -> None:
        if not rows:
            return
        values = np.concatenate(rows)
        for view, name in enumerate(self.view_names):
            selected = values[values[:, 0] == view]
            if not len(selected):
                continue
            for axis, column in (("x", 1), ("y", 2)):
                self._log_epoch_metric(
                    f"{prefix}window_center_{name}_{axis}_mean_mm",
                    float(selected[:, column].mean()),
                    1,
                )
                self._log_epoch_metric(
                    f"{prefix}window_center_{name}_{axis}_std_mm",
                    float(selected[:, column].std()),
                    1,
                )
            self._log_epoch_metric(
                f"{prefix}window_rule_distance_{name}_mean_mm",
                float(selected[:, 3].mean()),
                1,
            )
            self._log_epoch_metric(
                f"{prefix}window_rule_displaced_share_{name}",
                float((selected[:, 3] > 1.0).mean()),
                1,
            )
        if prefix and self.validation_window_studies:
            centers = np.concatenate([row[0] for row in self.validation_window_studies])
            present = np.concatenate([row[1] for row in self.validation_window_studies])
            targets = np.concatenate([row[2] for row in self.validation_window_studies])
            for view, name in enumerate(self.view_names):
                for index, label in enumerate(self.targets):
                    positive = present[:, view] & (targets[:, index] >= 0.5)
                    negative = present[:, view] & (targets[:, index] < 0.5)
                    if positive.any() and negative.any():
                        distance = np.linalg.norm(
                            centers[positive, view].mean(0)
                            - centers[negative, view].mean(0)
                        )
                        self._log_epoch_metric(
                            f"val_window_center_group_distance_{name}_{label.lower().replace(' ', '_')}_mm",
                            float(distance),
                            1,
                        )

    def training_step(self, batch: dict, batch_index: int) -> torch.Tensor:
        encoded = self.model.encode_batch(batch)
        self._log_augmentation_activity(len(batch["study_uid"]))
        logits = self.model.pool_encoded(encoded).to(torch.float32)
        self._log_views(batch, encoded, "")
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
        if self.scanner_confusion_head is not None:
            loss = loss + self._scanner_confusion_step(encoded, batch)
        if self.anchor_parameters:
            anchor_penalty = sum(
                alpha * (parameter.float() - getattr(self, buffer_name)).square().sum()
                for _, parameter, buffer_name, alpha in self.anchor_parameters
            )
            loss = loss + anchor_penalty
            self._log_epoch_metric(
                "anchor_penalty", anchor_penalty, len(batch["study_uid"])
            )
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

    def _scanner_confusion_step(self, encoded: dict, batch: dict) -> torch.Tensor:
        head = self.scanner_confusion_head
        tokens = self.model.half_knee_tokens(encoded)
        args = (
            encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            encoded["visual_valid"].flatten(),
            encoded["batch_size"],
        )
        groups = scanner_group_labels(
            batch["study_uid"], self.scanner_targets, tokens.device
        )
        head_logits = head(tokens.detach(), *args)
        head_loss = F.cross_entropy(head_logits, groups)
        accuracy = (head_logits.argmax(-1) == groups).float().mean()
        detached_parameters = {
            name: parameter.detach() for name, parameter in head.named_parameters()
        }
        confusion_logits = functional_call(head, detached_parameters, (tokens, *args))
        confusion_kl = F.kl_div(
            F.log_softmax(confusion_logits, dim=-1),
            self.scanner_prior.expand_as(confusion_logits),
            reduction="batchmean",
        )
        ramp = confusion_ramp(self.trainer.global_step, self.total_steps)
        weight = self.config.trainer.scanner_confusion_weight
        batch_size = len(batch["study_uid"])
        self._log_epoch_metric("scanner_head_loss", head_loss, batch_size)
        self._log_epoch_metric("scanner_head_accuracy", accuracy, batch_size)
        self._log_epoch_metric("scanner_confusion_kl", confusion_kl, batch_size)
        self._log_epoch_metric(
            "scanner_confusion_loss", weight * ramp * confusion_kl, batch_size
        )
        self._log_epoch_metric("scanner_confusion_ramp", ramp, batch_size)
        self._log_epoch_metric(
            "scanner_confusion_effective_weight", weight * ramp, batch_size
        )
        self._log_epoch_metric("scanner_head_studies", batch_size, 1, "sum")
        return head_loss + weight * ramp * confusion_kl

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
        backbone = dict(self.model.backbone.named_parameters())
        for stage, names in self.backbone_stages.items():
            stage_gradients = [
                backbone[name].grad for name in names if backbone[name].grad is not None
            ]
            if stage_gradients:
                self._log_epoch_metric(
                    f"backbone_grad_rms_{stage}", _gradient_rms(stage_gradients), 1
                )
        head_gradients = [
            parameter.grad
            for name, parameter in self.model.named_parameters()
            if not name.startswith("backbone.") and parameter.grad is not None
        ]
        if head_gradients:
            self._log_epoch_metric("head_grad_rms", _gradient_rms(head_gradients), 1)

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
        for rows in self.view_validation_rows.values():
            rows.clear()
        self.view_missing_studies.update(dict.fromkeys(self.view_names, 0))
        self.validation_window_rows.clear()
        self.validation_window_studies.clear()
        self.validation_rows.clear()
        self.live_validation_rows.clear()

    def validation_step(self, batch: dict, batch_index: int) -> None:
        saved_model = self.averaged if self.weight_averaging is not None else self.model
        encoded = saved_model.encode_batch(batch)
        prediction = torch.sigmoid(saved_model.pool_encoded(encoded).to(torch.float32))
        self._log_views(batch, encoded, "val_")
        for view, (view_name, rows) in enumerate(self.view_validation_rows.items()):
            view_prediction = torch.sigmoid(
                saved_model.pool_encoded(encoded, view).float()
            )
            present = encoded["visual_valid"].any(-1) & (encoded["view_index"] == view)
            self.view_missing_studies[view_name] += encoded["batch_size"] - len(
                encoded["study"][present].unique()
            )
            rows.append(
                (
                    batch["study_uid"],
                    batch["target"].cpu().numpy(),
                    view_prediction.cpu().numpy(),
                    batch["is_gold"].cpu().numpy(),
                )
            )
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
        self._log_window_stats("val_", self.validation_window_rows)
        for view_name, rows in self.view_validation_rows.items():
            result = validation_metrics(
                np.concatenate([row[1] for row in rows]),
                np.concatenate([row[2] for row in rows]),
                self.targets,
                np.concatenate([row[3] for row in rows]),
            )
            self._log_epoch_metric(
                f"val_macro_auc_view_{view_name}", result["macro_auc"], 1
            )
            self._log_epoch_metric(
                f"val_studies_missing_view_{view_name}",
                float(self.view_missing_studies[view_name]),
                1,
            )
            for label, auc in result["per_target_auc"].items():
                self._log_epoch_metric(
                    f"val_auc_view_{view_name}_{label.lower().replace(' ', '_')}",
                    auc,
                    1,
                )
        self._log_and_write_validation(self.validation_rows, "val_", "")
        if self.weight_averaging is not None:
            self._log_and_write_validation(
                self.live_validation_rows, "val_live_", "live_"
            )

    def configure_optimizers(self):
        settings = self.config.optimizer
        optimizer = build_optimizer(
            self.model,
            settings,
            tuple(
                head
                for head in (self.report_attribute_heads, self.scanner_confusion_head)
                if head is not None
            ),
            frozenset(f"backbone.{name}" for name, _, _, _ in self.anchor_parameters),
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
