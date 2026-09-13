import json
import math
from pathlib import Path

import numpy as np
import pytorch_lightning as L
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from knee_abnormality.checkpoints import checkpoint_sampling_epoch
from knee_abnormality.config import ExperimentConfig, OptimizerConfig
from knee_abnormality.dataset import (
    KneeStudyDataset,
    build_data_loader,
    build_series_index,
)
from knee_abnormality.inference import make_prediction_frame
from knee_abnormality.labels import FoldLabels, load_fold_labels
from knee_abnormality.metrics import validation_metrics
from knee_abnormality.model import KneeAbnormalityModel
from knee_abnormality.preprocessing_identity import preprocessing_identity


def build_optimizer(model: KneeAbnormalityModel, settings: OptimizerConfig) -> AdamW:
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate = (
            settings.backbone_lr if name.startswith("backbone.") else settings.head_lr
        )
        no_decay = (
            parameter.ndim < 2 or name.endswith("cls_token") or "type_embedding" in name
        )
        weight_decay = 0.0 if no_decay else settings.weight_decay
        groups.setdefault((learning_rate, weight_decay), []).append(parameter)
    return AdamW(
        [
            {"params": parameters, "lr": lr, "weight_decay": decay}
            for (lr, decay), parameters in groups.items()
        ],
        betas=(settings.beta1, settings.beta2),
        eps=settings.epsilon,
    )


def masked_binary_cross_entropy_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    if not observed.any():
        raise RuntimeError("Training batch has no observed targets")
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return losses[observed].mean()


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
            True,
            self.config.seed,
        )
        self.validation_dataset = KneeStudyDataset(
            validation,
            index,
            self.config.data,
            self.labels.targets,
            False,
            self.config.seed,
            self.config.data.validation_image_multiplier,
        )

    def _loader(self, dataset: KneeStudyDataset, shuffle: bool) -> DataLoader:
        return build_data_loader(dataset, self.config.trainer, shuffle=shuffle)

    def train_dataloader(self) -> DataLoader:
        module = getattr(self.trainer, "lightning_module", None)
        if module is not None:
            self.train_dataset.set_epoch(
                max(module.sampling_epoch, self.trainer.current_epoch)
            )
        return self._loader(self.train_dataset, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.validation_dataset, False)


class BaselineModule(L.LightningModule):
    def __init__(
        self,
        config: ExperimentConfig,
        targets: tuple[str, ...],
        fold: int,
        training_studies: int,
        validation_studies: int,
        prediction_dir: Path,
        load_pretrained: bool | None = None,
        input_identity: dict | None = None,
    ) -> None:
        super().__init__()
        if config.model.checkpoint_backbone and not config.model.backbone_eval:
            raise ValueError(
                "Training with activation checkpointing requires backbone_eval=true; "
                "disable checkpoint_backbone to preserve training-mode BatchNorm"
            )
        self.config_model = config
        self.sampling_epoch = 0
        self.targets = targets
        self.validation_studies = validation_studies
        self.prediction_dir = prediction_dir
        self.model = KneeAbnormalityModel(
            config.model, config.data, len(targets), load_pretrained=load_pretrained
        )
        self.validation_rows: list[
            tuple[list[str], np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self.save_hyperparameters(
            {
                "config": config.model_dump(mode="json"),
                "targets": list(targets),
                "fold": fold,
                "training_studies": training_studies,
                "validation_studies": validation_studies,
                "input_identity": input_identity,
                "preprocessing_identity": preprocessing_identity(),
            }
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.sampling_epoch = checkpoint_sampling_epoch(checkpoint)

    def on_train_epoch_start(self) -> None:
        self.sampling_epoch = self.current_epoch
        dataset = getattr(self.trainer.train_dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    def training_step(self, batch: dict, batch_index: int) -> torch.Tensor:
        logits = self(batch).to(torch.float32)
        loss = masked_binary_cross_entropy_with_logits(
            logits, batch["target"], batch["target_observed"]
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

    def on_validation_epoch_start(self) -> None:
        self.validation_rows.clear()

    def validation_step(self, batch: dict, batch_index: int) -> None:
        prediction = torch.sigmoid(self(batch).to(torch.float32))
        self.validation_rows.append(
            (
                batch["study_uid"],
                batch["target"].cpu().numpy(),
                prediction.cpu().numpy(),
                batch["is_gold"].cpu().numpy(),
            )
        )

    def on_validation_epoch_end(self) -> None:
        study_ids = [study for ids, _, _, _ in self.validation_rows for study in ids]
        targets = np.concatenate([values for _, values, _, _ in self.validation_rows])
        predictions = np.concatenate(
            [values for _, _, values, _ in self.validation_rows]
        )
        is_gold = np.concatenate([values for _, _, _, values in self.validation_rows])
        if len(study_ids) != self.validation_studies or len(set(study_ids)) != len(
            study_ids
        ):
            raise RuntimeError(
                "Validation epoch did not gather every study exactly once"
            )
        result = validation_metrics(targets, predictions, self.targets, is_gold)
        self.log("val_macro_auc", result["macro_auc"], prog_bar=True, sync_dist=False)
        self.log("val_soft_target_bce", result["soft_target_bce"], sync_dist=False)
        if result["gold"]["macro_auc"] is not None:
            self.log("val_gold_macro_auc", result["gold"]["macro_auc"], sync_dist=False)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        make_prediction_frame(study_ids, predictions, self.targets).write_csv(
            self.prediction_dir / f"validation_epoch_{self.current_epoch:03d}.csv"
        )
        (
            self.prediction_dir / f"metrics_epoch_{self.current_epoch:03d}.json"
        ).write_text(json.dumps(result, indent=2) + "\n")

    def configure_optimizers(self):
        settings = self.config_model.optimizer
        optimizer = build_optimizer(self.model, settings)
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = max(
            1, math.ceil(total_steps / self.config_model.trainer.max_epochs)
        )
        warmup_steps = settings.warmup_epochs * steps_per_epoch

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return 0.1 + 0.9 * step / warmup_steps
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
