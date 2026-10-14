import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytorch_lightning as L
import torch
from timm.models import group_parameters
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from knee_abnormality.config import ExperimentConfig, OptimizerConfig
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader
from knee_abnormality.data.labels import FoldLabels, distilled, load_fold_labels
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.evaluation.metrics import validation_metrics
from knee_abnormality.evaluation.predictions import make_prediction_frame
from knee_abnormality.identity import preprocessing_identity
from knee_abnormality.modeling.augmentation import AugmentationActivity
from knee_abnormality.modeling.model import (
    FRAME_ROTATION_RANGE_RAD,
    FRAME_ROTATIONS,
    FRAME_SIZES,
    POSITION_AXES,
    ROUTES,
    ROUTING_WIDTHS,
    TARGET_ROUTES,
    KneeAbnormalityModel,
)
from knee_abnormality.training.anatomy import (
    ANATOMY_CLASSES,
    ANATOMY_POINTS,
    MENISCUS_POINTS,
    SOFT_TISSUE_CLASSES,
    AnatomyTargets,
    anatomy_loss,
)
from knee_abnormality.training.checkpoints import checkpoint_sampling_epoch

PRECISION = "bf16-mixed"
GRADIENT_DIAGNOSTIC_INTERVAL = 10
HEIGHT_TARGET_LIMIT_MM = 80.0
JOINT_PLANE_ABOVE_PLATEAU_MM = 6.4
SIDE_LOSS_WEIGHT = 0.1
EARLY_ANCHOR_PREFIXES = ("conv1.", "bn1.", "layer1.", "stem.", "stages.0.")
LR_DECAYS = {
    "cosine": lambda progress: 0.5 * (1 + math.cos(math.pi * progress)),
    "linear": lambda progress: 1 - progress,
}


def build_optimizer(
    model: KneeAbnormalityModel,
    settings: OptimizerConfig,
    anchored_backbone_names: frozenset[str],
) -> AdamW:
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    stages = group_parameters(model.backbone, model.backbone.group_matcher(coarse=True))
    deepest = max(
        index
        for index, names in stages.items()
        if not all(name.startswith("head.") for name in names)
    )
    depth = {
        f"backbone.{parameter_name}": max(deepest - index, 0)
        for index, names in stages.items()
        for parameter_name in names
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate = (
            settings.backbone_lr * settings.backbone_lr_stage_decay ** depth[name]
            if name.startswith("backbone.")
            else settings.head_lr
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
        self.middle_only = config.model.slice_channels == "repeated_middle"
        self.labels: FoldLabels | None = None
        self.train_dataset: KneeStudyDataset | None = None
        self.validation_dataset: KneeStudyDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.labels is not None:
            return
        self.labels = distilled(self.config, load_fold_labels(self.config, self.fold))
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
            middle_only=self.middle_only,
        )

    def val_dataloader(self) -> DataLoader:
        return build_data_loader(
            self.validation_dataset,
            self.config.trainer,
            self.config.trainer.batch_size,
            middle_only=self.middle_only,
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
        self.train_sampling_sources = Counter()
        self.train_fit_centre_outside_slab_sources = 0
        self.train_feature_statistics_applied_clamped_channels = 0.0
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
        self.anatomy_targets = AnatomyTargets(
            config.paths.anatomy_targets, len(ANATOMY_CLASSES), ANATOMY_POINTS
        )
        self.meniscus_targets = AnatomyTargets(
            config.paths.meniscus_targets, len(SOFT_TISSUE_CLASSES), MENISCUS_POINTS
        )
        self.validation_rows: list[
            tuple[list[str], np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self.epoch_totals: dict[bool, dict[str, list]] = {False: {}, True: {}}
        self.epoch_spreads: dict[bool, dict[tuple[str, str], list]] = {
            False: {},
            True: {},
        }
        self.validating = False
        self.view_names = tuple(
            name
            for index, name in enumerate(config.data.view_names)
            if config.data.view_sources[index] == index
        )
        self.shared_view_names = tuple(
            (index, name)
            for index, name in enumerate(config.data.view_names)
            if config.data.view_sources[index] != index
        )
        self.series_type_names = tuple(
            f"{kind.plane.lower()}_{'fluid' if kind.fluid_sensitive else 't1pd'}"
            for kind in config.data.series_types
        )
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
        self.scheduled_anchors: set[str] = set()
        self.backbone_origin = {}
        early_anchors = 0
        for index, (name, parameter) in enumerate(backbone.named_parameters()):
            if parameter.ndim < 2:
                continue
            early = name.startswith(EARLY_ANCHOR_PREFIXES)
            early_anchors += early
            alpha = settings.anchor_alpha_early if early else settings.anchor_alpha_late
            if alpha > 0:
                buffer_name = f"anchor_weight_{index}"
                self.register_buffer(buffer_name, parameter.detach().float().clone())
                self.anchor_parameters.append((name, parameter, buffer_name, alpha))
                if settings.late_anchor_follows_lr and not early:
                    self.scheduled_anchors.add(buffer_name)
            else:
                self.backbone_origin[name] = parameter.detach().float().cpu().clone()
        if settings.anchor_alpha_early > 0 and not early_anchors:
            raise ValueError(
                "optimizer.anchor_alpha_early matches no backbone layer; the early "
                "anchor is defined by the stem and first stage names"
            )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.sampling_epoch = checkpoint_sampling_epoch(checkpoint)

    def on_train_epoch_start(self) -> None:
        self.train_window_rows.clear()
        self.train_sampling_sources.clear()
        self.train_fit_centre_outside_slab_sources = 0
        self.train_feature_statistics_applied_clamped_channels = 0.0
        self.sampling_epoch = self.current_epoch
        dataset = getattr(self.trainer.train_dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    def on_train_epoch_end(self) -> None:
        for region, scale in zip(
            ("medial", "lateral", "pf"),
            self.model.oa_detector_scale.detach().float(),
            strict=True,
        ):
            self._log_epoch_metric(f"oa_detector_scale_{region}", scale, 1)
        for outcome, count in sorted(self.train_sampling_sources.items()):
            self._log_epoch_metric(f"sampling_sources_{outcome}", float(count), 1)
        self._log_epoch_metric(
            "sampling_fit_centre_outside_slab_sources",
            float(self.train_fit_centre_outside_slab_sources),
            1,
        )
        self._log_epoch_metric(
            "slab_rule_and_positive_std_activity",
            torch.as_tensor(
                self.train_feature_statistics_applied_clamped_channels,
                device=self.device,
            ).clamp_max(float(self.train_fit_centre_outside_slab_sources)),
            1,
        )
        self._log_window_stats("", self.train_window_rows)
        for name, bias in zip(
            self.view_names,
            self.model.view_label_bias[: len(self.view_names)].detach().float(),
            strict=True,
        ):
            for channel, value in zip(self.model.channels, bias, strict=True):
                self._log_epoch_metric(
                    f"view_label_bias_{name}_{channel.lower().replace(' ', '_')}",
                    value,
                    1,
                )
        self._log_epoch_metric(
            "plane_readout_drift", self.model.head.divergence().mean(), 1
        )
        for name, width in zip(
            ROUTING_WIDTHS, self.model.routing_widths().detach(), strict=True
        ):
            self._log_epoch_metric(f"routing_width_{name}_mm", width, 1)
        for channel, scale, routed in zip(
            self.model.channels,
            self.model.knee_level_scale.detach(),
            self.model.routed_channel,
            strict=True,
        ):
            if routed:
                self._log_epoch_metric(
                    f"knee_level_scale_{channel.lower().replace(' ', '_')}", scale, 1
                )
        for name, value in zip(
            FRAME_ROTATIONS,
            torch.rad2deg(
                FRAME_ROTATION_RANGE_RAD
                * torch.tanh(self.model.frame_rotation_logit.detach().float())
            ),
            strict=True,
        ):
            self._log_epoch_metric(f"frame_rotation_{name}_deg", value, 1)
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
        self._flush_epoch_metrics()

    def _log_epoch_metric(
        self,
        name: str,
        value: torch.Tensor | float,
        batch_size: int,
        reduce_fx: str = "mean",
    ) -> None:
        weight = float(batch_size) if reduce_fx == "mean" else 1.0
        entry = self.epoch_totals[self.validating].setdefault(
            name, [0.0, 0.0, reduce_fx, [], []]
        )
        if isinstance(value, torch.Tensor):
            entry[3].append(value.detach())
            entry[4].append(weight)
        else:
            entry[0] += float(value) * weight
        entry[1] += weight

    def _flush_epoch_metrics(self) -> None:
        totals = self.epoch_totals[self.validating]
        for name, (total, weight, reduce_fx, values, weights) in totals.items():
            if values:
                stacked = torch.stack([value.float().reshape(()) for value in values])
                total = total + (stacked * stacked.new_tensor(weights)).sum()
            self.log(
                name,
                total / weight if reduce_fx == "mean" else total,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
        totals.clear()
        spreads = self.epoch_spreads[self.validating]
        for (name, suffix), (total, square, count) in spreads.items():
            self.log(
                f"{name}{suffix}",
                (square / count - (total / count).square()).clamp_min(0).sqrt(),
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
        spreads.clear()

    def _log_epoch_spread(
        self, name: str, values: torch.Tensor, suffix: str = "_std"
    ) -> None:
        values = values.detach().float()
        entry = self.epoch_spreads[self.validating].setdefault(
            (name, suffix), [0.0, 0.0, 0]
        )
        entry[0] = entry[0] + values.sum()
        entry[1] = entry[1] + values.square().sum()
        entry[2] += values.numel()

    def _log_crop_frames(self, encoded: dict, prefix: str) -> None:
        studies = encoded["batch_size"]
        member = encoded["study"]
        angles = torch.rad2deg(encoded["crop_angle"].detach())
        applied = (
            torch.zeros(studies, device=angles.device).index_add(
                0, member, (angles.abs() > 0).float()
            )
            > 0
        )
        self._log_epoch_metric(
            f"{prefix}crop_turned_study_fraction", applied.float().mean(), studies
        )
        moved = encoded["image_plane"] > 0
        if moved.any():
            self._log_epoch_metric(
                f"{prefix}crop_shift_abs_px",
                encoded["crop_shift"].detach().float()[moved].abs().mean(),
                int(moved.sum()),
            )
        for plane, name in ((1, "coronal"), (2, "sagittal")):
            selected = encoded["image_plane"] == plane
            counts = torch.zeros(studies, device=angles.device).index_add(
                0, member[selected], torch.ones_like(angles[selected])
            )
            total = torch.zeros_like(counts).index_add(
                0, member[selected], angles[selected]
            )
            absolute = torch.zeros_like(counts).index_add(
                0, member[selected], angles[selected].abs()
            )
            present = counts > 0
            number = int(present.sum())
            if number:
                values = (total / counts.clamp_min(1))[present]
                metric = f"{prefix}crop_turn_{name}_deg"
                self._log_epoch_metric(metric, values.mean(), number)
                self._log_epoch_spread(metric, values)
                self._log_epoch_metric(
                    f"{prefix}crop_turn_{name}_abs_deg",
                    (absolute / counts.clamp_min(1))[present].mean(),
                    number,
                )
                for axis, label in enumerate(("across", "down")):
                    total_shift = torch.zeros_like(counts).index_add(
                        0, member[selected], encoded["crop_shift"][selected, axis]
                    )
                    values = (total_shift / counts.clamp_min(1))[present]
                    metric = f"{prefix}crop_shift_{name}_{label}_px"
                    self._log_epoch_metric(metric, values.mean(), number)
                    self._log_epoch_spread(metric, values)

    def _log_smart_crop_pose(self, model: KneeAbnormalityModel, encoded: dict) -> None:
        windows = model.view_windows
        view = encoded["view_index"]
        angle = encoded["crop_angle"].detach().float()
        shift = encoded["crop_shift"].detach().float()
        scale = encoded["crop_affine"][:, :2, :2].detach().norm(dim=-2).mean(-1)
        axial = encoded["image_plane"] == 0
        if axial.any():
            count = int(axial.sum())
            self._log_epoch_metric(
                "val_axial_crop_shift_abs_mm",
                (shift[axial].abs() * windows.pixel_spacing[view[axial], None]).mean(),
                count,
            )
            self._log_epoch_metric(
                "val_axial_crop_turn_abs_deg",
                torch.rad2deg(angle[axial]).abs().mean(),
                count,
            )
        counts = torch.zeros(encoded["batch_size"], device=scale.device).index_add(
            0, encoded["study"], torch.ones_like(scale)
        )
        total = torch.zeros_like(counts).index_add(0, encoded["study"], scale)
        applied = (total / counts.clamp_min(1))[counts > 0]
        self._log_epoch_metric(
            "val_knee_crop_scale_mean", applied.mean(), applied.numel()
        )
        self._log_epoch_spread("val_knee_crop_scale", applied - 1.0)
        at_limit = (
            (shift.abs() >= windows.shift_limit[view, None] - 1e-6).any(-1)
            | (angle.abs() >= windows.rotation_limit[encoded["image_type"]] - 1e-6)
            | (scale <= 0.87 + 1e-6)
            | (scale >= 1.13 - 1e-6)
        )
        self._log_epoch_metric(
            "val_crop_pose_limit_fraction", at_limit.float().mean(), scale.numel()
        )

    def _log_canvas_pixel_size(self, batch: dict) -> None:
        maps = batch["patient_maps"].detach().float()[:, :, 1]
        spacing = maps[..., :2].norm(dim=-2)
        valid = (
            batch["slice_mask"]
            & batch["geometry_valid"]
            & torch.isfinite(maps).all((-1, -2))
            & (spacing > 0).all(-1)
        )
        for view, name in enumerate(self.view_names):
            selected = valid & (batch["view_index"] == view)[:, None]
            count = int(selected.sum())
            if count:
                self._log_epoch_metric(
                    f"val_canvas_pixel_size_{name}_mm",
                    spacing[selected].mean(),
                    count,
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
            activity = self.model.feature_statistics_activity
            self.train_feature_statistics_applied_clamped_channels += (
                activity.applied_clamped_channels.float()
            )
            self._log_family_activity(
                "feature_statistics",
                activity,
                "mean_absolute_change",
            )
            self._log_epoch_metric(
                "feature_statistics_std_clamped_fraction",
                activity.clamped_channels.float() / activity.total_channels,
                activity.total_channels,
            )
            self._log_epoch_metric(
                "feature_statistics_std_applied_clamped_channels",
                activity.applied_clamped_channels.float(),
                1,
                "sum",
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
                ("alignment_change", mixing.mean_alignment_change),
            ):
                self._log_epoch_metric(f"contrast_mixing_{name}", value, study_count)

    def _log_views(self, batch: dict, encoded: dict, prefix: str) -> None:
        centre_without_side = (batch["geometry_valid"] & batch["slice_mask"]).any(
            -1
        ) & (batch["knee_side_sign"] == 0)
        centre_without_side_counts = torch.zeros(
            len(batch["study_uid"]), device=self.device
        )
        centre_without_side_counts.scatter_add_(
            0, batch["series_study_index"], centre_without_side.float()
        )
        centre_without_side_studies = centre_without_side_counts > 0
        detail_views = (
            self.model.detail_view[batch["view_index"]]
            & ~self.model.shared_slice_view[batch["view_index"]]
            & batch["slice_mask"].any(-1)
        )
        for name, value in (
            ("studies", centre_without_side_studies.sum()),
            (
                "detail_views",
                (
                    detail_views
                    & centre_without_side_studies[batch["series_study_index"]]
                ).sum(),
            ),
            ("anchored_detail_views", (detail_views & centre_without_side).sum()),
        ):
            self._log_epoch_metric(
                f"{prefix}centre_without_side_{name}", value.float(), 1, "sum"
            )
        valid = (
            encoded["visual_valid"].any(-1)
            & ~self.model.shared_slice_view[encoded["view_index"]]
        )
        count = int(valid.sum())
        mass = self.model.attention_mass_view[: len(self.view_names)].mean(1)
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
        if prefix:
            for view, name in self.shared_view_names:
                values = self.model.attention_mass_view[view].mean(0)
                for label, value in zip(self.targets, values, strict=True):
                    if label in (*TARGET_ROUTES, "ACL"):
                        self._log_epoch_metric(
                            f"val_attention_mass_{name}_{label.lower().replace(' ', '_')}",
                            value,
                            len(batch["study_uid"]),
                        )
        selected = (
            self.model.detail_view[batch["view_index"]]
            & ~self.model.shared_slice_view[batch["view_index"]]
            & batch["slice_mask"].any(-1)
        )
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

    def _record_sampling_sources(self, batch: dict) -> None:
        self._log_epoch_metric(
            "preparation_failures",
            float(sum(len(failures) for failures in batch["failures"])),
            1,
            "sum",
        )
        for sampling in batch["sampling"]:
            sources = {
                uid: (outcome, metadata["reference_method"])
                for metadata in sampling
                for uid, outcome in metadata["sources"].items()
            }
            self.train_sampling_sources.update(
                outcome for outcome, _ in sources.values()
            )
            self.train_fit_centre_outside_slab_sources += sum(
                outcome == "centre_outside_slab" and method == "fit"
                for outcome, method in sources.values()
            )

    def training_step(self, batch: dict, batch_index: int) -> torch.Tensor:
        self._record_sampling_sources(batch)
        encoded = self.model.encode_batch(batch)
        self._log_crop_frames(encoded, "")
        self._log_augmentation_activity(len(batch["study_uid"]))
        logits = self.model.pool_encoded(encoded).to(torch.float32)
        self._log_views(batch, encoded, "")
        self._log_epoch_metric(
            "knee_level_logit_abs_mean",
            self.model.knee_level_logit_abs_mean,
            len(batch["study_uid"]),
        )
        loss = masked_binary_cross_entropy_with_logits(
            logits,
            batch["target"],
            batch["target_observed"],
            self.class_balance_weights,
        )
        loss = loss + SIDE_LOSS_WEIGHT * self._side_step(self.model, batch, encoded, "")
        loss = loss + self._anatomy_step(self.model, encoded, batch, "")
        loss = loss + self._meniscus_step(encoded, batch, "")
        if self.anchor_parameters:
            backbone_group = self.trainer.optimizers[0].param_groups[
                self.backbone_optimizer_group_index
            ]
            lr_ratio = backbone_group["lr"] / backbone_group["initial_lr"]
            anchor_penalty = sum(
                alpha
                * lr_ratio ** (buffer_name in self.scheduled_anchors)
                * (parameter.float() - getattr(self, buffer_name)).square().sum()
                for _, parameter, buffer_name, alpha in self.anchor_parameters
            )
            loss = loss + anchor_penalty
            self._log_epoch_metric(
                "anchor_penalty", anchor_penalty, len(batch["study_uid"])
            )
        teacher = self.trainer.datamodule.labels.provenance.get("teacher")
        if teacher is not None:
            self._log_epoch_metric(
                "teacher_target_shift",
                teacher["target_shift"],
                len(batch["study_uid"]),
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

    def _side_step(
        self, model: KneeAbnormalityModel, batch: dict, encoded: dict, prefix: str
    ) -> torch.Tensor:
        studies = len(batch["study_uid"])
        logit = encoded["side_logit"].float()
        present = encoded["side_present"]
        target = batch["knee_side_left"] ^ encoded["mirrored_study"]
        correct = ((logit > 0) == target).float()
        certain = batch["knee_side_certain"] & present
        coordinates = batch["knee_side_coordinates"]
        for name, selected, values in (
            ("side_head_accuracy", certain, correct),
            ("side_head_coordinate_agreement", coordinates & present, correct),
            (
                "side_probability_coordinate_agreement",
                coordinates,
                (
                    (encoded["study_left_probability"] > 0.5) == batch["knee_side_left"]
                ).float(),
            ),
        ):
            count = int(selected.sum())
            if count:
                self._log_epoch_metric(
                    f"{prefix}{name}", values[selected].mean(), count
                )
        for name, selected in (
            ("certain", batch["knee_side_certain"]),
            ("coordinates", coordinates),
            ("none", ~(batch["knee_side_certain"] | coordinates)),
        ):
            self._log_epoch_metric(
                f"{prefix}side_state_{name}_share", selected.float().mean(), studies
            )
        axis = encoded["study_axis"]
        self._log_epoch_metric(
            f"{prefix}knee_rotation_abs_deg",
            torch.rad2deg(torch.atan2(axis[:, 1], axis[:, 0])).abs().mean(),
            studies,
        )
        study_angles = model.rotation_angles(encoded["study_frame_rotation"].detach())
        for number, name in enumerate(FRAME_ROTATIONS):
            self._log_epoch_spread(
                f"{prefix}study_frame_{name}_deg", study_angles[:, number]
            )
            self._log_epoch_metric(
                f"{prefix}study_frame_{name}_deg",
                study_angles[:, number].mean(),
                studies,
            )
            self._log_epoch_metric(
                f"{prefix}study_frame_{name}_abs_deg",
                study_angles[:, number].abs().mean(),
                studies,
            )
        study_sizes = encoded["study_frame_sizes"].detach().float()
        for number, name in enumerate(FRAME_SIZES):
            self._log_epoch_spread(
                f"{prefix}study_frame_{name}_mm", study_sizes[:, number]
            )
            self._log_epoch_metric(
                f"{prefix}study_frame_{name}_mm", study_sizes[:, number].mean(), studies
            )
        study_shift = encoded["study_frame_shift"].detach().float()
        for number, name in enumerate(POSITION_AXES):
            self._log_epoch_spread(
                f"{prefix}study_frame_shift_{name}_mm", study_shift[:, number]
            )
            self._log_epoch_metric(
                f"{prefix}study_frame_shift_{name}_mm",
                study_shift[:, number].mean(),
                studies,
            )
            self._log_epoch_metric(
                f"{prefix}study_frame_shift_{name}_abs_mm",
                study_shift[:, number].abs().mean(),
                studies,
            )
        valid = encoded["visual_valid"]
        tokens = int(valid.sum())
        for route, value in zip(ROUTES, model.routing_mean, strict=True):
            self._log_epoch_metric(f"{prefix}routing_weight_{route}", value, tokens)
        if not certain.any():
            return logit.sum() * 0
        loss = F.binary_cross_entropy_with_logits(
            logit[certain], target[certain].float()
        )
        self._log_epoch_metric(f"{prefix}side_head_loss", loss, int(certain.sum()))
        return loss

    def _meniscus_step(self, encoded: dict, batch: dict, prefix: str) -> torch.Tensor:
        rows = self.meniscus_targets.token_rows(
            batch["study_uid"],
            encoded["study"],
        )
        shares, covered = self.meniscus_targets.token_shares(
            rows,
            encoded["meniscus_patient_position"],
            encoded["meniscus_token_steps"],
        )
        mask = covered & encoded["meniscus_visual_valid"] & encoded["routable"][:, None]
        loss, metrics = anatomy_loss(
            encoded["meniscus_scores"], shares, mask, SOFT_TISSUE_CLASSES
        )
        tokens = int(mask.sum())
        if tokens:
            self._log_epoch_metric(f"{prefix}meniscus_loss", loss, tokens)
            for name, value in metrics.items():
                self._log_epoch_metric(f"{prefix}meniscus_{name}", value, tokens)
        self._log_epoch_metric(f"{prefix}meniscus_tokens", float(tokens), 1, "sum")
        self._log_soft_tissue_routing(encoded, prefix)
        return self.config.trainer.meniscus_loss_weight * loss

    def _log_soft_tissue_routing(self, encoded: dict, prefix: str) -> None:
        stats = self.model.soft_tissue_routing_stats
        studies = encoded["batch_size"]
        for number, target in enumerate(TARGET_ROUTES):
            label = target.lower().replace(" ", "_")
            for name in ("open_cells", "weight", "routed", "logit_abs"):
                self._log_epoch_metric(
                    f"{prefix}soft_tissue_{name}_{label}",
                    stats[name][:, number].mean(),
                    studies,
                )
        for number, name in enumerate(("medial", "lateral", "kneecap")):
            self._log_epoch_metric(
                f"{prefix}soft_tissue_compartment_found_{name}",
                stats["compartment_found"][:, number].mean(),
                studies,
            )

    def _anatomy_step(
        self, model, encoded: dict, batch: dict, prefix: str
    ) -> torch.Tensor:
        rows = self.anatomy_targets.token_rows(
            batch["study_uid"],
            encoded["study"],
        )
        shares, covered = self.anatomy_targets.token_shares(
            rows,
            encoded["patient_position"],
            encoded["token_steps"],
        )
        mask = covered & encoded["visual_valid"] & encoded["routable"][:, None]
        loss, metrics = anatomy_loss(
            encoded["anatomy_scores"].view(*mask.shape, -1),
            shares,
            mask,
            ANATOMY_CLASSES,
        )
        tokens = int(mask.sum())
        if tokens:
            self._log_epoch_metric(f"{prefix}anatomy_loss", loss, tokens)
            for name, value in metrics.items():
                self._log_epoch_metric(f"{prefix}anatomy_{name}", value, tokens)
        self._log_epoch_metric(f"{prefix}anatomy_tokens", float(tokens), 1, "sum")
        self._log_epoch_metric(
            f"{prefix}anatomy_centre_displacement_mm",
            encoded["anatomy_centre_delta_mm"].norm(dim=-1).mean(),
            encoded["batch_size"],
        )
        self._log_epoch_metric(
            f"{prefix}anatomy_centre_applied_fraction",
            encoded["anatomy_centre_applied"].float().mean(),
            encoded["batch_size"],
        )
        centre, known_centre = self.anatomy_targets.study_centres(batch["study_uid"])
        member = encoded["study"]
        offset = centre[member] - encoded["patient_position"][:, 0]
        cosine, sine = encoded["study_axis"][member].unbind(-1)
        sign = 1 - 2 * encoded["left_probability"].detach().float()
        truth_centre = encoded["fitted_position"][:, 0, :2] + torch.stack(
            (
                sign * (offset[:, 0] * cosine + offset[:, 1] * sine),
                offset[:, 0] * sine - offset[:, 1] * cosine,
            ),
            -1,
        )
        valid = encoded["routable"] & encoded["visual_valid"].any(-1)
        counts = torch.zeros(encoded["batch_size"], device=self.device).index_add(
            0, member, valid.float()
        )
        errors = (
            torch.zeros_like(centre[:, :2]).index_add(
                0,
                member,
                torch.where(
                    valid[:, None],
                    encoded["frame_shift"][:, :2].detach() - truth_centre,
                    0.0,
                ),
            )
            / counts.clamp_min(1)[:, None]
        )
        seen_centre = known_centre & (counts > 0)
        if seen_centre.any():
            for axis, name in enumerate(("across", "front")):
                values = errors[seen_centre, axis]
                metric = f"{prefix}anatomy_centre_error_{name}_mm"
                self._log_epoch_spread(metric, values, suffix="")
                self._log_epoch_metric(f"{metric}_mean", values.mean(), len(values))
        truth, known = self.anatomy_targets.token_heights(
            batch["study_uid"], encoded["study"], encoded["patient_position"]
        )
        seen = (
            known[:, None]
            & encoded["visual_valid"]
            & encoded["routable"][:, None]
            & (truth.abs() < HEIGHT_TARGET_LIMIT_MM)
        )
        if seen.any():
            self._log_epoch_metric(
                f"{prefix}anatomy_plane_error_mm",
                (
                    encoded["position"][..., 2].float()
                    - truth
                    + JOINT_PLANE_ABOVE_PLATEAU_MM
                )[seen]
                .abs()
                .mean(),
                int(seen.sum()),
            )
        return self.config.trainer.anatomy_loss_weight * loss

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.trainer.global_step % GRADIENT_DIAGNOSTIC_INTERVAL:
            return
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

    def on_validation_epoch_start(self) -> None:
        self.validating = True
        self.epoch_totals[True].clear()
        self.epoch_spreads[True].clear()
        for rows in self.view_validation_rows.values():
            rows.clear()
        self.view_missing_studies.update(dict.fromkeys(self.view_names, 0))
        self.validation_window_rows.clear()
        self.validation_window_studies.clear()
        self.validation_rows.clear()

    def validation_step(self, batch: dict, batch_index: int) -> None:
        model = self.model
        encoded = model.encode_batch(batch)
        self._log_canvas_pixel_size(batch)
        self._log_crop_frames(encoded, "val_")
        self._log_smart_crop_pose(model, encoded)
        prediction = torch.sigmoid(model.pool_encoded(encoded).to(torch.float32))
        self._log_views(batch, encoded, "val_")
        self._side_step(model, batch, encoded, "val_")
        self._anatomy_step(model, encoded, batch, "val_")
        self._meniscus_step(encoded, batch, "val_")
        for view, (view_name, rows) in enumerate(self.view_validation_rows.items()):
            view_prediction = torch.sigmoid(model.pool_encoded(encoded, view).float())
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
        self._flush_epoch_metrics()
        self.validating = False

    def configure_optimizers(self):
        settings = self.config.optimizer
        optimizer = build_optimizer(
            self.model,
            settings,
            frozenset(f"backbone.{name}" for name, _, _, _ in self.anchor_parameters),
        )
        backbone_parameter_ids = {
            id(parameter) for parameter in self.model.backbone.parameters()
        }
        self.backbone_optimizer_group_index = next(
            index
            for index, group in enumerate(optimizer.param_groups)
            if any(
                id(parameter) in backbone_parameter_ids for parameter in group["params"]
            )
        )
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = max(
            1, math.ceil(total_steps / self.config.trainer.max_epochs)
        )
        warmup_steps = settings.warmup_epochs * steps_per_epoch
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_steps

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                start = settings.warmup_start_ratio
                return start + (1 - start) * step / warmup_steps
            progress = min(
                1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)
            )
            return settings.minimum_lr_ratio + (
                1 - settings.minimum_lr_ratio
            ) * LR_DECAYS[settings.lr_decay](progress)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, schedule),
                "interval": "step",
            },
        }
