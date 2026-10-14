import json
import math
import secrets
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    CliPositionalArg,
    CliSettingsSource,
    SettingsConfigDict,
)

SLICES_PER_INPUT = 3
TEST_IMAGE_MULTIPLIER = 2
ATTENTION_TEMPERATURE = 1.0
INFERENCE_BATCH_SIZE = 4
INFERENCE_NUM_WORKERS = 8
INFERENCE_IO_THREADS = 8
INFERENCE_LOADER_NICENESS = 10
SERIES_LISTING_THREADS = 16
STUDY_HEADERS = Path(
    "outputs/analysis/s/sharing-disagreement-20260914/study_headers.parquet"
)

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Fold = Annotated[StrictInt, Field(ge=0)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Device = Literal["auto", "cpu", "cuda"]
RuntimePrecision = Literal["auto", "fp32", "fp16", "bf16"]


def validate_attention_temperature(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Attention temperature must be a positive finite float")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Attention temperature must be a positive finite float")
    return result


def ordered_range(value: tuple[float, float]) -> tuple[float, float]:
    if value[0] > value[1]:
        raise ValueError("augmentation range must be ordered")
    return value


PositiveFiniteRange = Annotated[
    tuple[PositiveFiniteFloat, PositiveFiniteFloat], AfterValidator(ordered_range)
]
UnitFloat = Annotated[float, Field(ge=0, le=1)]
UnitFloatRange = Annotated[tuple[UnitFloat, UnitFloat], AfterValidator(ordered_range)]
KeepFraction = Annotated[float, Field(gt=0, le=1)]
KeepFractionRange = Annotated[
    tuple[KeepFraction, KeepFraction], AfterValidator(ordered_range)
]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(ConfigModel):
    data_root: Path
    output_root: Path
    label_manifest: Path
    report_attribute_manifest: Path
    anatomy_targets: Path | None = None
    meniscus_targets: Path | None = None
    teacher_table: Path | None = None

    @field_validator(
        "data_root",
        "output_root",
        "label_manifest",
        "report_attribute_manifest",
        "anatomy_targets",
        "meniscus_targets",
        "teacher_table",
        mode="before",
    )
    @classmethod
    def resolve_path(cls, value: object, info: ValidationInfo) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("path must be nonempty")
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            base = (info.context or {}).get("config_dir", Path.cwd())
            return (path if path.is_absolute() else base / path).resolve()
        return value


class DistillationConfig(ConfigModel):
    teacher_share: UnitFloat = 0.0


class LabelsConfig(ConfigModel):
    source: Identifier


class ViewAxisConfig(ConfigModel):
    anchor: Literal["image", "knee"]
    offset_mm: FiniteFloat


class ViewConfig(ConfigModel):
    field_mm: PositiveFiniteFloat
    slices: Annotated[StrictInt, Field(gt=0)]
    horizontal: ViewAxisConfig
    vertical: ViewAxisConfig

    @property
    def shifted(self) -> bool:
        return any(
            axis.anchor == "knee" or axis.offset_mm != 0
            for axis in (self.horizontal, self.vertical)
        )


class SeriesTypeConfig(ConfigModel):
    plane: Literal["Axial", "Coronal", "Sagittal"]
    fluid_sensitive: Literal[0, 1]
    fat_suppression: Literal[0, 1]
    views: Annotated[tuple[ViewConfig, ...], Field(min_length=1)]

    @property
    def slices(self) -> int:
        return sum(view.slices for view in self.views)


class AugmentationConfig(ConfigModel):
    probability: UnitFloat = 0.0


class ReflectionAugmentationConfig(AugmentationConfig):
    pass


class ContrastMixingAugmentationConfig(AugmentationConfig):
    target_weight_range: UnitFloatRange = (0.5, 1.0)


class IntensityAugmentationConfig(AugmentationConfig):
    gamma_range: PositiveFiniteRange = (1.0, 1.0)
    contrast_range: PositiveFiniteRange = (1.0, 1.0)


class AffineAugmentationConfig(AugmentationConfig):
    rotation_degrees: Annotated[float, Field(ge=0, le=180)] = 0.0
    translation_fraction: UnitFloat = 0.0
    scale_range: PositiveFiniteRange = (1.0, 1.0)


class NoiseAugmentationConfig(AugmentationConfig):
    std_range: PositiveFiniteRange = (0.005, 0.02)


class SmoothingAugmentationConfig(AugmentationConfig):
    spatial_sigma: PositiveFiniteFloat = 1.0
    range_sigma_range: PositiveFiniteRange = (0.05, 0.1)


class ResolutionAugmentationConfig(AugmentationConfig):
    keep_fraction_range: KeepFractionRange = (0.70, 1.0)


class GinAugmentationConfig(AugmentationConfig):
    blend_max: Annotated[float, Field(gt=0, le=1)] = 0.5


class DataConfig(ConfigModel):
    reflection_augmentation: ReflectionAugmentationConfig = Field(
        default_factory=ReflectionAugmentationConfig
    )
    contrast_mixing_augmentation: ContrastMixingAugmentationConfig = Field(
        default_factory=ContrastMixingAugmentationConfig
    )
    series_types: Annotated[tuple[SeriesTypeConfig, ...], Field(min_length=1)]
    sampling_sigma_fraction: PositiveFiniteFloat = 0.867
    validation_image_multiplier: Annotated[StrictInt, Field(gt=0)] = 1
    study_slice_total: Annotated[StrictInt, Field(gt=0)]
    intensity_augmentation: IntensityAugmentationConfig = Field(
        default_factory=IntensityAugmentationConfig
    )
    affine_augmentation: AffineAugmentationConfig = Field(
        default_factory=AffineAugmentationConfig
    )
    noise_augmentation: NoiseAugmentationConfig = Field(
        default_factory=NoiseAugmentationConfig
    )
    smoothing_augmentation: SmoothingAugmentationConfig = Field(
        default_factory=SmoothingAugmentationConfig
    )
    resolution_augmentation: ResolutionAugmentationConfig = Field(
        default_factory=ResolutionAugmentationConfig
    )
    gin_augmentation: GinAugmentationConfig = Field(
        default_factory=GinAugmentationConfig
    )
    reference_field_mm: Annotated[float, Field(gt=0)]
    image_size: Annotated[StrictInt, Field(gt=0)]
    lower_quantile: Annotated[float, Field(ge=0, le=1)]
    upper_quantile: Annotated[float, Field(ge=0, le=1)]
    mean: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    std: tuple[PositiveFiniteFloat, PositiveFiniteFloat, PositiveFiniteFloat]

    @model_validator(mode="after")
    def shifts_inside_frame(self) -> "DataConfig":
        if any(
            view.shifted and view.field_mm >= self.reference_field_mm
            for _, view in self.views
        ):
            raise ValueError(
                "A view with an anchor or offset must be narrower than "
                "data.reference_field_mm, which bounds every window"
            )
        return self

    @property
    def reference_spacing_mm(self) -> float:
        return self.reference_field_mm / self.image_size

    @property
    def views(self) -> tuple[tuple[int, ViewConfig], ...]:
        return tuple(
            (slot, view)
            for slot, kind in enumerate(self.series_types)
            for view in kind.views
        )

    @property
    def view_sources(self) -> tuple[int, ...]:
        sources = []
        for kind in self.series_types:
            first = len(sources)
            for order, _ in enumerate(kind.views):
                shared = (
                    kind.plane == "Sagittal" and kind.fluid_sensitive == 1 and order > 0
                )
                sources.append(first if shared else len(sources))
        return tuple(sources)

    @property
    def view_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for slot, view in self.views:
            kind = self.series_types[slot]
            stem = (
                f"{kind.plane.lower()}_{'fluid' if kind.fluid_sensitive else 't1pd'}"
                f"_{round(view.field_mm)}mm"
            )
            repeats = sum(name.split("mm")[0] + "mm" == stem for name in names)
            names.append(f"{stem}_{repeats + 1}" if repeats else stem)
        return tuple(names)

    @field_validator("series_types")
    @classmethod
    def distinct_series_types(
        cls, value: tuple[SeriesTypeConfig, ...]
    ) -> tuple[SeriesTypeConfig, ...]:
        keys = [
            (item.plane, item.fluid_sensitive, item.fat_suppression) for item in value
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("series types must be distinct")
        return value

    @field_validator("upper_quantile")
    @classmethod
    def ordered_quantiles(cls, value: float, info: ValidationInfo) -> float:
        lower = info.data.get("lower_quantile")
        if lower is not None and value <= lower:
            raise ValueError("upper quantile must exceed lower quantile")
        return value


class FeatureStatisticsAugmentationConfig(AugmentationConfig):
    pass


class ModelConfig(ConfigModel):
    backbone: Identifier
    pretrained: bool
    pretrained_weights: Literal["timm", "radimagenet"] = "timm"
    slice_channels: Literal["adjacent", "repeated_middle"] = "adjacent"
    channels_last: bool = False
    checkpoint_backbone: bool = False
    backbone_eval: bool = False
    dropout: Annotated[float, Field(ge=0, lt=1)]
    stochastic_depth: Annotated[float, Field(ge=0, lt=1)] = 0.0
    series_dropout: Annotated[float, Field(ge=0, lt=1)]
    plane_readout: bool = False
    view_label_bias: bool = False
    anatomical_frame: bool = False
    meniscus_map: bool = False
    soft_tissue_routing: bool = False
    feature_statistics_augmentation: FeatureStatisticsAugmentationConfig = Field(
        default_factory=FeatureStatisticsAugmentationConfig
    )

    @model_validator(mode="after")
    def validate_model_options(self) -> "ModelConfig":
        if self.checkpoint_backbone and not self.backbone_eval:
            raise ValueError(
                "checkpoint_backbone requires backbone_eval=true to preserve training-mode BatchNorm"
            )
        if self.soft_tissue_routing and not (
            self.meniscus_map and self.anatomical_frame
        ):
            raise ValueError(
                "soft_tissue_routing requires meniscus_map and anatomical_frame"
            )
        return self


class EMAConfig(ConfigModel):
    method: Literal["ema"]
    decay: Annotated[float, Field(gt=0, lt=1)]


class SWAConfig(ConfigModel):
    method: Literal["swa"]
    epochs: Annotated[StrictInt, Field(gt=0)]


WeightAveragingConfig = Annotated[EMAConfig | SWAConfig, Field(discriminator="method")]


class OptimizerConfig(ConfigModel):
    backbone_lr: Annotated[float, Field(gt=0)]
    backbone_lr_stage_decay: Annotated[float, Field(gt=0, le=1)] = 1.0
    head_lr: Annotated[float, Field(gt=0)]
    weight_decay: Annotated[float, Field(ge=0)]
    beta1: Annotated[float, Field(gt=0, lt=1)]
    beta2: Annotated[float, Field(gt=0, lt=1)]
    epsilon: Annotated[float, Field(gt=0)]
    warmup_epochs: Annotated[StrictInt, Field(ge=0)]
    warmup_start_ratio: Annotated[float, Field(gt=0, le=1)]
    minimum_lr_ratio: Annotated[float, Field(gt=0, le=1)]
    lr_decay: Literal["cosine", "linear"] = "cosine"
    weight_averaging: WeightAveragingConfig | None
    anchor_alpha_early: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    anchor_alpha_late: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    late_anchor_follows_lr: bool = False


class TrainerConfig(ConfigModel):
    max_epochs: Annotated[StrictInt, Field(gt=0)]
    batch_size: Annotated[StrictInt, Field(gt=0)]
    accumulate_grad_batches: Annotated[StrictInt, Field(gt=0)]
    gradient_clip_val: Annotated[float, Field(gt=0)]
    num_workers: Annotated[StrictInt, Field(ge=0)]
    persistent_workers: bool
    prefetch_factor: Annotated[StrictInt, Field(gt=0)]
    patience: Annotated[StrictInt, Field(ge=0)]
    min_delta: Annotated[float, Field(ge=0)]
    deterministic: bool
    class_balance_exponent: Annotated[float, Field(ge=0, le=1)]
    report_attribute_loss_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    anatomy_loss_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    meniscus_loss_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    scanner_confusion_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ValidationConfig(ConfigModel):
    split: Identifier
    folds: Annotated[tuple[Fold, ...], Field(min_length=1)]
    prediction_image_multiplier: Annotated[StrictInt, Field(gt=0)] = 1
    unused_folds: tuple[Fold, ...] = ()

    @field_validator("folds", "unused_folds")
    @classmethod
    def distinct_folds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("folds must be distinct")
        return value

    @model_validator(mode="after")
    def unused_folds_are_not_validated(self) -> "ValidationConfig":
        if set(self.folds) & set(self.unused_folds):
            raise ValueError("unused_folds must not hold a validation fold")
        return self


class ExperimentConfig(ConfigModel):
    name: Identifier
    seed: Annotated[StrictInt, Field(ge=0, lt=2**32)] | None = None
    paths: PathsConfig
    labels: LabelsConfig
    distillation: DistillationConfig = DistillationConfig()
    validation: ValidationConfig
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig

    @model_validator(mode="after")
    def swa_window_fits_schedule(self) -> "ExperimentConfig":
        averaging = self.optimizer.weight_averaging
        if isinstance(averaging, SWAConfig) and averaging.epochs > (
            self.trainer.max_epochs - self.optimizer.warmup_epochs
        ):
            raise ValueError(
                "optimizer.weight_averaging.epochs must not exceed "
                "trainer.max_epochs minus optimizer.warmup_epochs"
            )
        return self

    @property
    def split_dir(self) -> Path:
        return self.paths.data_root / "folds" / self.validation.split

    @property
    def raw_dir(self) -> Path:
        return self.paths.data_root / "raw"


def merge(values: dict, changes: dict) -> dict:
    merged = dict(values)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: Path,
    *,
    data_root: Path | None = None,
    label_manifest: Path | None = None,
    overrides: dict | None = None,
) -> ExperimentConfig:
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if isinstance(raw, dict) and set(raw) == {
        "schema_version",
        "experiment",
        "targets",
        "fold",
        "inference",
    }:
        raw = raw["experiment"]
    if overrides:
        raw = merge(raw, overrides)
    config = ExperimentConfig.model_validate(raw, context={"config_dir": path.parent})

    if data_root is not None or label_manifest is not None:
        effective = config.model_dump()
        for key, value in (
            ("data_root", data_root),
            ("label_manifest", label_manifest),
        ):
            if value is not None:
                effective["paths"][key] = value.expanduser().resolve()
        config = ExperimentConfig.model_validate(effective)
    if config.seed is None:
        config = config.model_copy(update={"seed": secrets.randbits(32)})
    return config


def parse_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: parse_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [parse_value(item) for item in value]
    return yaml.safe_load(value) if isinstance(value, str) else value


def parse_experiment_arguments[T: BaseModel](options: type[T]) -> tuple[T, dict]:
    parser = type(
        f"{options.__name__}Parser",
        (BaseSettings, ExperimentConfig, options),
        {
            "model_config": SettingsConfigDict(
                cli_kebab_case=True,
                cli_implicit_flags=True,
                cli_hide_none_type=True,
                cli_avoid_json=True,
                json_schema_extra=options.model_config.get("json_schema_extra"),
            )
        },
    )
    given = CliSettingsSource(parser, cli_parse_args=True)()
    arguments = options.model_validate(
        {name: given.pop(name) for name in options.model_fields if name in given}
    )
    overrides = parse_value(given)
    for key, value in overrides.get("paths", {}).items():
        overrides["paths"][key] = str(Path(value).expanduser().resolve())
    return arguments, overrides


class ConfigOptions(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Print the effective experiment configuration. A dotted "
            "flag overrides that field of the YAML; the defaults and required marks "
            "shown are the schema's, not the YAML's"
        }
    )

    config: CliPositionalArg[Path] = Field(description="Experiment YAML file")


def main() -> None:
    args, overrides = parse_experiment_arguments(ConfigOptions)
    config = load_config(args.config, overrides=overrides)
    print(
        json.dumps(
            {
                "config_file": str(args.config.expanduser().resolve()),
                "overrides": overrides,
                **config.model_dump(mode="json"),
                "split_dir": str(config.split_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
