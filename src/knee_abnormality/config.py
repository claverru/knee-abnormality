import json
import math
import sys
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
from pydantic_settings import CliPositionalArg, SettingsError

from knee_abnormality.cli import CLISettings

SLICES_PER_INPUT = 3
TEST_IMAGE_MULTIPLIER = 2
ATTENTION_TEMPERATURE = 1.0
INFERENCE_BATCH_SIZE = 4
INFERENCE_NUM_WORKERS = 16

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Fold = Annotated[StrictInt, Field(ge=0)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


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


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(ConfigModel):
    data_root: Path
    output_root: Path
    label_manifest: Path

    @field_validator("data_root", "output_root", "label_manifest", mode="before")
    @classmethod
    def resolve_path(cls, value: object, info: ValidationInfo) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("path must be nonempty")
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            base = (info.context or {}).get("config_dir", Path.cwd())
            return (path if path.is_absolute() else base / path).resolve()
        return value


class LabelsConfig(ConfigModel):
    source: Identifier


class SeriesTypeConfig(ConfigModel):
    plane: Literal["Axial", "Coronal", "Sagittal"]
    fluid_sensitive: Literal[0, 1]
    fat_suppression: Literal[0, 1]
    max_images: Annotated[StrictInt, Field(gt=0)]


class IntensityAugmentationConfig(ConfigModel):
    probability: Annotated[float, Field(ge=0, le=1)] = 0.0
    gamma_range: PositiveFiniteRange = (1.0, 1.0)
    contrast_range: PositiveFiniteRange = (1.0, 1.0)


class AffineAugmentationConfig(ConfigModel):
    probability: Annotated[float, Field(ge=0, le=1)] = 0.0
    rotation_degrees: Annotated[float, Field(ge=0, le=180)] = 0.0
    translation_fraction: Annotated[float, Field(ge=0, le=1)] = 0.0
    scale_range: PositiveFiniteRange = (1.0, 1.0)


class NoiseAugmentationConfig(ConfigModel):
    probability: Annotated[float, Field(ge=0, le=1)] = 0.0
    std_range: PositiveFiniteRange = (0.005, 0.02)


class ReflectionAugmentationConfig(ConfigModel):
    probability: Annotated[float, Field(ge=0, le=1)] = 0.0


class DataConfig(ConfigModel):
    reflection_augmentation: ReflectionAugmentationConfig = Field(
        default_factory=ReflectionAugmentationConfig
    )
    series_types: Annotated[tuple[SeriesTypeConfig, ...], Field(min_length=1)]
    sampling_sigma_mm: PositiveFiniteFloat = 80.0
    validation_image_multiplier: Annotated[StrictInt, Field(gt=0)] = 1
    robust_center_outlier_mm: PositiveFiniteFloat | None
    intensity_augmentation: IntensityAugmentationConfig = Field(
        default_factory=IntensityAugmentationConfig
    )
    affine_augmentation: AffineAugmentationConfig = Field(
        default_factory=AffineAugmentationConfig
    )
    noise_augmentation: NoiseAugmentationConfig = Field(
        default_factory=NoiseAugmentationConfig
    )
    spacing_mm: Annotated[float, Field(gt=0)]
    image_size: Annotated[StrictInt, Field(gt=0)]
    lower_quantile: Annotated[float, Field(ge=0, le=1)]
    upper_quantile: Annotated[float, Field(ge=0, le=1)]
    mean: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    std: tuple[PositiveFiniteFloat, PositiveFiniteFloat, PositiveFiniteFloat]

    @model_validator(mode="after")
    def triplet_inputs(self) -> "DataConfig":
        if any(kind.max_images % SLICES_PER_INPUT for kind in self.series_types):
            raise ValueError(
                "Triplet max_images limits count channel appearances and must be multiples of three"
            )
        return self

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


class ModelConfig(ConfigModel):
    backbone: Identifier
    pretrained: bool
    channels_last: bool = False
    checkpoint_backbone: bool = False
    backbone_eval: bool = False
    dropout: Annotated[float, Field(ge=0, lt=1)]
    series_dropout: Annotated[float, Field(ge=0, lt=1)]

    @model_validator(mode="after")
    def checkpointing_requires_backbone_eval(self) -> "ModelConfig":
        if self.checkpoint_backbone and not self.backbone_eval:
            raise ValueError(
                "checkpoint_backbone requires backbone_eval=true to preserve training-mode BatchNorm"
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
    head_lr: Annotated[float, Field(gt=0)]
    weight_decay: Annotated[float, Field(ge=0)]
    beta1: Annotated[float, Field(gt=0, lt=1)]
    beta2: Annotated[float, Field(gt=0, lt=1)]
    epsilon: Annotated[float, Field(gt=0)]
    warmup_epochs: Annotated[StrictInt, Field(ge=0)]
    minimum_lr_ratio: Annotated[float, Field(gt=0, le=1)]
    weight_averaging: WeightAveragingConfig | None


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


class ValidationConfig(ConfigModel):
    split: Identifier
    folds: Annotated[tuple[Fold, ...], Field(min_length=1)]

    @field_validator("folds")
    @classmethod
    def distinct_folds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("folds must be distinct")
        return value


class ExperimentConfig(ConfigModel):
    name: Identifier
    seed: Annotated[StrictInt, Field(ge=0, lt=2**32)]
    paths: PathsConfig
    labels: LabelsConfig
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


def load_config(
    path: Path,
    *,
    data_root: Path | None = None,
    output_root: Path | None = None,
    label_manifest: Path | None = None,
    folds: list[int] | None = None,
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
    config = ExperimentConfig.model_validate(raw, context={"config_dir": path.parent})

    if (
        data_root is not None
        or output_root is not None
        or label_manifest is not None
        or folds is not None
    ):
        effective = config.model_dump()
        for key, value in (
            ("data_root", data_root),
            ("output_root", output_root),
            ("label_manifest", label_manifest),
        ):
            if value is not None:
                effective["paths"][key] = value.expanduser().resolve()
        if folds is not None:
            effective["validation"]["folds"] = folds
        config = ExperimentConfig.model_validate(effective)
    return config


class ConfigCLI(CLISettings):
    config: CliPositionalArg[Path] = Field(description="Experiment YAML file")
    data_root: Path | None = Field(
        default=None, description="Data location, relative to the working directory"
    )
    output_root: Path | None = Field(
        default=None, description="Output location, relative to the working directory"
    )
    folds: list[Fold] | None = Field(
        default=None, description="Saved folds to run, e.g. 0,1,2,3,4"
    )


def main() -> None:
    try:
        args = ConfigCLI()
        config = load_config(
            args.config,
            data_root=args.data_root,
            output_root=args.output_root,
            folds=args.folds,
        )
    except (OSError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        json.dumps(
            {
                "config_file": str(args.config.expanduser().resolve()),
                **config.model_dump(mode="json"),
                "split_dir": str(config.split_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
