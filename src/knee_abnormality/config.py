import json
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

from knee_abnormality.cli import NonExitingCLISettings

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Fold = Annotated[StrictInt, Field(ge=0)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


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


class DataConfig(ConfigModel):
    series_types: Annotated[tuple[SeriesTypeConfig, ...], Field(min_length=1)]
    position_encoding: Literal["index", "physical_mm"] = "index"
    position_reference: Literal["origin", "center"] = "origin"
    acquisition_sampling: Literal["shortest", "pooled"] = "shortest"
    slice_sampling: Literal["uniform", "center_weighted"] = "uniform"
    center_sampling_weight: Annotated[float, Field(ge=1, allow_inf_nan=False)] = 3.0
    validation_image_multiplier: Annotated[StrictInt, Field(gt=0)] = 1
    input_channels: Literal[1, 3] = 1
    input_mode: Literal["single_slice", "adjacent_triplets"] = "single_slice"
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
    mean: FiniteFloat | tuple[FiniteFloat, ...]
    std: PositiveFiniteFloat | tuple[PositiveFiniteFloat, ...]

    @property
    def slices_per_input(self) -> int:
        return 3 if self.input_mode == "adjacent_triplets" else 1

    @model_validator(mode="after")
    def triplet_inputs(self) -> "DataConfig":
        if (
            self.position_reference == "center"
            and self.position_encoding != "physical_mm"
        ):
            raise ValueError("Centered positions require physical_mm encoding")
        if self.input_mode == "adjacent_triplets":
            if self.input_channels != 3:
                raise ValueError("Adjacent triplets require three input channels")
            if any(kind.max_images % 3 for kind in self.series_types):
                raise ValueError(
                    "Triplet max_images limits count raw slices and must be multiples of three"
                )
        return self

    @field_validator("mean", "std")
    @classmethod
    def normalization_channels(
        cls, value: float | tuple[float, ...], info: ValidationInfo
    ) -> float | tuple[float, ...]:
        channels = len(value) if isinstance(value, tuple) else 1
        if channels != info.data.get("input_channels", 1):
            raise ValueError("normalization must provide one value per input channel")
        return value

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
    feature_width: Annotated[StrictInt, Field(gt=0)]
    projection_width: Annotated[StrictInt, Field(gt=0)]
    channels_last: bool = False
    checkpoint_backbone: bool = False
    backbone_eval: bool = False
    pooling: Literal["transformer", "attention_mil"] = "transformer"
    mil_attention_width: Annotated[StrictInt, Field(gt=0)] = 128
    transformer_layers: Annotated[StrictInt, Field(gt=0)] = 1
    attention_heads: Annotated[StrictInt, Field(gt=0)] = 8
    feedforward_width: Annotated[StrictInt, Field(gt=0)] = 2048
    dropout: Annotated[float, Field(ge=0, lt=1)]
    series_dropout: Annotated[float, Field(ge=0, lt=1)]

    @field_validator("attention_heads")
    @classmethod
    def divisible_heads(cls, value: int, info: ValidationInfo) -> int:
        width = info.data.get("projection_width")
        if (
            info.data.get("pooling") == "transformer"
            and width is not None
            and width % value
        ):
            raise ValueError("projection width must be divisible by attention heads")
        return value


class OptimizerConfig(ConfigModel):
    backbone_lr: Annotated[float, Field(gt=0)]
    head_lr: Annotated[float, Field(gt=0)]
    weight_decay: Annotated[float, Field(ge=0)]
    beta1: Annotated[float, Field(gt=0, lt=1)]
    beta2: Annotated[float, Field(gt=0, lt=1)]
    epsilon: Annotated[float, Field(gt=0)]
    warmup_epochs: Annotated[StrictInt, Field(ge=0)]
    minimum_lr_ratio: Annotated[float, Field(gt=0, le=1)]


class TrainerConfig(ConfigModel):
    max_epochs: Annotated[StrictInt, Field(gt=0)]
    precision: Literal["bf16-mixed"]
    accelerator: Literal["auto", "cpu", "gpu"]
    devices: Literal[1]
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
    bundle_fields = {
        "schema_version",
        "experiment",
        "targets",
        "fold",
    }
    if isinstance(raw, dict) and set(raw) in (
        bundle_fields,
        bundle_fields | {"inference"},
    ):
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


class ConfigCLI(NonExitingCLISettings):
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
