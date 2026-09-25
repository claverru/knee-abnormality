from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from knee_abnormality.config import Device, RuntimePrecision


class CLISettings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_enforce_required=True,
        cli_exit_on_error=True,
        cli_implicit_flags=True,
        cli_hide_none_type=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


def describe(text: str) -> ConfigDict:
    return ConfigDict(json_schema_extra={"description": text})


def command(model: type[BaseModel], **kwargs):
    return Field(
        description=model.model_config["json_schema_extra"]["description"], **kwargs
    )


class Runtime(BaseModel):
    device: Device = Field(default="auto", description="Inference device")
    precision: RuntimePrecision = Field(
        default="auto", description="Inference precision"
    )


class Bootstrap(BaseModel):
    bootstrap_draws: int = Field(
        default=2000, ge=100, description="Paired bootstrap resamples"
    )
    bootstrap_seed: int = Field(default=20260914, description="Bootstrap seed")


class RunFold(BaseModel):
    fold: int | None = Field(
        default=None,
        ge=0,
        description="Fold to use; required when the run holds several",
    )

    def run_fold(self, folds: Sequence[int]) -> int:
        if self.fold is None:
            if len(folds) != 1:
                raise ValueError(
                    f"Run has several validation folds {list(folds)}; pass --fold"
                )
            return folds[0]
        if self.fold not in folds:
            raise ValueError(
                f"Fold {self.fold} is absent from the run's folds {list(folds)}"
            )
        return self.fold
