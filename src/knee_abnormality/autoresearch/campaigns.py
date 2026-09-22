from pathlib import Path, PurePosixPath

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CampaignPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    editable_files: list[str]
    editable_directories: list[str]
    config_keys: list[str]
    config_prefixes: list[str]
    list_item_fields: dict[str, list[str]]
    input_identity_keys: list[str]
    require_source_change: bool
    developer_instructions: str = Field(min_length=1)
    researcher_instructions: str = Field(min_length=1)

    @field_validator("editable_files", "editable_directories")
    @classmethod
    def source_paths(cls, values, info):
        for value in values:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                raise ValueError(
                    f"Source path must be normalized and relative: {value}"
                )
            if value == "configs/training.yaml" and info.field_name == "editable_files":
                continue
            package = PurePosixPath("src/knee_abnormality")
            if not path.is_relative_to(package) or path == package:
                raise ValueError(f"Editable source must be inside {package}: {value}")
            relative = path.relative_to(package)
            if relative.parts[0] in {"autoresearch", "evaluation", "identity.py"}:
                raise ValueError(
                    f"Campaign cannot edit acceptance or identity code: {value}"
                )
            if info.field_name == "editable_files" and path.suffix != ".py":
                raise ValueError(f"Editable source must be Python: {value}")
        return values

    @field_validator("config_keys", "config_prefixes", "input_identity_keys")
    @classmethod
    def field_paths(cls, values):
        for value in values:
            if not value or any(not part.isidentifier() for part in value.split(".")):
                raise ValueError(f"Invalid dotted field path: {value}")
        return values

    @field_validator("list_item_fields")
    @classmethod
    def list_fields(cls, values):
        cls.field_paths(list(values))
        for fields in values.values():
            if not fields or any(not field.isidentifier() for field in fields):
                raise ValueError(
                    "Mutable list-item fields must be nonempty field names"
                )
        return values


class BacklogPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refill_below: int = Field(ge=1)
    refill_to: int = Field(ge=1, le=5)

    @model_validator(mode="after")
    def thresholds(self):
        if self.refill_below > self.refill_to:
            raise ValueError("Refill threshold exceeds target")
        return self


class Campaign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    status: str
    recipe: str
    reference_run: str
    reference_equivalences: dict[str, bool | int | float | str]
    additional_evidence_runs: list[str]
    backlog: BacklogPolicy
    artifacts_root: str
    policy: CampaignPolicy

    @field_validator("reference_equivalences")
    @classmethod
    def equivalence_paths(cls, values):
        CampaignPolicy.field_paths(list(values))
        return values


def load_campaign(path: Path) -> Campaign:
    return Campaign.model_validate(yaml.safe_load(path.read_text()))
