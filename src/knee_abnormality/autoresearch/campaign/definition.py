from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

PACKAGE = PurePosixPath("src/knee_abnormality")
RECIPE = "configs/campaign.yaml"
PROTECTED_SOURCES = tuple(
    PACKAGE / name
    for name in (
        "autoresearch",
        "evaluation",
        "identity.py",
        "training/checkpoints.py",
        "data/splits.py",
    )
)
PROTECTED_CONFIG = (
    "name",
    "seed",
    "paths",
    "labels",
    "validation",
    "data.validation_image_multiplier",
)
EPOCHS = "trainer.max_epochs"
BATCH_SIZE = "trainer.batch_size"
RANGE_BUDGET_CONFIG = (EPOCHS, BATCH_SIZE, "trainer.accumulate_grad_batches")
VALIDATION_IDENTITY = (
    "validation_rows_sha256",
    "membership_sha256",
    "split_protocol_sha256",
    "label_manifest_sha256",
    "label_source",
)


def overlaps(path: str, other: str):
    return path == other or path.startswith(f"{other}.") or other.startswith(f"{path}.")


def dotted_paths(values):
    for value in values:
        if not value or any(not part.isidentifier() for part in value.split(".")):
            raise ValueError(f"Invalid dotted field path: {value}")
    return values


def permitted_config(values):
    for value in dotted_paths(values):
        if any(overlaps(value, protected) for protected in PROTECTED_CONFIG):
            raise ValueError(
                f"Campaign cannot change the split, labels, paths, seed or training budget: {value}"
            )
    return values


class CampaignPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    editable_files: list[str]
    editable_directories: list[str]
    config_keys: list[str]
    config_prefixes: list[str]
    list_item_fields: dict[str, list[str]]
    input_identity_keys: list[str]
    require_source_change: bool
    configuration_only_ideas: bool
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
            if value == RECIPE and info.field_name == "editable_files":
                continue
            if not path.is_relative_to(PACKAGE) or path == PACKAGE:
                raise ValueError(f"Editable source must be inside {PACKAGE}: {value}")
            if any(
                path.is_relative_to(protected) or protected.is_relative_to(path)
                for protected in PROTECTED_SOURCES
            ):
                raise ValueError(
                    f"Campaign cannot edit harness, evaluation, split, evaluation-label or identity code: {value}"
                )
            if info.field_name == "editable_files" and path.suffix != ".py":
                raise ValueError(f"Editable source must be Python: {value}")
        return values

    @field_validator("config_keys", "config_prefixes")
    @classmethod
    def config_paths(cls, values):
        return permitted_config(values)

    @field_validator("input_identity_keys")
    @classmethod
    def identity_paths(cls, values):
        for value in dotted_paths(values):
            if any(overlaps(value, key) for key in VALIDATION_IDENTITY):
                raise ValueError(
                    f"Campaign cannot permit a different validation frame, split or label source: {value}"
                )
        return values

    @field_validator("list_item_fields")
    @classmethod
    def list_fields(cls, values):
        permitted_config(list(values))
        for fields in values.values():
            if not fields or any(not field.isidentifier() for field in fields):
                raise ValueError(
                    "Mutable list-item fields must be nonempty field names"
                )
        return values

    @model_validator(mode="after")
    def configuration_only(self):
        if self.configuration_only_ideas and self.require_source_change:
            raise ValueError(
                "A campaign that allows configuration-only ideas cannot require a source change"
            )
        return self

    def permits_config(self, path: str):
        return any(
            overlaps(path, item)
            for item in (
                *self.config_keys,
                *self.config_prefixes,
                *self.list_item_fields,
            )
        )


class SealedBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["sealed"]


class RangeBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["range"]
    min_fraction: float = Field(gt=0)
    max_fraction: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.min_fraction >= self.max_fraction:
            raise ValueError("The budget range needs min_fraction below max_fraction")
        return self


Budget = Annotated[SealedBudget | RangeBudget, Field(discriminator="mode")]


class BacklogPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refill_below: int = Field(ge=1)
    refill_to: int = Field(ge=1, le=5)
    seed_order: Literal["ranked", "forced"] = "ranked"
    rotate_after: int = Field(ge=1)

    @model_validator(mode="after")
    def thresholds(self):
        if self.refill_below > self.refill_to:
            raise ValueError("Refill threshold exceeds target")
        return self


CAMPAIGN_ID = r"^[a-z][a-z0-9-]*$"


class ChainReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predecessor: str = Field(pattern=CAMPAIGN_ID)
    incumbent: str = Field(min_length=1)
    source: str = Field(min_length=1)
    run: str = Field(min_length=1)
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Campaign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=CAMPAIGN_ID)
    recipe: str
    reference_run: str
    reference_equivalences: dict[str, bool | int | float | str | list[str]]
    reference_config_exceptions: list[str] = Field(default_factory=list)
    reference_identity_exceptions: list[str] = Field(default_factory=list)
    additional_evidence_runs: list[str]
    backlog: BacklogPolicy
    budget: Budget
    policy: CampaignPolicy
    chain: ChainReference | None = None

    @computed_field
    @property
    def artifacts_root(self) -> str:
        return f"outputs/autoresearch/{self.id}"

    @field_validator("reference_equivalences")
    @classmethod
    def equivalence_paths(cls, values):
        return dotted_paths(values)

    @field_validator("reference_config_exceptions", "reference_identity_exceptions")
    @classmethod
    def exception_paths(cls, values):
        for value in values:
            parts = value.split(".")
            if parts[-1].endswith("[]") or any(
                not part.removesuffix("[]").isidentifier() for part in parts
            ):
                raise ValueError(f"Invalid reference exception path: {value}")
        return values

    @model_validator(mode="after")
    def epoch_budget(self):
        for path in RANGE_BUDGET_CONFIG:
            if self.budget.mode == "sealed" and self.policy.permits_config(path):
                raise ValueError(
                    f"Only a campaign with a wall-clock range budget can change {path}"
                )
        return self

    @model_validator(mode="after")
    def chained_reference(self):
        if self.chain is None:
            return self
        if self.reference_run != self.chain.run or self.recipe != chain_recipe(
            self.chain.run
        ):
            raise ValueError(
                "A chained campaign's recipe and reference are its incumbent's effective configuration and run"
            )
        if (
            self.reference_equivalences
            or self.reference_config_exceptions
            or self.reference_identity_exceptions
        ):
            raise ValueError(
                "A chained campaign takes no reference equivalences or exceptions"
            )
        return self


def chain_recipe(run: str):
    return f"{run}/effective_config.json"


def load_campaign(path: Path) -> Campaign:
    return Campaign.model_validate(yaml.safe_load(path.read_text()))
