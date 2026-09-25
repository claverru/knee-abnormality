from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from knee_abnormality.data.labels import STUDY, require_unique_ids
from knee_abnormality.provenance import sha256_file

PROBABILITY_TOLERANCE = 1e-4


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Question(FrozenModel):
    id: str
    group: str
    criterion: str
    options: dict[str, str] = Field(min_length=2)


class QuestionSet(FrozenModel):
    version: int
    preamble: str
    questions: tuple[Question, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "QuestionSet":
        ids = [question.id for question in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("Question ids must be unique")
        return self


class FileRecord(FrozenModel):
    path: str
    bytes: int
    sha256: str

    @classmethod
    def of(cls, path: Path) -> "FileRecord":
        return cls(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path))


class TargetQuestion(FrozenModel):
    id: str
    options: tuple[str, ...] = Field(min_length=2)


class Gate(FrozenModel):
    child: str
    parent: str
    parent_options: tuple[str, ...] = Field(min_length=1)
    empty_option: str
    fallback_option: str


class Merge(FrozenModel):
    question: str
    option: str
    sources: tuple[str, ...] = Field(min_length=2)


class TargetManifest(FrozenModel):
    version: int
    prepared_at: str
    method: str
    source_run: dict
    attributes: FileRecord
    question_set: FileRecord
    targets: FileRecord
    excluded: tuple[str, ...]
    merges: tuple[Merge, ...]
    gates: tuple[Gate, ...]
    questions: tuple[TargetQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_questions(self) -> "TargetManifest":
        ids = [question.id for question in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("Target questions must be unique")
        return self


@dataclass(frozen=True)
class ReportAttributeTargets:
    questions: tuple[TargetQuestion, ...]
    frame: pl.DataFrame
    provenance: dict

    @property
    def value_columns(self) -> tuple[str, ...]:
        return tuple(
            value_column(question.id, option)
            for question in self.questions
            for option in question.options
        )

    @property
    def weight_columns(self) -> tuple[str, ...]:
        return tuple(weight_column(question.id) for question in self.questions)


def value_column(question: str, option: str) -> str:
    return f"attribute_{question}__{option}"


def weight_column(question: str) -> str:
    return f"weight_{question}"


def check_distributions(
    frame: pl.DataFrame, questions: tuple[TargetQuestion, ...], name: str
) -> None:
    for question in questions:
        values = frame.select(
            value_column(question.id, option) for option in question.options
        ).to_numpy()
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{name} {question.id} must hold finite probabilities")
        if np.abs(values.sum(1) - 1).max() > PROBABILITY_TOLERANCE:
            raise ValueError(f"{name} {question.id} probabilities must sum to one")
        weights = frame[weight_column(question.id)].to_numpy()
        if not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
            raise ValueError(f"{name} {question.id} weights must lie in [0, 1]")


def load_report_attribute_targets(
    manifest_path: Path, data_root: Path
) -> ReportAttributeTargets:
    manifest = TargetManifest.model_validate_json(manifest_path.read_text())
    recorded = Path(manifest.targets.path)
    if recorded.is_absolute():
        path = recorded
    elif recorded.parts and recorded.parts[0] == "data":
        path = data_root.joinpath(*recorded.parts[1:])
    else:
        raise ValueError(
            "Report attribute targets must be absolute or under the project data/ directory"
        )
    if sha256_file(path) != manifest.targets.sha256:
        raise ValueError(f"Report attribute target hash mismatch: {path}")
    frame = pl.read_parquet(path)
    require_unique_ids(frame, "Report attribute targets")
    expected = [
        STUDY,
        *(
            value_column(question.id, option)
            for question in manifest.questions
            for option in question.options
        ),
        *(weight_column(question.id) for question in manifest.questions),
    ]
    if frame.columns != expected:
        raise ValueError("Report attribute target columns differ from the manifest")
    check_distributions(frame, manifest.questions, "Report attribute targets")
    return ReportAttributeTargets(
        manifest.questions,
        frame,
        {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "targets_sha256": manifest.targets.sha256,
            "question_set_sha256": manifest.question_set.sha256,
        },
    )
