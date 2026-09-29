import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from knee_abnormality.provenance import sha256_file

STUDY = "StudyInstanceUID"
TARGET_THRESHOLD = 0.5


@dataclass(frozen=True)
class FrozenSplit:
    protocol: dict
    membership: pl.DataFrame
    evaluation: pl.DataFrame


def require_unique_ids(
    frame: pl.DataFrame,
    name: str,
    *,
    reject_blank: bool = False,
    require_string: bool = True,
) -> None:
    if STUDY not in frame.columns or (
        require_string and frame.schema[STUDY] != pl.String
    ):
        raise ValueError(f"{name} must contain string study IDs")
    if frame[STUDY].null_count() or frame[STUDY].n_unique() != frame.height:
        raise ValueError(f"{name} contains missing or duplicate study IDs")
    if reject_blank and frame.filter(pl.col(STUDY).str.strip_chars() == "").height:
        raise ValueError(f"{name} contains blank study IDs")


def load_evaluation_targets(split_dir: Path, protocol: dict) -> pl.DataFrame:
    path = split_dir / "targets.csv"
    if sha256_file(path) != protocol["evaluation_targets_sha256"]:
        raise ValueError("Saved evaluation targets differ from the frozen protocol")
    frame = pl.read_csv(path, infer_schema_length=None)
    require_unique_ids(frame, "Frozen evaluation targets")
    if frame.columns != [STUDY, *protocol["targets"]]:
        raise ValueError("Frozen evaluation target schema differs from the protocol")
    values = frame.select(protocol["targets"]).to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Frozen evaluation targets must be finite probabilities")
    return frame


def load_split(raw_dir: Path, split_dir: Path) -> FrozenSplit:
    protocol_path = split_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_version"] != 2:
        raise ValueError(f"Unsupported split protocol: {protocol_path}")
    if protocol["validation_target_threshold"] != TARGET_THRESHOLD:
        raise ValueError("Unsupported frozen validation target threshold")
    studies_path = split_dir / "studies.csv"
    if sha256_file(studies_path) != protocol["manifest_sha256"]:
        raise ValueError("Saved study membership differs from the frozen protocol")
    for name, expected in protocol["input_sha256"].items():
        if sha256_file(raw_dir / name) != expected:
            raise ValueError(f"{name} differs from the frozen split source")
    membership = pl.read_csv(studies_path)
    require_unique_ids(membership, "Saved membership")
    if (
        membership["fold"].null_count()
        or not membership["fold"].is_in(range(protocol["n_splits"])).all()
    ):
        raise ValueError("All-study membership must assign every study to one fold")
    evaluation = load_evaluation_targets(split_dir, protocol)
    if set(evaluation[STUDY]) != set(membership[STUDY]):
        raise ValueError("Frozen evaluation targets must cover every study")
    return FrozenSplit(protocol, membership, evaluation)
