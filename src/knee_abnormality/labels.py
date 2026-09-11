import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from knee_abnormality.config import ExperimentConfig
from knee_abnormality.provenance import sha256_file

STUDY = "StudyInstanceUID"


@dataclass(frozen=True)
class FoldLabels:
    targets: tuple[str, ...]
    training: pl.DataFrame
    validation: pl.DataFrame
    protocol: dict
    provenance: dict


def _data_path(config: ExperimentConfig, recorded: str) -> Path:
    path = Path(recorded)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return config.paths.data_root.joinpath(*path.parts[1:])
    return config.paths.label_manifest.parent.parent / path


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


def load_source(
    config: ExperimentConfig, manifest: dict, source_name: str, targets: tuple[str, ...]
) -> tuple[pl.DataFrame, dict]:
    matches = []
    for source in manifest["sources"]:
        for file in source["files"]:
            if file.get("label_table") == source_name:
                matches.append((source, file))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one pinned label table named {source_name}, found {len(matches)}"
        )
    source, record = matches[0]
    path = _data_path(config, record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Label source hash mismatch: {path}")
    frame = pl.read_csv(path, infer_schema_length=None)
    require_unique_ids(frame, source_name)
    if frame.columns != [STUDY, *targets]:
        raise ValueError(
            f"{source_name} target schema differs from the frozen protocol"
        )
    values = frame.select(targets).to_numpy().astype(np.float32)
    finite = np.isfinite(values)
    if np.isinf(values).any() or ((values[finite] < 0) | (values[finite] > 1)).any():
        raise ValueError(f"{source_name} labels must be finite probabilities or null")
    provenance = {
        "label_table": source_name,
        "dataset": source["dataset"],
        "dataset_version": source["dataset_version"],
        "path": str(path),
        "sha256": record["sha256"],
    }
    return frame, provenance


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


def load_fold_labels(config: ExperimentConfig, fold: int) -> FoldLabels:
    protocol_path = config.split_dir / "protocol.json"
    studies_path = config.split_dir / "studies.csv"
    protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_version"] not in {1, 2} or fold not in range(
        protocol["n_splits"]
    ):
        raise ValueError(f"Fold {fold} is not defined by {protocol_path}")
    if sha256_file(studies_path) != protocol["manifest_sha256"]:
        raise ValueError("Saved study membership differs from the frozen protocol")
    for name, expected in protocol["input_sha256"].items():
        if sha256_file(config.raw_dir / name) != expected:
            raise ValueError(f"{name} differs from the frozen split source")

    targets = tuple(protocol["targets"])
    source_manifest = json.loads(config.paths.label_manifest.read_text())
    selected, selected_provenance = load_source(
        config, source_manifest, config.labels.source, targets
    )
    optional_sources = {}
    for name in ("steven_v1", "steven_v2"):
        try:
            optional_sources[name] = load_source(
                config, source_manifest, name, targets
            )[0]
        except ValueError:
            if config.labels.source.startswith("steven_"):
                raise

    membership = pl.read_csv(studies_path)
    require_unique_ids(membership, "Saved membership")
    official = pl.read_csv(
        config.raw_dir / "train.csv", infer_schema_length=None
    ).select(STUDY, *targets)
    require_unique_ids(official, "Official training table")
    base = membership.join(
        official.rename({target: f"gold_{target}" for target in targets}),
        on=STUDY,
        how="left",
        validate="1:1",
    ).join(
        selected.rename({target: f"source_{target}" for target in targets}),
        on=STUDY,
        how="left",
        validate="1:1",
    )
    for name, frame in optional_sources.items():
        base = base.join(
            frame.rename({target: f"{name}_{target}" for target in targets}),
            on=STUDY,
            how="left",
            validate="1:1",
        )
    if {"steven_v1", "steven_v2"}.issubset(optional_sources):
        base = base.with_columns(
            (pl.col(f"steven_v1_{target}") == 0.5).alias(
                f"steven_v1_unaddressed_{target}"
            )
            for target in targets
        ).with_columns(
            (pl.col(f"steven_v2_{target}") != pl.col(f"steven_v1_{target}"))
            .fill_null(False)
            .alias(f"steven_v2_repaired_{target}")
            for target in targets
        )
    base = base.with_columns(
        pl.when(pl.col("is_gold"))
        .then(pl.col(f"gold_{target}"))
        .otherwise(pl.col(f"source_{target}"))
        .cast(pl.Float32)
        .alias(f"target_{target}")
        for target in targets
    ).with_columns(
        pl.col(f"target_{target}").is_not_null().alias(f"observed_{target}")
        for target in targets
    )
    training = base.filter(pl.col("fold") != fold).sort(STUDY)
    if protocol["protocol_version"] == 2:
        if protocol["validation_target_threshold"] != 0.5:
            raise ValueError("Unsupported frozen validation target threshold")
        if (
            membership["fold"].null_count()
            or not membership["fold"].is_in(range(protocol["n_splits"])).all()
        ):
            raise ValueError("All-study membership must assign every study to one fold")
        evaluation = load_evaluation_targets(config.split_dir, protocol)
        if set(evaluation[STUDY]) != set(membership[STUDY]):
            raise ValueError("Frozen evaluation targets must cover every study")
        validation = (
            base.filter(pl.col("fold") == fold)
            .drop([f"target_{target}" for target in targets])
            .join(
                evaluation.rename({target: f"target_{target}" for target in targets}),
                on=STUDY,
                how="left",
                validate="1:1",
            )
            .with_columns(
                pl.col(f"target_{target}").cast(pl.Float32) for target in targets
            )
            .with_columns(
                pl.lit(True).alias(f"observed_{target}") for target in targets
            )
            .sort(STUDY)
        )
    else:
        validation = base.filter((pl.col("fold") == fold) & pl.col("is_gold")).sort(
            STUDY
        )
    if validation.height == 0:
        raise ValueError(f"Fold {fold} has no validation studies")
    if (
        training.select(f"observed_{target}" for target in targets).to_numpy().sum()
        == 0
    ):
        raise ValueError(f"Fold {fold} has no observed training targets")
    provenance = {
        "manifest": str(config.paths.label_manifest),
        "manifest_sha256": sha256_file(config.paths.label_manifest),
        "split_protocol": str(protocol_path),
        "split_protocol_sha256": sha256_file(protocol_path),
        "membership_sha256": protocol["manifest_sha256"],
        "selected": selected_provenance,
    }
    return FoldLabels(targets, training, validation, protocol, provenance)
