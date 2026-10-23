import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from knee_abnormality.config import ExperimentConfig
from knee_abnormality.data.splits import STUDY, load_split, require_unique_ids
from knee_abnormality.provenance import sha256_file


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
    if not path.parts or path.parts[0] != "data":
        raise ValueError(
            f"Label manifest paths must be absolute or under the project data/ directory: {recorded}"
        )
    return config.paths.data_root.joinpath(*path.parts[1:])


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


def load_fold_labels(config: ExperimentConfig, fold: int) -> FoldLabels:
    split = load_split(config.raw_dir, config.split_dir)
    protocol = split.protocol
    protocol_path = config.split_dir / "protocol.json"
    if fold not in range(protocol["n_splits"]):
        raise ValueError(f"Fold {fold} is not defined by {protocol_path}")
    targets = tuple(protocol["targets"])
    selected, selected_provenance = load_source(
        config,
        json.loads(config.paths.label_manifest.read_text()),
        config.labels.source,
        targets,
    )
    official = pl.read_csv(
        config.raw_dir / "train.csv", infer_schema_length=None
    ).select(STUDY, *targets)
    require_unique_ids(official, "Official training table")
    base = (
        split.membership.join(
            official.rename({target: f"gold_{target}" for target in targets}),
            on=STUDY,
            how="left",
            validate="1:1",
        )
        .join(
            selected.rename({target: f"source_{target}" for target in targets}),
            on=STUDY,
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.when(pl.col("is_gold"))
            .then(pl.col(f"gold_{target}"))
            .otherwise(pl.col(f"source_{target}"))
            .cast(pl.Float32)
            .alias(f"target_{target}")
            for target in targets
        )
        .with_columns(
            pl.col(f"target_{target}").is_not_null().alias(f"observed_{target}")
            for target in targets
        )
    )
    training = (
        base
        if config.validation.train_on_validation
        else base.filter(pl.col("fold") != fold)
    ).sort(STUDY)
    validation = (
        base.filter(pl.col("fold") == fold)
        .drop([f"target_{target}" for target in targets])
        .join(
            split.evaluation.rename({target: f"target_{target}" for target in targets}),
            on=STUDY,
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col(f"target_{target}").cast(pl.Float32) for target in targets)
        .with_columns(pl.lit(True).alias(f"observed_{target}") for target in targets)
        .sort(STUDY)
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


def distilled(config: ExperimentConfig, labels: FoldLabels) -> FoldLabels:
    share = config.distillation.teacher_share
    if share == 0:
        return labels
    path = config.paths.teacher_table
    if path is None:
        raise ValueError(
            "distillation.teacher_share above zero needs paths.teacher_table"
        )
    targets = labels.targets
    table = pl.read_parquet(path).filter(pl.col("reading") == "joint")
    require_unique_ids(table, "Teacher table")
    if table.columns != [STUDY, "fold", "reading", *targets]:
        raise ValueError("Teacher table schema differs from the frozen protocol")
    teacher = labels.training.select(STUDY, "fold").join(
        table.select(
            STUDY,
            pl.col("fold").alias("teacher_fold"),
            *(pl.col(target).alias(f"teacher_{target}") for target in targets),
        ),
        on=STUDY,
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    if teacher["teacher_fold"].null_count():
        raise ValueError("Teacher table must give every training study a value")
    logits = teacher.select(f"teacher_{target}" for target in targets).to_numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Teacher table logits must be finite")
    training = (
        labels.training.join(
            teacher.drop("fold", "teacher_fold"),
            on=STUDY,
            how="left",
            validate="1:1",
            maintain_order="left",
        )
        .with_columns(
            pl.when(pl.col("is_gold"))
            .then(pl.col(f"target_{target}"))
            .otherwise(
                (1 - share) * pl.col(f"target_{target}")
                + share / (1 + (-pl.col(f"teacher_{target}")).exp())
            )
            .cast(pl.Float32)
            .alias(f"target_{target}")
            for target in targets
        )
        .drop([f"teacher_{target}" for target in targets])
    )
    columns = [f"target_{target}" for target in targets]
    shift = float(
        np.abs(
            training.select(columns).to_numpy()
            - labels.training.select(columns).to_numpy()
        ).mean()
    )
    return FoldLabels(
        targets,
        training,
        labels.validation,
        labels.protocol,
        {
            **labels.provenance,
            "teacher": {
                "path": str(path),
                "sha256": sha256_file(path),
                "share": share,
                "target_shift": shift,
            },
        },
    )
