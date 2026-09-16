import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import ExperimentConfig, load_config
from knee_abnormality.data.labels import (
    STUDY,
    TARGET_THRESHOLD,
    load_source,
    require_unique_ids,
)
from knee_abnormality.evaluation.scoring import score
from knee_abnormality.provenance import sha256_file


def make_folds(
    train: pl.DataFrame,
    source: pl.DataFrame,
    targets: list[str],
    n_splits: int = 5,
    seed: int = 42,
):
    require_unique_ids(train, "Training table", reject_blank=True)
    if n_splits < 2 or not targets or len(set(targets)) != len(targets):
        raise ValueError("Need at least two folds and unique target columns")
    if train.columns != [STUDY, "Report", *targets]:
        raise ValueError(
            "Training columns must match the report and submission target schema"
        )
    if train.filter(pl.col("Report").fill_null("").str.strip_chars() == "").height:
        raise ValueError("Blank reports need an explicit grouping policy")
    y = train.select(targets).to_numpy().astype(float)
    observed = np.isfinite(y)
    if np.isinf(y).any() or not np.isin(y[observed], [0, 1]).all():
        raise ValueError("Observed reference labels must be binary")
    if ((observed.sum(axis=1) != 0) & (observed.sum(axis=1) != len(targets))).any():
        raise ValueError("Partially labeled studies need a revised split protocol")

    def report_group(text: str) -> str:
        normalized = " ".join(text.split()).casefold()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    studies = (
        train.with_columns(
            pl.Series("is_gold", observed.all(axis=1)),
            pl.col("Report")
            .map_elements(report_group, return_dtype=pl.String)
            .alias("report_group"),
        )
        .join(
            source.rename({target: f"source_{target}" for target in targets}),
            on=STUDY,
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.when(pl.col("is_gold"))
            .then(pl.col(target))
            .otherwise(pl.col(f"source_{target}"))
            .cast(pl.Float32)
            .alias(target)
            for target in targets
        )
    )
    effective = studies.select(targets).to_numpy()
    if not np.isfinite(effective).all() or ((effective < 0) | (effective > 1)).any():
        raise ValueError(
            "All studies require complete finite labels for stratification"
        )
    indicators = []
    for target in targets:
        indicators.extend(
            [
                (pl.col(target) >= TARGET_THRESHOLD).max().alias(f"positive_{target}"),
                (pl.col(target) < TARGET_THRESHOLD).max().alias(f"negative_{target}"),
                (pl.col("is_gold") & (pl.col(target) >= TARGET_THRESHOLD))
                .max()
                .alias(f"gold_positive_{target}"),
                (pl.col("is_gold") & (pl.col(target) < TARGET_THRESHOLD))
                .max()
                .alias(f"gold_negative_{target}"),
            ]
        )
    groups = (
        studies.group_by("report_group")
        .agg(*indicators, pl.col("is_gold").any())
        .sort("report_group")
    )
    if groups.height < n_splits:
        raise ValueError("Too few independent report groups")
    group_indicators = groups.drop("report_group")
    group_classes = group_indicators.to_numpy().astype(int)
    for indicator, count in zip(
        group_indicators.columns, group_classes.sum(axis=0), strict=True
    ):
        if count < n_splits:
            raise ValueError(
                f"{indicator}: {count} report groups cannot cover {n_splits} folds"
            )
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    assignment = np.full(groups.height, -1, dtype=np.int64)
    for fold, (_, valid) in enumerate(
        splitter.split(np.zeros((groups.height, 1)), group_classes)
    ):
        assignment[valid] = fold
    groups = groups.select("report_group").with_columns(pl.Series("fold", assignment))
    manifest = (
        studies.select(STUDY, "report_group", "is_gold")
        .join(groups, on="report_group", how="left", validate="m:1")
        .with_columns(pl.col("fold").cast(pl.Int64))
        .sort(STUDY)
    )

    counts = []
    for fold in range(n_splits):
        valid = studies.join(manifest.select(STUDY, "fold"), on=STUDY).filter(
            pl.col("fold") == fold
        )
        gold = valid.filter(pl.col("is_gold"))
        for target in targets:
            positive = int((valid[target] >= TARGET_THRESHOLD).sum())
            negative = valid.height - positive
            gold_positive = int((gold[target] >= TARGET_THRESHOLD).sum())
            gold_negative = gold.height - gold_positive
            if min(positive, negative) == 0:
                raise ValueError(
                    f"Fold {fold}, {target}: both classes are required; revise the split design"
                )
            if min(gold_positive, gold_negative) == 0:
                raise ValueError(
                    f"Fold {fold}, gold {target}: both classes are required; revise the split design"
                )
            counts.append(
                {
                    "fold": fold,
                    "target": target,
                    "positive": positive,
                    "negative": negative,
                    "gold_positive": gold_positive,
                    "gold_negative": gold_negative,
                }
            )
    return manifest, pl.DataFrame(counts), studies.select(STUDY, *targets).sort(STUDY)


def prepare(
    data_dir: Path,
    split_dir: Path,
    n_splits: int,
    seed: int,
    config: ExperimentConfig,
) -> dict:
    train = pl.read_csv(data_dir / "train.csv", infer_schema_length=None)
    sample = pl.read_csv(data_dir / "sample_submission.csv")
    if sample.columns[0] != STUDY:
        raise ValueError("Submission template must start with the study ID")
    targets = sample.columns[1:]
    source, source_provenance = load_source(
        config,
        json.loads(config.paths.label_manifest.read_text()),
        config.labels.source,
        tuple(targets),
    )
    manifest, counts, evaluation = make_folds(train, source, targets, n_splits, seed)
    csv = manifest.write_csv()
    evaluation_csv = evaluation.write_csv()
    metadata = {
        "protocol_version": 2,
        "n_splits": n_splits,
        "seed": seed,
        "targets": targets,
        "stratification": "iterative multilabel over all report groups; positive and negative effective-label and gold-only target indicators plus is_gold; both classes required per target in every full and gold validation fold",
        "grouping": "SHA-256 of whitespace-collapsed, casefolded Report",
        "validation_scope": "all_studies",
        "validation_target_threshold": TARGET_THRESHOLD,
        "evaluation_target_policy": "gold where available, frozen report-derived probabilities otherwise; threshold only for AUC",
        "label_source": {
            key: value for key, value in source_provenance.items() if key != "path"
        },
        "label_manifest_sha256": sha256_file(config.paths.label_manifest),
        "input_sha256": {
            name: sha256_file(data_dir / name)
            for name in ("train.csv", "sample_submission.csv")
        },
        "manifest_sha256": hashlib.sha256(csv.encode()).hexdigest(),
        "evaluation_targets_sha256": hashlib.sha256(
            evaluation_csv.encode()
        ).hexdigest(),
        "generator_sha256": {
            "validation.py": sha256_file(Path(__file__)),
            "labels.py": sha256_file(
                Path(__file__).resolve().parents[2]
                / "src/knee_abnormality/data/labels.py"
            ),
        },
        "versions": {
            name: version(name)
            for name in ("iterative-stratification", "numpy", "polars", "scikit-learn")
        },
        "folds": [
            {
                "fold": fold,
                "validation_studies": manifest.filter(pl.col("fold") == fold).height,
                "training_studies": manifest.filter(pl.col("fold") != fold).height,
                "validation_gold": manifest.filter(
                    (pl.col("fold") == fold) & pl.col("is_gold")
                ).height,
                "validation_report_only": manifest.filter(
                    (pl.col("fold") == fold) & ~pl.col("is_gold")
                ).height,
                "training_gold": manifest.filter(
                    (pl.col("fold") != fold) & pl.col("is_gold")
                ).height,
                "training_report_only": manifest.filter(
                    (pl.col("fold") != fold) & ~pl.col("is_gold")
                ).height,
            }
            for fold in range(n_splits)
        ],
    }
    artifacts = {
        "studies.csv": csv,
        "label_counts.csv": counts.write_csv(),
        "targets.csv": evaluation_csv,
        "protocol.json": json.dumps(metadata, indent=2) + "\n",
    }
    for name, content in artifacts.items():
        path = split_dir / name
        if path.exists() and path.read_text() != content:
            raise FileExistsError(
                f"Refusing to replace {path}; use a new --split-dir for a revised protocol"
            )
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        path = split_dir / name
        if not path.exists():
            with path.open("x") as stream:
                stream.write(content)
    return metadata


class ValidationCLI(CLISettings):
    command: CliPositionalArg[Literal["prepare", "score"]]
    config: Path = Path("configs/training.yaml")
    data_dir: Path | None = None
    split_dir: Path | None = None
    n_splits: int = Field(default=5, ge=2)
    seed: int | None = Field(default=None, ge=0, lt=2**32)
    predictions: Path | None = None


def main() -> None:
    try:
        args = ValidationCLI()
        config = load_config(
            args.config,
            data_root=args.data_dir.expanduser().resolve().parent
            if args.data_dir is not None
            else None,
        )
        data_dir = (
            args.data_dir.expanduser().resolve()
            if args.data_dir is not None
            else config.raw_dir
        )
        split_dir = (
            args.split_dir.expanduser().resolve()
            if args.split_dir is not None
            else config.split_dir
        )
        if args.command == "prepare":
            result = prepare(
                data_dir,
                split_dir,
                args.n_splits,
                config.seed if args.seed is None else args.seed,
                config,
            )
        else:
            if args.predictions is None:
                raise ValueError("score requires --predictions")
            result = score(data_dir, split_dir, args.predictions)
    except (OSError, TypeError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Validation error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
