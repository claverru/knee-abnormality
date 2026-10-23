import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliSubCommand

from knee_abnormality.cli import CLISettings, command, describe
from knee_abnormality.config import STUDY_HEADERS, ExperimentConfig, load_config
from knee_abnormality.data.labels import load_source
from knee_abnormality.data.splits import (
    STUDY,
    TARGET_THRESHOLD,
    load_split,
    require_unique_ids,
)
from knee_abnormality.evaluation.scoring import score
from knee_abnormality.provenance import sha256_file

MAJOR_VENDORS = ("Siemens", "Philips", "GE")


def scanner_groups(headers: pl.DataFrame) -> pl.DataFrame:
    field = pl.col("MagneticFieldStrength").cast(pl.Float64, strict=False)
    return headers.select(
        STUDY,
        pl.when(~pl.col("vendor").is_in(MAJOR_VENDORS))
        .then(pl.lit("minor"))
        .when(field.is_null())
        .then(pl.col("vendor") + "/unknown")
        .when(field >= 2)
        .then(pl.col("vendor") + "/3T")
        .otherwise(pl.col("vendor") + "/1.5T")
        .alias("scanner_group"),
    ).sort(STUDY)


def make_folds(
    train: pl.DataFrame,
    source: pl.DataFrame,
    scanners: pl.DataFrame,
    targets: list[str],
    n_splits: int = 5,
    seed: int = 42,
    gold_holdout: bool = False,
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
        .join(scanners, on=STUDY, how="left", validate="1:1")
    )
    if studies["scanner_group"].null_count():
        raise ValueError("Every study needs a scanner group")
    groups_present = sorted(studies["scanner_group"].unique().to_list())
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
    indicators.extend(
        (pl.col("scanner_group") == group).max().alias(f"scanner_{group}")
        for group in groups_present
    )
    groups = (
        studies.group_by("report_group")
        .agg(*indicators, pl.col("is_gold").any())
        .sort("report_group")
    )
    if groups.height < n_splits:
        raise ValueError("Too few independent report groups")
    group_indicators = groups.drop("report_group")
    stratified = np.ones(groups.height, dtype=bool)
    if gold_holdout:
        stratified = ~groups["is_gold"].to_numpy()
        group_indicators = group_indicators.select(
            name
            for name in group_indicators.columns
            if name != "is_gold" and not name.startswith("gold_")
        )
    group_classes = group_indicators.to_numpy().astype(int)[stratified]
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
    assignment = np.zeros(groups.height, dtype=np.int64)
    members = np.flatnonzero(stratified)
    for fold, (_, valid) in enumerate(
        splitter.split(np.zeros((len(members), 1)), group_classes)
    ):
        assignment[members[valid]] = fold
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
            if gold_holdout and fold and gold.height:
                raise ValueError(
                    f"Fold {fold} holds gold studies outside the held-out fold"
                )
            if min(gold_positive, gold_negative) == 0 and not (gold_holdout and fold):
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
    scanner_counts = (
        manifest.join(studies.select(STUDY, "scanner_group"), on=STUDY)
        .group_by("fold", "scanner_group")
        .len()
        .sort("fold", "scanner_group")
    )
    return (
        manifest,
        pl.DataFrame(counts),
        studies.select(STUDY, *targets).sort(STUDY),
        scanner_counts,
    )


def prepare(
    data_dir: Path,
    split_dir: Path,
    n_splits: int,
    seed: int,
    config: ExperimentConfig,
    scanner_table: Path,
    label_source: str,
    gold_holdout: bool,
) -> dict:
    train = pl.read_csv(data_dir / "train.csv", infer_schema_length=None)
    sample = pl.read_csv(data_dir / "sample_submission.csv")
    if sample.columns[0] != STUDY:
        raise ValueError("Submission template must start with the study ID")
    targets = sample.columns[1:]
    source, source_provenance = load_source(
        config,
        json.loads(config.paths.label_manifest.read_text()),
        label_source,
        tuple(targets),
    )
    scanners = scanner_groups(pl.read_parquet(scanner_table))
    manifest, counts, evaluation, scanner_counts = make_folds(
        train, source, scanners, targets, n_splits, seed, gold_holdout
    )
    csv = manifest.write_csv()
    evaluation_csv = evaluation.write_csv()
    metadata = {
        "protocol_version": 2,
        "n_splits": n_splits,
        "seed": seed,
        "targets": targets,
        "stratification": "every report group with a gold study in fold 0; iterative multilabel over the other report groups; positive and negative effective-label target indicators and scanner-group indicators; both classes required per target in every fold and in fold 0's gold studies"
        if gold_holdout
        else "iterative multilabel over all report groups; positive and negative effective-label and gold-only target indicators, scanner-group indicators plus is_gold; both classes required per target in every full and gold validation fold",
        **({"gold_holdout_fold": 0} if gold_holdout else {}),
        "scanner_groups": {
            "table": str(scanner_table),
            "sha256": sha256_file(scanner_table),
            "rule": "Siemens, Philips and GE split by MagneticFieldStrength into 3T (>= 2), 1.5T (< 2) and unknown; every other vendor pooled as minor",
        },
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
            **{
                name: sha256_file(
                    Path(__file__).resolve().parents[2]
                    / "src/knee_abnormality/data"
                    / name
                )
                for name in ("labels.py", "splits.py")
            },
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
                "validation_scanner_groups": dict(
                    scanner_counts.filter(pl.col("fold") == fold)
                    .select("scanner_group", "len")
                    .iter_rows()
                ),
            }
            for fold in range(n_splits)
        ],
    }
    write_split(
        split_dir,
        {
            "studies.csv": csv,
            "label_counts.csv": counts.write_csv(),
            "targets.csv": evaluation_csv,
            "protocol.json": json.dumps(metadata, indent=2) + "\n",
        },
    )
    return metadata


def write_split(split_dir: Path, artifacts: dict[str, str]) -> None:
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


def retarget(
    data_dir: Path,
    source_dir: Path,
    split_dir: Path,
    config: ExperimentConfig,
    label_manifest: Path,
    label_source: str,
) -> dict:
    source = load_split(data_dir, source_dir)
    targets = source.protocol["targets"]
    labels, provenance = load_source(
        config, json.loads(label_manifest.read_text()), label_source, tuple(targets)
    )
    official = pl.read_csv(data_dir / "train.csv", infer_schema_length=None).select(
        STUDY, *targets
    )
    evaluation = (
        source.membership.select(STUDY, "is_gold")
        .join(official, on=STUDY, how="left", validate="1:1")
        .join(
            labels.rename({t: f"source_{t}" for t in targets}),
            on=STUDY,
            how="left",
            validate="1:1",
        )
        .join(
            source.evaluation.rename({t: f"frozen_{t}" for t in targets}),
            on=STUDY,
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.when(pl.col("is_gold"))
            .then(pl.col(t))
            .otherwise(pl.coalesce(f"source_{t}", f"frozen_{t}"))
            .cast(pl.Float32)
            .alias(t)
            for t in targets
        )
        .select(STUDY, *targets)
        .sort(STUDY)
    )
    evaluation_csv = evaluation.write_csv()
    folds = source.membership.select(STUDY, "fold", "is_gold").join(
        evaluation, on=STUDY, validate="1:1"
    )
    counts = []
    for fold in range(source.protocol["n_splits"]):
        valid = folds.filter(pl.col("fold") == fold)
        gold = valid.filter(pl.col("is_gold"))
        for target in targets:
            positive = int((valid[target] >= TARGET_THRESHOLD).sum())
            gold_positive = int((gold[target] >= TARGET_THRESHOLD).sum())
            if min(positive, valid.height - positive) == 0:
                raise ValueError(f"Fold {fold}, {target}: both classes are required")
            counts.append(
                {
                    "fold": fold,
                    "target": target,
                    "positive": positive,
                    "negative": valid.height - positive,
                    "gold_positive": gold_positive,
                    "gold_negative": gold.height - gold_positive,
                }
            )
    metadata = {
        **source.protocol,
        "evaluation_target_policy": "gold where available, frozen report-derived probabilities otherwise, the source split's frozen value where the label source is null; threshold only for AUC",
        "label_source": {
            key: value for key, value in provenance.items() if key != "path"
        },
        "label_manifest_sha256": sha256_file(label_manifest),
        "evaluation_targets_sha256": hashlib.sha256(
            evaluation_csv.encode()
        ).hexdigest(),
        "retargeted_from": {
            "split_dir": str(source_dir),
            "protocol_sha256": sha256_file(source_dir / "protocol.json"),
            "targets_sha256": sha256_file(source_dir / "targets.csv"),
        },
        "generator_sha256": {
            **source.protocol["generator_sha256"],
            "validation.py": sha256_file(Path(__file__)),
        },
    }
    write_split(
        split_dir,
        {
            "studies.csv": (source_dir / "studies.csv").read_text(),
            "label_counts.csv": pl.DataFrame(counts).write_csv(),
            "targets.csv": evaluation_csv,
            "protocol.json": json.dumps(metadata, indent=2) + "\n",
        },
    )
    return metadata


class ValidationCommand(BaseModel):
    config: Path = Field(
        description="Experiment YAML supplying the data, split, seed and label defaults",
    )
    data_dir: Path | None = Field(
        default=None,
        description="Official competition CSV directory; defaults to the config's raw data",
    )
    split_dir: Path | None = Field(
        default=None, description="Fold directory; defaults to the config's split"
    )

    def locations(self) -> tuple[ExperimentConfig, Path, Path]:
        data_dir = (
            self.data_dir.expanduser().resolve() if self.data_dir is not None else None
        )
        config = load_config(
            self.config, data_root=data_dir.parent if data_dir is not None else None
        )
        split_dir = (
            self.split_dir.expanduser().resolve()
            if self.split_dir is not None
            else config.split_dir
        )
        return config, data_dir or config.raw_dir, split_dir


class Prepare(ValidationCommand):
    model_config = describe(
        "Create scanner-stratified report-group folds with full and gold target coverage"
    )

    n_splits: int = Field(default=5, ge=2, description="Number of folds")
    seed: int | None = Field(
        default=None, ge=0, lt=2**32, description="Fold seed; defaults to the config's"
    )
    scanner_table: Path = Field(
        default=STUDY_HEADERS,
        description="Per-study header table supplying the scanner strata",
    )
    label_source: str | None = Field(
        default=None, description="Label source; defaults to the config's"
    )
    gold_holdout: bool = Field(
        default=False,
        description="Put every report group with a gold study in fold 0 and stratify the other groups over all folds",
    )

    def cli_cmd(self) -> None:
        config, data_dir, split_dir = self.locations()
        result = prepare(
            data_dir,
            split_dir,
            self.n_splits,
            config.seed if self.seed is None else self.seed,
            config,
            self.scanner_table,
            config.labels.source if self.label_source is None else self.label_source,
            self.gold_holdout,
        )
        print(json.dumps(result, indent=2))


class Retarget(ValidationCommand):
    model_config = describe(
        "Copy a frozen split's folds and rebuild its frozen evaluation targets from another label source"
    )

    source_split_dir: Path = Field(description="Frozen split whose folds are kept")
    label_manifest: Path = Field(
        description="Label manifest pinning the new label source"
    )
    label_source: str = Field(description="Label table name inside the manifest")

    def cli_cmd(self) -> None:
        config, data_dir, split_dir = self.locations()
        result = retarget(
            data_dir,
            self.source_split_dir.expanduser().resolve(),
            split_dir,
            config,
            self.label_manifest,
            self.label_source,
        )
        print(json.dumps(result["label_source"], indent=2))


class Score(ValidationCommand):
    model_config = describe(
        "Score complete out-of-fold predictions with separate gold diagnostics"
    )

    predictions: Path = Field(description="Out-of-fold predictions CSV")

    def cli_cmd(self) -> None:
        _, data_dir, split_dir = self.locations()
        print(json.dumps(score(data_dir, split_dir, self.predictions), indent=2))


class ValidationCLI(CLISettings):
    model_config = describe(
        "Create the validation folds, retarget frozen folds to a new label source, or score out-of-fold predictions"
    )

    prepare: CliSubCommand[Prepare] = command(Prepare)
    retarget: CliSubCommand[Retarget] = command(Retarget)
    score: CliSubCommand[Score] = command(Score)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(ValidationCLI)


if __name__ == "__main__":
    main()
