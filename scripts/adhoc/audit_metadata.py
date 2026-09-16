import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY
from knee_abnormality.data.sources import SERIES
from knee_abnormality.provenance import sha256_file

PROFILE = ["Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
FILES = [
    "train.csv",
    "train_series.csv",
    "test.csv",
    "test_series.csv",
    "sample_submission.csv",
]


class AuditCLI(CLISettings):
    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/metadata")


def series_summary(studies: pl.DataFrame, series: pl.DataFrame) -> dict:
    study_ids = studies.select(STUDY).unique()
    series_studies = series.select(STUDY).unique()
    counts = series.group_by(STUDY).len()
    repeated = series.group_by([STUDY, *PROFILE]).len().filter(pl.col("len") > 1)
    return {
        "unique_studies": series[STUDY].n_unique(),
        "duplicate_series_ids": series.height - series[SERIES].n_unique(),
        "duplicate_study_series_pairs": series.height
        - series.unique([STUDY, SERIES]).height,
        "studies_without_series": study_ids.join(
            series_studies, on=STUDY, how="anti"
        ).height,
        "series_studies_missing_from_study_table": series_studies.join(
            study_ids, on=STUDY, how="anti"
        ).height,
        "series_per_represented_study": counts.select(
            pl.col("len").min().alias("min"),
            pl.col("len").median().alias("median"),
            pl.col("len").quantile(0.95).alias("p95"),
            pl.col("len").max().alias("max"),
        ).to_dicts()[0],
        "studies_with_repeated_profiles": repeated[STUDY].n_unique(),
        "repeated_profile_groups": repeated.height,
        "profiles": series.group_by(PROFILE)
        .agg(pl.len().alias("series"), pl.col(STUDY).n_unique().alias("studies"))
        .sort(PROFILE)
        .to_dicts(),
    }


def audit(data_dir: Path) -> tuple[dict, pl.DataFrame]:
    frames = {
        name: pl.read_csv(
            data_dir / name,
            infer_schema_length=None,
            schema_overrides={STUDY: pl.String},
        )
        for name in FILES
    }
    train, test = frames["train.csv"], frames["test.csv"]
    submission = frames["sample_submission.csv"]
    targets = [column for column in submission.columns if column != STUDY]
    if not targets or train.columns != [STUDY, "Report", *targets]:
        raise ValueError(
            "Training columns differ from the expected report and submission target schema"
        )
    if test.columns != [STUDY] or submission.columns[0] != STUDY:
        raise ValueError("Unexpected example test or submission schema")
    for name in ("train_series.csv", "test_series.csv"):
        if frames[name].columns != [STUDY, SERIES, *PROFILE]:
            raise ValueError(f"Unexpected series schema: {name}")

    observed = pl.sum_horizontal(pl.col(target).is_not_null() for target in targets)
    gold = train.filter(observed == len(targets))
    report_groups = train.group_by("Report").len().filter(pl.col("len") > 1)
    conflicting_reports = (
        gold.group_by("Report")
        .agg(pl.col(target).n_unique().alias(target) for target in targets)
        .filter(pl.any_horizontal(pl.col(target) > 1 for target in targets))
    )
    labels = pl.DataFrame(
        [
            {
                "target": target,
                "observed": train[target].count(),
                "missing": train[target].null_count(),
                "positive": (train[target] == 1).sum(),
                "negative": (train[target] == 0).sum(),
                "invalid": train.filter(
                    pl.col(target).is_not_null() & ~pl.col(target).is_in([0, 1])
                ).height,
            }
            for target in targets
        ]
    )
    test_ids = test.select(STUDY).unique()
    submission_ids = submission.select(STUDY).unique()
    report = {
        "competition": "rsna-knee-abnormality-detection",
        "audited_at": datetime.now(UTC).isoformat(),
        "polars_version": pl.__version__,
        "targets_in_submission_order": targets,
        "files": {
            name: {
                "bytes": (data_dir / name).stat().st_size,
                "sha256": sha256_file(data_dir / name),
                "rows": frame.height,
                "schema": {
                    column: str(dtype) for column, dtype in frame.schema.items()
                },
                "nulls": frame.null_count().to_dicts()[0],
                "duplicate_rows": frame.height - frame.unique().height,
            }
            for name, frame in frames.items()
        },
        "studies": {
            "duplicate_train_ids": train.height - train[STUDY].n_unique(),
            "duplicate_test_ids": test.height - test[STUDY].n_unique(),
            "train_test_overlap": train.select(STUDY)
            .unique()
            .join(test_ids, on=STUDY)
            .height,
            "fully_labeled": gold.height,
            "unlabeled": train.filter(observed == 0).height,
            "partially_labeled": train.filter(
                (observed > 0) & (observed < len(targets))
            ).height,
            "blank_or_null_reports": train.filter(
                pl.col("Report").fill_null("").str.strip_chars() == ""
            ).height,
            "duplicate_report_groups": report_groups.height,
            "studies_in_duplicate_report_groups": int(report_groups["len"].sum()),
            "duplicate_reports_with_conflicting_reference_labels": conflicting_reports.height,
        },
        "labels": labels.to_dicts(),
        "train_series": series_summary(train, frames["train_series.csv"]),
        "test_series": series_summary(test, frames["test_series.csv"]),
        "submission": {
            "duplicate_study_ids": submission.height - submission[STUDY].n_unique(),
            "missing_test_ids": test_ids.join(
                submission_ids, on=STUDY, how="anti"
            ).height,
            "extra_ids": submission_ids.join(test_ids, on=STUDY, how="anti").height,
            "same_id_order_as_test": submission[STUDY].equals(test[STUDY]),
            "invalid_prediction_cells": sum(
                submission.filter(
                    pl.col(target).is_null()
                    | ~pl.col(target).is_finite()
                    | ~pl.col(target).is_between(0, 1)
                ).height
                for target in targets
            ),
        },
        "limitations": [
            "Only the five named CSVs are audited; discovery was not a complete image-file inventory.",
            "Example test rows do not establish the hidden evaluation set size or composition.",
            "Repeated series metadata profiles do not establish duplicate pixel data.",
            "No patient identifier, image contents, external silver labels, or official evaluator were audited.",
        ],
    }
    return report, labels


def main() -> None:
    try:
        args = AuditCLI()
        report, labels = audit(args.data_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        labels.write_csv(args.output_dir / "label_counts.csv")
    except (OSError, ValueError, SettingsError) as error:
        print(f"Metadata audit error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "files"}, indent=2
        )
    )
    print(
        f"Full schemas, null counts, and SHA-256 hashes: {args.output_dir / 'audit.json'}"
    )


if __name__ == "__main__":
    main()
