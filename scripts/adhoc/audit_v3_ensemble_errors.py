import json
from pathlib import Path

import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY, require_unique_ids
from scripts.tools.prepare_combined_labels import fingerprint


class AuditCLI(CLISettings):
    model_config = describe(
        "Rank v3 label errors against the frozen three-family image ensemble and group "
        "them by scanner and report style to locate repeated labeling weaknesses"
    )
    output_dir: Path = Field(
        default=Path(".artifacts/v3_ensemble_audit"),
        description="Fresh private directory for ranked cells, group summaries and provenance",
    )


def summarize(frame: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
    return (
        frame.group_by(groups)
        .agg(
            pl.len().alias("cells"),
            pl.col(STUDY).n_unique().alias("studies"),
            pl.col("abs_error").mean().alias("mae"),
            pl.col("abs_error").pow(2).mean().alias("mse"),
            pl.col("cross_entropy").mean(),
            pl.col("v3").mean().alias("mean_label"),
            pl.col("ensemble").mean().alias("mean_prediction"),
            pl.col("severe").sum().alias("severe_cells"),
            pl.col("severe").mean().alias("severe_rate"),
            pl.col("label_high_models_low").sum(),
            pl.col("label_low_models_high").sum(),
            pl.col("report_conflict").sum(),
        )
        .sort(["severe_rate", "cells"], descending=True)
    )


def audit(settings: AuditCLI) -> dict:
    output = settings.output_dir
    if not output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Outputs must stay inside this worktree")
    if output.exists():
        raise FileExistsError(output)
    label_path = Path("data/labels/prepared/rebuild-state-v3/labels.csv")
    split_path = Path("data/folds/all_5fold_scanner_v2/studies.csv")
    header_path = Path(
        "outputs/analysis/s/sharing-disagreement-20260914/study_headers.parquet"
    )
    audit_path = Path("outputs/labels-v3-reading/v3_1-audit/ranked_cells.parquet")
    raw_path = Path("data/raw/train.csv")
    repair_path = Path(".artifacts/v3_2/prepared/rebuild-state-v3-2/labels.csv")
    labels = pl.read_csv(label_path)
    targets = labels.columns[1:]
    split = pl.read_csv(split_path).select(STUDY, "fold", "is_gold")
    headers = pl.read_parquet(header_path).select(
        STUDY,
        "vendor",
        "ManufacturerModelName",
        "MagneticFieldStrength",
        "scanner_group",
    )
    for name, frame in [("labels", labels), ("split", split), ("headers", headers)]:
        require_unique_ids(frame, name)
        if set(frame[STUDY]) != set(labels[STUDY]):
            raise ValueError(f"Study coverage differs: {name}")
    paths = [label_path, split_path, header_path, audit_path, raw_path, repair_path]
    values = labels.unpivot(index=STUDY, variable_name="label", value_name="v3")
    for family in ("convnext", "resnet50", "nextvit"):
        family_paths = [
            Path(
                "outputs/predictions/rebuild/"
                f"{family}-{'silver' if family == 'nextvit' and fold != 2 else 'gen1'}-fold-{fold}.parquet"
            )
            for fold in range(5)
        ]
        paths.extend(family_paths)
        predictions = pl.concat([pl.read_parquet(path) for path in family_paths])
        require_unique_ids(predictions, family)
        if set(predictions[STUDY]) != set(labels[STUDY]):
            raise ValueError(f"Prediction coverage differs: {family}")
        aligned = predictions.join(split, on=STUDY, validate="1:1", suffix="_expected")
        if aligned.filter(pl.col("fold") != pl.col("fold_expected")).height:
            raise ValueError(f"Fold mismatch: {family}")
        values = values.join(
            predictions.select(STUDY, *targets).unpivot(
                index=STUDY, variable_name="label", value_name=family
            ),
            on=[STUDY, "label"],
            validate="1:1",
        )
    details = pl.read_parquet(audit_path).select(
        STUDY, "label", "batch", "n", "verdict", "rule_changed", "Report"
    )
    raw = pl.read_csv(raw_path).select(STUDY, pl.col("Report").alias("raw_report"))
    if (
        details.join(raw, on=STUDY)
        .filter(pl.col("Report") != pl.col("raw_report"))
        .height
    ):
        raise ValueError("Cached report text differs from the official source")
    values = (
        values.join(split, on=STUDY, validate="m:1")
        .join(headers, on=STUDY, validate="m:1")
        .join(details, on=[STUDY, "label"], validate="1:1")
        .join(
            pl.read_csv(repair_path).unpivot(
                index=STUDY, variable_name="label", value_name="v3_2"
            ),
            on=[STUDY, "label"],
            validate="1:1",
        )
        .filter(~pl.col("is_gold") & pl.col("v3").is_not_null())
        .with_columns(
            pl.mean_horizontal("convnext", "resnet50", "nextvit").alias("ensemble"),
            pl.min_horizontal("convnext", "resnet50", "nextvit").alias("family_min"),
            pl.max_horizontal("convnext", "resnet50", "nextvit").alias("family_max"),
            pl.col("batch").str.replace(r"^batch-\d+-", "").alias("report_style"),
            pl.concat_str(
                "vendor",
                pl.col("ManufacturerModelName").str.to_uppercase(),
                "MagneticFieldStrength",
                separator=" / ",
            )
            .fill_null("unknown")
            .alias("scanner"),
        )
        .with_columns(
            (pl.col("v3") - pl.col("ensemble")).abs().alias("abs_error"),
            (pl.col("v3_2") - pl.col("ensemble")).abs().alias("v32_abs_error"),
            (pl.col("v3") - pl.col("v3_2")).abs().gt(1e-12).alias("already_repaired"),
            (pl.col("v3") >= 0.5).alias("label_positive"),
            ((pl.col("v3") >= 0.5) & (pl.col("family_max") <= 0.3)).alias(
                "label_high_models_low"
            ),
            ((pl.col("v3") < 0.4) & (pl.col("family_min") >= 0.6)).alias(
                "label_low_models_high"
            ),
            (
                ((pl.col("v3") >= 0.5) & pl.col("verdict").is_in(["no", "unlikely"]))
                | ((pl.col("v3") < 0.4) & pl.col("verdict").is_in(["yes", "probable"]))
            ).alias("report_conflict"),
            -(
                pl.col("v3") * pl.col("ensemble").clip(1e-6, 1 - 1e-6).log()
                + (1 - pl.col("v3"))
                * (1 - pl.col("ensemble").clip(1e-6, 1 - 1e-6)).log()
            ).alias("cross_entropy"),
        )
        .with_columns((pl.col("abs_error") >= 0.5).alias("severe"))
        .sort(["abs_error", STUDY, "label"], descending=[True, False, False])
        .with_row_index("rank", offset=1)
    )
    output.mkdir(parents=True)
    values.write_parquet(output / "ranked_cells.parquet")
    values.head(100).write_csv(output / "top100.csv")
    baseline = summarize(values, ["label"])
    baseline.write_csv(output / "targets.csv")
    for name, groups in {
        "scanners": ["scanner"],
        "scanner_targets": ["scanner", "label"],
        "vendor_field_targets": ["scanner_group", "label"],
        "report_style_targets": ["report_style", "label"],
        "scanner_report_targets": ["scanner", "report_style", "label"],
        "verdict_targets": ["verdict", "label"],
    }.items():
        summary = summarize(values, groups)
        if "label" in groups:
            summary = summary.join(
                baseline.select(
                    "label", pl.col("severe_rate").alias("global_target_rate")
                ),
                on="label",
            ).with_columns(
                (pl.col("severe_rate") / pl.col("global_target_rate")).alias(
                    "enrichment"
                )
            )
        summary.sort("severe_rate", descending=True).write_csv(output / f"{name}.csv")
    values.filter(pl.col("report_conflict") & (pl.col("abs_error") >= 0.4)).write_csv(
        output / "report_conflict_candidates.csv"
    )
    report = {
        "baseline": fingerprint(label_path),
        "cells": values.height,
        "studies": values[STUDY].n_unique(),
        "severe_cells": values["severe"].sum(),
        "ranking": "Absolute difference between v3 and arithmetic mean of three families; severe >=0.5; no report-verdict or previous-review filter",
        "nextvit": "gen1 fold 2; available silver folds 0,1,3,4; not a complete all-gen1 ensemble",
        "grouping": "Observed scanner vendor/model/field strength and report-style proxy; no institution/site ID available",
        "fold_alignment": "All three families cover each of 4407 studies once with expected frozen fold",
        "inputs": [fingerprint(path) for path in paths],
        "preparer": fingerprint(Path(__file__).resolve()),
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return {
        key: value for key, value in report.items() if key not in {"inputs", "preparer"}
    }


def main() -> None:
    print(json.dumps(audit(AuditCLI()), indent=2))


if __name__ == "__main__":
    main()
