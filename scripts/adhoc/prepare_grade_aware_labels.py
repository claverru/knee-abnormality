import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic_settings import SettingsError
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD, require_unique_ids
from knee_abnormality.provenance import sha256_file
from scripts.tools.prepare_combined_labels import (
    CONSENSUS_TARGETS,
    SNAPSHOT,
    SNAPSHOTS,
    consensus_sources,
    fingerprint,
)

LABEL_TABLE = "combined_v2"
BASE_TABLE = "combined_v1"
MINIMUM_SOURCES = 5
CONSENSUS_TOLERANCE = 1e-6
RAMIT_OWNER, RAMIT_SLUG, RAMIT_FILE = SNAPSHOTS["ramit_luna"]
RAMIT = Path(str(SNAPSHOT).format(owner=RAMIT_OWNER, slug=RAMIT_SLUG, name=RAMIT_FILE))


class PrepareCLI(CLISettings):
    base_manifest: Path = Path("configs/labels/combined-label-sources.json")
    comparison_manifest: Path = Path(
        "configs/labels/public-label-comparison-sources.json"
    )
    output_dir: Path = Path("data/labels/prepared/combined-v2")
    output_manifest: Path = Path("configs/labels/combined-v2-label-sources.json")


def base_labels(manifest: dict) -> tuple[pl.DataFrame, dict]:
    records = [
        record
        for source in manifest["sources"]
        for record in source["files"]
        if record.get("label_table") == BASE_TABLE
    ]
    if len(records) != 1:
        raise ValueError(f"Expected one {BASE_TABLE} record in the base manifest")
    record = records[0]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Pinned input changed: {path}")
    frame = pl.read_csv(path, infer_schema_length=None)
    require_unique_ids(frame, BASE_TABLE)
    return frame, record


def low_grade_cells(targets: list[str]) -> pl.DataFrame:
    classes = {t.replace(" ", "_").replace("'", ""): t for t in targets}
    calls = pl.read_csv(RAMIT, infer_schema_length=0)
    if not set(calls["CLASS"].unique()) <= set(classes):
        raise ValueError("Ramit classes differ from the competition targets")
    cells = calls.filter(
        (pl.col("classification") == "POSITIVE") & (pl.col("severity") == "LOW")
    ).select(STUDY, pl.col("CLASS").replace(classes).alias("target"))
    if cells.select(STUDY, "target").is_duplicated().any():
        raise ValueError("Ramit holds duplicate study-target calls")
    return cells


def consensus_long(
    sources: dict[str, pl.DataFrame], targets: list[str]
) -> pl.DataFrame:
    stacked = pl.concat(
        frame.unpivot(index=STUDY, on=targets, variable_name="target", value_name="v")
        for frame in sources.values()
    )
    return stacked.group_by(STUDY, "target").agg(
        pl.col("v").mean().alias("consensus"),
        pl.col("v").is_not_null().sum().alias("sources"),
    )


def prepare(settings: PrepareCLI) -> dict:
    if settings.output_dir.exists() or settings.output_manifest.exists():
        raise FileExistsError("Preparation requires fresh output paths")
    base_manifest = json.loads(settings.base_manifest.read_text())
    comparison = json.loads(settings.comparison_manifest.read_text())
    base, base_record = base_labels(base_manifest)
    targets = [c for c in base.columns if c != STUDY]
    if targets != pl.read_csv("data/raw/sample_submission.csv").columns[1:]:
        raise ValueError("Base label columns differ from the competition targets")
    inputs, sources = consensus_sources(comparison, targets)
    consensus = consensus_long(sources, targets)
    cells = low_grade_cells(targets).join(consensus, on=[STUDY, "target"], how="left")
    replaced = cells.filter(pl.col("sources") >= MINIMUM_SOURCES)
    base_long = base.unpivot(
        index=STUDY, on=targets, variable_name="target", value_name="base"
    )
    long = base_long.join(
        replaced.select(STUDY, "target", "consensus"),
        on=[STUDY, "target"],
        how="left",
    ).with_columns(pl.coalesce("consensus", "base").alias("value"))
    drift = long.filter(
        pl.col("target").is_in(CONSENSUS_TARGETS) & pl.col("consensus").is_not_null()
    ).select((pl.col("value") - pl.col("base")).abs().max())
    if (drift.item() or 0.0) > CONSENSUS_TOLERANCE:
        raise ValueError("Consensus targets differ from their combined_v1 values")
    labels = (
        long.pivot(on="target", index=STUDY, values="value")
        .select(STUDY, *targets)
        .sort(STUDY)
    )
    require_unique_ids(labels, LABEL_TABLE, reject_blank=True)
    if labels.height != base.height:
        raise ValueError("Grade-aware table lost or gained studies")
    values = labels.select(targets).to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Grade-aware targets must be finite probabilities")

    settings.output_dir.mkdir(parents=True)
    output_path = settings.output_dir / "labels.csv"
    labels.write_csv(output_path)
    prepared = {**fingerprint(output_path), "label_table": LABEL_TABLE}
    manifest = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "competition": "rsna-knee-abnormality-detection",
        "method": (
            f"{BASE_TABLE}, except that every study-target cell Ramit's GPT-luna extraction "
            "calls a LOW-severity positive takes the per-study mean of the available "
            "probabilities from the nine consensus extractors, whatever its target, when "
            f"at least {MINIMUM_SOURCES} of them cover the cell. The consensus targets "
            f"({', '.join(CONSENSUS_TARGETS)}) already equal that mean and do not change. "
            "Official gold labels still override these values in training and evaluation."
        ),
        "preparer": fingerprint(Path(__file__).resolve()),
        "base_label_table": BASE_TABLE,
        "minimum_sources": MINIMUM_SOURCES,
        "consensus_sources": list(sources),
        "inputs": {
            "base_manifest": fingerprint(settings.base_manifest),
            BASE_TABLE: base_record,
            "ramit_calls": fingerprint(RAMIT),
            **inputs,
        },
        "rows": labels.height,
        "sources": [
            {
                "dataset": "local/combined-labels",
                "dataset_version": 2,
                "files": [prepared],
            }
        ],
    }
    settings.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    official = pl.read_csv("data/raw/train.csv", infer_schema_length=None)
    silver_ids = official.filter(pl.col(targets[0]).is_null()).select(STUDY)
    gold = (
        official.filter(pl.col(targets[0]).is_not_null())
        .unpivot(index=STUDY, on=targets, variable_name="target", value_name="gold")
        .join(long, on=[STUDY, "target"])
    )
    gold_rows = []
    for target in targets:
        part = gold.filter(pl.col("target") == target)
        truth = part["gold"].to_numpy() >= TARGET_THRESHOLD
        gold_rows.append(
            {
                "target": target,
                "gold_positive": int(truth.sum()),
                "changed_cells": int(part["consensus"].is_not_null().sum()),
                "auc_base": float(roc_auc_score(truth, part["base"].to_numpy())),
                "auc_grade_aware": float(
                    roc_auc_score(truth, part["value"].to_numpy())
                ),
            }
        )
    silver = long.join(silver_ids, on=STUDY)
    changed = silver.filter(pl.col("consensus").is_not_null())
    per_target = []
    for target in targets:
        part = silver.filter(pl.col("target") == target)
        moved = changed.filter(pl.col("target") == target)
        per_target.append(
            {
                "target": target,
                "changed_cells": moved.height,
                "mean_before": float(moved["base"].mean()) if moved.height else None,
                "mean_after": float(moved["value"].mean()) if moved.height else None,
                "crossed_below_threshold": int(
                    (
                        (moved["base"] >= TARGET_THRESHOLD)
                        & (moved["value"] < TARGET_THRESHOLD)
                    ).sum()
                ),
                "positive_rate_before": float(
                    (part["base"] >= TARGET_THRESHOLD).mean()
                ),
                "positive_rate_after": float(
                    (part["value"] >= TARGET_THRESHOLD).mean()
                ),
            }
        )
    report = {
        "label_manifest": fingerprint(settings.output_manifest),
        "prepared_labels": prepared,
        "low_grade_cells": cells.height,
        "replaced_cells": replaced.height,
        "kept_for_too_few_sources": cells.height - replaced.height,
        "silver_changed_cells": changed.height,
        "silver_per_target": per_target,
        "gold_studies": gold[STUDY].n_unique(),
        "gold_macro_auc_base": float(np.mean([r["auc_base"] for r in gold_rows])),
        "gold_macro_auc_grade_aware": float(
            np.mean([r["auc_grade_aware"] for r in gold_rows])
        ),
        "gold_per_target": gold_rows,
    }
    (settings.output_dir / "preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    try:
        report = prepare(PrepareCLI())
    except (OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Grade-aware label preparation error: {error!r}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
