import json
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import scipy
import sklearn
from pydantic_settings import SettingsError
from scipy.stats import spearmanr

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, require_unique_ids
from knee_abnormality.evaluation.metrics import aucs
from knee_abnormality.provenance import sha256_file


class AuditCLI(CLISettings):
    manifest: Path = Path("configs/labels/public-label-sources.json")
    output_dir: Path = Path("data/labels/audit/public-v1")


def matrix(frame: pl.DataFrame, targets: list[str]) -> np.ndarray:
    return frame.select(targets).to_numpy().astype(float)


def auc_summary(gold: pl.DataFrame, frame: pl.DataFrame, targets: list[str]) -> dict:
    joined = gold.select(STUDY, *targets).join(
        frame.select(STUDY, *targets), on=STUDY, suffix="__source", validate="1:1"
    )
    result = aucs(
        matrix(joined, targets),
        matrix(joined, [target + "__source" for target in targets]),
        targets,
        strict=False,
    )
    return {
        "studies": joined.height,
        "macro_auc": result["macro_auc"],
        "per_target": [
            {"target": target, "auc": result["per_target_auc"][target]}
            for target in targets
        ],
    }


def audit(root: Path, manifest_path: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    pinned_files = manifest["inputs"] + [
        file for source in manifest["sources"] for file in source["files"]
    ]
    for item in pinned_files:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Pinned input changed: {item['path']}")

    train = pl.read_csv(root / "data/raw/train.csv", infer_schema_length=None)
    targets = pl.read_csv(root / "data/raw/sample_submission.csv").columns[1:]
    gold = train.drop_nulls(targets)
    folds = pl.read_csv(root / "data/folds/gold_5fold_v1/studies.csv")
    frames, table_files = {}, {}
    for source in manifest["sources"]:
        for file in source["files"]:
            if "label_table" in file:
                name = file["label_table"]
                frame = pl.read_csv(root / file["path"], infer_schema_length=None)
                if any(column not in frame.columns for column in [STUDY, *targets]):
                    raise ValueError(f"Missing required columns: {name}")
                require_unique_ids(frame, name, require_string=False)
                frames[name], table_files[name] = frame, file

    common_gold = gold
    for frame in frames.values():
        common_gold = common_gold.join(frame.select(STUDY), on=STUDY, how="semi")

    summaries, target_rows, value_rows, verdict_rows, group_rows = {}, [], [], [], []
    for name, frame in frames.items():
        values = matrix(frame, targets)
        invalid = ~np.isfinite(values) | (values < 0) | (values > 1)
        missing = train.select(STUDY).join(frame.select(STUDY), on=STUDY, how="anti")
        extra = frame.select(STUDY).join(train.select(STUDY), on=STUDY, how="anti")
        joined_gold = gold.select(STUDY, *targets).join(
            frame.select(STUDY, *targets), on=STUDY, suffix="__source", validate="1:1"
        )
        truth = matrix(joined_gold, targets)
        pred = matrix(joined_gold, [target + "__source" for target in targets])
        report_groups = (
            frame.select(STUDY, *targets)
            .join(folds.select(STUDY, "report_group"), on=STUDY, validate="1:1")
            .group_by("report_group")
            .agg(
                pl.len().alias("studies"),
                *[pl.col(target).n_unique().alias(target) for target in targets],
            )
            .filter(pl.col("studies") > 1)
        )
        inconsistent = report_groups.filter(
            pl.any_horizontal(pl.col(target) > 1 for target in targets)
        )
        for row in inconsistent.to_dicts():
            group_rows.append({"source": name, **row})
        summary = {
            "path": table_files[name]["path"],
            "sha256": table_files[name]["sha256"],
            "rows": frame.height,
            "unique_studies": frame[STUDY].n_unique(),
            "blank_ids": frame.filter(pl.col(STUDY).str.strip_chars() == "").height,
            "duplicate_rows": frame.height - frame.unique().height,
            "schema": {key: str(value) for key, value in frame.schema.items()},
            "target_order_matches_submission": [
                c for c in frame.columns if c in targets
            ]
            == targets,
            "extra_columns": [c for c in frame.columns if c not in [STUDY, *targets]],
            "missing_studies": missing.join(folds, on=STUDY).to_dicts(),
            "extra_studies": extra[STUDY].to_list(),
            "invalid_target_cells": int(invalid.sum()),
            "exact_half_cells": int((values == 0.5).sum()),
            "binary_cells": int(np.isin(values, [0, 1]).sum()),
            "gold_agreement": {
                "available_studies": joined_gold.height,
                "exact_cells": int((truth == pred).sum()),
                "exact_rows": int(np.all(truth == pred, axis=1).sum()),
                "fully_binary_source_rows": int(
                    np.all(np.isin(pred, [0, 1]), axis=1).sum()
                ),
                "threshold_0_5_disagreements": int(((pred > 0.5) != truth).sum()),
                "interpretation": "Exact matches cannot prove gold replacement; nonbinary rows rule out wholesale binary replacement for those rows. Prior gold tuning is not detectable from values alone.",
            },
            "available_gold": auc_summary(gold, frame, targets),
            "common_gold": auc_summary(common_gold, frame, targets),
            "repeated_normalized_report_groups": report_groups.height,
            "repeated_report_groups_with_different_targets": inconsistent.height,
            "studies_in_inconsistent_groups": int(inconsistent["studies"].sum()),
        }
        summaries[name] = summary
        for i, target in enumerate(targets):
            col, vals = frame[target], values[:, i]
            conf_name, verdict_name = target + "__conf", target + "__verdict"
            stats = {
                "source": name,
                "target": target,
                "rows": frame.height,
                "nulls": col.null_count(),
                "invalid": int(invalid[:, i].sum()),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "mean": float(np.nanmean(vals)),
                "unique_values": col.n_unique(),
                "zero": int((vals == 0).sum()),
                "half": int((vals == 0.5).sum()),
                "one": int((vals == 1).sum()),
                "below_half": int((vals < 0.5).sum()),
                "above_half": int((vals > 0.5).sum()),
                "confidence_nulls": None,
                "confidence_invalid": None,
                "confidence_min": None,
                "confidence_max": None,
                "confidence_mean": None,
                "verdict_unk": None,
                "verdict_nulls": None,
                "verdict_invalid": None,
            }
            if conf_name in frame.columns:
                conf = frame[conf_name].to_numpy()
                stats.update(
                    {
                        "confidence_nulls": frame[conf_name].null_count(),
                        "confidence_invalid": int(
                            (~np.isfinite(conf) | (conf < 0) | (conf > 1)).sum()
                        ),
                        "confidence_min": float(np.nanmin(conf)),
                        "confidence_max": float(np.nanmax(conf)),
                        "confidence_mean": float(np.nanmean(conf)),
                    }
                )
            if verdict_name in frame.columns:
                stats["verdict_unk"] = int((frame[verdict_name] == "UNK").sum())
                stats["verdict_nulls"] = frame[verdict_name].null_count()
                stats["verdict_invalid"] = frame.filter(
                    pl.col(verdict_name).is_null()
                    | ~pl.col(verdict_name).is_in(["YES", "NO", "UNK"])
                ).height
                for row in (
                    frame.group_by(verdict_name, target, conf_name).len().to_dicts()
                ):
                    verdict_rows.append(
                        {
                            "source": name,
                            "target": target,
                            "verdict": row[verdict_name],
                            "value": row[target],
                            "confidence": row[conf_name],
                            "cells": row["len"],
                        }
                    )
            target_rows.append(stats)
            for kind, column in [("target", target), ("confidence", conf_name)]:
                if column in frame.columns:
                    for row in frame.group_by(column).len().sort(column).to_dicts():
                        value_rows.append(
                            {
                                "source": name,
                                "target": target,
                                "kind": kind,
                                "value": row[column],
                                "cells": row["len"],
                            }
                        )

    pair_rows, review_rows = [], []
    for left_name, right_name in combinations(
        [n for n in frames if n != "lixin_with_conf"], 2
    ):
        joined = (
            frames[left_name]
            .select(STUDY, *targets)
            .join(
                frames[right_name].select(STUDY, *targets),
                on=STUDY,
                suffix="__right",
                validate="1:1",
            )
        )
        for target in targets:
            left, right = (
                joined[target].to_numpy(),
                joined[target + "__right"].to_numpy(),
            )
            valid = (
                np.isfinite(left)
                & np.isfinite(right)
                & (left >= 0)
                & (left <= 1)
                & (right >= 0)
                & (right <= 1)
            )
            a, b = left[valid], right[valid]
            correlation = (
                float(spearmanr(a, b).statistic)
                if len(np.unique(a)) > 1 and len(np.unique(b)) > 1
                else None
            )
            pair_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "target": target,
                    "shared_studies": joined.height,
                    "valid_pairs": int(valid.sum()),
                    "mean_absolute_difference": float(np.mean(np.abs(a - b))),
                    "spearman": correlation,
                    "opposite_sides_of_half": int(((a - 0.5) * (b - 0.5) < 0).sum()),
                    "one_or_both_half": int(((a == 0.5) | (b == 0.5)).sum()),
                    "absolute_difference_gt_half": int((np.abs(a - b) > 0.5).sum()),
                }
            )
            if (left_name, right_name) == ("steven_v4", "pilkwang_v2"):
                for row in (
                    joined.filter(
                        (pl.col(target) - pl.col(target + "__right")).abs() > 0.5
                    )
                    .select(STUDY, target, target + "__right")
                    .to_dicts()
                ):
                    review_rows.append(
                        {
                            STUDY: row[STUDY],
                            "target": target,
                            "steven_v4": row[target],
                            "pilkwang_v2": row[target + "__right"],
                        }
                    )

    aligned = {name: frame.sort(STUDY) for name, frame in frames.items()}
    all_study_names = [
        "steven_v1",
        "steven_v2",
        "steven_v4",
        "lixin_targets",
        "lixin_with_conf",
    ]
    for name in all_study_names:
        if not aligned[name][STUDY].equals(aligned["steven_v1"][STUDY]):
            raise ValueError("Source relationships require identical study membership")
    v1, v2, v4, lixin = [matrix(aligned[n], targets) for n in all_study_names[:4]]
    syn, eff = targets.index("Synovitis"), targets.index("Effusion")
    expected_v2 = v1.copy()
    silent_syn = v1[:, syn] == 0.5
    expected_v2[silent_syn, syn] = 0.25 + 0.5 * v1[silent_syn, eff]
    relationships = {
        "steven_v2": {
            "formula": "v1 Synovitis == 0.5 -> 0.25 + 0.5 * v1 Effusion; other cells unchanged",
            "max_absolute_residual": float(np.max(np.abs(expected_v2 - v2))),
            "v1_silent_synovitis_studies": int(silent_syn.sum()),
            "changed_cells_by_target": dict(
                zip(targets, (v1 != v2).sum(axis=0).tolist())
            ),
        },
        "steven_v4": {
            "formula": "0.5 * steven_v2 + 0.5 * lixin_targets, in probability space",
            "max_absolute_residual": float(np.max(np.abs(v4 - (v2 + lixin) / 2))),
            "v1_half_cells_no_longer_half": int(((v1 == 0.5) & (v4 != 0.5)).sum()),
            "v4_half_cells_not_half_in_v1": int(((v1 != 0.5) & (v4 == 0.5)).sum()),
            "v1_half_v4_quarter_cells": int(((v1 == 0.5) & (v4 == 0.25)).sum()),
        },
        "lixin_duplicate_target_files": {
            "max_absolute_residual": float(
                np.max(np.abs(lixin - matrix(aligned["lixin_with_conf"], targets)))
            ),
        },
    }
    fold_coverage = []
    for name, frame in frames.items():
        for fold in range(5):
            membership = folds.filter((pl.col("fold") == fold) & pl.col("is_gold"))
            fold_coverage.append(
                {
                    "source": name,
                    "fold": fold,
                    "reference_studies": membership.height,
                    "reference_studies_with_source": membership.join(
                        frame.select(STUDY), on=STUDY, how="semi"
                    ).height,
                }
            )

    report = {
        "audited_at": datetime.now(UTC).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": sha256_file(manifest_path),
        "versions": {
            "polars": pl.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "targets_in_submission_order": targets,
        "training_studies": train.height,
        "reference_studies": gold.height,
        "common_reference_studies": common_gold.height,
        "sources": summaries,
        "source_relationships": relationships,
        "fold_reference_coverage": fold_coverage,
        "limitations": [
            "Scores measure label-table agreement, not image-model performance or independent OOF quality.",
            "Public source selection, repair, and blending have used reference-set feedback; current folds cannot undo that history.",
            "A value of 0.5, an UNK verdict, and a confidence field have source-specific meanings; value counts are not clinical prevalence.",
            "Missing report evidence is not established absence, and source confidence is not verified calibration.",
            "No per-report clinical adjudication, upstream LLM rerun, training, or tests performed.",
            "Identical-report groups do not establish patient or image identity; exact value differences can include rounding.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for name, rows in [
        ("target_statistics", target_rows),
        ("value_counts", value_rows),
        ("verdict_mapping", verdict_rows),
        ("pairwise", pair_rows),
        ("report_group_disagreements", group_rows),
        ("large_disagreements", review_rows),
        ("fold_reference_coverage", fold_coverage),
    ]:
        pl.DataFrame(rows).write_csv(output_dir / f"{name}.csv")
    return report


def main() -> None:
    try:
        args = AuditCLI()
        report = audit(Path.cwd(), args.manifest, args.output_dir)
    except (KeyError, OSError, ValueError, SettingsError) as error:
        print(f"Public label audit error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "common_reference_studies": report["common_reference_studies"],
                "sources": {
                    name: {
                        "rows": source["rows"],
                        "invalid_cells": source["invalid_target_cells"],
                        "missing_studies": len(source["missing_studies"]),
                        "available_gold": source["available_gold"]["studies"],
                        "available_gold_auc": source["available_gold"]["macro_auc"],
                        "common_gold_auc": source["common_gold"]["macro_auc"],
                    }
                    for name, source in report["sources"].items()
                },
                "source_relationships": report["source_relationships"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
