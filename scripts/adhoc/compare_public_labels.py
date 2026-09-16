import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsError
from scipy.stats import spearmanr

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, require_unique_ids
from knee_abnormality.evaluation.metrics import aucs, bootstrap_auc
from knee_abnormality.provenance import sha256_file


class CompareCLI(CLISettings):
    manifest: Path = Path("configs/labels/public-label-comparison-sources.json")
    output_dir: Path = Path("data/labels/audit/new-public-20260913/results")
    seed: int = 20260913
    bootstrap_draws: int = Field(default=5000, ge=100)


def matrix(frame, targets):
    return frame.select(targets).to_numpy().astype(float)


def paired_interval(truth, left, right, draws, seed):
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(truth), np.full(len(truth), 1 / len(truth)), size=draws
    )
    positives = counts @ truth
    valid = ((positives > 0) & (positives < len(truth))).all(axis=1)
    delta = bootstrap_auc(truth, right, counts[valid]) - bootstrap_auc(
        truth, left, counts[valid]
    )
    return {
        "draws": draws,
        "valid_draws": int(valid.sum()),
        "delta_percentile_95": np.percentile(delta, [2.5, 97.5]).tolist(),
    }


def compare(settings):
    manifest = json.loads(settings.manifest.read_text())
    pinned = manifest["inputs"] + [
        item for source in manifest["sources"] for item in source["files"]
    ]
    for item in pinned:
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Pinned input changed: {path}")
    train = pl.read_csv("data/raw/train.csv", infer_schema_length=None)
    targets = pl.read_csv("data/raw/sample_submission.csv").columns[1:]
    gold = train.drop_nulls(targets).select(STUDY, *targets).sort(STUDY)
    silver = train.select(STUDY).join(gold.select(STUDY), on=STUDY, how="anti")
    frames = {}
    for table in manifest["tables"]:
        frame = (
            pl.read_csv(table["path"], infer_schema_length=None)
            .select(
                STUDY,
                *[
                    pl.col(target + table["target_suffix"]).alias(target)
                    for target in targets
                ],
            )
            .sort(STUDY)
        )
        require_unique_ids(frame, table["name"], require_string=False)
        values = matrix(frame, targets)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid target values: {table['name']}")
        frames[table["name"]] = frame
    summaries, target_rows, pair_rows, disagreements = {}, [], [], []
    baseline = frames["steven_v4"]
    for table in manifest["tables"]:
        name = table["name"]
        frame = frames[name]
        joined = gold.join(frame, on=STUDY, suffix="__pred", validate="1:1").sort(STUDY)
        truth = matrix(joined, targets)
        pred = matrix(joined, [target + "__pred" for target in targets])
        current = (
            joined.select(STUDY).join(baseline, on=STUDY, validate="1:1").sort(STUDY)
        )
        base = matrix(current, targets)
        score = aucs(truth, pred, targets)
        baseline_score = aucs(truth, base, targets)
        gold_copied = bool(np.array_equal(truth, pred))
        nongold = silver.join(frame, on=STUDY, validate="1:1")
        values = matrix(nongold, targets)
        summary = {
            "path": table["path"],
            "sha256": table["sha256"],
            "rows": frame.height,
            "reference_rows": joined.height,
            "missing_gold": gold.join(frame.select(STUDY), on=STUDY, how="anti")[
                STUDY
            ].to_list(),
            "missing_silver": silver.join(frame.select(STUDY), on=STUDY, how="anti")[
                STUDY
            ].to_list(),
            "extra_studies": frame.join(train.select(STUDY), on=STUDY, how="anti")[
                STUDY
            ].to_list(),
            "claimed_macro_auc": table["claimed_macro_auc"],
            "claim_kind": table["claim_kind"],
            "measured_macro_auc": score["macro_auc"],
            "steven_on_same_reference_rows": baseline_score["macro_auc"],
            "delta_vs_steven": score["macro_auc"] - baseline_score["macro_auc"],
            "gold_exact_rows": int(np.all(truth == pred, axis=1).sum()),
            "gold_binary_rows": int(np.isin(pred, [0, 1]).all(axis=1).sum()),
            "all_reference_values_equal_official": gold_copied,
            "silver_binary_fraction": float(np.isin(values, [0, 1]).mean()),
            "silver_half_fraction": float((values == 0.5).mean()),
        }
        if not gold_copied and name != "steven_v4":
            summary["paired_study_bootstrap"] = paired_interval(
                truth, base, pred, settings.bootstrap_draws, settings.seed
            )
        for target in targets:
            target_rows.append(
                {
                    "source": name,
                    "target": target,
                    "reference_rows": joined.height,
                    "auc": score["per_target_auc"][target],
                    "steven_auc": baseline_score["per_target_auc"][target],
                    "delta": score["per_target_auc"][target]
                    - baseline_score["per_target_auc"][target],
                }
            )
        summaries[name] = summary

    pairs = [("steven_v4", name) for name in frames if name != "steven_v4"] + [
        ("pilkwang_v2", "zhen_v4"),
        ("pilkwang_v2", "rayan_release1"),
        ("zhen_v4", "zhen_v7"),
        ("rayan_release1", "rayan_release2"),
    ]
    for left_name, right_name in pairs:
        shared = frames[left_name].join(
            frames[right_name], on=STUDY, suffix="__right", validate="1:1"
        )
        for pool, ids in [("all", train.select(STUDY)), ("silver", silver)]:
            joined = ids.join(shared, on=STUDY, validate="1:1")
            for target in targets:
                left = joined[target].to_numpy()
                right = joined[target + "__right"].to_numpy()
                difference = np.abs(right - left)
                pair_rows.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "pool": pool,
                        "target": target,
                        "studies": joined.height,
                        "changed_cells_at_1e_12": int((difference > 1e-12).sum()),
                        "mean_absolute_difference": float(difference.mean()),
                        "max_absolute_difference": float(difference.max()),
                        "spearman": float(spearmanr(left, right).statistic),
                        "threshold_0_5_disagreements": int(
                            ((left >= 0.5) != (right >= 0.5)).sum()
                        ),
                        "difference_gt_half": int((difference > 0.5).sum()),
                        "left_mean": float(left.mean()),
                        "right_mean": float(right.mean()),
                    }
                )
                if left_name == "steven_v4" and pool == "silver":
                    for row in (
                        joined.filter(
                            (pl.col(target) - pl.col(target + "__right")).abs() > 0.5
                        )
                        .select(STUDY, target, target + "__right")
                        .iter_rows(named=True)
                    ):
                        disagreements.append(
                            {
                                "source": right_name,
                                STUDY: row[STUDY],
                                "target": target,
                                "steven_v4": row[target],
                                "candidate": row[target + "__right"],
                            }
                        )
    steven_rayan = frames["steven_v4"].join(
        frames["rayan_release1"], on=STUDY, suffix="__rayan", validate="1:1"
    )
    other_targets = [target for target in targets if target != "Fracture"]
    rayan_pilkwang = (
        frames["pilkwang_v2"]
        .select(STUDY, "Fracture")
        .join(
            frames["rayan_release1"].select(STUDY, "Fracture"),
            on=STUDY,
            suffix="__rayan",
            validate="1:1",
        )
    )
    relationships = {
        "rayan_release1_nonfracture_vs_clipped_steven": {
            "studies": steven_rayan.height,
            "targets": other_targets,
            "clip_bounds": [0.005, 0.995],
            "maximum_absolute_residual": float(
                np.abs(
                    np.clip(matrix(steven_rayan, other_targets), 0.005, 0.995)
                    - matrix(
                        steven_rayan, [target + "__rayan" for target in other_targets]
                    )
                ).max()
            ),
        },
        "rayan_release1_fracture_vs_pilkwang": {
            "shared_studies": rayan_pilkwang.height,
            "maximum_absolute_residual": float(
                (rayan_pilkwang["Fracture"] - rayan_pilkwang["Fracture__rayan"])
                .abs()
                .max()
            ),
        },
    }
    report = {
        "audited_at": datetime.now(UTC).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": sha256_file(settings.manifest),
        "configuration": settings.model_dump(mode="json"),
        "versions": {
            name: version(name) for name in ["numpy", "polars", "scipy", "scikit-learn"]
        },
        "reference_studies": gold.height,
        "silver_studies": silver.height,
        "targets": targets,
        "sources": summaries,
        "source_relationships": relationships,
        "limitations": [
            "Reference AUC measures label agreement, not image-model performance.",
            "Whole-reference official-value replacement makes direct label AUC uninformative.",
            "Nonbinary reference values do not exclude tuning on the official reference set.",
            "Paired percentile bootstrap resamples studies and excludes draws lacking a class in any target; it does not account for source selection or prior tuning.",
            "Source scores and confidence fields have different uncertainty and calibration semantics.",
            "Training, label generation and tests were not run; report text is not emitted.",
        ],
    }
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    (settings.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for name, rows in [
        ("gold_per_target", target_rows),
        ("pairwise", pair_rows),
        ("large_silver_disagreements", disagreements),
    ]:
        pl.DataFrame(rows).write_csv(settings.output_dir / f"{name}.csv")
    return report


def main() -> None:
    try:
        settings = CompareCLI()
        report = compare(settings)
    except (KeyError, OSError, ValueError, SettingsError) as error:
        print(f"Public label comparison error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        json.dumps(
            {"output_dir": str(settings.output_dir), "sources": report["sources"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
