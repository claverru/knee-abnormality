import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from pydantic_settings import SettingsError
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD
from knee_abnormality.provenance import copy_sources, sha256_file

SNAPSHOT = "downloaded-20260921"
WIDE = {
    "jev_hybrid": "riadmohamed42/jev-knee-labels/HYBRID_labels_scores.csv",
    "jev_text": "riadmohamed42/jev-knee-labels/FINAL2_labels_scores.csv",
    "barun_folds": "barun2104/rsna-knee-stratified-folds-and-llm-soft-labels/train_folds.csv",
    "dreaddev_soft": "dreaddevelopment/rsna-knee-labels/labels_llm_soft.csv",
    "flight_v4hybrid": "flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv",
    "freshtime_pseudo": "freshtime/rsna-knee-pseudo-labels-betta/pseudo_labels.csv",
    "giovanni_prob": "giovannicr/rodilla-pseudo-labels/pseudo_label_probabilities.csv",
    "jezreal_pseudo": "jezrealmomoh/rsna-knee-pseudo-labels/pseudo_labeled_train.csv",
    "saidul_super": "mohammadsaidulislam/rsna-knee-super-labels-v1/super_labels_v1.csv",
    "mtito_v4hybrid": "mtitoadhip/rsna-knee-labels-v4hybrid/report_labels_v4hybrid.csv",
    "niklita_v3": "niklitacheporev/rsna-knee-gold-labels-v3/report_labels_v3.csv",
    "noisy_calibrated": "noisyislands/rsna-knee-calibrated-labels/master_calibrated_labels.parquet",
    "shingo_calibrated": "shingo257/rsna-knee-calibrated-labels-v1/calibrated_silver_labels.csv",
    "tsuyu_balanced": "tsuyu122/rsna-knee-mri-balanced-labels-public/labels.parquet",
    "vmohit_weak": "vmohitrao/rsna-knee-report-labels/weak_labels.csv",
    "yunus_v3merged": "yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels/report_labels_v3.csv",
    "yunus_v2merged": "yunusgmsoy/rsna-knee-llm-labels-2-source-merged/report_labels_v2.csv",
    "yunus_llm": "yunusgmsoy/rsna-knee-llm-report-labels/yunus_llm_labels.csv",
    "zaid_llm": "zaidaliiq1000/rsna-knee-llm-labels/llm_labels.csv",
    "zhen_v8": "zhen931/rsna-knee-weak-labels-v8/report_labels_v8.csv",
}
RAMIT = "ramitnandi/rsna-knee-labels-gpt-luna/report_labels.csv"
JOSH = "joshuaziel/rsna-knee-mri-report-labels-gpt-5-6-luna/all_raw_predictions.json"
AUDITED = {
    "steven_v4": ("stevenleehans/version-6/llm_labels_v4_blend.csv", ""),
    "pilkwang_v2": ("pilkwang/version-1/report_labels_v2.csv", ""),
    "tonylin_soft": ("tonylin1026/version-3/refined_soft_labels.csv", "_soft"),
}
STRICT_MCL = ("josh_luna", "ramit_luna", "jev_text", "yunus_llm", "dreaddev_soft")
RUNS = {
    "control_B": "outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat",
    "learned_strength": "outputs/training/anatomical-attention/learned-strength",
}


class ScreenCLI(CLISettings):
    public_dir: Path = Path("data/labels/public")
    split_dir: Path = Path("data/folds/all_5fold_v3")
    output_dir: Path


def standard(frame: pl.DataFrame, targets: list[str], suffix: str = "") -> pl.DataFrame:
    return frame.select(
        pl.col(STUDY).cast(pl.Utf8),
        *[pl.col(t + suffix).cast(pl.Float64).alias(t) for t in targets],
    )


def snapshot(settings: ScreenCLI, relative: str) -> Path:
    owner, slug, name = relative.split("/", 2)
    return settings.public_dir / owner / slug / SNAPSHOT / name


def load_tables(settings: ScreenCLI, targets: list[str]) -> tuple[dict, dict]:
    tables, files = {}, {}
    for name, (relative, suffix) in AUDITED.items():
        path = settings.public_dir / relative
        tables[name] = standard(
            pl.read_csv(path, infer_schema_length=None), targets, suffix
        )
        files[name] = path
    for name, relative in WIDE.items():
        path = snapshot(settings, relative)
        frame = (
            pl.read_parquet(path)
            if path.suffix == ".parquet"
            else pl.read_csv(path, infer_schema_length=None)
        )
        tables[name], files[name] = standard(frame, targets), path
    names = {t.replace(" ", "_").replace("'", ""): t for t in targets}
    path = snapshot(settings, RAMIT)
    ramit = (
        pl.read_csv(path, infer_schema_length=None)
        .with_columns(pl.col("CLASS").replace(names))
        .pivot(on="CLASS", index=STUDY, values="SCORE", aggregate_function="mean")
    )
    tables["ramit_luna"], files["ramit_luna"] = standard(ramit, targets), path
    path = snapshot(settings, JOSH)
    josh = pl.DataFrame(json.loads(path.read_text()))
    top = josh["confidence"].max()
    josh = josh.with_columns(
        pl.col("Finding").replace(
            {t.replace(" ", "_").replace("'s", ""): t for t in targets}
        ),
        pl.when(pl.col("annotation") == 1)
        .then(0.5 + pl.col("confidence") / (2 * top))
        .otherwise(0.5 - pl.col("confidence") / (2 * top))
        .alias("score"),
    ).pivot(on="Finding", index=STUDY, values="score", aggregate_function="mean")
    tables["josh_luna"], files["josh_luna"] = standard(josh, targets), path
    return tables, files


def macro(truth: np.ndarray, values: np.ndarray, rows: np.ndarray) -> float:
    return float(
        np.mean(
            [
                roc_auc_score(truth[rows, i], values[rows, i])
                for i in range(truth.shape[1])
            ]
        )
    )


def screen(settings: ScreenCLI) -> None:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory exists: {settings.output_dir}")
    targets = pl.read_csv("data/raw/sample_submission.csv").columns[1:]
    train = pl.read_csv("data/raw/train.csv", infer_schema_length=None)
    gold = train.drop_nulls(targets).select(STUDY, *targets).sort(STUDY)
    truth = gold.select(targets).to_numpy().astype(float)
    silver_ids = train.select(STUDY).join(gold.select(STUDY), on=STUDY, how="anti")
    tables, files = load_tables(settings, targets)
    steven = tables["steven_v4"]
    steven_gold = gold.select(STUDY).join(steven, on=STUDY, how="left")
    steven_gold = steven_gold.select(targets).to_numpy().astype(float)
    mcl = targets.index("MCL")
    rows, label_rows = [], []
    for name, table in tables.items():
        values = table.select(targets).to_numpy()
        joined = gold.select(STUDY).join(table, on=STUDY, how="left")
        labels = joined.select(targets).to_numpy().astype(float)
        available = np.isfinite(labels).all(axis=1)
        copied = (np.abs(labels - truth) < 1e-9).all(axis=1) & available
        row = {
            "table": name,
            "rows": len(table),
            "train_coverage": train.select(STUDY).join(table, on=STUDY).height,
            "nonfinite_cells": int((~np.isfinite(values)).sum()),
            "gold_rows": int(available.sum()),
            "gold_rows_copied": int(copied.sum()),
        }
        if available.sum() and not copied.any():
            hard = labels[available, mcl] >= TARGET_THRESHOLD
            row |= {
                "gold_macro": macro(truth, labels, available),
                "steven_same_rows": macro(truth, steven_gold, available),
                "gold_macro_without_fracture": macro(
                    truth[:, :-1], labels[:, :-1], available
                ),
                "steven_without_fracture": macro(
                    truth[:, :-1], steven_gold[:, :-1], available
                ),
                "gold_mcl_auc": roc_auc_score(
                    truth[available, mcl], labels[available, mcl]
                ),
                "gold_mcl_called": int(hard.sum()),
                "gold_mcl_false_positive": int(
                    (hard & (truth[available, mcl] == 0)).sum()
                ),
                "gold_mcl_missed": int((~hard & (truth[available, mcl] == 1)).sum()),
                "per_target_delta": json.dumps(
                    {
                        t: round(
                            roc_auc_score(truth[available, i], labels[available, i])
                            - roc_auc_score(
                                truth[available, i], steven_gold[available, i]
                            ),
                            4,
                        )
                        for i, t in enumerate(targets)
                    }
                ),
            }
        silver = silver_ids.join(table, on=STUDY).join(
            steven, on=STUDY, suffix="_steven"
        )
        left = silver.select(targets).to_numpy().astype(float)
        right = silver.select([t + "_steven" for t in targets]).to_numpy().astype(float)
        finite = np.isfinite(left)
        row["silver_cells_crossing_steven"] = (
            float(
                ((left >= TARGET_THRESHOLD) != (right >= TARGET_THRESHOLD))[
                    finite
                ].mean()
            )
            if finite.any()
            else None
        )
        row["silver_mcl_positive_rate"] = float(
            (left[finite[:, mcl], mcl] >= TARGET_THRESHOLD).mean()
        )
        rows.append(row)
        for i, target in enumerate(targets):
            called = labels[available, i] >= TARGET_THRESHOLD
            positive = truth[available, i] == 1
            benchmarked = bool(available.sum()) and not copied.any()
            label_rows.append(
                {
                    "table": name,
                    "target": target,
                    "gold_auc": roc_auc_score(positive, labels[available, i])
                    if benchmarked
                    else None,
                    "gold_false_positive": int((called & ~positive).sum())
                    if benchmarked
                    else None,
                    "gold_missed": int((~called & positive).sum())
                    if benchmarked
                    else None,
                    "silver_positive_rate": float(
                        (left[finite[:, i], i] >= TARGET_THRESHOLD).mean()
                    )
                    if finite[:, i].any()
                    else None,
                }
            )

    silver = silver_ids
    for name in ("steven_v4", *STRICT_MCL):
        silver = silver.join(
            tables[name].select(STUDY, pl.col("MCL").alias(name)), on=STUDY, how="left"
        )
    calls = {
        name: silver[name].to_numpy().astype(float) >= TARGET_THRESHOLD
        for name in ("steven_v4", *STRICT_MCL)
    }
    present = {
        name: np.isfinite(silver[name].to_numpy().astype(float)) for name in calls
    }
    votes = sum(calls[name] & present[name] for name in STRICT_MCL)
    voters = sum(present[name] for name in STRICT_MCL)
    majority = votes * 2 > voters
    strict = {
        "sources": list(STRICT_MCL),
        "silver_studies": len(silver),
        "steven_positives": int(calls["steven_v4"].sum()),
        "majority_positives": int(majority.sum()),
        "steven_positives_without_strict_vote": int(
            (calls["steven_v4"] & (votes == 0)).sum()
        ),
        "positives_inside_steven": {
            name: [
                int(calls[name].sum()),
                int((calls[name] & calls["steven_v4"]).sum()),
            ]
            for name in STRICT_MCL
        },
        "pairwise_jaccard": {
            f"{a}|{b}": float(
                (calls[a] & calls[b]).sum() / max((calls[a] | calls[b]).sum(), 1)
            )
            for a, b in combinations(STRICT_MCL, 2)
        },
    }
    majority_frame = silver.select(STUDY).with_columns(
        pl.Series("strict_mcl", majority.astype(float))
    )
    split_labels = pl.read_csv(settings.split_dir / "targets.csv")
    fold = pl.read_csv(settings.split_dir / "studies.csv").filter(pl.col("fold") == 0)
    rescored = {}
    for run, root in RUNS.items():
        frame = (
            pl.read_csv(Path(root) / "fold_0" / "validation_predictions.csv")
            .join(fold.select(STUDY), on=STUDY)
            .join(
                split_labels.select(STUDY, pl.col("MCL").alias("split_mcl")), on=STUDY
            )
            .join(majority_frame, on=STUDY)
        )
        rescored[run] = {
            "silver_fold0_studies": len(frame),
            "strict_positives": int(frame["strict_mcl"].sum()),
            "split_positives": int((frame["split_mcl"] >= TARGET_THRESHOLD).sum()),
            "mcl_auc_split_labels": roc_auc_score(
                frame["split_mcl"] >= TARGET_THRESHOLD, frame["MCL"]
            ),
            "mcl_auc_strict_majority": roc_auc_score(frame["strict_mcl"], frame["MCL"]),
        }
    strict["fold0_rescore"] = rescored

    settings.output_dir.mkdir(parents=True)
    screen_frame = pl.DataFrame(rows, infer_schema_length=None)
    screen_frame.write_csv(settings.output_dir / "screen.csv")
    pl.DataFrame(label_rows).write_csv(settings.output_dir / "per_label.csv")
    majority_frame.write_csv(settings.output_dir / "strict_mcl_majority.csv")
    (settings.output_dir / "strict_mcl.json").write_text(json.dumps(strict, indent=2))
    sources = copy_sources((Path(__file__),), settings.output_dir / "source")
    manifest = {
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                *files.values(),
                Path("data/raw/train.csv"),
                settings.split_dir / "targets.csv",
                settings.split_dir / "studies.csv",
            )
        },
        "runs": RUNS,
        "sources": sources,
    }
    (settings.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=260, float_precision=4):
        print(
            screen_frame.drop("per_target_delta").sort(
                "gold_macro", descending=True, nulls_last=True
            )
        )
    print(json.dumps(strict, indent=2))


def main() -> None:
    try:
        screen(ScreenCLI())
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Label screen error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
