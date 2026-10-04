import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD, require_unique_ids
from knee_abnormality.kaggle import COMPETITION
from knee_abnormality.provenance import sha256_file
from scripts.tools.prepare_combined_labels import fingerprint

TEACHER_SPLIT = Path("data/folds/all_5fold_scanner_v2")
REXNET_FOLDS_1_4 = Path("outputs/training/20261003-224423-rexnet-200-views-label-bias-folds-1-4")
LABEL_TABLE = "oof_blend_v2"
BASE_TABLE = "rebuild_state_v2"
TEACHER_WEIGHT = 0.25
TEACHER_RUNS = {
    "refined_resnet50": {
        0: Path("outputs/training/20261003-110045-refined-rn50-views-label-bias"),
        **{
            fold: Path(
                "outputs/training/20261003-122711-refined-rn50-views-label-bias-folds-1-4"
            )
            for fold in (1, 2, 3, 4)
        },
    },
    "rexnet_200": {
        0: Path("outputs/training/20261003-114512-rexnet-200-views-label-bias"),
        **{fold: REXNET_FOLDS_1_4 for fold in (1, 2, 3, 4)},
    },
}
BLEND_WEIGHT = TEACHER_WEIGHT * len(TEACHER_RUNS)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 2263738815


class PrepareCLI(CLISettings):
    model_config = describe(
        "Build the oof_blend_v2 label table (half rebuild_state_v2, a quarter each the x4 "
        "out-of-fold predictions of the Refined ResNet-50 and ReXNet-200 five-fold runs "
        "of 2026-10-03) and its pinned manifest"
    )

    base_manifest: Path = Field(
        default=Path("configs/labels/rebuild-state-v2-label-sources.json"),
        description="Pinned rebuild_state_v2 label manifest",
    )
    output_dir: Path = Field(
        default=Path("data/labels/prepared/oof-blend-v2"),
        description="Directory for the prepared label table",
    )
    output_manifest: Path = Field(
        default=Path("configs/labels/oof-blend-v2-label-sources.json"),
        description="Manifest to write for the prepared table",
    )


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


def teacher_predictions(
    runs: dict[int, Path], targets: list[str], split: pl.DataFrame
) -> tuple[pl.DataFrame, dict]:
    folds = sorted(split["fold"].unique().to_list())
    if folds != sorted(runs):
        raise ValueError(f"Teacher runs do not cover the folds of {TEACHER_SPLIT}")
    parts, provenance = [], {}
    for fold in folds:
        run = runs[fold]
        config = json.loads((run / "effective_config.json").read_text())
        if (
            Path(config["validation"]["split"]).name != TEACHER_SPLIT.name
            or fold not in config["validation"]["folds"]
            or config["labels"]["source"] != BASE_TABLE
        ):
            raise ValueError(
                f"{run} is not a {BASE_TABLE} run of fold {fold} of {TEACHER_SPLIT}"
            )
        path = run / f"fold_{fold}" / "validation_predictions.csv"
        frame = pl.read_csv(path, infer_schema_length=None)
        if frame.columns != [STUDY, "fold", *targets]:
            raise ValueError(f"{path} has an unexpected schema")
        expected = set(split.filter(pl.col("fold") == fold)[STUDY].to_list())
        if set(frame[STUDY].to_list()) != expected or (frame["fold"] != fold).any():
            raise ValueError(f"{path} does not hold exactly its held-out fold")
        parts.append(frame.select(STUDY, *targets))
        provenance[f"fold{fold}"] = {
            "run": str(run),
            "predictions": fingerprint(path),
            "selected_checkpoint": json.loads(
                (run / f"fold_{fold}" / "selected_checkpoint.json").read_text()
            ),
        }
    teacher = pl.concat(parts)
    require_unique_ids(teacher, "Teacher out-of-fold predictions")
    values = teacher.select(targets).to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Teacher predictions must be finite probabilities")
    return teacher, provenance


def macro_auc(truth: np.ndarray, scores: np.ndarray, rows: np.ndarray) -> float | None:
    values = []
    for column in range(truth.shape[1]):
        positive = truth[rows, column]
        if positive.all() or not positive.any():
            continue
        values.append(roc_auc_score(positive, scores[rows, column]))
    return float(np.mean(values)) if values else None


def gold_check(
    official: pl.DataFrame,
    tables: dict[str, pl.DataFrame],
    targets: list[str],
) -> dict:
    gold = official.filter(pl.col(targets[0]).is_not_null()).sort(STUDY)
    truth = gold.select(targets).to_numpy() >= TARGET_THRESHOLD
    scores = {
        name: gold.select(STUDY)
        .join(table, on=STUDY, how="left")
        .select(targets)
        .to_numpy()
        for name, table in tables.items()
    }
    everything = np.arange(gold.height)
    per_target = [
        {
            "target": target,
            "gold_positive": int(truth[:, column].sum()),
            **{
                name: float(roc_auc_score(truth[:, column], values[:, column]))
                for name, values in scores.items()
            },
        }
        for column, target in enumerate(targets)
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(BOOTSTRAP_DRAWS):
        rows = rng.integers(0, gold.height, gold.height)
        deltas.append(
            macro_auc(truth, scores[LABEL_TABLE], rows)
            - macro_auc(truth, scores[BASE_TABLE], rows)
        )
    deltas = np.array(deltas)
    return {
        "studies": gold.height,
        "macro_auc": {
            name: macro_auc(truth, values, everything)
            for name, values in scores.items()
        },
        "blend_minus_base": {
            "observed": macro_auc(truth, scores[LABEL_TABLE], everything)
            - macro_auc(truth, scores[BASE_TABLE], everything),
            "interval_80": [float(np.quantile(deltas, q)) for q in (0.1, 0.9)],
            "interval_95": [float(np.quantile(deltas, q)) for q in (0.025, 0.975)],
            "share_positive": float((deltas > 0).mean()),
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "per_target": per_target,
    }


def prepare(settings: PrepareCLI) -> dict:
    if settings.output_dir.exists() or settings.output_manifest.exists():
        raise FileExistsError("Preparation requires fresh output paths")
    base_manifest = json.loads(settings.base_manifest.read_text())
    base, base_record = base_labels(base_manifest)
    targets = [c for c in base.columns if c != STUDY]
    if targets != pl.read_csv("data/raw/sample_submission.csv").columns[1:]:
        raise ValueError("Base label columns differ from the competition targets")
    split = pl.read_csv(TEACHER_SPLIT / "studies.csv", infer_schema_length=None)
    if set(split[STUDY].to_list()) != set(base[STUDY].to_list()):
        raise ValueError("Teacher split and base table cover different studies")
    teachers, teacher_provenance = {}, {}
    for family, runs in TEACHER_RUNS.items():
        teachers[family], teacher_provenance[family] = teacher_predictions(
            runs, targets, split
        )
    joined = base
    for family, teacher in teachers.items():
        joined = joined.join(
            teacher, on=STUDY, how="left", suffix=f"_{family}", validate="1:1"
        )
    labels = joined.select(
        STUDY,
        *[
            (
                (1 - BLEND_WEIGHT) * pl.col(t)
                + TEACHER_WEIGHT
                * pl.sum_horizontal(pl.col(f"{t}_{family}") for family in teachers)
            ).alias(t)
            for t in targets
        ],
    ).sort(STUDY)
    require_unique_ids(labels, LABEL_TABLE, reject_blank=True)
    values = labels.select(targets).to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Blended targets must be finite probabilities")

    settings.output_dir.mkdir(parents=True)
    output_path = settings.output_dir / "labels.csv"
    labels.write_csv(output_path)
    prepared = {**fingerprint(output_path), "label_table": LABEL_TABLE}
    manifest = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "competition": COMPETITION,
        "method": (
            f"{1 - BLEND_WEIGHT:g} times {BASE_TABLE} plus {TEACHER_WEIGHT:g} times each of two "
            "image teachers: the x4 out-of-fold probabilities of the Refined-RadImageNet "
            "ResNet-50 and of ReXNet-200 on the 2026-10-03 recipes (84/93 mm views, view "
            f"label bias) on {TEACHER_SPLIT.name} ({BASE_TABLE} labels, seed 1533697213). "
            "Every study's teacher value comes from the model that held it out. Each fold's "
            "teachers trained on the other folds, so no fold's validation is clean for this "
            "table. Official gold labels still override these values in training and "
            "evaluation."
        ),
        "preparer": fingerprint(Path(__file__).resolve()),
        "base_label_table": BASE_TABLE,
        "blend_weight": BLEND_WEIGHT,
        "teacher_weight": TEACHER_WEIGHT,
        "inputs": {
            "base_manifest": fingerprint(settings.base_manifest),
            BASE_TABLE: base_record,
            "teacher_split": fingerprint(TEACHER_SPLIT / "studies.csv"),
            "teachers": teacher_provenance,
        },
        "rows": labels.height,
        "sources": [
            {
                "dataset": "local/oof-blend-labels",
                "dataset_version": 1,
                "files": [prepared],
            }
        ],
    }
    settings.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    official = pl.read_csv("data/raw/train.csv", infer_schema_length=None).select(
        STUDY, *targets
    )
    silver_ids = official.filter(pl.col(targets[0]).is_null())[STUDY].to_list()
    before = base.filter(pl.col(STUDY).is_in(silver_ids)).sort(STUDY)
    after = labels.filter(pl.col(STUDY).is_in(silver_ids)).sort(STUDY)
    silver = [
        {
            "target": t,
            "mean_before": float(before[t].mean()),
            "mean_after": float(after[t].mean()),
            "mean_absolute_change": float((after[t] - before[t]).abs().mean()),
            "positive_rate_before": float((before[t] >= TARGET_THRESHOLD).mean()),
            "positive_rate_after": float((after[t] >= TARGET_THRESHOLD).mean()),
            "cells_crossing_threshold": int(
                (
                    (before[t] >= TARGET_THRESHOLD) != (after[t] >= TARGET_THRESHOLD)
                ).sum()
            ),
        }
        for t in targets
    ]
    report = {
        "label_manifest": fingerprint(settings.output_manifest),
        "prepared_labels": prepared,
        "silver_studies": len(silver_ids),
        "silver_per_target": silver,
        "gold": gold_check(
            official,
            {BASE_TABLE: base, **teachers, LABEL_TABLE: labels},
            targets,
        ),
    }
    (settings.output_dir / "preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    report = prepare(PrepareCLI())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
