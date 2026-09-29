import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from scipy.stats import binomtest

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD, load_split
from knee_abnormality.evaluation.metrics import aucs, soft_aucs

LABEL_SPLITS = {
    "steven_v4": "all_5fold_v3",
    "combined_v1": "all_5fold_scanner_v1",
    "rebuild_state_v2": "all_5fold_scanner_v2",
}
CHECKPOINT = re.compile(r"(?P<run>.*)/fold_(?P<fold>\d)/(?P<name>[^/]+)$")


class SubmissionMetricsCLI(CLISettings):
    model_config = describe(
        "Score every submitted checkpoint's saved validation predictions against each frozen "
        "label set and correlate the local metrics with the public leaderboard within "
        "comparable cohorts"
    )

    ledger: Path = Field(default=Path("outputs/submission/kaggle-ledger.json"), description="Machine submission ledger")
    raw_dir: Path = Field(default=Path("data/raw"), description="Official competition files")
    folds_dir: Path = Field(default=Path("data/folds"), description="Frozen split directories")
    output_dir: Path = Field(description="Fresh output directory")
    permutations: int = Field(default=20000, ge=1000, description="Within-cohort permutations")
    seed: int = Field(default=20260928, description="Permutation seed")
    decisive: float = Field(default=0.003, gt=0, description="Public difference counted as beyond seed noise")


def submitted_checkpoint(entry: dict) -> dict | None:
    bundle = Path(entry.get("bundle") or "")
    manifest = bundle / "manifest.json"
    if entry.get("public_score") in (None, "") or not manifest.exists():
        return None
    record = json.loads(manifest.read_text())
    checkpoint = record.get("checkpoint")
    match = CHECKPOINT.match(checkpoint["source_path"]) if isinstance(checkpoint, dict) else None
    if match is None or not Path(match["run"]).exists():
        return None
    run, fold, name = Path(match["run"]), int(match["fold"]), match["name"]
    runtime = record["runtime"]
    config = json.loads((run / "effective_config.json").read_text())
    epochs = sorted((run / f"fold_{fold}/epochs").glob("metrics_epoch_*.json"))
    epoch = int(re.search(r"epoch=(\d+)", name)[1]) if "epoch=" in name else len(epochs) - 1
    selected = json.loads((run / f"fold_{fold}/selected_checkpoint.json").read_text())
    return {
        "submission": entry["sequence"],
        "public": float(entry["public_score"]),
        "run": str(run),
        "fold": fold,
        "epoch": epoch,
        "split": config["validation"]["split"],
        "labels": config["labels"]["source"],
        "validation_multiplier": config["data"].get("validation_image_multiplier"),
        "inference": f"x{runtime.get('image_multiplier', 2)} tau{runtime.get('attention_temperature', 1.0)}"
        + (" tta" if runtime.get("reflection_tta") else ""),
        "is_selected": Path(selected["path"]).name == name,
        "prediction_multiplier": selected.get("prediction_image_multiplier", config["data"].get("validation_image_multiplier")),
    }


def recorded_metrics(row: dict) -> dict:
    path = Path(row["run"]) / f"fold_{row['fold']}/epochs/metrics_epoch_{row['epoch']:03d}.json"
    stored = json.loads(path.read_text())
    return {
        "own_x1_macro_auc": stored["macro_auc"],
        "own_x1_soft_macro_auc": stored.get("soft_macro_auc"),
        "own_x1_soft_bce": -stored["soft_target_bce"],
        "own_x1_gold_auc": stored["gold"]["macro_auc"],
        "own_x1_report_auc": stored["report_derived"]["macro_auc"],
    }


def label_sets(settings: SubmissionMetricsCLI) -> tuple[dict[str, pl.DataFrame], list[str], pl.DataFrame]:
    frames, targets = {}, None
    for labels, split_name in LABEL_SPLITS.items():
        split = load_split(settings.raw_dir, settings.folds_dir / split_name)
        targets = split.protocol["targets"]
        frames[labels] = split.evaluation
    membership = load_split(settings.raw_dir, settings.folds_dir / LABEL_SPLITS["rebuild_state_v2"]).membership
    return frames, targets, membership.select(STUDY, "is_gold")


def score(predictions: pl.DataFrame, truth: pl.DataFrame, gold: pl.DataFrame, targets: list[str], prefix: str) -> dict:
    frame = predictions.join(truth.rename({t: f"y_{t}" for t in targets}), on=STUDY).join(gold, on=STUDY)
    probability = frame.select(targets).to_numpy().astype(np.float64)
    soft = frame.select([f"y_{t}" for t in targets]).to_numpy().astype(np.float64)
    hard = (soft >= TARGET_THRESHOLD).astype(np.int8)
    is_gold = frame["is_gold"].to_numpy().astype(bool)
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    bce = -(soft * np.log(clipped) + (1 - soft) * np.log1p(-clipped)).mean()
    return {
        f"{prefix}_macro_auc": aucs(hard, probability, targets)["macro_auc"],
        f"{prefix}_soft_macro_auc": soft_aucs(soft, probability, targets)["soft_macro_auc"],
        f"{prefix}_soft_bce": -float(bce),
        f"{prefix}_report_auc": aucs(hard[~is_gold], probability[~is_gold], targets)["macro_auc"],
    }


def consensus_auc(predictions: pl.DataFrame, frames: dict[str, pl.DataFrame], targets: list[str], prefix: str) -> dict:
    joined = predictions
    for labels, frame in frames.items():
        joined = joined.join(frame.rename({t: f"{labels}_{t}" for t in targets}), on=STUDY)
    per_target = []
    for target in targets:
        hard = np.stack([(joined[f"{labels}_{target}"].to_numpy() >= TARGET_THRESHOLD) for labels in frames])
        agree = (hard == hard[0]).all(axis=0)
        per_target.append(
            aucs(hard[0][agree, None].astype(np.int8), joined[target].to_numpy()[agree, None], [target])["macro_auc"]
        )
    return {f"{prefix}_consensus_auc": float(np.mean(per_target))}


def gold_auc(predictions: pl.DataFrame, official: pl.DataFrame, targets: list[str], prefix: str) -> dict:
    frame = predictions.join(official.rename({t: f"y_{t}" for t in targets}), on=STUDY)
    result = aucs(
        frame.select([f"y_{t}" for t in targets]).to_numpy().astype(np.int8),
        frame.select(targets).to_numpy().astype(np.float64),
        targets,
        strict=False,
    )
    return {f"{prefix}_gold_auc": result["macro_auc"]}


def prediction_metrics(row: dict, frames: dict, targets: list[str], gold: pl.DataFrame, official: pl.DataFrame) -> dict:
    run, fold = Path(row["run"]), row["fold"]
    sources = {f"val_x{row['validation_multiplier']}": run / f"fold_{fold}/epochs/validation_epoch_{row['epoch']:03d}.csv"}
    if row["is_selected"]:
        sources[f"pred_x{row['prediction_multiplier']}"] = run / "oof_predictions.csv"
    result = {}
    for source, path in sources.items():
        if not path.exists():
            continue
        predictions = pl.read_csv(path).select(STUDY, *targets)
        for labels, truth in frames.items():
            result |= score(predictions, truth, gold, targets, f"{source}_{labels}")
        result |= consensus_auc(predictions, frames, targets, source)
        result |= gold_auc(predictions, official, targets, source)
    return result


def official_gold(settings: SubmissionMetricsCLI, targets: list[str], gold: pl.DataFrame) -> pl.DataFrame:
    frame = pl.read_csv(settings.raw_dir / "train.csv", infer_schema_length=None)
    return frame.join(gold.filter(pl.col("is_gold")), on=STUDY).select(STUDY, *targets)


def cohort_key(row: dict, metric: str) -> tuple:
    key = (row["split"], row["fold"], row["inference"])
    if metric.startswith("own_") or metric.startswith("val_"):
        key += (row["validation_multiplier"],)
    if metric.startswith("own_"):
        key += (row["labels"],)
    return key


def evaluate_metric(table: pl.DataFrame, metric: str, settings: SubmissionMetricsCLI) -> dict | None:
    rows = [row for row in table.iter_rows(named=True) if row.get(metric) is not None and np.isfinite(row[metric])]
    cohorts: dict[tuple, list[dict]] = {}
    for row in rows:
        cohorts.setdefault(cohort_key(row, metric), []).append(row)
    cohorts = {key: members for key, members in cohorts.items() if len(members) >= 2}
    if not cohorts:
        return None
    concordant = ranked = decisive_concordant = decisive_ranked = 0
    centered_public, centered_metric, groups = [], [], []
    for index, members in enumerate(cohorts.values()):
        public = np.array([m["public"] for m in members])
        values = np.array([m[metric] for m in members])
        centered_public.extend(public - public.mean())
        centered_metric.extend(values - values.mean())
        groups.extend([index] * len(members))
        for a, b in combinations(range(len(members)), 2):
            if public[a] == public[b] or values[a] == values[b]:
                continue
            agrees = int((public[a] > public[b]) == (values[a] > values[b]))
            ranked += 1
            concordant += agrees
            if abs(public[a] - public[b]) >= settings.decisive - 1e-9:
                decisive_ranked += 1
                decisive_concordant += agrees
    x, y, groups = np.array(centered_metric), np.array(centered_public), np.array(groups)
    observed = float(np.corrcoef(x, y)[0, 1])
    generator = np.random.default_rng(settings.seed)
    exceed = 0
    for _ in range(settings.permutations):
        shuffled = y.copy()
        for group in np.unique(groups):
            index = np.flatnonzero(groups == group)
            shuffled[index] = y[generator.permutation(index)]
        exceed += int(abs(np.corrcoef(x, shuffled)[0, 1]) >= abs(observed) - 1e-12)
    return {
        "metric": metric,
        "submissions": len(x),
        "cohorts": len(cohorts),
        "within_cohort_pearson": observed,
        "pearson_permutation_p": (exceed + 1) / (settings.permutations + 1),
        "concordant_pairs": concordant,
        "ranked_pairs": ranked,
        "concordance": concordant / ranked if ranked else None,
        "concordance_binomial_p": binomtest(concordant, ranked, 0.5).pvalue if ranked else None,
        "decisive_concordant": decisive_concordant,
        "decisive_pairs": decisive_ranked,
        "decisive_concordance": decisive_concordant / decisive_ranked if decisive_ranked else None,
    }


def run(settings: SubmissionMetricsCLI) -> None:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory already exists: {settings.output_dir}")
    frames, targets, gold = label_sets(settings)
    official = official_gold(settings, targets, gold)
    rows = []
    for entry in json.loads(settings.ledger.read_text()):
        row = submitted_checkpoint(entry)
        if row is None:
            continue
        rows.append(row | recorded_metrics(row) | prediction_metrics(row, frames, targets, gold, official))
    table = pl.DataFrame(rows, infer_schema_length=None)
    metrics = [c for c in table.columns if c.endswith(("_auc", "_bce"))]
    results = [r for r in (evaluate_metric(table, m, settings) for m in metrics) if r]
    summary = pl.DataFrame(results).sort("concordance", descending=True, nulls_last=True)
    settings.output_dir.mkdir(parents=True)
    table.write_csv(settings.output_dir / "submissions.csv")
    summary.write_csv(settings.output_dir / "metrics.csv")
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, fmt_str_lengths=40):
        print(summary.select("metric", "submissions", "cohorts", "within_cohort_pearson", "pearson_permutation_p", "concordance", "ranked_pairs", "decisive_concordance", "decisive_pairs"))
        print(
            table.group_by("split", "fold", "inference")
            .agg(n=pl.len(), submissions=pl.col("submission"), public_sd=pl.col("public").std())
            .sort("split", "fold")
        )


if __name__ == "__main__":
    run(SubmissionMetricsCLI())
