import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from pydantic_settings import SettingsError
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD
from knee_abnormality.provenance import copy_sources, sha256_file
from scripts.tools.prepare_combined_labels import CONSENSUS_TARGETS

RAMIT = "ramitnandi/rsna-knee-labels-gpt-luna/downloaded-20260921/report_labels.csv"
ZAID = "zaidaliiq1000/rsna-knee-llm-labels/downloaded-20260921/llm_labels.csv"
PILKWANG = "pilkwang/version-1/report_labels_v2.csv"
RAMIT_LEVELS = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
PILKWANG_STATES = {"YES": "positive", "NO": "negative", "UNK": "unaddressed"}
RAMIT_STATES = {"POSITIVE": "positive", "NEGATIVE": "negative", "NA": "unaddressed"}


class AttributeCLI(CLISettings):
    public_dir: Path = Path("data/labels/public")
    split_dir: Path = Path("data/folds/all_5fold_scanner_v1")
    predictions: Path = Path("outputs/training/k_05/fold_0/validation_predictions.csv")
    silver_labels: Path = Path("data/labels/prepared/combined-v1/labels.csv")
    validation_fold: int = 0
    bootstrap_draws: int = 1000
    seed: int = 2263738815
    output_dir: Path


def load_ramit(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=0).select(
        pl.col(STUDY),
        pl.col("CLASS")
        .str.replace_all("_", " ")
        .str.replace("Bakers", "Baker's")
        .alias("target"),
        pl.col("classification").replace_strict(RAMIT_STATES).alias("ramit_state"),
        pl.col("severity")
        .replace_strict(RAMIT_LEVELS, default=None, return_dtype=pl.Int64)
        .alias("ramit_severity"),
        (pl.col("hedged") == "true").alias("ramit_hedged"),
    )


def load_zaid(path: Path, targets: list[str]) -> pl.DataFrame:
    frame = pl.read_csv(path, infer_schema_length=0).filter(pl.col("n_models") != "0")
    return pl.concat(
        frame.select(
            pl.col(STUDY),
            pl.lit(target).alias("target"),
            pl.col(f"{target}_sev").cast(pl.Float64).alias("zaid_severity_raw"),
            (pl.col(f"{target}_mentioned") == "True").alias("zaid_mentioned"),
            pl.col(target).cast(pl.Float64).alias("zaid_label"),
        )
        for target in targets
    ).with_columns(
        pl.when(pl.col("zaid_severity_raw") > 0)
        .then(pl.col("zaid_severity_raw"))
        .alias("zaid_severity"),
        pl.when(~pl.col("zaid_mentioned"))
        .then(pl.lit("unaddressed"))
        .when(pl.col("zaid_label") >= TARGET_THRESHOLD)
        .then(pl.lit("positive"))
        .otherwise(pl.lit("negative"))
        .alias("zaid_state"),
    )


def load_pilkwang(path: Path, targets: list[str]) -> pl.DataFrame:
    frame = pl.read_csv(path, infer_schema_length=0)
    return pl.concat(
        frame.select(
            pl.col(STUDY),
            pl.lit(target).alias("target"),
            pl.col(f"{target}__verdict")
            .replace_strict(PILKWANG_STATES)
            .alias("pilkwang_state"),
        )
        for target in targets
    )


def long_targets(path: Path, targets: list[str], name: str) -> pl.DataFrame:
    frame = pl.read_csv(path, infer_schema_length=None)
    return frame.unpivot(
        index=STUDY, on=targets, variable_name="target", value_name=name
    ).with_columns(pl.col(name).cast(pl.Float64))


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 10 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    return float(spearmanr(left, right).statistic)


def bootstrap_spearman(
    left: np.ndarray, right: np.ndarray, draws: int, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    if len(left) < 10:
        return None, None
    values = []
    for _ in range(draws):
        index = rng.integers(0, len(left), len(left))
        value = spearman(left[index], right[index])
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def coverage(frame: pl.DataFrame, targets: list[str]) -> pl.DataFrame:
    rows = []
    for target in targets:
        part = frame.filter(pl.col("target") == target)
        row = {"target": target, "studies": part.height}
        for source in ("ramit", "zaid", "pilkwang"):
            counts = dict(part[f"{source}_state"].value_counts().rows())
            for state in ("positive", "negative", "unaddressed"):
                row[f"{source}_{state}"] = counts.get(state, 0)
        levels = dict(part["ramit_severity"].value_counts().rows())
        for name, level in RAMIT_LEVELS.items():
            row[f"ramit_{name.lower()}"] = levels.get(level, 0)
        row["ramit_hedged"] = int(part["ramit_hedged"].sum())
        row["zaid_severity_rated"] = int(part["zaid_severity"].is_not_null().sum())
        rows.append(row)
    return pl.DataFrame(rows)


def severity_agreement(frame: pl.DataFrame, targets: list[str]) -> pl.DataFrame:
    rows = []
    for target in targets:
        part = frame.filter(
            (pl.col("target") == target)
            & pl.col("ramit_severity").is_not_null()
            & pl.col("zaid_severity").is_not_null()
        )
        ramit = part["ramit_severity"].to_numpy()
        zaid = part["zaid_severity"].to_numpy()
        binned = np.clip(np.rint(zaid), 1, 3).astype(int)
        rows.append(
            {
                "target": target,
                "both_rated": part.height,
                "spearman": spearman(ramit, zaid),
                "exact_level_agreement": float((ramit == binned).mean())
                if part.height
                else None,
                "quadratic_kappa": float(
                    cohen_kappa_score(ramit, binned, weights="quadratic")
                )
                if part.height >= 10 and np.unique(binned).size > 1
                else None,
            }
        )
    return pl.DataFrame(rows)


def state_agreement(frame: pl.DataFrame, targets: list[str]) -> pl.DataFrame:
    rows = []
    for target in targets:
        part = frame.filter(pl.col("target") == target).drop_nulls(
            ["ramit_state", "zaid_state", "pilkwang_state"]
        )
        for left, right in combinations(("ramit", "zaid", "pilkwang"), 2):
            a = part[f"{left}_state"].to_numpy()
            b = part[f"{right}_state"].to_numpy()
            addressed_a = a != "unaddressed"
            addressed_b = b != "unaddressed"
            both = addressed_a & addressed_b
            rows.append(
                {
                    "target": target,
                    "pair": f"{left}-{right}",
                    "studies": len(a),
                    "addressed_kappa": float(
                        cohen_kappa_score(addressed_a, addressed_b)
                    ),
                    "addressed_agreement": float((addressed_a == addressed_b).mean()),
                    "both_addressed": int(both.sum()),
                    "polarity_agreement": float((a[both] == b[both]).mean())
                    if both.any()
                    else None,
                }
            )
    return pl.DataFrame(rows)


def target_relation(frame: pl.DataFrame, targets: list[str]) -> pl.DataFrame:
    rows = []
    for target in targets:
        part = frame.filter(pl.col("target") == target)
        positive = part.filter(pl.col("label") >= TARGET_THRESHOLD)
        rated = positive.filter(pl.col("ramit_severity").is_not_null())
        levels = rated["ramit_severity"].value_counts()["count"].to_numpy()
        share = levels / levels.sum() if levels.size else levels
        rows.append(
            {
                "target": target,
                "label_positive": positive.height,
                "label_positive_with_ramit_severity": rated.height,
                "ramit_positive_with_label_positive": float(
                    (
                        part.filter(pl.col("ramit_state") == "positive")["label"]
                        >= TARGET_THRESHOLD
                    ).mean()
                ),
                "modal_level_share": float(share.max()) if share.size else None,
                "level_entropy_bits": float(-(share * np.log2(share)).sum())
                if share.size
                else None,
                "spearman_severity_soft_label": spearman(
                    rated["ramit_severity"].to_numpy(), rated["label"].to_numpy()
                ),
            }
        )
    return pl.DataFrame(rows)


def grounding(
    frame: pl.DataFrame,
    targets: list[str],
    draws: int,
    rng: np.random.Generator,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, negatives = [], []
    for target in targets:
        part = frame.filter(pl.col("target") == target)
        negative = part.filter(pl.col("label") < TARGET_THRESHOLD)
        positive = part.filter(pl.col("label") >= TARGET_THRESHOLD)
        negative_scores = negative["prediction"].to_numpy()
        for source, column in (
            ("ramit", "ramit_severity"),
            ("zaid", "zaid_severity"),
        ):
            rated = positive.filter(pl.col(column).is_not_null())
            severity = rated[column].to_numpy()
            scores = rated["prediction"].to_numpy()
            low, high = bootstrap_spearman(scores, severity, draws, rng)
            row = {
                "target": target,
                "source": source,
                "positives_rated": rated.height,
                "negatives": negative.height,
                "spearman_prediction_severity": spearman(scores, severity),
                "spearman_ci_low": low,
                "spearman_ci_high": high,
            }
            if source == "ramit":
                for name, level in RAMIT_LEVELS.items():
                    stratum = rated.filter(pl.col(column) == level)
                    row[f"{name.lower()}_positives"] = stratum.height
                    row[f"{name.lower()}_mean_prediction"] = (
                        float(stratum["prediction"].mean()) if stratum.height else None
                    )
                    row[f"{name.lower()}_auc_vs_negatives"] = (
                        float(
                            roc_auc_score(
                                np.r_[
                                    np.ones(stratum.height),
                                    np.zeros(len(negative_scores)),
                                ],
                                np.r_[
                                    stratum["prediction"].to_numpy(), negative_scores
                                ],
                            )
                        )
                        if stratum.height >= 5
                        else None
                    )
            rows.append(row)
        by_state = {
            state: negative.filter(pl.col("ramit_state") == state)
            for state in ("negative", "unaddressed")
        }
        positive_scores = positive["prediction"].to_numpy()
        entry = {"target": target, "positives": positive.height}
        for state, stratum in by_state.items():
            entry[f"{state}_negatives"] = stratum.height
            entry[f"{state}_mean_prediction"] = (
                float(stratum["prediction"].mean()) if stratum.height else None
            )
            entry[f"positives_auc_vs_{state}"] = (
                float(
                    roc_auc_score(
                        np.r_[np.ones(len(positive_scores)), np.zeros(stratum.height)],
                        np.r_[positive_scores, stratum["prediction"].to_numpy()],
                    )
                )
                if stratum.height >= 5 and len(positive_scores) >= 5
                else None
            )
        negatives.append(entry)
    return pl.DataFrame(rows), pl.DataFrame(negatives)


def silver_by_call(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by("target_group", "target", "ramit_call")
        .agg(
            pl.len().alias("cells"),
            pl.col("silver").mean().alias("silver_mean"),
            (pl.col("silver") >= TARGET_THRESHOLD).mean().alias("silver_at_or_above"),
        )
        .sort("target_group", "target", "ramit_call")
    )


def gold_by_call(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns((pl.col("label") >= TARGET_THRESHOLD).alias("gold_positive"))
        .group_by("target_group", "ramit_call")
        .agg(
            pl.len().alias("cells"),
            pl.col("gold_positive").sum().alias("gold_positive"),
            pl.col("gold_positive").mean().alias("gold_positive_rate"),
            pl.col("silver").mean().alias("silver_mean"),
        )
        .sort("target_group", "ramit_call")
    )


def check(settings: AttributeCLI) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((settings.split_dir / "protocol.json").read_text())
    targets = protocol["targets"]
    rng = np.random.default_rng(settings.seed)
    files = {
        "ramit": settings.public_dir / RAMIT,
        "zaid": settings.public_dir / ZAID,
        "pilkwang": settings.public_dir / PILKWANG,
        "targets": settings.split_dir / "targets.csv",
        "studies": settings.split_dir / "studies.csv",
        "predictions": settings.predictions,
        "silver": settings.silver_labels,
    }
    studies = pl.read_csv(files["studies"], infer_schema_length=None).select(
        STUDY, "fold", pl.col("is_gold").cast(pl.Utf8).str.to_lowercase() == "true"
    )
    frame = (
        long_targets(files["targets"], targets, "label")
        .join(studies, on=STUDY, how="left")
        .join(load_ramit(files["ramit"]), on=[STUDY, "target"], how="left")
        .join(load_zaid(files["zaid"], targets), on=[STUDY, "target"], how="left")
        .join(
            load_pilkwang(files["pilkwang"], targets), on=[STUDY, "target"], how="left"
        )
        .join(
            long_targets(files["silver"], targets, "silver"),
            on=[STUDY, "target"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("ramit_state") == "positive")
            .then(
                pl.col("ramit_severity").replace_strict(
                    {v: k for k, v in RAMIT_LEVELS.items()},
                    default="UNRATED",
                    return_dtype=pl.Utf8,
                )
            )
            .otherwise(pl.col("ramit_state").str.to_uppercase())
            .alias("ramit_call"),
            pl.when(pl.col("target").is_in(CONSENSUS_TARGETS))
            .then(pl.lit("consensus"))
            .otherwise(pl.lit("steven_v4"))
            .alias("target_group"),
        )
    )
    if frame["fold"].null_count():
        raise ValueError("targets.csv holds studies missing from studies.csv")
    training = frame.filter(pl.col("fold") != settings.validation_fold)
    predictions = long_targets(files["predictions"], targets, "prediction")
    validation = frame.filter(pl.col("fold") == settings.validation_fold).join(
        predictions, on=[STUDY, "target"], how="inner"
    )
    if (
        validation.height
        != frame.filter(pl.col("fold") == settings.validation_fold).height
    ):
        raise ValueError("predictions do not cover the validation fold")
    gold = frame.filter(pl.col("is_gold"))
    outputs = {
        "coverage_training": coverage(training, targets),
        "severity_agreement_training": severity_agreement(training, targets),
        "state_agreement_training": state_agreement(training, targets),
        "target_relation_training": target_relation(training, targets),
    }
    outputs["grounding_validation"], outputs["negatives_by_report_state_validation"] = (
        grounding(validation, targets, settings.bootstrap_draws, rng)
    )
    outputs["silver_by_ramit_call_non_gold"] = silver_by_call(
        frame.filter(~pl.col("is_gold"))
    )
    outputs["gold_by_ramit_call"] = gold_by_call(gold)
    for name, table in outputs.items():
        table.write_csv(settings.output_dir / f"{name}.csv")
    summary = {
        "training_studies": training[STUDY].n_unique(),
        "validation_studies": validation[STUDY].n_unique(),
        "gold_studies": gold[STUDY].n_unique(),
        "validation_fold": settings.validation_fold,
        "bootstrap_draws": settings.bootstrap_draws,
        "seed": settings.seed,
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    manifest = {
        "inputs": {str(path): sha256_file(path) for path in files.values()},
        "sources": copy_sources((Path(__file__),), settings.output_dir / "source"),
    }
    (settings.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with pl.Config(tbl_rows=60, tbl_cols=30, tbl_width_chars=300, float_precision=3):
        print(json.dumps(summary, indent=2))
        for name, table in outputs.items():
            print(name)
            print(table)


def main() -> None:
    try:
        check(AttributeCLI())
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Report attribute check error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
