import json
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD
from scripts.adhoc.correlate_grouped_auc import (
    CURRENT,
    cohort_pairs,
    decisive_concordance,
    permuted_differences,
    prediction_sources,
    search,
    study_groups,
    submission_rows,
    target_tables,
    weighted_concordance,
)

GROUPINGS = ("pooled", "vendor", "scanner_group", "model", "model_field")
LABELS = ("mean3", "rebuild_state_v2", "steven_v4", "combined_v1")
PRIMARY = ("mean3|confidence|scanner_group", "mean3|confidence|vendor")
BASELINES = (
    "rebuild_state_v2|none|pooled",
    "rebuild_state_v2|confidence|pooled",
    "rebuild_state_v2|none|scanner_group",
)


class ConfidencePairCLI(CLISettings):
    model_config = describe(
        "Score submitted checkpoints with the same-group, confidence-weighted pair AUC "
        "(pair weight |2s-1|^2 of both studies, positive-negative pairs within one scanner "
        "grouping) and test whether it orders submissions like the public leaderboard better "
        "than the current metric, in and out of sample"
    )

    ledger: Path = Field(
        default=Path("outputs/submission/kaggle-ledger.json"),
        description="Machine submission ledger",
    )
    raw_dir: Path = Field(
        default=Path("data/raw"), description="Official competition files"
    )
    folds_dir: Path = Field(
        default=Path("data/folds"), description="Frozen split directories"
    )
    output_dir: Path = Field(description="Fresh output directory")
    permutations: int = Field(
        default=5000, ge=100, description="Within-cohort public-score permutations"
    )
    decisive: float = Field(
        default=0.003,
        gt=0,
        description="Public difference counted as beyond seed noise",
    )
    seed: int = Field(default=20260929, description="Permutation seed")


def pair_auc(
    truth: np.ndarray, weight: np.ndarray, score: np.ndarray, groups: np.ndarray
) -> float:
    numerator = denominator = 0.0
    positive = truth >= TARGET_THRESHOLD
    for group in np.unique(groups):
        mask = groups == group
        pos, neg = positive[mask], ~positive[mask]
        w, p = weight[mask], score[mask]
        total_pos, total_neg = w[pos].sum(), w[neg].sum()
        if total_pos <= 0 or total_neg <= 0:
            continue
        values, inverse = np.unique(p, return_inverse=True)
        negative_mass = np.bincount(
            inverse, weights=np.where(neg, w, 0.0), minlength=len(values)
        )
        below = np.cumsum(negative_mass) - negative_mass
        numerator += (
            w[pos] * (below[inverse[pos]] + 0.5 * negative_mass[inverse[pos]])
        ).sum()
        denominator += total_pos * total_neg
    return numerator / denominator if denominator > 0 else np.nan


def metric_values(
    predictions: pl.DataFrame, frames: dict, targets: list[str], groups: pl.DataFrame
) -> dict:
    result = {}
    for labels in LABELS:
        joined = predictions.join(
            frames[labels].rename({t: f"y_{t}" for t in targets}), on=STUDY
        ).join(groups, on=STUDY)
        for grouping, weighting in product(GROUPINGS, ("none", "confidence")):
            group = joined[grouping].to_numpy()
            values = []
            for target in targets:
                soft = joined[f"y_{target}"].to_numpy().astype(np.float64)
                weight = (
                    np.abs(2 * soft - 1) ** 2
                    if weighting == "confidence"
                    else np.ones_like(soft)
                )
                values.append(
                    pair_auc(
                        soft,
                        weight,
                        joined[target].to_numpy().astype(np.float64),
                        group,
                    )
                )
            result[f"{labels}|{weighting}|{grouping}"] = float(np.nanmean(values))
    return result


def run(settings: ConfidencePairCLI) -> None:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory already exists: {settings.output_dir}")
    frames, targets = target_tables(settings)
    groups = study_groups().with_columns(
        pl.lit("all").alias("pooled"),
        (pl.col("scanner") + "/" + pl.col("field")).alias("model_field"),
        pl.col("scanner").alias("model"),
    )
    records = []
    for row in submission_rows(settings):
        for source, path in prediction_sources(
            Path(row["run"]),
            row["fold"],
            row["epoch"],
            row["validation_multiplier"],
            row["is_selected"],
            row["prediction_multiplier"],
        ).items():
            predictions = pl.read_csv(path).select(STUDY, *targets)
            records.append(
                row
                | {"source": source}
                | metric_values(predictions, frames, list(targets), groups)
            )
    table = pl.DataFrame(records, infer_schema_length=None).with_row_index("row")
    settings.output_dir.mkdir(parents=True)
    table.write_parquet(settings.output_dir / "submission_metrics.parquet")
    names = [c for c in table.columns if c.count("|") == 2]
    pairs = [
        (a, b, f) for source in ("val", "x4") for a, b, f in cohort_pairs(table, source)
    ]
    left, right = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    public = table["public"].to_numpy()
    public_difference = public[left] - public[right]
    source_of = table["source"].to_numpy()[left]
    family_of = np.array([p[2] for p in pairs])
    differences = {}
    for name in names:
        values = table[name].to_numpy()
        difference = values[left] - values[right]
        if np.isfinite(difference).all():
            differences[name] = difference
    reference = "rebuild_state_v2|none|pooled"
    rows = []
    for name, difference in differences.items():
        signs = np.sign(difference)
        entry = {
            "variant": name,
            "weighted": float(weighted_concordance(signs, public_difference)),
        }
        for source in ("val", "x4"):
            mask = source_of == source
            entry[f"{source}_weighted"] = float(
                weighted_concordance(signs[mask], public_difference[mask])
            )
            entry[f"{source}_decisive"], entry[f"{source}_decisive_pairs"] = (
                decisive_concordance(
                    signs[mask], public_difference[mask], settings.decisive
                )
            )
        fold0 = np.char.find(family_of.astype(str), "scanner_v1/f0/") >= 0
        entry["folds_1_4_weighted"] = float(
            weighted_concordance(signs[~fold0], public_difference[~fold0])
        )
        rows.append(entry)
    ranking = pl.DataFrame(rows).sort("weighted", descending=True)
    ranking.write_csv(settings.output_dir / "ranking.csv")
    permutations = permuted_differences(
        table, pairs, np.random.default_rng(settings.seed), settings.permutations
    )
    members = list(differences)
    _, best_score, _, _ = search(differences, public_difference, members, 2, False)
    current = np.sign(differences[reference])
    gain = best_score - weighted_concordance(current, public_difference)
    exceed = sum(
        search(differences, p, members, 2, False)[1] - weighted_concordance(current, p)
        >= gain - 1e-12
        for p in permutations
    )
    primary_p = {}
    for name in PRIMARY:
        signs = np.sign(differences[name])
        observed = weighted_concordance(
            signs, public_difference
        ) - weighted_concordance(current, public_difference)
        primary_p[name] = {
            "gain_over_current": float(observed),
            "p": (
                sum(
                    weighted_concordance(signs, p) - weighted_concordance(current, p)
                    >= observed - 1e-12
                    for p in permutations
                )
                + 1
            )
            / (len(permutations) + 1),
        }
    loco = []
    numerator = weight = 0.0
    for held_out in sorted(set(family_of)):
        train, test = family_of != held_out, family_of == held_out
        if np.abs(public_difference[test]).sum() == 0:
            continue
        picked = search(
            {k: v[train] for k, v in differences.items()},
            public_difference[train],
            members,
            2,
            False,
        )[0][0]
        held = public_difference[test]
        score = float(weighted_concordance(np.sign(differences[picked][test]), held))
        numerator += score * np.abs(held).sum()
        weight += np.abs(held).sum()
        loco.append(
            {
                "held_out": held_out,
                "chosen": picked,
                "held_out_weighted": score,
                "current_weighted": float(weighted_concordance(current[test], held)),
            }
        )
    summary = {
        "variants": len(members),
        "pairs": len(pairs),
        "family_best": str(ranking["variant"][0]),
        "family_max_statistic_p": (exceed + 1) / (len(permutations) + 1),
        "family_leave_one_cohort_out": numerator / weight,
        "primary": primary_p,
    }
    pl.DataFrame(loco).write_csv(settings.output_dir / "leave_one_cohort_out.csv")
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with pl.Config(
        tbl_rows=-1, tbl_width_chars=220, fmt_str_lengths=60, float_precision=3
    ):
        print(ranking)
        print(pl.DataFrame(loco))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(ConfidencePairCLI())
