import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsError
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD
from knee_abnormality.provenance import copy_sources
from scripts.adhoc.screen_public_label_tables import ScreenCLI, load_tables
from scripts.tools.prepare_combined_labels import (
    BASE_TABLE,
    CONSENSUS_TARGETS,
    consensus_sources,
    standard,
)

GOLD_TUNED = ("jev_text", "josh_luna")
EXTRA_TABLES = {
    "steven_v1": "data/labels/public/stevenleehans/version-6/llm_labels_full.csv",
    "steven_v2": "data/labels/public/stevenleehans/version-6/llm_labels_v2.csv",
    "lixin_gpt56sol": "data/labels/public/lixin73/version-1/labels_llm_gpt56sol.csv",
}


class AnalyzeCLI(CLISettings):
    comparison_manifest: Path = Path(
        "configs/labels/public-label-comparison-sources.json"
    )
    public_dir: Path = Path("data/labels/public")
    output_dir: Path
    draws: int = Field(default=2000, ge=200)
    seed: int = 0
    probability_threshold: float = Field(default=0.9, gt=0.5, lt=1)


def binary_votes(values: np.ndarray) -> np.ndarray:
    return np.where(
        np.isfinite(values), (values >= TARGET_THRESHOLD).astype(float), np.nan
    )


def dawid_skene(
    votes: np.ndarray, fit: np.ndarray, iterations: int = 300
) -> np.ndarray:
    observed, value = np.isfinite(votes), np.nan_to_num(votes)
    seen, vote = observed[fit], value[fit]
    posterior = np.nanmean(votes[fit], axis=1)
    posterior = np.where(np.isfinite(posterior), posterior, 0.1)

    def likelihoods(obs, val, prior, sensitivity, specificity):
        positive = np.log(prior) + (
            obs * (val * np.log(sensitivity) + (1 - val) * np.log(1 - sensitivity))
        ).sum(axis=1)
        negative = np.log(1 - prior) + (
            obs * ((1 - val) * np.log(specificity) + val * np.log(1 - specificity))
        ).sum(axis=1)
        return 1 / (1 + np.exp(negative - positive))

    for _ in range(iterations):
        prior = posterior.mean()
        sensitivity = ((posterior[:, None] * vote * seen).sum(0) + 1) / (
            (posterior[:, None] * seen).sum(0) + 2
        )
        specificity = (((1 - posterior)[:, None] * (1 - vote) * seen).sum(0) + 1) / (
            ((1 - posterior)[:, None] * seen).sum(0) + 2
        )
        posterior = likelihoods(seen, vote, prior, sensitivity, specificity)
    return likelihoods(observed, value, prior, sensitivity, specificity)


def analyze(settings: AnalyzeCLI) -> None:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory exists: {settings.output_dir}")
    targets = pl.read_csv("data/raw/sample_submission.csv").columns[1:]
    train = pl.read_csv("data/raw/train.csv", infer_schema_length=None)
    ids = train.select(STUDY).sort(STUDY)
    gold_frame = train.drop_nulls(targets).select(STUDY, *targets)
    gold = ids.join(
        gold_frame.with_columns(pl.lit(True).alias("gold")), on=STUDY, how="left"
    )
    is_gold = gold["gold"].fill_null(False).to_numpy()
    truth = gold.select(targets).to_numpy().astype(float)
    comparison = json.loads(settings.comparison_manifest.read_text())
    _, sources = consensus_sources(comparison, targets)
    tables, _ = load_tables(
        ScreenCLI(public_dir=settings.public_dir, output_dir=settings.output_dir),
        targets,
    )
    for name, path in EXTRA_TABLES.items():
        tables[name] = standard(pl.read_csv(path, infer_schema_length=None), targets)
    tables |= {f"consensus:{name}": frame for name, frame in sources.items()}

    def matrix(frame: pl.DataFrame) -> np.ndarray:
        return (
            ids.join(frame, on=STUDY, how="left")
            .sort(STUDY)
            .select(targets)
            .to_numpy()
            .astype(float)
        )

    values = {name: matrix(frame) for name, frame in tables.items()}
    silver = ~is_gold
    names = sorted(values)
    agreement_rows = []
    for first in names:
        for second in names:
            if first >= second:
                continue
            shares = []
            for index in range(len(targets)):
                a, b = values[first][silver, index], values[second][silver, index]
                both = np.isfinite(a) & np.isfinite(b)
                if both.sum() >= 100:
                    shares.append(
                        float(
                            (
                                (a[both] >= TARGET_THRESHOLD)
                                == (b[both] >= TARGET_THRESHOLD)
                            ).mean()
                        )
                    )
            if shares:
                agreement_rows.append(
                    {
                        "first": first,
                        "second": second,
                        "silver_agreement": np.mean(shares),
                    }
                )
    agreement = pl.DataFrame(agreement_rows).sort("silver_agreement", descending=True)

    steven = values[BASE_TABLE]
    consensus = {name: values[f"consensus:{name}"] for name in sources}
    gold_rows = np.flatnonzero(is_gold)
    rng = np.random.default_rng(settings.seed)
    draws = rng.integers(0, len(gold_rows), (settings.draws, len(gold_rows)))

    def gold_auc(score: np.ndarray, index: int) -> float:
        return float(roc_auc_score(truth[gold_rows, index], score[gold_rows]))

    def probability_better(score: np.ndarray, index: int) -> float:
        y, s, b = truth[gold_rows, index], score[gold_rows], steven[gold_rows, index]
        wins = [
            roc_auc_score(y[d], s[d]) > roc_auc_score(y[d], b[d])
            for d in draws
            if y[d].min() != y[d].max()
        ]
        return float(np.mean(wins))

    def mean_of(members, index: int) -> np.ndarray:
        stacked = np.stack([consensus[name][:, index] for name in members], axis=1)
        return np.nan_to_num(np.nanmean(stacked, axis=1))

    rows = []
    for index, target in enumerate(targets):
        base = gold_auc(steven[:, index], index)
        average = mean_of(sources, index)
        leave_one_out = [
            gold_auc(mean_of([n for n in sources if n != left], index), index)
            for left in sources
        ]
        untuned = gold_auc(
            mean_of([n for n in sources if n not in GOLD_TUNED], index), index
        )
        votes = np.stack(
            [binary_votes(steven[:, index])]
            + [binary_votes(consensus[name][:, index]) for name in sources],
            axis=1,
        )
        row = {
            "target": target,
            "steven_gold_auc": base,
            "consensus_gold_auc": gold_auc(average, index),
            "p_consensus_better": probability_better(average, index),
            "leave_one_out_min": min(leave_one_out),
            "leave_one_out_max": max(leave_one_out),
            "without_gold_tuned": untuned,
            "dawid_skene_gold_auc": gold_auc(dawid_skene(votes, silver), index),
            "steven_silver_rate": float(
                (steven[silver, index] >= TARGET_THRESHOLD).mean()
            ),
            "consensus_silver_rate": float(
                (average[silver] >= TARGET_THRESHOLD).mean()
            ),
        }
        row["selected"] = (
            row["p_consensus_better"] > settings.probability_threshold
            and row["leave_one_out_min"] > base
            and untuned > base
        )
        rows.append(row)
    per_target = pl.DataFrame(rows)
    selected = per_target.filter("selected")["target"].to_list()
    if tuple(selected) != CONSENSUS_TARGETS:
        raise ValueError(
            f"Selection {selected} differs from the prepared table {CONSENSUS_TARGETS}"
        )
    combined = [
        r["consensus_gold_auc"] if r["selected"] else r["steven_gold_auc"] for r in rows
    ]

    settings.output_dir.mkdir(parents=True)
    agreement.write_csv(settings.output_dir / "silver_agreement.csv")
    per_target.write_csv(settings.output_dir / "per_target.csv")
    summary = {
        "consensus_sources": list(sources),
        "gold_tuned_sources": list(GOLD_TUNED),
        "rule": (
            f"use the consensus mean when P(consensus beats Steven v4 on gold) > "
            f"{settings.probability_threshold} over {settings.draws} gold-study bootstrap draws, "
            "the leave-one-source-out minimum still beats Steven v4, and the mean without "
            "gold-tuned sources still beats Steven v4"
        ),
        "selected_targets": selected,
        "gold_macro_auc": {
            BASE_TABLE: float(per_target["steven_gold_auc"].mean()),
            "combined_v1": float(np.mean(combined)),
            "dawid_skene_all_targets": float(per_target["dawid_skene_gold_auc"].mean()),
        },
        "sources": copy_sources((Path(__file__),), settings.output_dir / "source"),
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=220, float_precision=3):
        print(per_target)
    print(
        json.dumps(
            {k: summary[k] for k in ("selected_targets", "gold_macro_auc")}, indent=2
        )
    )


def main() -> None:
    try:
        analyze(AnalyzeCLI())
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Label combination analysis error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
