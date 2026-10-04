import json
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD
from knee_abnormality.evaluation.metrics import aucs, bootstrap_auc

KEYS = ["family", "multiplier", "tau", "tta"]


class BlendCLI(CLISettings):
    model_config = describe(
        "Score the swept out-of-fold predictions of two model families: the attention "
        "temperature per family and image multiplier, the family weight of their blend, "
        "and paired report-group bootstrap intervals against the equal blend at tau 1"
    )

    sweep_dir: Path = Field(description="Directory written by sweep_child_temperatures")
    split_dir: Path = Path("data/folds/all_5fold_scanner_v2")
    families: tuple[str, str] = ("rn50", "rexnet-200")
    weight_step: float = Field(default=0.05, gt=0, le=0.5)
    draws: int = Field(default=2000, ge=200)
    seed: int = 20261004
    output: Path = Field(description="JSON report to write")


def main() -> None:
    settings = BlendCLI()
    labels = pl.read_csv(settings.split_dir / "targets.csv")
    targets = labels.columns[1:]
    frame = (
        pl.read_csv(settings.split_dir / "studies.csv")
        .join(labels, on=STUDY, validate="1:1")
        .sort(STUDY)
    )
    hard = (frame.select(targets).to_numpy() >= TARGET_THRESHOLD).astype(np.float64)
    folds = frame["fold"].to_numpy()
    swept = pl.concat(
        pl.read_parquet(path) for path in sorted(settings.sweep_dir.glob("*.parquet"))
    )
    scores = {}
    for key, part in swept.partition_by(KEYS, as_dict=True).items():
        aligned = frame.select(STUDY).join(
            part, on=STUDY, how="left", validate="1:1", maintain_order="left"
        )
        values = aligned.select(targets).to_numpy()
        if not np.isfinite(values).all() or (aligned["fold"] != frame["fold"]).any():
            raise ValueError(f"{key} does not cover every study out of fold")
        scores[key] = values.astype(np.float64)

    def macro(values: np.ndarray) -> float:
        return aucs(hard, values, targets)["macro_auc"]

    def fold_mean(values: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    aucs(hard[folds == fold], values[folds == fold], targets)[
                        "macro_auc"
                    ]
                    for fold in np.unique(folds)
                ]
            )
        )

    singles = [
        {
            **dict(zip(KEYS, key, strict=True)),
            "pooled_macro_auc": macro(values),
            "fold_mean_macro_auc": fold_mean(values),
        }
        for key, values in sorted(scores.items())
    ]
    first, second = settings.families
    multipliers = sorted({key[1] for key in scores})
    taus = sorted({key[2] for key in scores})
    weights = np.round(np.arange(0, 1 + 1e-9, settings.weight_step), 4)
    blends = []
    for multiplier, tta, tau_a, tau_b in product(
        multipliers, (False, True), taus, taus
    ):
        a = scores[(first, multiplier, tau_a, tta)]
        b = scores[(second, multiplier, tau_b, tta)]
        for weight in weights:
            blends.append(
                {
                    "multiplier": multiplier,
                    "tta": tta,
                    f"tau_{first}": tau_a,
                    f"tau_{second}": tau_b,
                    f"weight_{first}": float(weight),
                    "pooled_macro_auc": macro(weight * a + (1 - weight) * b),
                }
            )
    table = pl.DataFrame(blends)
    single_table = pl.DataFrame(singles)

    groups, inverse = np.unique(frame["report_group"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(settings.seed)
    counts = rng.multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=settings.draws
    )[:, inverse]
    positive = counts @ hard
    counts = counts[
        ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(axis=1)
    ]

    def interval(candidate: np.ndarray, reference: np.ndarray) -> dict:
        delta = bootstrap_auc(hard, candidate, counts) - bootstrap_auc(
            hard, reference, counts
        )
        return {
            "observed": macro(candidate) - macro(reference),
            "interval_80": [float(np.quantile(delta, q)) for q in (0.1, 0.9)],
            "share_positive": float((delta > 0).mean()),
        }

    choices = []
    for multiplier in multipliers:
        per_family = {
            family: single_table.filter(
                (pl.col("family") == family)
                & (pl.col("multiplier") == multiplier)
                & pl.col("tta")
            )
            .sort("pooled_macro_auc", descending=True)
            .row(0, named=True)
            for family in settings.families
        }
        tau_a, tau_b = (per_family[family]["tau"] for family in settings.families)
        a = scores[(first, multiplier, tau_a, True)]
        b = scores[(second, multiplier, tau_b, True)]
        by_weight = table.filter(
            (pl.col("multiplier") == multiplier)
            & pl.col("tta")
            & (pl.col(f"tau_{first}") == tau_a)
            & (pl.col(f"tau_{second}") == tau_b)
        ).sort(f"weight_{first}")
        best = by_weight.sort("pooled_macro_auc", descending=True).row(0, named=True)
        weight = best[f"weight_{first}"]
        chosen = weight * a + (1 - weight) * b
        equal = (a + b) / 2
        reference = (
            scores[(first, multiplier, 1.0, True)]
            + scores[(second, multiplier, 1.0, True)]
        ) / 2
        joint = (
            table.filter((pl.col("multiplier") == multiplier) & pl.col("tta"))
            .sort("pooled_macro_auc", descending=True)
            .row(0, named=True)
        )
        choices.append(
            {
                "multiplier": multiplier,
                "family_tau": per_family,
                "weights": by_weight.select(
                    f"weight_{first}", "pooled_macro_auc"
                ).to_dicts(),
                "chosen": best,
                "chosen_per_target": aucs(hard, chosen, targets)["per_target_auc"],
                "equal_weight_macro_auc": macro(equal),
                "chosen_minus_equal_weight": interval(chosen, equal),
                "chosen_minus_equal_weight_tau_1": interval(chosen, reference),
                "joint_grid_best": joint,
                "no_tta_macro_auc": macro(
                    weight * scores[(first, multiplier, tau_a, False)]
                    + (1 - weight) * scores[(second, multiplier, tau_b, False)]
                ),
            }
        )
    base = choices[0]
    base_values = (
        base["chosen"][f"weight_{first}"]
        * scores[(first, base["multiplier"], base["chosen"][f"tau_{first}"], True)]
        + (1 - base["chosen"][f"weight_{first}"])
        * scores[(second, base["multiplier"], base["chosen"][f"tau_{second}"], True)]
    )
    for choice in choices[1:]:
        weight = choice["chosen"][f"weight_{first}"]
        values = (
            weight
            * scores[
                (first, choice["multiplier"], choice["chosen"][f"tau_{first}"], True)
            ]
            + (1 - weight)
            * scores[
                (second, choice["multiplier"], choice["chosen"][f"tau_{second}"], True)
            ]
        )
        choice[f"minus_x{base['multiplier']}"] = interval(values, base_values)
    report = {
        "studies": frame.height,
        "bootstrap_draws": len(counts),
        "seed": settings.seed,
        "singles": singles,
        "choices": choices,
    }
    settings.output.write_text(json.dumps(report, indent=2) + "\n")
    table.write_csv(settings.output.with_suffix(".blends.csv"))
    print(
        single_table.filter(pl.col("tta")).pivot(
            "tau", index=["family", "multiplier"], values="pooled_macro_auc"
        )
    )
    for choice in choices:
        print(
            json.dumps(
                {
                    key: choice[key]
                    for key in choice
                    if key not in {"weights", "family_tau", "chosen_per_target"}
                }
            )
        )


if __name__ == "__main__":
    main()
