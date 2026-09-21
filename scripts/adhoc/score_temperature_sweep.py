import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD
from knee_abnormality.evaluation.metrics import aucs, bootstrap_auc

NAME = re.compile(r"x(?P<multiplier>\d+)-tau(?P<tau>\d+\.\d+)\.csv")


class SweepCLI(CLISettings):
    sweep_dir: Path
    split_dir: Path = Path("data/folds/all_5fold_scanner_v1")
    fold: int = Field(default=0, ge=0)
    reference_tau: float = 1.0
    draws: int = Field(default=2000, ge=200)
    seed: int = 20260921


def score(settings: SweepCLI) -> None:
    labels = pl.read_csv(settings.split_dir / "targets.csv")
    targets = labels.columns[1:]
    frame = (
        pl.read_csv(settings.split_dir / "studies.csv")
        .filter(pl.col("fold") == settings.fold)
        .join(labels, on=STUDY, validate="1:1")
        .sort(STUDY)
    )
    hard = (frame.select(targets).to_numpy() >= TARGET_THRESHOLD).astype(np.float64)
    groups, inverse = np.unique(frame["report_group"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(settings.seed)
    counts = rng.multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=settings.draws
    )[:, inverse]
    positive = counts @ hard
    counts = counts[
        ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(axis=1)
    ]
    runs = {}
    for path in sorted(settings.sweep_dir.glob("x*-tau*.csv")):
        match = NAME.fullmatch(path.name)
        if match is None:
            continue
        prediction = pl.read_csv(path).sort(STUDY)
        if prediction[STUDY].to_list() != frame[STUDY].to_list():
            raise ValueError(f"{path} does not cover fold {settings.fold} exactly")
        values = prediction.select(targets).to_numpy().astype(np.float64)
        runs[(int(match["multiplier"]), float(match["tau"]))] = values
    rows, draws = [], {}
    for (multiplier, tau), values in runs.items():
        draws[(multiplier, tau)] = bootstrap_auc(hard, values, counts)
        result = aucs(hard.astype(np.int8), values, targets)
        rows.append(
            {
                "multiplier": multiplier,
                "tau": tau,
                "macro_auc": result["macro_auc"],
                **result["per_target_auc"],
            }
        )
    table = pl.DataFrame(rows).sort("multiplier", "tau")
    comparisons = []
    for multiplier, tau in runs:
        reference = (multiplier, settings.reference_tau)
        if reference not in runs or tau == settings.reference_tau:
            continue
        delta = draws[(multiplier, tau)] - draws[reference]
        comparisons.append(
            {
                "multiplier": multiplier,
                "tau": tau,
                "delta_vs_reference": float(delta.mean()),
                "central_80": np.percentile(delta, [10, 90]).tolist(),
                "central_95": np.percentile(delta, [2.5, 97.5]).tolist(),
                "positive_fraction": float((delta > 0).mean()),
            }
        )
    table.write_csv(settings.sweep_dir / "scores.csv")
    summary = {
        "split": str(settings.split_dir),
        "fold": settings.fold,
        "studies": frame.height,
        "reference_tau": settings.reference_tau,
        "bootstrap": {"unit": "report_group", "valid_draws": len(counts)},
        "best": {
            str(m): table.filter(pl.col("multiplier") == m)
            .sort("macro_auc", descending=True)
            .row(0, named=True)
            for m in sorted({m for m, _ in runs})
        },
        "comparisons": sorted(comparisons, key=lambda r: (r["multiplier"], r["tau"])),
    }
    (settings.sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with pl.Config(tbl_rows=40, tbl_cols=16, tbl_width_chars=240, float_precision=4):
        print(table)
    for row in summary["comparisons"]:
        print(
            f"x{row['multiplier']} tau {row['tau']:.2f}: {row['delta_vs_reference']:+.4f} "
            f"80% [{row['central_80'][0]:+.4f}, {row['central_80'][1]:+.4f}] "
            f"P>0 {row['positive_fraction']:.2f}"
        )


def main() -> None:
    try:
        score(SweepCLI())
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Temperature sweep scoring error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
