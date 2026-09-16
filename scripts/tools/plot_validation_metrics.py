import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsError

from knee_abnormality.cli import CLISettings

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EPOCH_PATTERN = re.compile(r"metrics_epoch_(\d+)\.json$")


class PlotCLI(CLISettings):
    runs: CliPositionalArg[list[str]] = Field(min_length=1)
    output: Path = Path("outputs/analysis/validation_metrics.png")


@dataclass(frozen=True)
class MetricSeries:
    label: str
    epochs: list[int]
    macro_auc: list[float]
    gold_macro_auc: list[float]
    report_macro_auc: list[float]
    soft_target_bce: list[float]


def resolve_run(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    under_training = ROOT / "outputs" / "training" / value
    if under_training.is_dir():
        return under_training
    raise ValueError(f"Run folder not found: {value}")


def metric_files(run: Path) -> Iterable[tuple[str, Path]]:
    folds = sorted(path for path in run.glob("fold_*") if path.is_dir())
    if not folds:
        raise ValueError(f"No fold directories found in {run}")
    for fold in folds:
        files = []
        for path in fold.glob("epochs/metrics_epoch_*.json"):
            match = EPOCH_PATTERN.search(path.name)
            if match is not None:
                files.append((int(match.group(1)), path))
        if not files:
            raise ValueError(f"No epoch metrics found in {fold}")
        for _, path in sorted(files):
            yield fold.name, path


def load_metric_series(run: Path) -> list[MetricSeries]:
    by_fold: dict[str, list[tuple[int, dict]]] = {}
    for fold, path in metric_files(run):
        match = EPOCH_PATTERN.search(path.name)
        if match is None:
            continue
        try:
            metric = json.loads(path.read_text(encoding="utf-8"))
            required_values = (
                metric["macro_auc"],
                metric["soft_target_bce"],
            )
            subgroup_values = (
                metric["gold"]["macro_auc"],
                metric["report_derived"]["macro_auc"],
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid metric file {path}: {error}") from error
        if not all(
            isinstance(value, int | float) for value in required_values
        ) or not all(
            value is None or isinstance(value, int | float) for value in subgroup_values
        ):
            raise ValueError(f"Invalid metric values in {path}")
        by_fold.setdefault(fold, []).append((int(match.group(1)), metric))
    series = []
    for fold, records in by_fold.items():
        label = run.name if len(by_fold) == 1 else f"{run.name}/{fold}"
        series.append(
            MetricSeries(
                label=label,
                epochs=[epoch for epoch, _ in records],
                macro_auc=[float(metric["macro_auc"]) for _, metric in records],
                gold_macro_auc=[
                    float(metric["gold"]["macro_auc"])
                    if metric["gold"]["macro_auc"] is not None
                    else float("nan")
                    for _, metric in records
                ],
                report_macro_auc=[
                    float(metric["report_derived"]["macro_auc"])
                    if metric["report_derived"]["macro_auc"] is not None
                    else float("nan")
                    for _, metric in records
                ],
                soft_target_bce=[
                    float(metric["soft_target_bce"]) for _, metric in records
                ],
            )
        )
    return series


def plot(series: list[MetricSeries], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panels = (
        ("Validation macro AUC", "macro_auc", "ROC-AUC"),
        ("Gold macro AUC", "gold_macro_auc", "ROC-AUC"),
        ("Report-derived macro AUC", "report_macro_auc", "ROC-AUC"),
        ("Soft-target BCE", "soft_target_bce", "BCE"),
    )
    for axis, (title, attribute, ylabel) in zip(axes.flat, panels, strict=True):
        for item in series:
            axis.plot(
                item.epochs, getattr(item, attribute), marker="o", label=item.label
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
    axes[0, 0].legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    try:
        args = PlotCLI()
        runs = [resolve_run(value) for value in args.runs]
        if len(runs) != len(set(runs)):
            raise ValueError("Run folders must be unique")
        series = [item for run in runs for item in load_metric_series(run)]
        output = args.output.expanduser()
        if not output.is_absolute():
            output = ROOT / output
        plot(series, output)
    except (OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Validation metric plot error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(output)


if __name__ == "__main__":
    main()
