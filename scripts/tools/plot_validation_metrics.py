import json
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import polars as pl
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, load_split
from knee_abnormality.evaluation.metrics import soft_aucs

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

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
    soft_macro_auc: list[float]
    gold_macro_auc: list[float]
    training_bce: list[float]
    soft_target_bce: list[float]
    per_target_auc: list[dict[str, float]]


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


def evaluation_targets(run: Path) -> pl.DataFrame:
    config = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    split = load_split(
        ROOT / "data/raw", ROOT / "data/folds" / config["validation"]["split"]
    )
    return split.evaluation.join(split.membership.select(STUDY, "fold"), on=STUDY)


def soft_macro_aucs(
    fold: Path, epochs: list[int], targets: pl.DataFrame
) -> list[float]:
    index = int(fold.name.removeprefix("fold_"))
    names = [column for column in targets.columns if column not in (STUDY, "fold")]
    truth = targets.filter(pl.col("fold") == index).drop("fold").sort(STUDY)
    values = []
    for epoch in epochs:
        path = fold / "epochs" / f"validation_epoch_{epoch:03d}.csv"
        prediction = pl.read_csv(path).select(STUDY, *names).sort(STUDY)
        if prediction[STUDY].to_list() != truth[STUDY].to_list():
            raise ValueError(f"Validation studies differ from the frozen fold: {path}")
        values.append(
            soft_aucs(
                truth.select(names).to_numpy(),
                prediction.select(names).to_numpy(),
                names,
            )["soft_macro_auc"]
        )
    return values


def training_bce(fold: Path, epochs: list[int]) -> list[float]:
    by_epoch: dict[int, float] = {}
    for path in sorted(fold.glob("logs/version_*/metrics.csv")):
        frame = pl.read_csv(path, infer_schema_length=None)
        if "train_loss_epoch" not in frame.columns:
            continue
        for epoch, loss in (
            frame.select("epoch", "train_loss_epoch").drop_nulls().iter_rows()
        ):
            by_epoch[int(epoch)] = float(loss)
    if not by_epoch:
        raise ValueError(f"No epoch training loss logged in {fold}")
    return [by_epoch.get(epoch, float("nan")) for epoch in epochs]


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
            subgroup_values = (metric["gold"]["macro_auc"],)
            per_target = metric["per_target_auc"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid metric file {path}: {error}") from error
        if (
            not all(isinstance(value, int | float) for value in required_values)
            or not all(
                value is None or isinstance(value, int | float)
                for value in subgroup_values
            )
            or not isinstance(per_target, dict)
            or not all(
                value is None or isinstance(value, int | float)
                for value in per_target.values()
            )
        ):
            raise ValueError(f"Invalid metric values in {path}")
        by_fold.setdefault(fold, []).append((int(match.group(1)), metric))
    targets = evaluation_targets(run)
    series = []
    for fold, records in by_fold.items():
        label = run.name if len(by_fold) == 1 else f"{run.name}/{fold}"
        epochs = [epoch for epoch, _ in records]
        series.append(
            MetricSeries(
                label=label,
                epochs=epochs,
                macro_auc=[float(metric["macro_auc"]) for _, metric in records],
                soft_macro_auc=soft_macro_aucs(run / fold, epochs, targets),
                gold_macro_auc=[
                    float(metric["gold"]["macro_auc"])
                    if metric["gold"]["macro_auc"] is not None
                    else float("nan")
                    for _, metric in records
                ],
                training_bce=training_bce(run / fold, epochs),
                soft_target_bce=[
                    float(metric["soft_target_bce"]) for _, metric in records
                ],
                per_target_auc=[
                    {
                        target: float(value) if value is not None else float("nan")
                        for target, value in metric["per_target_auc"].items()
                    }
                    for _, metric in records
                ],
            )
        )
    return series


def ordered_targets(series: list[MetricSeries]) -> list[str]:
    targets: list[str] = []
    for item in series:
        for record in item.per_target_auc:
            for target in record:
                if target not in targets:
                    targets.append(target)
    return targets


def best_epoch_index(item: MetricSeries) -> int:
    return max(range(len(item.epochs)), key=lambda index: item.macro_auc[index])


def plot_per_target(axis: Axes, series: list[MetricSeries], palette: list[str]) -> None:
    targets = ordered_targets(series)
    positions = list(range(len(targets)))
    width = 0.8 / len(series)
    best_values = [
        [
            item.per_target_auc[best_epoch_index(item)].get(target, float("nan"))
            for target in targets
        ]
        for item in series
    ]
    defined = [
        value for values in best_values for value in values if not math.isnan(value)
    ]
    floor = math.floor(min(defined) * 20) / 20 if defined else 0.5
    ceiling = math.ceil(max([*defined, floor + 0.05]) * 20) / 20
    for index, (item, color, values) in enumerate(
        zip(series, palette, best_values, strict=True)
    ):
        offset = (index - (len(series) - 1) / 2) * width
        bars = axis.bar(
            [position + offset for position in positions],
            [value - floor for value in values],
            width=width * 0.9,
            bottom=floor,
            color=color,
            zorder=2,
        )
        axis.axhline(
            item.macro_auc[best_epoch_index(item)],
            color=color,
            linestyle="--",
            alpha=0.5,
            zorder=3,
        )
        if len(series) <= 2:
            axis.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    axis.set_xticks(positions, targets, rotation=30, ha="right")
    axis.set_ylim(floor, ceiling)
    axis.set_title("Per-target AUC at best macro-AUC epoch (dashed: macro AUC)")
    axis.set_ylabel(f"ROC-AUC (bars from {floor:.2f})")
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)


def plot(series: list[MetricSeries], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(19, 9), constrained_layout=True)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette = [cycle[index % len(cycle)] for index in range(len(series))]
    labels = [
        f"{item.label} (best epoch {item.epochs[best_epoch_index(item)]})"
        for item in series
    ]
    panels = (
        (
            axes[0, 0],
            "Validation macro AUC (hard targets at 0.5)",
            "macro_auc",
            "ROC-AUC",
        ),
        (axes[0, 1], "Validation soft macro AUC", "soft_macro_auc", "Soft ROC-AUC"),
        (axes[0, 2], "Gold macro AUC", "gold_macro_auc", "ROC-AUC"),
        (axes[1, 0], "Training BCE (epoch mean, as optimized)", "training_bce", "BCE"),
        (axes[1, 1], "Validation soft-target BCE", "soft_target_bce", "BCE"),
    )
    for axis, title, attribute, ylabel in panels:
        for item, color, label in zip(series, palette, labels, strict=True):
            axis.plot(
                item.epochs,
                getattr(item, attribute),
                marker="o",
                color=color,
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
    plot_per_target(axes[1, 2], series, palette)
    axes[0, 0].legend(loc="best", fontsize=9)
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
