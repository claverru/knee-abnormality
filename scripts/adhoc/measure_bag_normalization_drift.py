import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.labels import load_fold_labels
from knee_abnormality.data.preprocessing import device
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.data.splits import STUDY

DATASETS = {}
BOUNDS = []
PAIRS = (
    ("validation_x1", "validation_x4"),
    ("validation_x1", "validation_x6"),
    ("training_epoch0", "training_epoch1"),
    ("training_epoch0", "validation_x6"),
)


class Settings(CLISettings):
    model_config = describe(
        "Measure how a series' intensity bounds and its shared slices' pixels change "
        "with the bag: validation bags at x1, x4 and x6 and two training epochs at x1"
    )
    config: Path = Field(description="Experiment configuration")
    data_root: Path = Field(description="Data root holding raw/ and folds/")
    output_dir: Path = Field(description="Fresh output directory")
    fold: int = Field(default=0, description="Validation fold supplying the studies")
    studies: int = Field(default=300, gt=0, description="Studies to measure")
    workers: int = Field(default=6, gt=0, description="Worker processes")


def recording_bounds(values, lower, upper, fraction):
    low, high = ORIGINAL(values, lower, upper, fraction)
    BOUNDS.append((float(low), float(high)))
    return low, high


ORIGINAL = device.quantile_bounds


def start(config_path: str, data_root: str, uids: list[str]) -> None:
    config = load_config(Path(config_path), data_root=Path(data_root))
    index = build_series_index(config.paths.data_root, config.data, "train", uids)
    frame = pl.DataFrame({STUDY: uids})
    for name, seed, multiplier, epoch in (
        ("validation_x1", None, 1, 0),
        ("validation_x4", None, 4, 0),
        ("validation_x6", None, 6, 0),
        ("training_epoch0", config.seed, 1, 0),
        ("training_epoch1", config.seed, 1, 1),
    ):
        dataset = KneeStudyDataset(frame, index, config.data, (), seed, multiplier)
        dataset.set_epoch(epoch)
        DATASETS[name] = dataset
    device.quantile_bounds = recording_bounds


def prepared(name: str, row: int) -> dict:
    item = DATASETS[name][row]
    entries = [series for acquisitions in item["series"] for series in acquisitions]
    BOUNDS.clear()
    batch = device._prepare_on_cpu(collate_studies([item]))
    result = {}
    for index, (series, (low, high)) in enumerate(zip(entries, BOUNDS, strict=True)):
        slices = {}
        for triplet, paths in enumerate(series.selected_paths):
            slices.setdefault(paths[1], batch["images"][index, triplet, 1].numpy())
        result[(series.uid, series.view_index)] = (low, high, slices)
    return result


def measure(row: int) -> list[dict]:
    bags = {name: prepared(name, row) for name in DATASETS}
    rows = []
    for first, second in PAIRS:
        for key in bags[first].keys() & bags[second].keys():
            low_a, high_a, slices_a = bags[first][key]
            low_b, high_b, slices_b = bags[second][key]
            if not (high_a > low_a and high_b > low_b):
                continue
            shared = sorted(slices_a.keys() & slices_b.keys())
            differences = [
                np.abs(slices_a[path].astype(np.int16) - slices_b[path])
                for path in shared
            ]
            body = [slices_a[path] > 0 for path in shared]
            pixels = int(sum(mask.sum() for mask in body))
            rows.append(
                {
                    "row": row,
                    "pair": f"{first} -> {second}",
                    "series": key[0],
                    "view": key[1],
                    "gain_ratio": (high_a - low_a) / (high_b - low_b),
                    "low_shift_fraction": (low_b - low_a) / (high_a - low_a),
                    "high_shift_fraction": (high_b - high_a) / (high_a - low_a),
                    "slices_first": len(slices_a),
                    "slices_second": len(slices_b),
                    "shared_slices": len(shared),
                    "mean_level_difference": float(
                        sum(d[m].sum() for d, m in zip(differences, body, strict=True))
                        / pixels
                    )
                    if pixels
                    else None,
                    "max_level_difference": int(max(d.max() for d in differences))
                    if differences
                    else None,
                }
            )
    return rows


def main() -> None:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(settings.config, data_root=settings.data_root)
    validation = load_fold_labels(config, settings.fold).validation
    uids = sorted(validation[STUDY].to_list())[: settings.studies]
    with get_context("spawn").Pool(
        settings.workers,
        start,
        (str(settings.config), str(settings.data_root), uids),
    ) as pool:
        results = pool.map(measure, range(len(uids)), chunksize=2)
    frame = pl.DataFrame([row for rows in results for row in rows])
    frame.write_parquet(settings.output_dir / "series.parquet")
    quantiles = (0.5, 0.9, 0.99, 1.0)
    summary = {"studies": len(uids), "fold": settings.fold, "pairs": {}}
    for pair, part in frame.partition_by("pair", as_dict=True).items():
        gain = np.abs(np.log(part["gain_ratio"].to_numpy())) * 100
        levels = part["mean_level_difference"].drop_nulls().to_numpy()
        summary["pairs"][pair[0]] = {
            "series": part.height,
            "bounds_identical": int(
                ((part["gain_ratio"] == 1) & (part["low_shift_fraction"] == 0)).sum()
            ),
            "abs_gain_change_percent": {
                str(q): float(np.quantile(gain, q)) for q in quantiles
            },
            "gain_over_5_percent": int((gain > 5).sum()),
            "gain_over_10_percent": int((gain > 10).sum()),
            "mean_level_difference_of_255": {
                str(q): float(np.quantile(levels, q)) for q in quantiles
            }
            if len(levels)
            else None,
            "series_with_shared_slices": int((part["shared_slices"] > 0).sum()),
        }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
