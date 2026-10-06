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
QUANTILES = {}
PAIRS = (
    ("validation_x1", "validation_x4"),
    ("validation_x1", "validation_x6"),
    ("training_epoch0", "training_epoch1"),
    ("training_epoch0", "validation_x6"),
)
ESTIMATORS = (
    "pooled",
    "slice_median",
    "slice_mean",
    "pooled_upper_990",
    "slice_median_upper_990",
)


class Settings(CLISettings):
    model_config = describe(
        "Compare intensity-window estimators computed from the bag's own slices by how "
        "much a series' window moves between bags: validation at x1, x4 and x6 and two "
        "training epochs at x1"
    )
    config: Path = Field(description="Experiment configuration")
    data_root: Path = Field(description="Data root holding raw/ and folds/")
    output_dir: Path = Field(description="Fresh output directory")
    fold: int = Field(default=0, description="Validation fold supplying the studies")
    studies: int = Field(default=300, gt=0, description="Studies to measure")
    workers: int = Field(default=6, gt=0, description="Worker processes")


def start(config_path: str, data_root: str, uids: list[str]) -> None:
    config = load_config(Path(config_path), data_root=Path(data_root))
    index = build_series_index(config.paths.data_root, config.data, "train", uids)
    frame = pl.DataFrame({STUDY: uids})
    QUANTILES["bounds"] = (config.data.lower_quantile, config.data.upper_quantile)
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


def windows(slices: list[np.ndarray]) -> dict[str, tuple[float, float, float]]:
    lower, upper = QUANTILES["bounds"]
    pooled = np.concatenate(slices)
    per_slice = np.asarray(
        [np.quantile(values, [lower, upper, 0.99]) for values in slices]
    )
    low, high, high_990 = np.quantile(pooled, [lower, upper, 0.99])
    result = {
        "pooled": (low, high),
        "slice_median": tuple(np.median(per_slice[:, :2], axis=0)),
        "slice_mean": tuple(per_slice[:, :2].mean(axis=0)),
        "pooled_upper_990": (low, high_990),
        "slice_median_upper_990": (
            float(np.median(per_slice[:, 0])),
            float(np.median(per_slice[:, 2])),
        ),
    }
    return {
        name: (float(a), float(b), float((pooled > b).mean()))
        for name, (a, b) in result.items()
    }


def prepared(name: str, row: int) -> dict:
    item = DATASETS[name][row]
    entries = [series for acquisitions in item["series"] for series in acquisitions]
    batch = collate_studies([item])
    result = {}
    for series, (start_index, count, _) in zip(
        entries, batch["series_frames"], strict=True
    ):
        slices = [
            np.asarray(
                device._region(
                    batch["frames"][start_index + index], batch["frame_buffers"]
                )
            )
            .astype(np.float32)
            .reshape(-1)
            for index in range(count)
        ]
        result[(series.uid, series.view_index)] = (windows(slices), count)
    return result


def measure(row: int) -> list[dict]:
    bags = {name: prepared(name, row) for name in DATASETS}
    rows = []
    for first, second in PAIRS:
        for key in bags[first].keys() & bags[second].keys():
            (a, slices_a), (b, slices_b) = bags[first][key], bags[second][key]
            for estimator in ESTIMATORS:
                low_a, high_a, clipped_a = a[estimator]
                low_b, high_b, clipped_b = b[estimator]
                if not (high_a > low_a and high_b > low_b):
                    continue
                rows.append(
                    {
                        "row": row,
                        "pair": f"{first} -> {second}",
                        "series": key[0],
                        "view": key[1],
                        "estimator": estimator,
                        "gain_ratio": (high_a - low_a) / (high_b - low_b),
                        "high_shift_fraction": (high_b - high_a) / (high_a - low_a),
                        "low_shift_fraction": (low_b - low_a) / (high_a - low_a),
                        "clipped_first": clipped_a,
                        "clipped_second": clipped_b,
                        "slices_first": slices_a,
                        "slices_second": slices_b,
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
    summary = {"studies": len(uids), "fold": settings.fold, "pairs": {}}
    for (pair, estimator), part in frame.partition_by(
        "pair", "estimator", as_dict=True
    ).items():
        gain = np.abs(np.log(part["gain_ratio"].to_numpy())) * 100
        summary["pairs"].setdefault(pair, {})[estimator] = {
            "series": part.height,
            "abs_gain_change_percent": {
                str(q): float(np.quantile(gain, q)) for q in (0.5, 0.9, 0.99)
            },
            "signed_gain_change_percent_mean": float(
                np.log(part["gain_ratio"].to_numpy()).mean() * 100
            ),
            "gain_over_10_percent": int((gain > 10).sum()),
            "clipped_top_percent_median": float(
                np.median(part["clipped_first"].to_numpy()) * 100
            ),
            "clipped_top_percent_p90": float(
                np.quantile(part["clipped_first"].to_numpy(), 0.9) * 100
            ),
        }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
