import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.labels import load_fold_labels
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.data.splits import STUDY
from knee_abnormality.identity import preprocessing_identity
from knee_abnormality.provenance import copy_sources, sha256_file

LEVELS = (0.005, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995)
NAMES = ("q005", "q25", "q50", "q75", "q95", "q99", "q995")


class StabilityCLI(CLISettings):
    config: Path = Path("configs/training.yaml")
    output_dir: Path
    studies: int = Field(default=30, ge=2, le=200)
    epochs: int = Field(default=6, ge=2, le=20)
    io_threads: int = Field(default=8, ge=1, le=32)
    sample_seed: int = 20260920


def region(decoded) -> np.ndarray:
    top, left, rows, columns = decoded.rect
    return decoded.image[top : top + rows, left : left + columns].reshape(-1)


def pool_quantiles(paths, cache) -> dict | None:
    values = []
    for path in paths:
        decoded = cache.get(Path(path))
        if decoded is None or isinstance(decoded, str):
            return None
        values.append(region(decoded))
    if not values:
        return None
    pooled = np.concatenate(values)
    quantiles = np.quantile(pooled.astype(np.float64), LEVELS)
    row = {name: float(value) for name, value in zip(NAMES, quantiles, strict=True)}
    row["pixels"] = int(pooled.size)
    return row


class RecordingDataset(KneeStudyDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.series_rows: list[dict] = []
        self.triplet_rows: list[dict] = []
        self.study_uid = ""

    def _try_prepare_candidate(
        self, candidate, selected, cache, failures, physical, plane
    ):
        value = super()._try_prepare_candidate(
            candidate, selected, cache, failures, physical, plane
        )
        if value is None or not value.selected_paths:
            return value
        unique = sorted({path for group in value.selected_paths for path in group})
        pooled = pool_quantiles(unique, cache)
        if pooled is not None:
            self.series_rows.append(
                {
                    STUDY: self.study_uid,
                    "series": candidate.uid,
                    "plane": plane,
                    "epoch": int(self.epoch.value),
                    "low": float(value.intensity_bounds[0]),
                    "high": float(value.intensity_bounds[1]),
                    "slices": len(unique),
                    "triplets": len(value.selected_paths),
                    **pooled,
                }
            )
        for group in value.selected_paths:
            stats = pool_quantiles(group, cache)
            if stats is None:
                continue
            self.triplet_rows.append(
                {
                    STUDY: self.study_uid,
                    "series": candidate.uid,
                    "epoch": int(self.epoch.value),
                    "middle": group[1],
                    **stats,
                }
            )
        return value

    def __getitem__(self, index: int):
        self.study_uid = str(self.studies[STUDY][index])
        return super().__getitem__(index)


def relative_range(frame: pl.DataFrame, column: str, keys: list[str]) -> pl.DataFrame:
    return (
        frame.group_by(keys)
        .agg(
            pl.len().alias("draws"),
            pl.col(column).min().alias("minimum"),
            pl.col(column).max().alias("maximum"),
            pl.col(column).median().alias("middle"),
        )
        .filter((pl.col("draws") > 1) & (pl.col("middle") > 0))
        .with_columns(
            ((pl.col("maximum") - pl.col("minimum")) / pl.col("middle")).alias("spread")
        )
    )


def summarize(frame: pl.DataFrame, keys: list[str]) -> dict:
    result = {}
    for name in NAMES + ("high",):
        if name not in frame.columns:
            continue
        spreads = relative_range(frame, name, keys)["spread"]
        if not spreads.len():
            continue
        result[name] = {
            "median": round(float(spreads.median()), 4),
            "p90": round(float(spreads.quantile(0.9)), 4),
            "maximum": round(float(spreads.max()), 4),
            "groups": spreads.len(),
        }
    return result


def measure(args: StabilityCLI) -> dict:
    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")
    config = load_config(args.config)
    labels = load_fold_labels(config, config.validation.folds[0])
    index = build_series_index(config.paths.data_root, config.data, "train")
    pool = sorted(set(labels.training[STUDY].to_list()) & set(index))
    rng = np.random.default_rng(args.sample_seed)
    selected = sorted(
        rng.choice(pool, min(args.studies, len(pool)), replace=False).tolist()
    )
    rows = pl.DataFrame({STUDY: selected}).join(
        labels.training, on=STUDY, how="left", maintain_order="left"
    )
    dataset = RecordingDataset(
        rows,
        index,
        config.data,
        labels.targets,
        config.seed,
        io_threads=args.io_threads,
    )
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        for position in range(len(dataset)):
            dataset[position]
    args.output_dir.mkdir(parents=True)
    series = pl.DataFrame(dataset.series_rows)
    triplets = pl.DataFrame(dataset.triplet_rows)
    series.write_parquet(args.output_dir / "series_bounds.parquet")
    triplets.write_parquet(args.output_dir / "triplet_bounds.parquet")
    zero_low = float((series["low"] == 0.0).mean())
    within = (
        triplets.group_by([STUDY, "series", "epoch"])
        .agg(
            pl.len().alias("draws"),
            pl.col("q995").min().alias("minimum"),
            pl.col("q995").max().alias("maximum"),
            pl.col("q995").median().alias("middle"),
        )
        .filter((pl.col("draws") > 1) & (pl.col("middle") > 0))
        .with_columns(
            ((pl.col("maximum") - pl.col("minimum")) / pl.col("middle")).alias("spread")
        )
    )
    contrast = config.data.intensity_augmentation.contrast_range
    report = {
        "studies": len(selected),
        "epochs": args.epochs,
        "series_observations": series.height,
        "triplet_observations": triplets.height,
        "identity": preprocessing_identity(config.model.backbone),
        "low_bound_exactly_zero": round(zero_low, 4),
        "across_draws_same_series": summarize(series, [STUDY, "series"]),
        "across_triplets_within_series": {
            "median": round(float(within["spread"].median()), 4),
            "p90": round(float(within["spread"].quantile(0.9)), 4),
            "groups": within.height,
        },
        "contrast_augmentation_relative_range": round(
            (contrast[1] - contrast[0]) / ((contrast[0] + contrast[1]) / 2), 4
        ),
    }
    manifest = {
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "sample_seed": args.sample_seed,
        "training_seed": config.seed,
        "fold": config.validation.folds[0],
        "split": config.validation.split,
    }
    copy_sources((Path(__file__),), args.output_dir / "source")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest["report_sha256"] = sha256_file(args.output_dir / "report.json")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return report


def main() -> None:
    try:
        report = measure(StabilityCLI())
    except (OSError, RuntimeError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Intensity bound stability error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
