import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
import pydicom
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.labels import load_fold_labels
from knee_abnormality.data.preprocessing import device
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.data.splits import STUDY

STATE = {}


class Settings(CLISettings):
    model_config = describe(
        "Compare the DICOM header window (WindowCenter, WindowWidth) of every slice of a "
        "series with the pixel-percentile window of its x1 validation bag: presence, "
        "consistency across slices and agreement by vendor"
    )
    config: Path = Field(description="Experiment configuration")
    data_root: Path = Field(description="Data root holding raw/ and folds/")
    output_dir: Path = Field(description="Fresh output directory")
    fold: int = Field(default=0, description="Validation fold supplying the studies")
    studies: int = Field(default=300, gt=0, description="Studies to measure")
    workers: int = Field(default=4, gt=0, description="Worker processes")


def start(config_path: str, data_root: str, uids: list[str]) -> None:
    config = load_config(Path(config_path), data_root=Path(data_root))
    index = build_series_index(config.paths.data_root, config.data, "train", uids)
    STATE["bounds"] = (config.data.lower_quantile, config.data.upper_quantile)
    STATE["dataset"] = KneeStudyDataset(
        pl.DataFrame({STUDY: uids}), index, config.data, (), None, 1
    )


def first_value(value) -> float | None:
    if value is None:
        return None
    try:
        return float(
            value[0] if isinstance(value, pydicom.multival.MultiValue) else value
        )
    except (TypeError, ValueError, IndexError):
        return None


def header_windows(paths: list[Path]) -> list[tuple[float, float]]:
    windows = []
    for path in paths:
        header = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=["WindowCenter", "WindowWidth"],
        )
        center = first_value(header.get("WindowCenter"))
        width = first_value(header.get("WindowWidth"))
        if center is not None and width is not None and width > 0:
            windows.append((center - width / 2, center + width / 2))
    return windows


def measure(row: int) -> list[dict]:
    dataset = STATE["dataset"]
    item = dataset[row]
    uid = item["study_uid"]
    directories = {
        candidate.uid: candidate.directory
        for values in dataset.series_index[uid].slots
        for candidate in values
    }
    manufacturer = "|".join(dataset._study_sides[uid].manufacturers)
    entries = [series for acquisitions in item["series"] for series in acquisitions]
    batch = collate_studies([item])
    lower, upper = STATE["bounds"]
    rows = []
    for series, (start_index, count, _) in zip(
        entries, batch["series_frames"], strict=True
    ):
        pixels = np.concatenate(
            [
                np.asarray(
                    device._region(
                        batch["frames"][start_index + index], batch["frame_buffers"]
                    )
                )
                .astype(np.float32)
                .reshape(-1)
                for index in range(count)
            ]
        )
        low, high = (float(value) for value in np.quantile(pixels, [lower, upper]))
        records = dataset._slice_index[directories[series.uid]].records
        windows = header_windows([record.path for record in records])
        result = {
            STUDY: uid,
            "series": series.uid,
            "view": series.view_index,
            "manufacturer": manufacturer,
            "slices": len(records),
            "header_slices": len(windows),
            "pixel_low": low,
            "pixel_high": high,
        }
        if windows and high > low:
            values = np.asarray(windows)
            top = float(np.median(values[:, 1]))
            bottom = float(np.median(values[:, 0]))
            result |= {
                "header_low": bottom,
                "header_high": top,
                "header_top_spread": float(
                    (np.quantile(values[:, 1], 0.9) - np.quantile(values[:, 1], 0.1))
                    / top
                )
                if top > 0
                else None,
                "header_distinct_tops": len(np.unique(values[:, 1])),
                "log_top_ratio": float(np.log(top / high))
                if top > 0 and high > 0
                else None,
                "width_ratio": (top - bottom) / (high - low),
                "clipped_above_header": float((pixels > top).mean()),
                "clipped_below_header": float((pixels < bottom).mean()),
            }
        rows.append(result)
    return rows


def spread(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


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
    frame = pl.DataFrame(
        [row for rows in results for row in rows], infer_schema_length=None
    )
    frame.write_parquet(settings.output_dir / "series.parquet")
    summary = {"studies": len(uids), "series": frame.height, "by_manufacturer": {}}
    for (name,), part in sorted(
        frame.partition_by("manufacturer", as_dict=True).items()
    ):
        present = part.filter(pl.col("header_high").is_not_null())
        entry = {
            "series": part.height,
            "with_header_window": present.height,
            "all_slices_tagged": int((part["header_slices"] == part["slices"]).sum()),
        }
        if present.height:
            entry |= {
                "one_window_for_the_series": int(
                    (present["header_distinct_tops"] == 1).sum()
                ),
                "top_spread_across_slices": spread(
                    present["header_top_spread"].drop_nulls().to_numpy()
                ),
                "header_top_over_pixel_top": spread(
                    np.exp(present["log_top_ratio"].drop_nulls().to_numpy())
                ),
                "width_ratio": spread(present["width_ratio"].to_numpy()),
                "clipped_above_percent": spread(
                    present["clipped_above_header"].to_numpy() * 100
                ),
            }
        summary["by_manufacturer"][name] = entry
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
