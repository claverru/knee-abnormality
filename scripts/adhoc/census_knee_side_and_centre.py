import json
import re
import unicodedata
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.preprocessing.laterality import SIDE_NAMES, WORDS
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.data.splits import STUDY

DATASET = None


class Settings(CLISettings):
    model_config = describe(
        "Census of how each training study's knee side and centre are resolved: the "
        "side source by vendor with a report cross-check, and the centre-fit method, "
        "residuals and frame-of-reference count"
    )
    config: Path = Field(description="Experiment configuration")
    data_root: Path = Field(description="Data root holding raw/")
    output_dir: Path = Field(description="Fresh output directory")
    workers: int = Field(default=6, gt=0, description="Worker processes")


def start(config_path: str, data_root: str) -> None:
    global DATASET
    config = load_config(Path(config_path), data_root=Path(data_root))
    index = build_series_index(config.paths.data_root, config.data, "train")
    DATASET = KneeStudyDataset(
        pl.DataFrame({STUDY: sorted(index)}), index, config.data, (), None
    )


def census(uid: str) -> dict:
    DATASET._index_study_geometry(DATASET.series_index[uid])
    side = DATASET._study_sides[uid]
    reference = DATASET._study_references[uid]
    geometry = DATASET._series_geometry[uid]
    residuals = [value for _, value in reference.residuals_mm]
    return {
        STUDY: uid,
        "side": SIDE_NAMES[side.value],
        "side_source": side.source,
        "side_reason": side.reason,
        "coordinate_side": SIDE_NAMES[side.coordinate_side],
        "center_x_mm": side.center_x_mm,
        "manufacturer": "|".join(side.manufacturers),
        "series": len(geometry),
        "trusted_series": sum(valid for _, _, valid in geometry.values()),
        "centre_method": reference.method if reference.point is not None else "none",
        "centre_reason": reference.reason,
        "centre_rms_mm": reference.rms_mm,
        "centre_max_residual_mm": max(residuals) if residuals else None,
        "contributors": len(reference.contributors),
        "frame_uids": len(reference.frame_uids),
    }


def report_side(report: str | None) -> str:
    normalized = (
        unicodedata.normalize("NFKD", (report or "").upper())
        .encode("ascii", "ignore")
        .decode()
    )
    tokens = set(re.findall(r"[A-Z]+", normalized))
    found = [side for side in ("L", "R") if tokens & WORDS[side]]
    return {"L": "left", "R": "right"}[found[0]] if len(found) == 1 else "none"


def main() -> None:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(settings.config, data_root=settings.data_root)
    reports = pl.read_csv(config.raw_dir / "train.csv", infer_schema_length=None)
    uids = sorted(reports[STUDY].to_list())
    with get_context("spawn").Pool(
        settings.workers, start, (str(settings.config), str(settings.data_root))
    ) as pool:
        rows = pool.map(census, uids, chunksize=8)
    frame = pl.DataFrame(rows, infer_schema_length=None).join(
        reports.select(
            STUDY,
            pl.col("Report")
            .map_elements(report_side, return_dtype=pl.String)
            .alias("report_side"),
        ),
        on=STUDY,
        validate="1:1",
    )
    frame.write_parquet(settings.output_dir / "studies.parquet")
    checked = frame.filter(
        (pl.col("side") != "unknown") & (pl.col("report_side") != "none")
    ).with_columns((pl.col("side") == pl.col("report_side")).alias("agrees"))
    fitted = frame.filter(pl.col("centre_rms_mm").is_not_null())
    summary = {
        "studies": frame.height,
        "side_source": frame.group_by("side_source", "side_reason")
        .len()
        .sort("len", descending=True)
        .to_dicts(),
        "trusted_geometry_studies": int((frame["trusted_series"] > 0).sum()),
        "coordinate_source_by_manufacturer": frame.filter(
            pl.col("side_source") == "coordinates"
        )
        .group_by("manufacturer")
        .len()
        .sort("len", descending=True)
        .to_dicts(),
        "report_agreement": checked.group_by("side_source", "manufacturer")
        .agg(pl.len().alias("checked"), pl.col("agrees").sum().alias("agree"))
        .sort("side_source", "checked", descending=[False, True])
        .to_dicts(),
        "centre_method": frame.group_by("centre_method", "centre_reason")
        .len()
        .sort("len", descending=True)
        .to_dicts(),
        "centre_rms_mm": {
            str(q): float(np.quantile(fitted["centre_rms_mm"].to_numpy(), q))
            for q in (0.5, 0.9, 0.99, 1.0)
        },
        "centre_max_residual_mm": {
            str(q): float(np.quantile(fitted["centre_max_residual_mm"].to_numpy(), q))
            for q in (0.5, 0.9, 0.99, 1.0)
        },
        "frame_uids_per_study": {
            str(q): float(np.quantile(frame["frame_uids"].to_numpy(), q))
            for q in (0.0, 0.5, 1.0)
        },
        "contributors_equal_frame_uids": int(
            (frame["contributors"] == frame["frame_uids"]).sum()
        ),
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
