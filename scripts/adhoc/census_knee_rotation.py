import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.data.splits import STUDY

DATASET = None
PLANES = ("Axial", "Coronal", "Sagittal")
THRESHOLDS = (5, 10, 15, 20, 30)
DEPTH_MM = 30.0


class Settings(CLISettings):
    model_config = describe(
        "Census of how far each cleaned training series is rotated about the "
        "head-foot axis from the scanner's left-right axis, signed as external or "
        "internal rotation of the knee, with the agreement between a study's planes"
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


def rotation(orientation: tuple[float, ...], plane: str) -> float | None:
    axes = np.asarray(orientation, dtype=np.float64).reshape(2, 3)
    vector = np.cross(axes[0], axes[1]) if plane == "Sagittal" else axes[0]
    if not np.isfinite(vector).all() or np.hypot(vector[0], vector[1]) < 1e-6:
        return None
    if vector[0] < 0:
        vector = -vector
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def census(uid: str) -> list[dict]:
    study = DATASET._index_study_geometry(DATASET.series_index[uid])
    excluded = DATASET._excluded_series[uid]
    side = DATASET._study_sides[uid]
    rows = []
    seen = set()
    for slot, values in enumerate(study.slots):
        plane = DATASET.data.series_types[slot].plane
        for candidate in values:
            indexed = DATASET._slice_index.get(candidate.directory)
            if (
                candidate.uid in seen
                or candidate.uid in excluded
                or indexed is None
                or plane not in PLANES
            ):
                continue
            seen.add(candidate.uid)
            orientations = list(indexed.geometry.orientations) or sorted(
                {
                    record.pose.layout.orientation
                    for record in indexed.records
                    if record.pose is not None
                }
            )
            if not orientations:
                continue
            rows.append(
                {
                    STUDY: uid,
                    "series": candidate.uid,
                    "plane": plane,
                    "side": int(side.value),
                    "side_source": side.source,
                    "rotation_deg": rotation(orientations[0], plane),
                }
            )
    return rows


def spread(values: np.ndarray) -> dict:
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.1)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "abs_p50": float(np.quantile(np.abs(values), 0.5)),
        "abs_p90": float(np.quantile(np.abs(values), 0.9)),
        **{f"abs_above_{t}": float((np.abs(values) > t).mean()) for t in THRESHOLDS},
    }


def main() -> None:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(settings.config, data_root=settings.data_root)
    reports = pl.read_csv(config.raw_dir / "train.csv", infer_schema_length=None)
    uids = sorted(reports[STUDY].to_list())
    with get_context("spawn").Pool(
        settings.workers, start, (str(settings.config), str(settings.data_root))
    ) as pool:
        nested = pool.map(census, uids, chunksize=8)
    frame = pl.DataFrame(
        [row for rows in nested for row in rows], infer_schema_length=None
    ).drop_nulls("rotation_deg")
    frame = frame.with_columns(
        (
            pl.col("rotation_deg")
            * pl.col("side").replace_strict({0: 0.0, 1: 1.0, 2: -1.0})
        ).alias("external_deg")
    )
    frame.write_parquet(settings.output_dir / "series.parquet")
    sided = frame.filter(pl.col("side") != 0)
    studies = (
        frame.group_by(STUDY, "plane")
        .agg(pl.col("rotation_deg").mean())
        .pivot("plane", index=STUDY, values="rotation_deg")
    )
    pairs = {}
    for first, second in (
        ("Axial", "Coronal"),
        ("Axial", "Sagittal"),
        ("Coronal", "Sagittal"),
    ):
        both = studies.drop_nulls([first, second])
        difference = (both[first] - both[second]).to_numpy()
        pairs[f"{first}-{second}"] = {
            **spread(difference),
            "correlation": float(
                np.corrcoef(both[first].to_numpy(), both[second].to_numpy())[0, 1]
            ),
        }
    summary = {
        "studies": frame[STUDY].n_unique(),
        "series": frame.height,
        "side_sign_values": sorted(frame["side"].unique().to_list()),
        "scanner_frame_by_plane": {
            plane: spread(
                frame.filter(pl.col("plane") == plane)["rotation_deg"].to_numpy()
            )
            for plane in PLANES
        },
        "external_rotation_by_plane": {
            plane: {
                **spread(
                    sided.filter(pl.col("plane") == plane)["external_deg"].to_numpy()
                ),
                "share_external": float(
                    (sided.filter(pl.col("plane") == plane)["external_deg"] > 0).mean()
                ),
            }
            for plane in PLANES
        },
        "plane_difference_within_study": pairs,
        f"boundary_error_mm_at_{DEPTH_MM:.0f}mm": {
            str(t): float(DEPTH_MM * np.tan(np.radians(t))) for t in THRESHOLDS
        },
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
