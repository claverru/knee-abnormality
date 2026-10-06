import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.preprocessing.geometry import reflection_eligibility
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.data.splits import STUDY

DATASET = None
NOMINAL = {"Axial": (0, 1, 2), "Coronal": (0, 2, 1), "Sagittal": (1, 2, 0)}
ANGLES = ("flip_angle_deg", "normal_tilt_deg", "row_angle_deg", "column_angle_deg")
THRESHOLDS = (5, 10, 20, 30)
RADIUS_MM = 40.0


class Settings(CLISettings):
    model_config = describe(
        "Census of how far each cleaned training series is tilted from the patient "
        "axes that reflection augmentation and reflection TTA assume, by plane and "
        "reflection eligibility"
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


def angle(vector: np.ndarray, axis: int) -> float:
    unit = vector / np.linalg.norm(vector)
    return float(np.degrees(np.arccos(min(1.0, abs(unit[axis])))))


def tilt(orientations: list[tuple[float, ...]], plane: str) -> dict:
    row_axis, column_axis, normal_axis = NOMINAL[plane]
    rows = []
    for orientation in orientations:
        axes = np.asarray(orientation, dtype=np.float64).reshape(2, 3)
        normal = np.cross(axes[0], axes[1])
        if not np.isfinite(axes).all() or min(
            np.linalg.norm(axes[0]), np.linalg.norm(axes[1]), np.linalg.norm(normal)
        ) < 1e-6:
            continue
        rows.append(
            (
                angle(normal, 0) if plane == "Sagittal" else angle(axes[0], 0),
                angle(normal, normal_axis),
                angle(axes[0], row_axis),
                angle(axes[1], column_axis),
            )
        )
    if not rows:
        return dict.fromkeys(ANGLES)
    return dict(zip(ANGLES, np.max(np.asarray(rows), axis=0).tolist(), strict=True))


def census(uid: str) -> list[dict]:
    study = DATASET._index_study_geometry(DATASET.series_index[uid])
    excluded = DATASET._excluded_series[uid]
    rows = []
    seen = set()
    for slot, values in enumerate(study.slots):
        plane = DATASET.data.series_types[slot].plane
        for candidate in values:
            if candidate.uid in seen:
                continue
            seen.add(candidate.uid)
            indexed = DATASET._slice_index.get(candidate.directory)
            row = {
                STUDY: uid,
                "series": candidate.uid,
                "slot": slot,
                "plane": plane,
                "indexed": indexed is not None,
                "cleaned_out": candidate.uid in excluded,
            }
            if indexed is None:
                rows.append(
                    {
                        **row,
                        "eligible": False,
                        "reason": "unindexed",
                        "distinct_orientations": 0,
                        **dict.fromkeys(ANGLES),
                    }
                )
                continue
            geometry = indexed.geometry
            eligible, reason = reflection_eligibility(geometry, plane)
            orientations = list(geometry.orientations) or sorted(
                {
                    record.pose.layout.orientation
                    for record in indexed.records
                    if record.pose is not None
                }
            )
            rows.append(
                {
                    **row,
                    "eligible": eligible,
                    "reason": reason,
                    "distinct_orientations": len(orientations),
                    **(tilt(orientations, plane) if plane in NOMINAL else dict.fromkeys(ANGLES)),
                }
            )
    return rows


def misplacement(degrees: float) -> dict:
    radians = np.radians(degrees)
    return {
        "chord_2r_sin_half_a_mm": float(2 * RADIUS_MM * np.sin(radians / 2)),
        "mirror_line_2r_sin_a_mm": float(2 * RADIUS_MM * np.sin(radians)),
    }


def describe_angles(frame: pl.DataFrame) -> dict:
    result = {"series": frame.height}
    for name in ANGLES:
        values = frame[name].drop_nulls().to_numpy()
        if not len(values):
            continue
        quantiles = {
            label: float(np.quantile(values, q))
            for label, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))
        }
        result[name] = {
            **quantiles,
            **{f"above_{t}": int((values > t).sum()) for t in THRESHOLDS},
        }
        if name == "flip_angle_deg":
            result[name]["misplacement_at_40mm"] = {
                label: misplacement(value) for label, value in quantiles.items()
            }
    return result


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
    )
    kept = frame.filter(pl.col("indexed") & ~pl.col("cleaned_out"))
    kept = kept.with_columns(
        pl.col("eligible").all().over(STUDY).alias("study_eligible")
    )
    frame = frame.join(
        kept.select("series", STUDY, "study_eligible"),
        on=["series", STUDY],
        how="left",
    )
    frame.write_parquet(settings.output_dir / "series.parquet")
    studies = kept.group_by(STUDY).agg(
        pl.col("eligible").all().alias("eligible"),
        pl.col("flip_angle_deg").max().alias("flip"),
        pl.col("normal_tilt_deg").max().alias("tilt"),
        pl.max_horizontal("flip_angle_deg", "normal_tilt_deg").max().alias("either"),
        *[
            pl.col("flip_angle_deg")
            .filter(pl.col("plane") == plane)
            .max()
            .alias(f"flip_{plane}")
            for plane in NOMINAL
        ],
    )
    eligible_studies = studies.filter(pl.col("eligible"))
    reflected = kept.filter(pl.col("study_eligible"))
    planes = sorted(kept["plane"].unique().to_list())
    summary = {
        "studies": len(uids),
        "studies_with_surviving_series": studies.height,
        "series_listed": frame.height,
        "series_unindexed": int((~frame["indexed"]).sum()),
        "series_cleaned_out": int(frame["cleaned_out"].sum()),
        "series_surviving": kept.height,
        "series_eligible": int(kept["eligible"].sum()),
        "series_in_eligible_studies": reflected.height,
        "eligible_studies": eligible_studies.height,
        "eligible_study_share_above": {
            column: {
                f"above_{t}": {
                    "studies": int((eligible_studies[column] > t).sum()),
                    "share": float((eligible_studies[column] > t).mean()),
                }
                for t in THRESHOLDS
            }
            for column in (
                "flip",
                "tilt",
                "either",
                *[f"flip_{plane}" for plane in NOMINAL],
            )
        },
        "ineligible_by_reason": kept.filter(~pl.col("eligible"))
        .group_by("plane", "reason")
        .len()
        .sort("plane", "len", descending=[False, True])
        .to_dicts(),
        "eligible_series_by_plane": {
            plane: describe_angles(
                kept.filter(pl.col("eligible") & (pl.col("plane") == plane))
            )
            for plane in planes
        },
        "series_in_eligible_studies_by_plane": {
            plane: describe_angles(reflected.filter(pl.col("plane") == plane))
            for plane in planes
        },
        "ineligible_series_by_plane": {
            plane: describe_angles(
                kept.filter(~pl.col("eligible") & (pl.col("plane") == plane))
            )
            for plane in planes
        },
        "misplacement_at_40mm_by_angle": {
            str(t): misplacement(t) for t in (1, 2, 5, 10, 20, 30, 45)
        },
        "multiple_orientation_series": int((kept["distinct_orientations"] > 1).sum()),
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
