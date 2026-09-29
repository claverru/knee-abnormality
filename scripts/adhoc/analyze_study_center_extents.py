import json
import sys
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import SLICES_PER_INPUT, load_config
from knee_abnormality.data.preprocessing.dicom import PREPARATION_ERRORS
from knee_abnormality.data.preprocessing.filtering import clean_series
from knee_abnormality.data.preprocessing.geometry import estimate_reference
from knee_abnormality.data.preprocessing.series import index_series
from knee_abnormality.data.sources import SERIES, StudySeries, build_series_index
from knee_abnormality.data.splits import STUDY
from knee_abnormality.provenance import copy_sources, git_revision, sha256_file

PLANE_AXIS = {"Axial": 2, "Coronal": 1, "Sagittal": 0}
PLANE_SIDES = {
    "Axial": ("inferior", "superior"),
    "Coronal": ("anterior", "posterior"),
    "Sagittal": ("right", "left"),
}


class ExtentsCLI(CLISettings):
    config: Path = Path("configs/training.yaml")
    split: str = "train"
    output_dir: Path = Path("outputs/analysis/study-center-extents")
    workers: int = Field(default=12, ge=1, le=32)
    io_threads: int = Field(default=4, ge=1, le=16)
    studies: int | None = Field(default=None, ge=1)


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        **{
            f"p{q}": float(np.percentile(array, q)) for q in (0, 5, 25, 50, 75, 95, 100)
        },
    }


def measure_study(study: StudySeries, types: tuple, io_threads: int) -> tuple:
    candidates = {}
    slots = {}
    for slot, values in enumerate(study.slots):
        for candidate in values:
            candidates.setdefault(candidate.uid, (candidate, types[slot]))
            slots.setdefault(candidate.uid, slot)
    indexed = {}
    for uid, (candidate, kind) in sorted(candidates.items()):
        try:
            indexed[uid] = (
                index_series(
                    candidate.directory,
                    io_threads,
                    list(candidate.listing) if candidate.listing is not None else None,
                ),
                kind,
            )
        except PREPARATION_ERRORS as error:
            indexed[uid] = (None, kind, f"{type(error).__name__}: {error}")
    readable = {uid: value[0] for uid, value in indexed.items() if value[0] is not None}
    excluded, _ = clean_series(
        readable,
        slots,
        tuple(
            (kind.plane, kind.fluid_sensitive, kind.fat_suppression) for kind in types
        ),
    )
    eligible = tuple(
        (uid, kind[0], value.geometry)
        for uid, (value, kind, *_) in indexed.items()
        if value is not None
        and uid not in excluded
        and len(value.records) >= SLICES_PER_INPUT
    )
    reference = estimate_reference(eligible)
    point = np.asarray(reference.point, dtype=np.float64) if reference.point else None
    residuals = dict(reference.residuals_mm)
    rows = []
    for uid, value in sorted(indexed.items()):
        series, kind = value[0], value[1]
        plane, fluid, fat = kind
        row = {
            STUDY: study.uid,
            SERIES: uid,
            "plane": plane,
            "fluid_sensitive": fluid,
            "fat_suppression": fat,
            "slices": len(series.records) if series is not None else 0,
            "excluded": uid in excluded,
            "eligible": any(uid == item[0] for item in eligible),
            "geometry_reason": series.geometry.reason
            if series is not None
            else value[2],
            "residual_mm": residuals.get(uid),
            "slice_spacing_mm": None,
            "negative_mm": None,
            "positive_mm": None,
        }
        if series is not None and series.geometry.reason is None:
            geometry = series.geometry
            normal = np.asarray(geometry.normal, dtype=np.float64)
            scale = float(np.linalg.norm(normal))
            unit = normal / scale
            positions = (
                np.asarray(
                    [record.physical_position for record in series.records],
                    dtype=np.float64,
                )
                / scale
            )
            steps = np.abs(np.diff(positions))
            row["slice_spacing_mm"] = float(np.median(steps)) if steps.size else None
            if point is not None:
                offset = float(point @ unit)
                low, high = positions[0] - offset, positions[-1] - offset
                if unit[PLANE_AXIS[plane]] < 0:
                    low, high = -high, -low
                row["negative_mm"] = low
                row["positive_mm"] = high
        rows.append(row)
    return (
        {
            STUDY: study.uid,
            "accepted": reference.point is not None,
            "reason": reference.reason,
            "center_x_mm": float(point[0]) if point is not None else None,
            "center_y_mm": float(point[1]) if point is not None else None,
            "center_z_mm": float(point[2]) if point is not None else None,
            "rms_mm": reference.rms_mm,
            "condition": reference.condition,
            "contributors": len(reference.contributors),
            "series": len(indexed),
            "frames": len(reference.frame_uids),
        },
        rows,
    )


def derive_extents(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("positive_mm") - pl.col("negative_mm")).alias("span_mm"),
        ((pl.col("positive_mm") + pl.col("negative_mm")) / 2).alias(
            "midpoint_offset_mm"
        ),
    )


def union_extents(series: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    usable = series.filter(
        pl.col("eligible")
        & pl.col("negative_mm").is_not_null()
        & pl.col("positive_mm").is_not_null()
    )
    return derive_extents(
        usable.group_by(keys).agg(
            pl.len().alias("series"),
            pl.col("negative_mm").min(),
            pl.col("positive_mm").max(),
            pl.col("slice_spacing_mm").median(),
        )
    ).sort(keys)


def collapse_extents(union: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return derive_extents(
        union.group_by(keys).agg(
            pl.col("series").sum(),
            pl.col("negative_mm").min(),
            pl.col("positive_mm").max(),
            pl.col("slice_spacing_mm").median(),
        )
    ).sort(keys)


def extent_stats(frame: pl.DataFrame, plane: str, fraction: float) -> dict:
    negative, positive = PLANE_SIDES[plane]
    low, high = frame["negative_mm"], frame["positive_mm"]
    nearest = np.minimum(-low.to_numpy(), high.to_numpy())
    sigma = fraction * float(nearest.mean()) if frame.height else None
    return {
        f"distance_{negative}_mm": distribution((-low).to_list()),
        f"distance_{positive}_mm": distribution(high.to_list()),
        "distance_nearest_side_mm": distribution(nearest.tolist())
        if frame.height
        else {"n": 0},
        "centre_outside_span": float(((low > 0) | (high < 0)).mean())
        if frame.height
        else None,
        "sampling_sigma_mm": sigma,
        f"fraction_{negative}_under_sigma": float((-low < sigma).mean())
        if frame.height
        else None,
        f"fraction_{positive}_under_sigma": float((high < sigma).mean())
        if frame.height
        else None,
        "span_mm": distribution(frame["span_mm"].to_list()),
        "midpoint_offset_mm": distribution(frame["midpoint_offset_mm"].to_list()),
        "slice_spacing_mm": distribution(
            [v for v in frame["slice_spacing_mm"].to_list() if v is not None]
        ),
    }


def paired_fluid_delta(union: pl.DataFrame, plane: str) -> dict:
    columns = ["negative_mm", "positive_mm", "span_mm", "midpoint_offset_mm"]
    frames = [
        union.filter(
            (pl.col("plane") == plane) & (pl.col("fluid_sensitive") == fluid)
        ).select(STUDY, *columns)
        for fluid in (0, 1)
    ]
    paired = frames[0].join(frames[1], on=STUDY, suffix="_fluid")
    if not paired.height:
        return {"studies": 0}
    negative, positive = PLANE_SIDES[plane]
    return {
        "studies": paired.height,
        f"distance_{negative}_mm": distribution(
            (paired["negative_mm"] - paired["negative_mm_fluid"]).to_list()
        ),
        f"distance_{positive}_mm": distribution(
            (paired["positive_mm_fluid"] - paired["positive_mm"]).to_list()
        ),
        "span_mm": distribution(
            (paired["span_mm_fluid"] - paired["span_mm"]).to_list()
        ),
        "midpoint_offset_mm": distribution(
            (
                paired["midpoint_offset_mm_fluid"] - paired["midpoint_offset_mm"]
            ).to_list()
        ),
    }


def summarize(
    studies: pl.DataFrame, series: pl.DataFrame, planes: pl.DataFrame, fraction: float
) -> dict:
    pooled = collapse_extents(planes, [STUDY, "plane"])
    rows = derive_extents(
        series.filter(pl.col("eligible") & pl.col("negative_mm").is_not_null())
    )
    per_plane = {}
    for plane in sorted(PLANE_AXIS):
        negative, positive = PLANE_SIDES[plane]
        plane_union = pooled.filter(pl.col("plane") == plane)
        plane_rows = rows.filter(pl.col("plane") == plane)
        per_plane[plane] = {
            "sides": {"negative": negative, "positive": positive},
            "studies": plane_union.height,
            "series": plane_rows.height,
            "study_union": extent_stats(plane_union, plane, fraction),
            "per_series": extent_stats(plane_rows, plane, fraction),
            "by_fluid": {
                str(fluid): {
                    "studies": union.height,
                    "series": int(union["series"].sum()),
                    "study_union": extent_stats(union, plane, fraction),
                    "per_series": extent_stats(
                        plane_rows.filter(pl.col("fluid_sensitive") == fluid),
                        plane,
                        fraction,
                    ),
                }
                for fluid in (0, 1)
                for union in [
                    planes.filter(
                        (pl.col("plane") == plane)
                        & (pl.col("fluid_sensitive") == fluid)
                    )
                ]
            },
            "fluid_minus_nonfluid": paired_fluid_delta(planes, plane),
        }
    rejected = studies.filter(~pl.col("accepted"))
    return {
        "studies": studies.height,
        "accepted_studies": int(studies["accepted"].sum()),
        "rejection_reasons": dict(
            rejected.group_by("reason").len().sort("len", descending=True).iter_rows()
        ),
        "center_mm": {
            axis: distribution(
                studies.filter(pl.col("accepted"))[f"center_{axis}_mm"].to_list()
            )
            for axis in ("x", "y", "z")
        },
        "rms_mm": distribution(
            [v for v in studies["rms_mm"].to_list() if v is not None]
        ),
        "series": series.height,
        "eligible_series": int(series["eligible"].sum()),
        "series_geometry_reasons": dict(
            series.filter(pl.col("geometry_reason").is_not_null())
            .group_by("geometry_reason")
            .len()
            .sort("len", descending=True)
            .iter_rows()
        ),
        "sampling_sigma_fraction": fraction,
        "fluid_sensitive": "0 is the non fluid-sensitive, non fat-suppressed type; 1 is fluid-sensitive with fat suppression",
        "planes": per_plane,
        "limits": "Header geometry only; distances span slice centres, so add half the slice spacing for the outer faces. Sagittal right/left is patient laterality, not medial/lateral.",
    }


def analyze(settings: ExtentsCLI) -> None:
    output = settings.output_dir
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(settings.config)
    index = build_series_index(config.paths.data_root, config.data, settings.split)
    types = tuple(
        (kind.plane, kind.fluid_sensitive, kind.fat_suppression)
        for kind in config.data.series_types
    )
    ordered = [index[uid] for uid in sorted(index)]
    if settings.studies is not None:
        ordered = ordered[: settings.studies]
    source_files = [Path(__file__), *sorted(Path("src/knee_abnormality").rglob("*.py"))]
    inputs = [
        settings.config,
        config.paths.data_root / "raw" / f"{settings.split}_series.csv",
        config.paths.data_root
        / "metadata"
        / f"{settings.split}_series_image_counts.parquet",
        Path("uv.lock"),
    ]
    copy_sources(source_files, output / "source")
    revision, status = git_revision(Path.cwd())
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "git_revision": revision,
        "git_status": status,
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in inputs
            if path.exists()
        ],
        "versions": {
            name: version(name) for name in ("numpy", "polars", "pydantic-settings")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    worker = partial(measure_study, types=types, io_threads=settings.io_threads)
    study_rows, series_rows = [], []
    with get_context("spawn").Pool(settings.workers) as pool:
        for number, (study_row, rows) in enumerate(
            pool.imap_unordered(worker, ordered, chunksize=4), 1
        ):
            study_rows.append(study_row)
            series_rows.extend(rows)
            if number % 200 == 0 or number == len(ordered):
                print(
                    json.dumps({"studies": number, "total": len(ordered)}), flush=True
                )
    studies = pl.DataFrame(study_rows, infer_schema_length=None).sort(STUDY)
    series = pl.DataFrame(series_rows, infer_schema_length=None).sort(STUDY, SERIES)
    planes = union_extents(series, [STUDY, "plane", "fluid_sensitive"])
    studies.write_parquet(output / "studies.parquet")
    series.write_parquet(output / "series.parquet")
    planes.write_parquet(output / "planes.parquet")
    summary = summarize(studies, series, planes, config.data.sampling_sigma_fraction)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for plane, value in summary["planes"].items():
        negative, positive = value["sides"]["negative"], value["sides"]["positive"]
        for name, block in [("all", value), *value["by_fluid"].items()]:
            union = block["study_union"]
            print(
                json.dumps(
                    {
                        "plane": plane,
                        "fluid": name,
                        "studies": block["studies"],
                        negative: round(union[f"distance_{negative}_mm"]["mean"], 1),
                        positive: round(union[f"distance_{positive}_mm"]["mean"], 1),
                        "span": round(union["span_mm"]["mean"], 1),
                        "offset": round(union["midpoint_offset_mm"]["mean"], 1),
                    }
                ),
                flush=True,
            )


def main() -> None:
    try:
        analyze(ExtentsCLI())
    except (OSError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Study centre extent analysis error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
