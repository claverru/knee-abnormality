import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.preprocessing.geometry import within_acquired_range
from knee_abnormality.data.preprocessing.sampling import AXES, build_pool
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.data.splits import STUDY

DATASET = None
QUANTILES = (0.5, 0.9, 0.99, 1.0)


class Settings(CLISettings):
    model_config = describe(
        "Census of the acquired-slab guard on the fitted knee centre: per cleaned "
        "training series, whether the centre projects inside the slab, the miss "
        "distance along the slice normal and the sampling outcome of build_pool"
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


def census(uid: str) -> list[dict]:
    study = DATASET._index_study_geometry(DATASET.series_index[uid])
    reference = DATASET._study_references[uid]
    side = DATASET._study_sides[uid]
    residuals = dict(reference.residuals_mm)
    data = DATASET.data
    rows = []
    for slot in range(len(study.slots)):
        kind = data.series_types[slot]
        _, sources = DATASET._slot_sources(study, slot, [])
        if not sources:
            continue
        pool = build_pool(
            tuple(sources),
            kind.plane,
            reference,
            data.reference_spacing_mm,
            data.reference_field_mm / 2,
        )
        for series_uid, indexed in sources:
            geometry = indexed.geometry
            reliable = geometry.reason is None and geometry.native_poses_valid
            measurable = (
                reference.point is not None
                and geometry.reason is None
                and geometry.bounds is not None
                and geometry.normal is not None
            )
            inside = miss = thickness = agrees = tilt = None
            if reliable:
                normal = np.asarray(geometry.normal, dtype=float)
                agrees = int(np.argmax(np.abs(normal))) == AXES[kind.plane]
            if measurable:
                normal = np.asarray(geometry.normal, dtype=float)
                normal /= np.linalg.norm(normal)
                lower, upper = sorted(geometry.bounds)
                projection = float(np.dot(reference.point, normal))
                inside = within_acquired_range(geometry, reference.point)
                miss = (
                    projection - upper
                    if projection > upper
                    else projection - lower
                    if projection < lower
                    else 0.0
                )
                thickness = upper - lower
                tilt = float(
                    np.degrees(np.arccos(min(1.0, abs(normal[AXES[kind.plane]]))))
                )
            rows.append(
                {
                    STUDY: uid,
                    "series": series_uid,
                    "slot": slot,
                    "plane": kind.plane,
                    "fluid_sensitive": kind.fluid_sensitive,
                    "fat_suppression": kind.fat_suppression,
                    "slot_sources": len(sources),
                    "manufacturer": "|".join(side.manufacturers),
                    "centre_fitted": reference.point is not None,
                    "centre_method": reference.method
                    if reference.point is not None
                    else "none",
                    "centre_reason": reference.reason,
                    "centre_rms_mm": reference.rms_mm,
                    "residual_mm": residuals.get(series_uid),
                    "contributor": series_uid in reference.contributors,
                    "frame_uids": len(reference.frame_uids),
                    "geometry_reason": geometry.reason,
                    "reliable": reliable,
                    "plane_agrees": agrees,
                    "tilt_deg": tilt,
                    "slices": len(indexed.records),
                    "inside": inside,
                    "miss_mm": miss,
                    "slab_mm": thickness,
                    "guard_evaluated": reference.point is not None
                    and reference.method != "fit",
                    "outcome": pool.diagnostics["sources"].get(series_uid),
                    "pool_path": pool.diagnostics["path"],
                    "in_pool": series_uid in pool.ranking,
                }
            )
    return rows


def quantiles(values: pl.Series) -> dict:
    data = values.drop_nulls().to_numpy()
    if not len(data):
        return {"n": 0}
    return {"n": len(data)} | {str(q): float(np.quantile(data, q)) for q in QUANTILES}


def counts(frame: pl.DataFrame, *keys: str) -> list[dict]:
    return frame.group_by(*keys).len().sort(*keys).to_dicts()


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
    frame.write_parquet(settings.output_dir / "series.parquet")
    fitted = frame.filter(pl.col("centre_fitted"))
    measured = fitted.filter(pl.col("inside").is_not_null())
    outside = measured.filter(~pl.col("inside"))
    effective = outside.filter(pl.col("guard_evaluated"))
    per_plane = (
        measured.group_by(STUDY, "plane", "manufacturer", "centre_method")
        .agg(
            pl.len().alias("series"),
            (~pl.col("inside")).sum().alias("outside"),
        )
        .with_columns((pl.col("outside") == pl.col("series")).alias("all_outside"))
    )
    by_plane = (
        measured.group_by("plane")
        .agg(
            pl.len().alias("series"),
            (~pl.col("inside")).sum().alias("outside_series"),
            pl.col(STUDY).n_unique().alias("studies"),
            pl.col(STUDY).filter(~pl.col("inside")).n_unique().alias("outside_studies"),
            (~pl.col("inside") & pl.col("guard_evaluated"))
            .sum()
            .alias("guard_fired_series"),
            pl.col(STUDY)
            .filter(~pl.col("inside") & pl.col("guard_evaluated"))
            .n_unique()
            .alias("guard_fired_studies"),
            pl.col("slab_mm").median().alias("median_slab_mm"),
        )
        .sort("plane")
    )
    manufacturer = (
        measured.group_by("manufacturer")
        .agg(
            pl.len().alias("series"),
            (~pl.col("inside")).sum().alias("outside_series"),
            pl.col(STUDY).n_unique().alias("studies"),
            pl.col(STUDY).filter(~pl.col("inside")).n_unique().alias("outside_studies"),
            pl.col("miss_mm")
            .filter(~pl.col("inside"))
            .abs()
            .median()
            .alias("median_abs_miss_mm"),
        )
        .with_columns(
            (pl.col("outside_series") / pl.col("series")).alias("outside_rate")
        )
        .sort("series", descending=True)
    )
    summary = {
        "studies": len(uids),
        "studies_with_rows": frame[STUDY].n_unique(),
        "series_rows": frame.height,
        "fitted_studies": fitted[STUDY].n_unique(),
        "fitted_series": fitted.height,
        "measured_series": measured.height,
        "centre_method_studies": frame.unique(STUDY)
        .group_by("centre_method")
        .len()
        .sort("centre_method")
        .to_dicts(),
        "unmeasured_fitted_by_geometry_reason": counts(
            fitted.filter(pl.col("inside").is_null()), "geometry_reason"
        ),
        "outside_series": outside.height,
        "outside_studies": outside[STUDY].n_unique(),
        "guard_fired_series": effective.height,
        "guard_fired_studies": effective[STUDY].n_unique(),
        "by_plane": by_plane.to_dicts(),
        "outside_by_method": counts(outside, "centre_method", "plane"),
        "measured_by_method": counts(measured, "centre_method"),
        "abs_miss_mm_outside": quantiles(outside["miss_mm"].abs()),
        "abs_miss_mm_outside_by_plane": {
            plane: quantiles(outside.filter(pl.col("plane") == plane)["miss_mm"].abs())
            for plane in sorted(AXES)
        },
        "abs_miss_mm_guard_fired": quantiles(effective["miss_mm"].abs()),
        "abs_miss_mm_outside_fit_method": quantiles(
            outside.filter(pl.col("centre_method") == "fit")["miss_mm"].abs()
        ),
        "miss_sign_by_plane": counts(
            outside.with_columns((pl.col("miss_mm") > 0).alias("above")),
            "plane",
            "above",
        ),
        "outside_plane_agrees": counts(outside, "plane_agrees"),
        "outside_contributor": counts(outside, "contributor"),
        "outside_residual_mm": quantiles(outside["residual_mm"]),
        "inside_residual_mm": quantiles(
            measured.filter(pl.col("inside"))["residual_mm"]
        ),
        "outside_slab_mm": quantiles(outside["slab_mm"]),
        "inside_slab_mm": quantiles(measured.filter(pl.col("inside"))["slab_mm"]),
        "studies_all_series_of_plane_outside": per_plane.filter(pl.col("all_outside"))
        .group_by("plane")
        .agg(
            pl.len().alias("studies"),
            (pl.col("centre_method") != "fit").sum().alias("guard_evaluated_studies"),
        )
        .sort("plane")
        .to_dicts(),
        "studies_any_plane_all_outside": per_plane.filter(pl.col("all_outside"))[
            STUDY
        ].n_unique(),
        "manufacturer": manufacturer.to_dicts(),
        "outcome_by_plane": counts(frame, "plane", "outcome"),
        "outcome_by_plane_fitted": counts(fitted, "plane", "outcome"),
        "outcome_of_outside": counts(outside, "centre_method", "outcome", "in_pool"),
        "pool_path_by_plane": counts(
            frame.unique([STUDY, "slot"]), "plane", "pool_path"
        ),
        "pool_path_by_plane_fitted": counts(
            fitted.unique([STUDY, "slot"]), "plane", "pool_path"
        ),
        "local_series_dropped_from_physical_pool": frame.filter(
            (pl.col("pool_path") == "physical") & ~pl.col("in_pool")
        )
        .group_by("plane", "outcome")
        .len()
        .sort("plane", "outcome")
        .to_dicts(),
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
