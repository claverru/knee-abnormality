import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from scipy import ndimage

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.preprocessing.dicom import PREPARATION_ERRORS, read_frame
from knee_abnormality.data.preprocessing.geometry import image_center
from knee_abnormality.data.preprocessing.spatial import detail_position, record_layout
from knee_abnormality.data.sources import SERIES, build_series_index
from knee_abnormality.data.splits import STUDY

SIZE = 224
FREE_SPACING_MM = 4.0
RULE_WIDTHS = (70.0, 84.0, 280.0 / 3, 100.0, 112.0, 120.0)
AXIAL_WIDTHS = (100.0, 112.0, 120.0, 130.0, 140.0, 150.0, 160.0)
SLICES = (("q25", 0.25), ("q50", 0.5), ("q75", 0.75))
DIRECTIONS = {
    (0, 1): "patient_left",
    (0, -1): "patient_right",
    (1, 1): "posterior",
    (1, -1): "anterior",
    (2, 1): "superior",
    (2, -1): "inferior",
}
STATE = {}


class Settings(CLISettings):
    model_config = describe(
        "Measure, on a seeded sample of training series per series type, the tissue a "
        "square view window leaves outside on each side, the height left above and "
        "below the joint line, and the offset of the fitted knee centre from the image "
        "centre, for image-centred and rule-placed windows of several widths. Run with "
        "PYTHONPATH set to the archived source of the run whose rule placement is "
        "measured."
    )

    config: Path = Field(description="effective_config.json of the measured run")
    output_dir: Path = Field(description="Output directory")
    series_per_type: int = Field(
        default=1000, ge=1, description="Series sampled per series type"
    )
    seed: int = Field(default=20261002, description="Sampling seed")
    workers: int = Field(default=8, ge=1, le=8, description="Worker processes")
    studies: int | None = Field(
        default=None, ge=1, description="Stop after this many studies"
    )


def tissue_mask(image, spacing, fraction=0.08):
    smooth = ndimage.gaussian_filter(image, 1.5 / spacing)
    k0, k1 = max(2, smooth.shape[0] // 50), max(2, smooth.shape[1] // 50)
    background = min(
        float(np.median(x))
        for x in (smooth[:k0], smooth[-k0:], smooth[:, :k1], smooth[:, -k1:])
    )
    high = float(np.percentile(smooth, 99))
    mask = smooth > background + fraction * (high - background)
    mask = ndimage.binary_opening(mask, iterations=max(1, round(1.0 / spacing)))
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == int(np.argmax(sizes)) + 1
    rows = (
        np.maximum.accumulate(mask, 1)
        & np.maximum.accumulate(mask[:, ::-1], 1)[:, ::-1]
    )
    columns = (
        np.maximum.accumulate(mask, 0) & np.maximum.accumulate(mask[::-1], 0)[::-1]
    )
    return rows & columns


def joint_row(image, mask, spacing, fluid):
    smooth = ndimage.gaussian_filter(image.astype(np.float64), 1.0 / spacing)
    rows_any = np.where(mask.any(1))[0]
    cols_any = np.where(mask.any(0))[0]
    if len(rows_any) < 10 or len(cols_any) < 10:
        return None
    top, bottom = rows_any[0], rows_any[-1]
    left, right = cols_any[0], cols_any[-1]
    width = right - left
    lo, hi = left + int(0.25 * width), left + int(0.75 * width)
    scale = np.percentile(smooth[mask], 95)
    region = np.where(mask[:, lo:hi], smooth[:, lo:hi] / scale, np.nan)
    with np.errstate(all="ignore"):
        profile = np.nanmedian(region, axis=1)
    valid = np.isfinite(profile)
    if valid.sum() < 10:
        return None
    profile = np.interp(np.arange(len(profile)), np.where(valid)[0], profile[valid])
    half, flank = round(2.5 / spacing), round(10.0 / spacing)
    csum = np.concatenate(([0], np.cumsum(profile)))
    n = len(profile)
    idx = np.arange(n)

    def mean(a, b):
        a, b = np.clip(a, 0, n), np.clip(b, 0, n)
        return (csum[b] - csum[a]) / np.maximum(b - a, 1)

    center = mean(idx - half, idx + half + 1)
    sides = 0.5 * (
        mean(idx - half - flank, idx - half)
        + mean(idx + half + 1, idx + half + 1 + flank)
    )
    contrast = (center - sides) * (1 if fluid else -1)
    height = bottom - top
    search = (
        (idx >= top + 0.15 * height)
        & (idx <= bottom - 0.15 * height)
        & (idx - half - flank >= 0)
        & (idx + half + flank < n)
    )
    if search.sum() < 5:
        return None
    z = contrast[search]
    return int(idx[search][np.argmax(z)]), float(
        (z.max() - np.median(z)) / (np.std(z) + 1e-9)
    )


def direction_name(vector, sign):
    axis = int(np.argmax(np.abs(vector)))
    name = DIRECTIONS[(axis, 1 if vector[axis] > 0 else -1)]
    if axis != 0:
        return name
    if sign == 0:
        return name
    medial_positive = sign > 0
    return "medial" if (name == "patient_left") == medial_positive else "lateral"


def clamp_centre(value, width, extent):
    if width >= extent:
        return 0.0
    limit = (extent - width) / 2
    return float(np.clip(value, -limit, limit))


def window_rows(mask, xs, ys, half_x, half_y, centre, width, names):
    cx, cy = centre
    x0, x1 = cx - width / 2, cx + width / 2
    y0, y1 = cy - width / 2, cy + width / 2
    rows = (ys >= y0) & (ys <= y1)
    columns = (xs >= x0) & (xs <= x1)
    out = []
    for low_name, high_name, axis, selected, low, high, half, extent in (
        (names["left"], names["right"], xs, rows, x0, x1, half_x, mask.shape[1]),
        (names["top"], names["bottom"], ys, columns, y0, y1, half_y, mask.shape[0]),
    ):
        band = mask[selected] if axis is xs else mask[:, selected].T
        occupied = np.flatnonzero(band.any(0)) if band.size else np.empty(0, int)
        native_low, native_high = axis[0] - half, axis[-1] + half
        if len(occupied):
            tissue_low = axis[occupied[0]] - half
            tissue_high = axis[occupied[-1]] + half
            cuts = (max(0.0, low - tissue_low), max(0.0, tissue_high - high))
            touches = (occupied[0] == 0, occupied[-1] == extent - 1)
        else:
            cuts, touches = (0.0, 0.0), (False, False)
        pads = (max(0.0, native_low - low), max(0.0, high - native_high))
        for name, cut, touch, pad in zip(
            (low_name, high_name), cuts, touches, pads, strict=True
        ):
            out.append(
                {
                    "side": name,
                    "cut_mm": float(cut),
                    "tissue_reaches_stored_edge": bool(touch),
                    "padding_mm": float(pad),
                }
            )
    return out


def knee_offsets(record, geometry, plane):
    _, center_x, center_y, trusted = geometry
    if not trusted or record.pose is None:
        return None, None
    layout = record.pose.layout
    a = np.asarray(layout.orientation[:3])
    b = np.asarray(layout.orientation[3:])
    centre = image_center(record.pose)
    if plane == "Axial":
        matrix = np.array([[a[0], b[0]], [a[1], b[1]]])
        if abs(np.linalg.det(matrix)) < 1e-6:
            return None, None
        u, v = np.linalg.solve(
            matrix, np.array([center_x - centre[0], center_y - centre[1]])
        )
        return float(u), float(v)
    axis = 0 if plane == "Coronal" else 1
    target = center_x if plane == "Coronal" else center_y
    if abs(a[axis]) < 1e-6:
        return None, None
    return float((target - centre[axis]) / a[axis]), None


def measure_series(dataset, study_uid, candidate, kind, wide_spacing):
    plane, fluid = kind.plane, kind.fluid_sensitive
    base = {
        STUDY: study_uid,
        SERIES: candidate.uid,
        "plane": plane,
        "fluid_sensitive": fluid,
    }
    indexed = dataset._slice_index.get(candidate.directory)
    if indexed is None:
        return {**base, "status": "unindexed"}, [], []
    available = [r for r in indexed.records if record_layout(r) is not None]
    if not available:
        return {**base, "status": "no_layout"}, [], []
    geometry = dataset._series_geometry[study_uid].get(
        candidate.uid, (0.0, 0.0, 0.0, False)
    )
    geometry = (
        *geometry[:3],
        bool(
            geometry[3]
            and indexed.geometry.native_poses_valid
            and indexed.geometry.reason is None
        ),
    )
    middle = available[len(available) // 2]
    rows, columns, row_spacing, column_spacing = record_layout(middle)
    fov_x, fov_y = columns * column_spacing, rows * row_spacing
    sign = geometry[0] if geometry[3] else 0.0
    if middle.pose is not None:
        a = np.asarray(middle.pose.layout.orientation[:3])
        b = np.asarray(middle.pose.layout.orientation[3:])
        names = {
            "left": direction_name(-a, sign),
            "right": direction_name(a, sign),
            "top": direction_name(-b, sign),
            "bottom": direction_name(b, sign),
        }
    else:
        a = b = None
        names = {key: f"image_{key}" for key in ("left", "right", "top", "bottom")}
    knee_x, knee_y = knee_offsets(middle, geometry, plane)
    centres = {}
    if plane == "Axial":
        for width in AXIAL_WIDTHS:
            centres[("centred", width)] = (0.0, 0.0)
            if knee_x is not None:
                centres[("knee", width)] = (
                    clamp_centre(knee_x, width, fov_x),
                    clamp_centre(knee_y, width, fov_y),
                )
    else:
        for width in RULE_WIDTHS:
            centres[("centred", width)] = (0.0, 0.0)
            for name, wide in (
                ("rule", FREE_SPACING_MM),
                ("rule_in_120", wide_spacing),
            ):
                centre = detail_position(
                    indexed.records, plane, geometry, wide, width / SIZE, SIZE
                )[2]
                if centre is not None:
                    centres[(name, width)] = centre
    series = {
        **base,
        "status": "ok",
        "records": len(available),
        "rows": rows,
        "columns": columns,
        "row_spacing": row_spacing,
        "column_spacing": column_spacing,
        "fov_x_mm": fov_x,
        "fov_y_mm": fov_y,
        "trusted": geometry[3],
        "knee_side_sign": sign,
        "knee_dx_mm": knee_x,
        "knee_dy_mm": knee_y,
        "side_left": names["left"],
        "side_right": names["right"],
        "side_top": names["top"],
        "side_bottom": names["bottom"],
        "joint_y_mm": None,
        "joint_patient_z_mm": None,
        "joint_score": None,
    }
    cuts, margins = [], []
    for tag, quantile in SLICES:
        record = (
            middle
            if tag == "q50"
            else available[round(quantile * (len(available) - 1))]
        )
        try:
            frame = read_frame(record.path, record.descriptor)
        except PREPARATION_ERRORS as error:
            series[f"error_{tag}"] = f"{type(error).__name__}: {error}"
            continue
        image = frame.pixels * frame.slope + frame.intercept
        if image.shape != (rows, columns):
            series[f"error_{tag}"] = "shape_differs_from_layout"
            continue
        mask = tissue_mask(image, row_spacing)
        fraction = float(mask.mean())
        xs = (np.arange(columns) - (columns - 1) / 2) * column_spacing
        ys = (np.arange(rows) - (rows - 1) / 2) * row_spacing
        occupied_x = np.flatnonzero(mask.any(0))
        occupied_y = np.flatnonzero(mask.any(1))
        if len(occupied_x):
            series[f"tissue_x0_{tag}"] = float(xs[occupied_x[0]] - column_spacing / 2)
            series[f"tissue_x1_{tag}"] = float(xs[occupied_x[-1]] + column_spacing / 2)
            series[f"tissue_y0_{tag}"] = float(ys[occupied_y[0]] - row_spacing / 2)
            series[f"tissue_y1_{tag}"] = float(ys[occupied_y[-1]] + row_spacing / 2)
        series[f"mask_fraction_{tag}"] = fraction
        joint = None
        if tag == "q50" and plane != "Axial" and names["bottom"] == "inferior":
            found = joint_row(image, mask, row_spacing, fluid == 1)
            if found is not None:
                joint = float(ys[found[0]])
                series["joint_y_mm"] = joint
                series["joint_score"] = found[1]
                series["joint_patient_z_mm"] = float(
                    image_center(record.pose)[2] + b[2] * joint
                )
        for (placement, width), centre in centres.items():
            for entry in window_rows(
                mask,
                xs,
                ys,
                column_spacing / 2,
                row_spacing / 2,
                centre,
                width,
                names,
            ):
                cuts.append(
                    {
                        SERIES: candidate.uid,
                        "slice": tag,
                        "placement": placement,
                        "width_mm": width,
                        "centre_x_mm": centre[0],
                        "centre_y_mm": centre[1],
                        **entry,
                    }
                )
            if joint is not None:
                top = max(centre[1] - width / 2, -fov_y / 2)
                bottom = min(centre[1] + width / 2, fov_y / 2)
                margins.append(
                    {
                        SERIES: candidate.uid,
                        "placement": placement,
                        "width_mm": width,
                        "above_joint_mm": joint - top,
                        "below_joint_mm": bottom - joint,
                    }
                )
    return series, cuts, margins


def initialise(config_path, sample):
    config = load_config(config_path)
    index = build_series_index(config.paths.data_root, config.data, "train")
    studies = pl.DataFrame({STUDY: sorted(index)})
    STATE["config"] = config
    STATE["index"] = index
    STATE["sample"] = sample
    STATE["dataset"] = KneeStudyDataset(studies, index, config.data, (), None)


def measure_study(study_uid):
    dataset, config = STATE["dataset"], STATE["config"]
    study = dataset._index_study_geometry(STATE["index"][study_uid])
    excluded = dataset._excluded_series[study_uid]
    series_rows, cut_rows, margin_rows = [], [], []
    for slot, values in enumerate(study.slots):
        kind = config.data.series_types[slot]
        for candidate in values:
            if candidate.uid not in STATE["sample"]:
                continue
            if candidate.uid in excluded:
                series_rows.append(
                    {
                        STUDY: study_uid,
                        SERIES: candidate.uid,
                        "plane": kind.plane,
                        "fluid_sensitive": kind.fluid_sensitive,
                        "status": "excluded_by_cleaning",
                    }
                )
                continue
            series, cuts, margins = measure_series(
                dataset, study_uid, candidate, kind, config.data.spacing_mm
            )
            series_rows.append(series)
            cut_rows.extend(cuts)
            margin_rows.extend(margins)
    dataset._slice_index.clear()
    return series_rows, cut_rows, margin_rows


def run(settings: Settings) -> dict:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(settings.config)
    raw = pl.read_csv(config.paths.data_root / "raw" / "train_series.csv").sort(SERIES)
    drawn = pl.concat(
        [
            part.sample(
                min(settings.series_per_type, part.height),
                seed=settings.seed,
                shuffle=True,
            )
            for part in raw.partition_by(
                "Anatomical_Plane", "Fluid_Sensitive", maintain_order=True
            )
        ]
    )
    sample = frozenset(drawn[SERIES].to_list())
    studies = sorted(set(drawn[STUDY].to_list()))
    if settings.studies is not None:
        studies = studies[: settings.studies]
    series_rows, cut_rows, margin_rows = [], [], []
    with get_context("spawn").Pool(
        settings.workers, initialise, (settings.config, sample)
    ) as pool:
        for number, (series, cuts, margins) in enumerate(
            pool.imap_unordered(measure_study, studies, chunksize=4), 1
        ):
            series_rows.extend(series)
            cut_rows.extend(cuts)
            margin_rows.extend(margins)
            if number % 200 == 0 or number == len(studies):
                print(
                    json.dumps({"studies": number, "total": len(studies)}),
                    file=sys.stderr,
                    flush=True,
                )
    series = pl.DataFrame(series_rows, infer_schema_length=None).sort(SERIES)
    series.write_parquet(settings.output_dir / "window_series.parquet")
    pl.DataFrame(cut_rows, infer_schema_length=None).write_parquet(
        settings.output_dir / "window_cuts.parquet"
    )
    pl.DataFrame(margin_rows, infer_schema_length=None).write_parquet(
        settings.output_dir / "window_joint_margins.parquet"
    )
    summary = {
        "config": str(settings.config),
        "seed": settings.seed,
        "series_per_type": settings.series_per_type,
        "drawn_series": len(sample),
        "studies": len(studies),
        "status": dict(series.group_by("status").len().iter_rows()),
    }
    (settings.output_dir / "window_measurement.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    print(json.dumps(run(Settings()), indent=2))


if __name__ == "__main__":
    main()
