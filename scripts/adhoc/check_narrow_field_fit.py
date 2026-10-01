from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.preprocessing.dicom import read_frame
from knee_abnormality.data.preprocessing.series import index_series

FIELDS = (120, 130, 140, 160)
COLORS = {120: "red", 130: "yellow", 140: "cyan", 160: "dodgerblue"}
SIDES = {
    "Sagittal": ("anterior", "posterior"),
    "Coronal": ("right", "left"),
    "Axial": ("right", "left", "anterior", "posterior"),
}
DEPTHS = (5, 10, 15, 20, 25)
BELOW = (30, 40)


class NarrowFieldFitCLI(CLISettings):
    model_config = describe(
        "Measure what an image-centred narrow field cuts on the middle slice of the "
        "input-resolution series sample: tissue depth clipped from the skin on each side, "
        "the tibia left below the joint line, and contact sheets with the candidate fields drawn"
    )

    metrics: Path = Field(
        default=Path(
            "outputs/analysis/input-resolution-research-20260915/images/series_metrics_overhang.parquet"
        ),
        description="Per-series tissue edges and joint-line offsets of the sampled series",
    )
    series_root: Path = Field(
        default=Path("data/raw/train_series"), description="Training DICOM root"
    )
    output_dir: Path = Field(description="Fresh output directory")
    sheet_series: int = Field(
        default=24, ge=1, le=48, description="Series drawn per plane"
    )
    seed: int = Field(default=20261001, description="Contact-sheet sampling seed")


def weighted_share(frame: pl.DataFrame, mask: pl.Series) -> float:
    weight = frame["weight"]
    return float((weight * mask.cast(pl.Float64)).sum() / weight.sum())


def clip_table(frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for plane, sides in SIDES.items():
        series = frame.filter(pl.col("Anatomical_Plane") == plane)
        for side in sides:
            edge = series[f"{side}_edge"].abs()
            for field in FIELDS:
                clip = (edge - field / 2).clip(lower_bound=0)
                rows.append(
                    {
                        "plane": plane,
                        "side": side,
                        "field_mm": field,
                        "series": series.height,
                        "clip_median_mm": float(clip.median()),
                        "clip_p90_mm": float(clip.quantile(0.9)),
                        **{
                            f"share_over_{depth}mm": float((clip > depth).mean())
                            for depth in DEPTHS
                        },
                        **{
                            f"weighted_share_over_{depth}mm": weighted_share(
                                series, clip > depth
                            )
                            for depth in DEPTHS
                        },
                    }
                )
    return pl.DataFrame(rows)


def joint_table(frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for plane in ("Sagittal", "Coronal"):
        series = frame.filter(
            (pl.col("Anatomical_Plane") == plane)
            & pl.col("joint_band_offset_mm").is_not_null()
        )
        offset = series["joint_band_offset_mm"]
        for field in FIELDS:
            below = field / 2 + offset
            rows.append(
                {
                    "plane": plane,
                    "field_mm": field,
                    "series": series.height,
                    "below_joint_median_mm": float(below.median()),
                    "above_joint_median_mm": float((field / 2 - offset).median()),
                    **{
                        f"share_below_under_{limit}mm": float((below < limit).mean())
                        for limit in BELOW
                    },
                    **{
                        f"weighted_share_below_under_{limit}mm": weighted_share(
                            series, below < limit
                        )
                        for limit in BELOW
                    },
                }
            )
    return pl.DataFrame(rows)


def draw_sheet(rows: pl.DataFrame, plane: str, cli: NarrowFieldFitCLI) -> None:
    columns = 6
    lines = int(np.ceil(rows.height / columns))
    figure, axes = plt.subplots(lines, columns, figsize=(4 * columns, 4 * lines))
    for axis in np.asarray(axes).ravel():
        axis.axis("off")
    for axis, row in zip(np.asarray(axes).ravel(), rows.iter_rows(named=True)):
        records = index_series(
            cli.series_root / row["StudyInstanceUID"] / row["SeriesInstanceUID"]
        ).records
        record = records[round(0.5 * (len(records) - 1))]
        frame = read_frame(record.path, record.descriptor)
        image = frame.pixels * frame.slope + frame.intercept
        height, width = image.shape
        axis.imshow(image, cmap="gray", vmax=np.percentile(image, 99.5))
        for field in FIELDS:
            half_rows = field / 2 / frame.row_spacing
            half_columns = field / 2 / frame.column_spacing
            axis.add_patch(
                plt.Rectangle(
                    (width / 2 - half_columns, height / 2 - half_rows),
                    2 * half_columns,
                    2 * half_rows,
                    fill=False,
                    edgecolor=COLORS[field],
                    linewidth=1.0,
                )
            )
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_title(
            f"{row['index']} {row['vendor_group']} fs{row['Fluid_Sensitive']} "
            f"FOV {row['fov_x_mm']:.0f}x{row['fov_y_mm']:.0f}",
            fontsize=9,
        )
    figure.suptitle(
        f"{plane} middle slices: centred fields "
        + ", ".join(f"{field} mm {COLORS[field]}" for field in FIELDS)
    )
    figure.tight_layout()
    figure.savefig(cli.output_dir / f"sheet_{plane.lower()}.png", dpi=80)
    plt.close(figure)


def main() -> None:
    cli = NarrowFieldFitCLI()
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    frame = pl.read_parquet(cli.metrics).filter(pl.col("mask_fraction") <= 0.9)
    clip_table(frame).write_csv(cli.output_dir / "clip_depth.csv")
    joint_table(frame).write_csv(cli.output_dir / "joint_margin.csv")
    sheets = []
    for plane in SIDES:
        rows = (
            frame.filter(pl.col("Anatomical_Plane") == plane)
            .sort("SeriesInstanceUID")
            .sample(cli.sheet_series, seed=cli.seed, shuffle=True)
            .with_row_index()
        )
        draw_sheet(rows, plane, cli)
        sheets.append(
            rows.select(
                "Anatomical_Plane", "index", "StudyInstanceUID", "SeriesInstanceUID"
            )
        )
    pl.concat(sheets).write_csv(cli.output_dir / "sheet_series.csv")


if __name__ == "__main__":
    main()
