from pathlib import Path

import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe

SIZE = 224
SPACINGS = (0.3125, 0.375, 5 / 12, 100 / 224, 0.5, 120 / 224, 130 / 224, 0.625)
WIDTHS = (70.0, 84.0, 280 / 3, 100.0, 112.0, 120.0, 130.0, 140.0, 150.0, 160.0)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
TOLERANCE = 5e-4
KEYS = ["plane", "fluid_sensitive"]
AXES = {
    "finer_axis": "acq_res_best",
    "coarser_axis": "acq_res_worst",
    "frequency_axis": "acq_res_freq",
    "phase_axis": "acq_res_phase",
    "image_horizontal": "acq_res_h",
    "image_vertical": "acq_res_v",
    "stored_pixel_finer": "ps_min",
}


class Settings(CLISettings):
    model_config = describe(
        "Tabulate, per series type (plane by fluid sensitivity) over every training "
        "series of the acquisition-header audit, the acquired in-plane spacing "
        "distribution, the share of series fully resolved at each candidate spacing, "
        "the stored field of view and how much of each candidate window is padding"
    )

    acquisition: Path = Field(
        default=Path(
            "outputs/analysis/i/input-resolution-research-20260915/acquisition/"
            "series_acquisition.parquet"
        ),
        description="Per-series table written by the acquisition-header audit",
    )
    output_dir: Path = Field(description="Output directory")


def percentiles(frame: pl.DataFrame, column: str) -> dict:
    values = frame[column].drop_nulls()
    return {
        "n": len(values),
        **{
            f"p{round(q * 100)}": float(values.quantile(q, "linear")) for q in QUANTILES
        },
    }


def run(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train = (
        pl.read_parquet(settings.acquisition)
        .filter(pl.col("split") == "train")
        .rename({"Fluid_Sensitive": "fluid_sensitive"})
    )
    studies = train["StudyInstanceUID"].n_unique()
    counts, spacing_rows, resolved_rows, field_rows, padding_rows = [], [], [], [], []
    for key, part in sorted(train.partition_by(KEYS, as_dict=True).items()):
        labels = dict(zip(KEYS, key, strict=True))
        usable = part.filter(pl.col("acq_res_best").is_not_null())
        ok = part.filter(pl.col("acq_estimate_quality") == "ok")
        quality = dict(part.group_by("acq_estimate_quality").len().iter_rows())
        counts.append(
            {
                **labels,
                "series": part.height,
                "studies_with_type": part["StudyInstanceUID"].n_unique(),
                "share_of_studies": part["StudyInstanceUID"].n_unique() / studies,
                "acquisition_matrix_usable": usable.height,
                "acquisition_matrix_missing": part.height - usable.height,
                "quality_ok": quality.get("ok", 0),
                "quality_3d": quality.get("3d", 0),
                "quality_philips_rectangular_field": quality.get("philips_rect_fov", 0),
            }
        )
        for subset, frame in (("usable_matrix", usable), ("quality_ok", ok)):
            for axis, column in AXES.items():
                spacing_rows.append(
                    {
                        **labels,
                        "subset": subset,
                        "axis": axis,
                        **percentiles(frame, column),
                    }
                )
        missing = part.height - usable.height
        for spacing in SPACINGS:
            kept = (usable["acq_res_best"] >= spacing - TOLERANCE).sum()
            both = (usable["acq_res_worst"] >= spacing - TOLERANCE).sum()
            kept_ok = (ok["acq_res_best"] >= spacing - TOLERANCE).sum()
            resolved_rows.append(
                {
                    **labels,
                    "spacing_mm": spacing,
                    "width_mm": spacing * SIZE,
                    "series_with_matrix": usable.height,
                    "share_finer_axis_kept": kept / usable.height,
                    "share_if_missing_all_lost": kept / part.height,
                    "share_if_missing_all_kept": (kept + missing) / part.height,
                    "share_coarser_axis_kept": both / usable.height,
                    "series_quality_ok": ok.height,
                    "share_finer_axis_kept_quality_ok": kept_ok / ok.height,
                }
            )
        for axis, column in (("width", "fov_h"), ("height", "fov_v")):
            field_rows.append({**labels, "axis": axis, **percentiles(part, column)})
        sized = part.filter(
            pl.col("fov_h").is_not_null() & pl.col("fov_v").is_not_null()
        )
        for width in WIDTHS:
            inside = pl.min_horizontal(pl.col("fov_h"), pl.lit(width)) * (
                pl.min_horizontal(pl.col("fov_v"), pl.lit(width))
            )
            padded = sized.select(
                (pl.col("fov_h") < width - 0.5).mean().alias("w"),
                (pl.col("fov_v") < width - 0.5).mean().alias("h"),
                (pl.min_horizontal("fov_h", "fov_v") < width - 0.5).mean().alias("a"),
                (1 - inside / width**2).mean().alias("area"),
                ((1 - inside / width**2) > 0.1).mean().alias("tenth"),
            ).row(0)
            padding_rows.append(
                {
                    **labels,
                    "width_mm": width,
                    "series": sized.height,
                    "share_stored_width_below_window": padded[0],
                    "share_stored_height_below_window": padded[1],
                    "share_either_axis_below_window": padded[2],
                    "mean_padded_area_share": padded[3],
                    "share_over_10pct_padding": padded[4],
                }
            )
    for name, rows in (
        ("series_counts", counts),
        ("acquired_spacing_percentiles", spacing_rows),
        ("resolved_share", resolved_rows),
        ("native_field", field_rows),
        ("window_padding", padding_rows),
    ):
        frame = pl.DataFrame(rows)
        frame.write_csv(settings.output_dir / f"{name}.csv", float_precision=4)
        with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=250):
            print(name)
            print(frame)


def main() -> None:
    run(Settings())


if __name__ == "__main__":
    main()
