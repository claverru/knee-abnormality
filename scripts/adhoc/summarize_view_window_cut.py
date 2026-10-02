import json
from pathlib import Path

import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe

STUDY = "StudyInstanceUID"
SERIES = "SeriesInstanceUID"
KEYS = ["plane", "fluid_sensitive"]
SIDES = {
    "Axial": ("anterior", "posterior", "medial", "lateral"),
    "Coronal": ("medial", "lateral"),
    "Sagittal": ("anterior", "posterior"),
}
SLICES = ("q25", "q50", "q75")
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class Settings(CLISettings):
    model_config = describe(
        "Summarise the per-series window measurements of measure_view_window_cut by "
        "series type: tissue cut per side, height left around the joint line, tissue "
        "extent and the offset of the knee from the image centre"
    )

    directory: Path = Field(description="Directory written by measure_view_window_cut")
    mask_fraction_limit: float = Field(
        default=0.9, description="Largest tissue share of a slice with a usable mask"
    )
    joint_contributors: int = Field(
        default=3,
        ge=1,
        description="Coronal and sagittal series a study needs for a joint-line consensus",
    )


def quantile_columns(column: str, prefix: str) -> list[pl.Expr]:
    return [
        pl.col(column).quantile(q, "linear").alias(f"{prefix}p{round(q * 100)}")
        for q in QUANTILES
    ]


def run(settings: Settings) -> dict:
    out = settings.directory
    measured = pl.read_parquet(out / "window_series.parquet")
    series = measured.filter(pl.col("status") == "ok")
    masks = series.unpivot(
        [f"mask_fraction_{tag}" for tag in SLICES],
        index=[SERIES, *KEYS],
        variable_name="slice",
        value_name="mask_fraction",
    ).with_columns(pl.col("slice").str.replace("mask_fraction_", ""))
    usable = masks.filter(pl.col("mask_fraction") <= settings.mask_fraction_limit)
    allowed = pl.DataFrame(
        [
            {"plane": plane, "side": side}
            for plane, sides in SIDES.items()
            for side in sides
        ]
    )
    cuts = (
        pl.read_parquet(out / "window_cuts.parquet")
        .join(usable, on=[SERIES, "slice"])
        .join(allowed, on=["plane", "side"])
    )
    window = [SERIES, *KEYS, "placement", "width_mm"]
    deepest = (
        cuts.group_by(*window, "side")
        .agg(
            pl.col("cut_mm").max(),
            pl.col("tissue_reaches_stored_edge").any(),
            pl.col("padding_mm").max(),
        )
        .with_columns(slice=pl.lit("max3"))
    )
    per_side = pl.concat(
        [
            cuts.select(
                *window,
                "side",
                "slice",
                "cut_mm",
                "tissue_reaches_stored_edge",
                "padding_mm",
            ),
            deepest.select(
                *window,
                "side",
                "slice",
                "cut_mm",
                "tissue_reaches_stored_edge",
                "padding_mm",
            ),
        ]
    )
    expected = pl.DataFrame(
        [{"plane": plane, "sides": len(sides)} for plane, sides in SIDES.items()]
    )
    any_side = (
        per_side.group_by(*window, "slice")
        .agg(
            pl.col("cut_mm").max(),
            pl.col("tissue_reaches_stored_edge").any(),
            pl.col("padding_mm").max(),
            pl.len().alias("found"),
        )
        .join(expected, on="plane")
        .filter(pl.col("found") == pl.col("sides"))
        .with_columns(side=pl.lit("any"))
        .select(per_side.columns)
    )
    tissue_cut = (
        pl.concat([per_side, any_side])
        .group_by(*KEYS, "placement", "width_mm", "slice", "side")
        .agg(
            pl.len().alias("series"),
            (pl.col("cut_mm") > 10).mean().alias("share_cut_over_10mm"),
            (pl.col("cut_mm") > 20).mean().alias("share_cut_over_20mm"),
            pl.col("cut_mm").median().alias("cut_median_mm"),
            pl.col("cut_mm").quantile(0.9, "linear").alias("cut_p90_mm"),
            pl.col("tissue_reaches_stored_edge")
            .mean()
            .alias("share_tissue_reaches_stored_edge"),
            (pl.col("padding_mm") > 0.5).mean().alias("share_window_past_stored_edge"),
        )
        .sort(*KEYS, "placement", "slice", "side", "width_mm")
    )
    tissue_cut.write_csv(out / "tissue_cut.csv", float_precision=4)

    middle = usable.filter(pl.col("slice") == "q50").select(SERIES)
    located = (
        series.join(middle, on=SERIES)
        .filter(pl.col("joint_patient_z_mm").is_not_null() & pl.col("trusted"))
        .with_columns(
            contributors=pl.len().over(STUDY),
            consensus_z=pl.col("joint_patient_z_mm").median().over(STUDY),
        )
        .filter(pl.col("contributors") >= settings.joint_contributors)
        .with_columns(
            own_gap_mm=(pl.col("joint_patient_z_mm") - pl.col("consensus_z")).abs(),
            joint_consensus_y_mm=pl.col("joint_y_mm")
            + pl.col("joint_patient_z_mm")
            - pl.col("consensus_z"),
        )
    )
    agreement = (
        located.group_by(*KEYS)
        .agg(
            pl.len().alias("series"),
            (pl.col("own_gap_mm") <= 5).mean().alias("share_within_5mm_of_study"),
            (pl.col("own_gap_mm") <= 10).mean().alias("share_within_10mm_of_study"),
            pl.col("own_gap_mm").median().alias("median_gap_mm"),
        )
        .sort(*KEYS)
    )
    agreement.write_csv(out / "joint_line_agreement.csv", float_precision=4)
    windows = (
        pl.read_parquet(out / "window_cuts.parquet")
        .filter(pl.col("slice") == "q50")
        .unique([SERIES, "placement", "width_mm"])
        .select(SERIES, "placement", "width_mm", "centre_y_mm")
    )
    margins = windows.join(
        located.select(SERIES, *KEYS, "fov_y_mm", "joint_consensus_y_mm"), on=SERIES
    ).with_columns(
        above_joint_mm=pl.col("joint_consensus_y_mm")
        - pl.max_horizontal(
            pl.col("centre_y_mm") - pl.col("width_mm") / 2, -pl.col("fov_y_mm") / 2
        ),
        below_joint_mm=pl.min_horizontal(
            pl.col("centre_y_mm") + pl.col("width_mm") / 2, pl.col("fov_y_mm") / 2
        )
        - pl.col("joint_consensus_y_mm"),
    )
    joint_margin = (
        margins.group_by(*KEYS, "placement", "width_mm")
        .agg(
            pl.len().alias("series"),
            pl.col("above_joint_mm").median().alias("above_median_mm"),
            pl.col("above_joint_mm").quantile(0.1, "linear").alias("above_p10_mm"),
            pl.col("below_joint_mm").median().alias("below_median_mm"),
            pl.col("below_joint_mm").quantile(0.1, "linear").alias("below_p10_mm"),
            (pl.col("below_joint_mm") < 30).mean().alias("share_below_under_30mm"),
            (pl.col("below_joint_mm") < 40).mean().alias("share_below_under_40mm"),
            (pl.col("above_joint_mm") < 30).mean().alias("share_above_under_30mm"),
            (pl.col("above_joint_mm") < 40).mean().alias("share_above_under_40mm"),
            (pl.col("above_joint_mm") < 50).mean().alias("share_above_under_50mm"),
        )
        .sort(*KEYS, "placement", "width_mm")
    )
    joint_margin.write_csv(out / "joint_margin.csv", float_precision=4)

    good = series.join(middle, on=SERIES).join(
        located.select(SERIES, "joint_consensus_y_mm"), on=SERIES, how="left"
    )
    horizontal_medial = (
        pl.when(pl.col("side_right") == "medial")
        .then(1.0)
        .when(pl.col("side_left") == "medial")
        .then(-1.0)
    )
    horizontal_anterior = (
        pl.when(pl.col("side_left") == "anterior")
        .then(-1.0)
        .when(pl.col("side_right") == "anterior")
        .then(1.0)
    )
    vertical_anterior = (
        pl.when(pl.col("side_top") == "anterior")
        .then(-1.0)
        .when(pl.col("side_bottom") == "anterior")
        .then(1.0)
    )
    placed = good.with_columns(
        tissue_mid_x=(pl.col("tissue_x0_q50") + pl.col("tissue_x1_q50")) / 2,
        tissue_mid_y=(pl.col("tissue_y0_q50") + pl.col("tissue_y1_q50")) / 2,
        tissue_width_mm=pl.col("tissue_x1_q50") - pl.col("tissue_x0_q50"),
        tissue_height_mm=pl.col("tissue_y1_q50") - pl.col("tissue_y0_q50"),
    ).with_columns(
        knee_medial_mm=pl.col("knee_dx_mm") * horizontal_medial,
        knee_anterior_mm=pl.when(pl.col("plane") == "Axial")
        .then(pl.col("knee_dy_mm") * vertical_anterior)
        .otherwise(pl.col("knee_dx_mm") * horizontal_anterior),
        tissue_medial_mm=pl.col("tissue_mid_x") * horizontal_medial,
        tissue_anterior_mm=pl.when(pl.col("plane") == "Axial")
        .then(pl.col("tissue_mid_y") * vertical_anterior)
        .otherwise(pl.col("tissue_mid_x") * horizontal_anterior),
        joint_below_centre_mm=pl.col("joint_consensus_y_mm"),
    )
    offset_rows = []
    for key, part in sorted(placed.partition_by(KEYS, as_dict=True).items()):
        labels = dict(zip(KEYS, key, strict=True))
        for column in (
            "knee_medial_mm",
            "knee_anterior_mm",
            "tissue_medial_mm",
            "tissue_anterior_mm",
            "joint_below_centre_mm",
            "tissue_width_mm",
            "tissue_height_mm",
        ):
            values = part[column].drop_nulls()
            if values.is_empty():
                continue
            offset_rows.append(
                {
                    **labels,
                    "measure": column,
                    "series": len(values),
                    **{
                        f"p{round(q * 100)}": float(values.quantile(q, "linear"))
                        for q in QUANTILES
                    },
                    "abs_p50": float(values.abs().quantile(0.5, "linear")),
                    "abs_p90": float(values.abs().quantile(0.9, "linear")),
                    "abs_p95": float(values.abs().quantile(0.95, "linear")),
                    "share_abs_over_10mm": float((values.abs() > 10).mean()),
                }
            )
    pl.DataFrame(offset_rows).write_csv(out / "centre_offset.csv", float_precision=4)

    joints = placed.filter(
        pl.col("joint_patient_z_mm").is_not_null() & pl.col("trusted")
    )
    paired = (
        joints.filter(pl.col("plane") == "Coronal")
        .group_by(STUDY)
        .agg(pl.col("joint_patient_z_mm").median().alias("coronal"))
        .join(
            joints.filter(pl.col("plane") == "Sagittal")
            .group_by(STUDY)
            .agg(pl.col("joint_patient_z_mm").median().alias("sagittal")),
            on=STUDY,
        )
        .with_columns(gap=(pl.col("coronal") - pl.col("sagittal")).abs())
    )
    coverage = (
        measured.group_by(*KEYS)
        .agg(
            pl.len().alias("series"),
            (pl.col("status") == "ok").sum().alias("measured"),
            (pl.col("status") == "excluded_by_cleaning")
            .sum()
            .alias("excluded_by_cleaning"),
            pl.col("trusted").sum().alias("trusted_geometry"),
            (pl.col("mask_fraction_q50") <= settings.mask_fraction_limit)
            .sum()
            .alias("usable_middle_mask"),
            pl.col("joint_y_mm").is_not_null().sum().alias("joint_line_found"),
            pl.col("side_left")
            .is_in(["patient_left", "patient_right"])
            .sum()
            .alias("knee_side_unknown_in_plane"),
        )
        .sort(*KEYS)
    )
    coverage.write_csv(out / "window_coverage.csv")
    summary = {
        "mask_fraction_limit": settings.mask_fraction_limit,
        "joint_cross_check_studies": paired.height,
        "joint_cross_check_within_10mm": float((paired["gap"] <= 10).mean()),
        "joint_cross_check_median_gap_mm": float(paired["gap"].median()),
    }
    (out / "window_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=250):
        print(coverage)
    return summary


def main() -> None:
    print(json.dumps(run(Settings()), indent=2))


if __name__ == "__main__":
    main()
