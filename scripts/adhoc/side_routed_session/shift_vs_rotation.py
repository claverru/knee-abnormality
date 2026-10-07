import numpy as np
import polars as pl

shifts = pl.read_csv("outputs/analysis/side-routed-views-20261005/studies.csv")
shifts = shifts.with_columns(left=pl.col("left_probability") > 0.5).with_columns(
    lateral=pl.when("left")
    .then(pl.col("shift_midline_mm"))
    .otherwise(-pl.col("shift_midline_mm")),
    sign=pl.when("left").then(1.0).otherwise(-1.0),
)
series = pl.read_parquet("outputs/analysis/knee-rotation-census-20261005/series.parquet")
key = shifts.columns[1]
rotation = (
    series.filter(pl.col("plane") != "Sagittal")
    .group_by(key)
    .agg(pl.col("rotation_deg").mean())
)
joined = shifts.join(rotation, on=key).with_columns(
    external=pl.col("rotation_deg") * pl.col("sign")
)
print(joined.height)
print(
    "corr lateral shift vs external rotation",
    np.corrcoef(joined["lateral"], joined["external"])[0, 1],
)
print(
    "corr scanner shift vs scanner rotation",
    np.corrcoef(joined["shift_midline_mm"], joined["rotation_deg"])[0, 1],
)
print(
    joined.with_columns(
        band=pl.col("external").cut([0, 5, 10], labels=["<0", "0-5", "5-10", ">10"])
    )
    .group_by("band")
    .agg(pl.len(), pl.col("lateral").mean().round(2), pl.col("external").mean().round(1))
    .sort("external")
)
studies = (
    series.filter(pl.col("plane") != "Sagittal")
    .group_by(key)
    .agg(pl.col("external_deg").mean(), pl.col("side").first())
    .filter(pl.col("side") != 0)
)
values = studies["external_deg"].to_numpy()
print(
    "studies",
    len(values),
    "mean",
    values.mean().round(2),
    "share external",
    (values > 0).mean().round(3),
    {t: float((np.abs(values) > t).mean().round(3)) for t in (5, 10, 15, 20)},
)
