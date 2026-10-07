import polars as pl

table = pl.read_csv("outputs/analysis/side-routed-views-20261005/studies.csv")
table = table.with_columns(left=pl.col("left_probability") > 0.5).with_columns(
    lateral=pl.when("left")
    .then(pl.col("shift_midline_mm"))
    .otherwise(-pl.col("shift_midline_mm"))
)
print(
    table.group_by("left").agg(
        pl.len(),
        pl.col("shift_midline_mm").mean().round(2),
        pl.col("lateral").mean().round(2),
    )
)
print(
    table.select(
        pl.col("lateral").mean().round(2).alias("mean"),
        pl.col("lateral").std().round(2).alias("std"),
        (pl.col("lateral") > 0).mean().round(2).alias("share_lateral"),
        pl.col("lateral").quantile(0.1).round(1).alias("p10"),
        pl.col("lateral").quantile(0.9).round(1).alias("p90"),
    )
)
print(table.group_by("state").len())
