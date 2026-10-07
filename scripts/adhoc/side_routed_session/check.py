import polars as pl

pl.Config.set_tbl_rows(40)
f = pl.read_parquet("outputs/analysis/slice-budget-shortfall-20261005/study_views.parquet").filter(pl.col("sources") > 0)
f = f.with_columns((pl.col("requested") - pl.col("delivered")).alias("short"), (pl.col("capacity") - pl.col("delivered")).alias("spare"))
for m in (4, 6):
    s = f.filter(pl.col("multiplier") == m).group_by("StudyInstanceUID").agg(
        pl.len().alias("views"),
        (pl.col("short") > 0).sum().alias("short_views"),
        pl.col("short").sum(),
        pl.col("spare").sum(),
        pl.col("requested").sum(),
        pl.col("capacity").sum(),
    ).filter(pl.col("short") > 0)
    print(m, s.with_columns((pl.col("short_views") == pl.col("views")).alias("all_views_short")).group_by("all_views_short").agg(pl.len(), pl.col("short").sum(), pl.col("spare").sum(), pl.min_horizontal("short", "spare").sum().alias("recoverable")))
    print(m, "studies whose total capacity < total requested", s.filter(pl.col("capacity") < pl.col("requested")).height, "spare quantiles", s["spare"].quantile(0.25), s["spare"].quantile(0.5), s["spare"].quantile(0.75), s["spare"].max())
print(f.filter(pl.col("multiplier") == 1).select(pl.col("capacity").quantile(0.01).alias("q01"), pl.col("capacity").quantile(0.1).alias("q10"), pl.col("capacity").quantile(0.5).alias("q50"), pl.col("capacity").quantile(0.9).alias("q90"), pl.col("capacity").min().alias("min"), pl.col("capacity").max().alias("max")))
print(f.filter(pl.col("delivered") < pl.min_horizontal("requested", "capacity")).select("multiplier", "view", "pool_path", "requested", "capacity", "delivered", "candidates"))
