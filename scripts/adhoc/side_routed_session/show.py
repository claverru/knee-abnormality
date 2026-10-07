import json

import polars as pl

s = json.load(open("outputs/analysis/slice-budget-shortfall-20261005/summary.json"))
for scope in ("all_studies", "fold_0"):
    for m in s[scope]:
        print(scope, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "views"})
        for v in m["views"]:
            print("   ", {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()})
f = pl.read_parquet("outputs/analysis/slice-budget-shortfall-20261005/study_views.parquet")
print(f.height, f["StudyInstanceUID"].n_unique(), f["index_failures"].max())
print(f.filter(pl.col("multiplier") == 1).group_by("pool_path", pl.col("sources") > 0).len())
print(f.filter((pl.col("multiplier") == 1) & (pl.col("fold") == 0)).select(pl.col("delivered").sum(), pl.col("StudyInstanceUID").n_unique()))
print(f.filter((pl.col("sources") > 0) & (pl.col("delivered") < pl.min_horizontal("requested", "capacity"))).group_by("multiplier").len())
print(f.filter((pl.col("multiplier") == 1) & (pl.col("sources") > 0)).group_by("StudyInstanceUID").agg(pl.len().alias("views")).group_by("views").len().sort("views"))
