import sys

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_fmt_str_lengths(30)
c = pl.read_parquet(sys.argv[1])
same = pl.read_parquet(sys.argv[2]).filter(
    (pl.col("kind") == "pair")
    & (pl.col("normal_angle_deg") <= 3)
    & pl.col("ngf_c_mag").is_not_null()
)


def dist(name, x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return
    print(
        f"{name:44s} n={len(x):5d} med={np.median(x):5.2f} p75={np.quantile(x, 0.75):5.2f} p90={np.quantile(x, 0.9):5.2f} "
        f"p95={np.quantile(x, 0.95):5.2f} p99={np.quantile(x, 0.99):6.2f} >3={100 * (x > 3).mean():5.2f}% >5={100 * (x > 5).mean():5.2f}% >10={100 * (x > 10).mean():5.2f}%"
    )


print(
    c.group_by("kind").len(),
    "studies",
    c["study"].n_unique(),
    "median study seconds",
    c.unique("study")["study_seconds"].median(),
)
pair = c.filter(pl.col("kind") == "pair").with_columns(
    combo=pl.col("plane_a") + ">" + pl.col("plane_b"),
    reliable=(pl.col("dev") < 2)
    & ((pl.col("along") - pl.col("ncc_along")).abs() < 2)
    & ~pl.col("edge"),
    rank_gap=pl.col("rank_b") - pl.col("rank_a"),
    same_contrast=(pl.col("fluid_a") == pl.col("fluid_b"))
    & (pl.col("fs_a") == pl.col("fs_b")),
)
print("pairs", len(pair), "reliable share", round(float(pair["reliable"].mean()), 4))
keys = ["study", "series_a", "series_b"]
syn = c.filter(pl.col("kind") == "synthetic").select(
    keys + [pl.col("sx").alias("tx"), pl.col("sy").alias("ty")]
)
if len(syn):
    j = pair.join(syn, on=keys).with_columns(
        err=(
            (pl.col("tx") - pl.col("sx") + 7.0) ** 2
            + (pl.col("ty") - pl.col("sy") - 4.5) ** 2
        ).sqrt()
    )
    dist("synthetic recovery err all", j["err"])
    dist("synthetic recovery err reliable", j.filter("reliable")["err"])

rel = pair.filter("reliable")
print(
    "\n== |along| (shift along the intersection line; both series resolve it in-plane)"
)
dist("all", pair["along"].abs())
dist("reliable", rel["along"].abs())
for combo in sorted(pair["combo"].unique().to_list()):
    dist(f"{combo} reliable", rel.filter(pl.col("combo") == combo)["along"].abs())
dist("same contrast reliable", rel.filter("same_contrast")["along"].abs())
dist("different contrast reliable", rel.filter(~pl.col("same_contrast"))["along"].abs())
print("\n== |across| (through-plane of B; resolved only at B's slice gap)")
dist("all", pair["across"].abs())
dist("reliable", rel["across"].abs())
print("\n== rank gap, reliable |along|")
for g in sorted(rel["rank_gap"].unique().to_list())[:8]:
    dist(f"rank gap {g}", rel.filter(pl.col("rank_gap") == g)["along"].abs())
total = pair["study"].n_unique()
for thr in (3, 5, 10):
    n = rel.filter(pl.col("along").abs() > thr)["study"].n_unique()
    print(
        f"studies with >=1 reliable cross pair |along| > {thr}: {n} ({100 * n / total:.2f}%) of {total}"
    )

m = {}
for r in same.iter_rows(named=True):
    v = np.array([r["ngf_c_lr"], r["ngf_c_ap"], r["ngf_c_si"]])
    m[(r["study"], r["series_a"], r["series_b"])] = v
    m[(r["study"], r["series_b"], r["series_a"])] = -v
x = {}
for r in pair.iter_rows(named=True):
    line = np.array(r["line"])
    x[(r["study"], r["series_a"], r["series_b"])] = (r["along"], line, r["reliable"])
    x[(r["study"], r["series_b"], r["series_a"])] = (-r["along"], line, r["reliable"])
res, direct, via = [], [], []
for (study, a, b), v in m.items():
    if a > b:
        continue
    for (s2, p, q), (along, line, ok) in x.items():
        if s2 != study or p != a or q in (a, b):
            continue
        other = x.get((study, b, q))
        if other is None or not ok or not other[2]:
            continue
        pred = float(v @ line) + other[0] * float(np.sign(other[1] @ line))
        res.append(pred - along)
        direct.append(along)
        via.append(pred)
res = np.array(res)
print(
    "\n== triangle closure on the along component: same-plane(a,b) + cross(b,c) - cross(a,c)"
)
dist("closure |residual| mm", np.abs(res))
if len(res) > 2:
    print(
        "corr direct vs via",
        round(float(np.corrcoef(direct, via)[0, 1]), 3),
        "n",
        len(res),
    )

cols = [
    "study",
    "combo",
    "number_a",
    "number_b",
    "fs_a",
    "fs_b",
    "along",
    "across",
    "ncc_along",
    "dev",
    "overlap",
    "reliable",
    "desc_a",
    "desc_b",
]
print(
    pair.sort(pl.col("along").abs(), descending=True)
    .head(20)
    .select(cols)
    .with_columns(pl.col(pl.Float64).round(2), pl.col("study").str.slice(-8))
)
