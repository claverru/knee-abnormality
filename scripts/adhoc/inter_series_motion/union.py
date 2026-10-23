import os

import polars as pl

out = "outputs/analysis/i/inter-series-motion-20261005/"
if os.path.exists(out + "smoke.parquet"):
    os.remove(out + "smoke.parquet")
c = pl.read_parquet(out + "cross_plane_sample200.parquet").filter(
    pl.col("kind") == "pair"
)
s = pl.read_parquet(out + "pairs_full.parquet").filter(
    (pl.col("normal_angle_deg") <= 3) & pl.col("ngf_c_mag").is_not_null()
)
studies = c["study"].unique()
s2 = s.filter(pl.col("study").is_in(studies.to_list()))
for thr in (3, 5, 10):
    a = set(s2.filter(pl.col("ngf_c_mag") > thr)["study"].to_list())
    b = set(c.filter(pl.col("along").abs() > thr)["study"].to_list())
    b2 = set(
        c.filter((pl.col("along") ** 2 + pl.col("across") ** 2).sqrt() > thr)[
            "study"
        ].to_list()
    )
    print(
        thr,
        "same-plane",
        len(a),
        "cross along",
        len(b),
        "union",
        len(a | b),
        "union incl. coarse across",
        len(a | b2),
        "of",
        len(studies),
    )
edge = s.filter(pl.col("ngf_c_edge"))
print("pairs at search limit", len(edge), "studies", edge["study"].n_unique())
print(
    "studies >=20mm-capped or >10",
    s.filter(pl.col("ngf_c_mag") > 10)["study"].n_unique(),
)
per = s.group_by("study").agg(pl.col("ngf_c_mag").max().alias("max_mag"))
print(
    per["max_mag"].quantile(0.5),
    per["max_mag"].quantile(0.9),
    per["max_mag"].quantile(0.95),
    per["max_mag"].quantile(0.99),
    len(per),
)
s.drop(
    "ua", "va", "ngf_sx_slices", "ngf_sy_slices", "ncc_sx_slices", "ncc_sy_slices"
).write_csv(out + "pairs_full_comparable.csv")
