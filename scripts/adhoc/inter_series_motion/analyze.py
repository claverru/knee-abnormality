import sys

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_fmt_str_lengths(40)
d = pl.read_parquet(sys.argv[1])
QS = [0.5, 0.75, 0.9, 0.95, 0.99]


def q(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return []
    return [round(float(np.quantile(x, p)), 2) for p in QS] + [round(float(x.max()), 2), len(x)]


def dist(name, x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return
    print(
        f"{name:42s} n={len(x):5d} med={np.median(x):5.2f} p75={np.quantile(x, .75):5.2f} p90={np.quantile(x, .9):5.2f} "
        f"p95={np.quantile(x, .95):5.2f} p99={np.quantile(x, .99):6.2f} >3={100 * (x > 3).mean():5.2f}% >5={100 * (x > 5).mean():5.2f}% >10={100 * (x > 10).mean():5.2f}%"
    )


print(d.group_by("kind").len())
print("studies", d["study"].n_unique(), "median study seconds", d.unique("study")["study_seconds"].median())
pair = d.filter(pl.col("kind") == "pair").with_columns(
    agree=((pl.col("ngf_c_sx") - pl.col("ncc_c_sx")) ** 2 + (pl.col("ngf_c_sy") - pl.col("ncc_c_sy")) ** 2).sqrt(),
    rank_gap=pl.col("rank_b") - pl.col("rank_a"),
    same_contrast=(pl.col("fluid_a") == pl.col("fluid_b")) & (pl.col("fs_a") == pl.col("fs_b")),
)
print("pairs", len(pair), "normal angle", q(pair["normal_angle_deg"]), "share>3deg", round(float((pair["normal_angle_deg"] > 3).mean()), 3))
comp = pair.filter((pl.col("normal_angle_deg") <= 3) & pl.col("ngf_c_mag").is_not_null())
comp = comp.with_columns(reliable=(pl.col("ngf_c_dev") < 2) & (pl.col("agree") < 2) & ~pl.col("ngf_c_edge"))
print("comparable pairs", len(comp), "studies", comp["study"].n_unique(), "reliable share", round(float(comp["reliable"].mean()), 4))
print("abs slice offset", q(comp["slice_offset_mm"].abs()), "overlap low q", [round(float(comp["overlap"].quantile(p)), 2) for p in (0.01, 0.05, 0.5)])
print("inplane angle", q(comp["inplane_angle_deg"]))

if (d["kind"] == "null_adjacent").any():
    print("\n== VALIDATION")
    n = d.filter(pl.col("kind") == "null_adjacent")
    for est in ("ngf_c", "ncc_c", "ngf", "ncc"):
        dist(f"null adjacent slices {est}", n[f"{est}_mag"])
    for plane in ("Sagittal", "Coronal", "Axial"):
        dist(f"null adjacent ngf_c {plane}", n.filter(pl.col("plane") == plane)["ngf_c_mag"])
    keys = ["study", "series_a", "series_b"]
    for est in ("ngf_c", "ncc_c"):
        s = d.filter(pl.col("kind") == "synthetic").select(keys + [pl.col(f"{est}_sx").alias("tx"), pl.col(f"{est}_sy").alias("ty")])
        j = comp.join(s, on=keys).with_columns(
            err=((pl.col("tx") - pl.col(f"{est}_sx") + 7.0) ** 2 + (pl.col("ty") - pl.col(f"{est}_sy") - 4.5) ** 2).sqrt()
        )
        dist(f"synthetic recovery err {est} all", j["err"])
        dist(f"synthetic recovery err {est} reliable", j.filter("reliable")["err"])
        print("   share err>1mm all", round(float((j["err"] > 1).mean()), 4), "reliable", round(float((j.filter("reliable")["err"] > 1).mean()), 4))

print("\n== SHIFT MAGNITUDE (in-plane 2D, mm), ngf combined")
dist("all comparable pairs", comp["ngf_c_mag"])
rel = comp.filter("reliable")
dist("reliable pairs", rel["ngf_c_mag"])
dist("unreliable pairs", comp.filter(~pl.col("reliable"))["ngf_c_mag"])
dist("ncc combined, all comparable", comp["ncc_c_mag"])
for plane in ("Sagittal", "Coronal", "Axial"):
    dist(f"{plane} all", comp.filter(pl.col("plane") == plane)["ngf_c_mag"])
    dist(f"{plane} reliable", rel.filter(pl.col("plane") == plane)["ngf_c_mag"])
print("\n== PER AXIS |component| mm (patient axes; only planes that contain the axis)")
for axis, planes in (("lr", ("Coronal", "Axial")), ("ap", ("Sagittal", "Axial")), ("si", ("Sagittal", "Coronal"))):
    for plane in planes:
        dist(f"{axis} from {plane} all", comp.filter(pl.col("plane") == plane)[f"ngf_c_{axis}"].abs())
        dist(f"{axis} from {plane} reliable", rel.filter(pl.col("plane") == plane)[f"ngf_c_{axis}"].abs())
    dist(f"{axis} pooled reliable", rel.filter(pl.col("plane").is_in(planes))[f"ngf_c_{axis}"].abs())
print("\nsigned mean shift (bias check), reliable")
print(rel.group_by("plane", "same_contrast").agg(pl.len(), pl.col("ngf_c_sx").mean().round(3), pl.col("ngf_c_sy").mean().round(3), pl.col("ngf_c_sx").median().round(3).alias("sx_med"), pl.col("ngf_c_sy").median().round(3).alias("sy_med")).sort("plane", "same_contrast"))

print("\n== STUDIES")
total = comp["study"].n_unique()
for thr in (3, 5, 10):
    a = comp.filter(pl.col("ngf_c_mag") > thr)["study"].n_unique()
    r = rel.filter(pl.col("ngf_c_mag") > thr)["study"].n_unique()
    print(f"studies with >=1 pair > {thr} mm: all-comparable {a} ({100 * a / total:.2f}%), reliable-only {r} ({100 * r / total:.2f}%) of {total}")

print("\n== ORDER / CONTRAST")
for g in sorted(comp["rank_gap"].unique().to_list()):
    dist(f"rank gap {g} reliable", rel.filter(pl.col("rank_gap") == g)["ngf_c_mag"])
dist("same contrast reliable", rel.filter("same_contrast")["ngf_c_mag"])
dist("different contrast reliable", rel.filter(~pl.col("same_contrast"))["ngf_c_mag"])
print("spearman rank_gap vs mag (reliable)", rel.select(pl.corr("rank_gap", "ngf_c_mag", method="spearman")).item())
print(rel.group_by("manufacturer").agg(pl.len(), pl.col("ngf_c_mag").median().round(2).alias("med"), pl.col("ngf_c_mag").quantile(0.95).round(2).alias("p95"), (pl.col("ngf_c_mag") > 5).mean().round(4).alias(">5")).sort("len", descending=True))

print("\n== TOP SHIFTS")
cols = ["study", "plane", "number_a", "number_b", "fs_a", "fs_b", "ngf_c_sx", "ngf_c_sy", "ngf_c_mag", "ngf_c_dev", "agree", "reliable", "overlap", "desc_a", "desc_b"]
print(rel.sort("ngf_c_mag", descending=True).head(25).select(cols).with_columns(pl.col(pl.Float64).round(2), pl.col("study").str.slice(-8)))
print(comp.filter(~pl.col("reliable")).sort("ngf_c_mag", descending=True).head(15).select(cols).with_columns(pl.col(pl.Float64).round(2), pl.col("study").str.slice(-8)))
