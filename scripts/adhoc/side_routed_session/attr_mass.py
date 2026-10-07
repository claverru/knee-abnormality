import polars as pl

v4 = pl.read_parquet("data/labels/prepared/jevk5-report-attributes-v4.0/targets.parquet")
v3 = pl.read_parquet("data/labels/prepared/jevk5-report-attributes-v3.1/targets.parquet")
print(v4.shape, v3.shape)
print([c for c in v4.columns if "marrow_edema_location" in c])
SPATIAL = {
    "v3 contusion_location": (v3, "contusion_location", ("none", "not_stated")),
    "v4 marrow_edema_location": (v4, "marrow_edema_location", ("none", "not_stated")),
    "v4 patellar_facet": (v4, "patellar_facet", ("no_patellar_damage", "not_stated")),
    "v3 pf_cartilage_surface": (v3, "pf_cartilage_surface", ("no_damage", "not_stated")),
    "v4 pf_cartilage_surface": (v4, "pf_cartilage_surface", ("no_damage", "not_stated")),
    "v3 medial tear location": (v3, "medial_meniscus_tear_location", ("no_tear", "not_stated")),
    "v4 medial tear location": (v4, "medial_meniscus_tear_location", ("no_tear", "not_stated")),
    "v3 lateral tear location": (v3, "lateral_meniscus_tear_location", ("no_tear", "not_stated")),
    "v4 lateral tear location": (v4, "lateral_meniscus_tear_location", ("no_tear", "not_stated")),
    "v3 baker_size": (v3, "baker_size", ("absent", "other_cyst_only")),
    "v4 baker_size": (v4, "baker_size", ("absent", "other_cyst_only")),
    "v3 medial_osteophytes": (v3, "osteophytes", ("none", "not_mentioned")),
    "v4 osteophytes": (v4, "osteophytes", ("none",)),
}
for name, (frame, question, skip) in SPATIAL.items():
    columns = [
        c
        for c in frame.columns
        if c.startswith(f"attribute_{question}__")
    ]
    informative = [c for c in columns if not c.endswith(skip)]
    if not informative:
        print(name, "no columns", [c for c in frame.columns if question in c][:8])
        continue
    weight = frame[f"weight_{question}"].fill_null(0.0) if f"weight_{question}" in frame.columns else 1.0
    mass = frame.select(pl.sum_horizontal(informative).alias("m"))["m"].fill_null(0.0)
    print(
        f"{name:28} options {len(columns):2}  studies with located mass > 0.5: {int((mass > 0.5).sum()):5}   total located mass {float(mass.sum()):8.1f}   weighted {float((mass * weight).sum()):8.1f}"
    )
