import polars as pl

frame = pl.read_parquet("data/labels/prepared/jevk5-report-attributes-v4.0/targets.parquet")
for question in (
    "medial_meniscus_tear_location",
    "lateral_meniscus_tear_location",
    "pf_cartilage_surface",
    "patellar_facet",
    "marrow_edema_location",
    "extra_meniscal_cyst",
    "extra_lateral_ligaments",
    "extra_pcl",
    "extra_extensor_tendons",
    "extra_bursitis",
    "medial_meniscus_extrusion",
    "extra_patellar_instability",
):
    columns = [c for c in frame.columns if c.startswith(f"attribute_{question}__")]
    weight = frame[f"weight_{question}"]
    parts = []
    for column in columns:
        confident = int((frame[column] > 0.5).sum())
        parts.append(f"{column.split('__')[1]} {confident}")
    print(f"{question:32} " + ", ".join(parts))
