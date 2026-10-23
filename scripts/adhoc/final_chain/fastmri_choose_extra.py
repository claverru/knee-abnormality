import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("/home/claudio/.cache/knee-segmenters") / sys.argv[1]
FRACTION = float(sys.argv[2])
SEED = 20261014
RAW = Path("/workspace/data/fastmri/raw")
EXISTING = Path(
    "/workspace/data/labels/prepared/anatomy-mrannotator-fastmri1230-joints/knees.json"
)

table = pl.read_parquet(
    "/workspace/data/labels/prepared/fastmri-rn50-main-x4-tau09-tta/training_table.parquet"
)
studies = set(table["StudyInstanceUID"].unique().to_list())
chosen = {knee["study"] for knee in json.loads(EXISTING.read_text())}
catalog = pl.read_parquet("/workspace/data/fastmri/series_catalog.parquet").select(
    "SeriesInstanceUID", "n_images"
)
series = (
    pl.read_csv(RAW / "train_series.csv")
    .filter(
        (pl.col("Anatomical_Plane") == "Sagittal")
        & pl.col("StudyInstanceUID").is_in(list(studies))
    )
    .join(catalog, on="SeriesInstanceUID", how="left")
    .sort("n_images", "SeriesInstanceUID", descending=[True, False])
    .unique("StudyInstanceUID", keep="first", maintain_order=True)
)
headers = pl.read_parquet("/workspace/data/fastmri/study_headers.parquet")
frame = series.join(headers, on="StudyInstanceUID", how="left").with_columns(
    pl.concat_str(
        [
            pl.col(name).fill_null("")
            for name in (
                "Manufacturer",
                "ManufacturerModelName",
                "MagneticFieldStrength",
            )
        ],
        separator="|",
    ).alias("scanner")
)
wanted = round(FRACTION * len(studies))
sizes = dict(frame.group_by("scanner").len().sort("scanner").iter_rows())
weights = {name: np.sqrt(size) for name, size in sizes.items()}
quota = {name: 0 for name in sizes}
remaining = wanted
open_groups = set(sizes)
while remaining > 0 and open_groups:
    total = sum(weights[name] for name in open_groups)
    shares = {name: remaining * weights[name] / total for name in open_groups}
    granted = 0
    for name in sorted(open_groups):
        add = min(int(np.floor(shares[name])), sizes[name] - quota[name])
        quota[name] += add
        granted += add
    open_groups = {name for name in open_groups if quota[name] < sizes[name]}
    if granted == 0:
        for name in sorted(open_groups, key=lambda item: -shares[item])[:remaining]:
            quota[name] += 1
            granted += 1
    remaining -= granted
have = dict(
    frame.filter(pl.col("StudyInstanceUID").is_in(list(chosen)))
    .group_by("scanner")
    .len()
    .iter_rows()
)
rng = np.random.default_rng(SEED)
jobs = []
for name in sorted(sizes):
    extra = max(quota[name] - have.get(name, 0), 0)
    rows = (
        frame.filter(
            (pl.col("scanner") == name)
            & ~pl.col("StudyInstanceUID").is_in(list(chosen))
        )
        .sort("StudyInstanceUID")
        .to_dicts()
    )
    extra = min(extra, len(rows))
    for index in sorted(rng.choice(len(rows), extra, replace=False).tolist()):
        row = rows[index]
        jobs.append(
            (
                row["StudyInstanceUID"],
                -1,
                {
                    "SeriesInstanceUID": row["SeriesInstanceUID"],
                    "Fluid_Sensitive": row["Fluid_Sensitive"],
                    "Fat_Suppression": row["Fat_Suppression"],
                    "scanner": name,
                },
            )
        )
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "pending.json").write_text(json.dumps(jobs))
print(
    "training studies",
    len(studies),
    "target",
    wanted,
    "already chosen",
    len(chosen),
    "new",
    len(jobs),
    "total",
    len(chosen) + len(jobs),
)
for name in sorted(sizes, key=lambda item: -sizes[item])[:12]:
    print(
        f"  {name:60} {sizes[name]:5d} studies, quota {quota[name]:4d}, had {have.get(name, 0):4d}"
    )
