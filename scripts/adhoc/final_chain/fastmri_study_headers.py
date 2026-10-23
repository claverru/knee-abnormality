from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pydicom

RAW = Path("/workspace/data/fastmri/raw/train_series")


def header(study: str) -> dict:
    directory = next(path for path in sorted((RAW / study).iterdir()))
    first = next(path for path in sorted(directory.resolve().glob("*.dcm")))
    dataset = pydicom.dcmread(first, stop_before_pixels=True)
    return {
        "StudyInstanceUID": study,
        "Manufacturer": str(dataset.get("Manufacturer") or ""),
        "ManufacturerModelName": str(dataset.get("ManufacturerModelName") or ""),
        "MagneticFieldStrength": str(dataset.get("MagneticFieldStrength") or ""),
    }


studies = sorted(path.name for path in RAW.iterdir())
with ProcessPoolExecutor(60) as pool:
    rows = list(pool.map(header, studies, chunksize=32))
frame = pl.DataFrame(rows)
frame.write_parquet("/workspace/data/fastmri/study_headers.parquet")
print(
    frame.height,
    frame.group_by("Manufacturer", "ManufacturerModelName", "MagneticFieldStrength")
    .len()
    .sort("len", descending=True)
    .head(8)
    .rows(),
)
