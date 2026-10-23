import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

sys.path.insert(0, "/root")
from fastmri_layout import TARGETS, describe  # noqa: E402

SOURCE = Path("/workspace/data/raw/oai/image03/00m")
OUTPUT = Path("/workspace/data/oai")
EXPECTED = {
    "SAG_IW_TSE": ("Sagittal", 1),
    "COR_IW_TSE": ("Coronal", 0),
    "COR_MPR": ("Coronal", 1),
    "AX_MPR": ("Axial", 1),
}


def rename(series: str) -> int:
    count = 0
    for entry in os.scandir(series):
        if entry.is_file() and not entry.name.endswith(".dcm"):
            os.rename(entry.path, entry.path + ".dcm")
            count += 1
    return count


def main() -> None:
    series = sorted(str(path) for path in SOURCE.glob("*/*/*/*") if path.is_dir())
    with ProcessPoolExecutor(60) as pool:
        renamed = sum(pool.map(rename, series, chunksize=64))
        rows = list(pool.map(describe, series, chunksize=64))
    print("series", len(series), "files renamed", renamed, flush=True)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    parts = [Path(value).parts for value in frame["directory"]]
    kind = frame["description"].str.replace(r"_(RIGHT|LEFT)$", "")
    side = frame["description"].str.extract(r"_(RIGHT|LEFT)$").str.to_lowercase()
    frame = frame.with_columns(
        kind.alias("kind"),
        side.alias("side"),
        pl.Series("patient", [part[-3] for part in parts]),
        pl.Series("visit_date", [part[-2] for part in parts]),
    ).with_columns(
        StudyInstanceUID=pl.format("oai-{}-{}", "patient", "side"),
    ).with_columns(
        SeriesInstanceUID=pl.format("{}-{}", "StudyInstanceUID", pl.col("kind").str.to_lowercase()),
    )
    expected_plane = pl.col("kind").replace_strict({key: value[0] for key, value in EXPECTED.items()}, default=None)
    expected_fs = pl.col("kind").replace_strict({key: value[1] for key, value in EXPECTED.items()}, default=None)
    frame = frame.with_columns(
        type_matches=(pl.col("plane") == expected_plane) & (pl.col("fat_suppressed") == expected_fs)
    )
    frame = (
        frame.sort("n_images", descending=True)
        .with_columns(copy=pl.int_range(pl.len()).over("SeriesInstanceUID"))
        .with_columns(
            reason=pl.when(pl.col("copy") > 0).then(pl.lit("duplicate series of the knee")).otherwise(pl.col("reason"))
        )
        .drop("copy")
        .sort("directory")
    )
    raw = OUTPUT / "raw"
    (raw / "test_series").mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUTPUT / "series_catalog.parquet")
    kept = frame.filter(pl.col("reason").is_null())
    for study, name, directory in kept.select("StudyInstanceUID", "SeriesInstanceUID", "directory").iter_rows():
        link = raw / "test_series" / study / name
        link.parent.mkdir(exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(directory)
    table = kept.select(
        "StudyInstanceUID",
        "SeriesInstanceUID",
        pl.col("fat_suppressed").alias("Fluid_Sensitive"),
        pl.col("fat_suppressed").alias("Fat_Suppression"),
        pl.col("plane").alias("Anatomical_Plane"),
    )
    table.write_csv(raw / "test_series.csv")
    table.write_csv(raw / "train_series.csv")
    if not (raw / "train_series").exists():
        (raw / "train_series").symlink_to("test_series")
    studies = kept.select("StudyInstanceUID").unique().sort("StudyInstanceUID")
    studies.with_columns(pl.lit(0.5).alias(target) for target in TARGETS).write_csv(raw / "sample_submission.csv")
    kept.group_by("StudyInstanceUID").agg(
        pl.col("patient").first(), pl.col("side").first(), pl.len().alias("n_series")
    ).sort("StudyInstanceUID").write_csv(OUTPUT / "studies.csv")
    print("kept series", kept.height, "studies", studies.height, "patients", kept["patient"].n_unique())
    print("excluded", frame.filter(pl.col("reason").is_not_null()).group_by("reason").len().rows())
    print("types", kept.group_by("kind", "plane", "fat_suppressed").len().sort("kind").rows())
    print("type mismatches", kept.filter(~pl.col("type_matches")).group_by("kind", "plane", "fat_suppressed", "contrast_source").len().rows())
    print("contrast source", kept["contrast_source"].value_counts(sort=True).rows())
    print("series per study", kept.group_by("StudyInstanceUID").len()["len"].value_counts().sort("len").rows())
    print("tilt degrees p50/p95/max", kept["tilt_degrees"].quantile(0.5), kept["tilt_degrees"].quantile(0.95), kept["tilt_degrees"].max())


if __name__ == "__main__":
    main()
