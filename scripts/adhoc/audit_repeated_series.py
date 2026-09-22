import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY
from knee_abnormality.data.preprocessing.dicom import (
    PREPARATION_ERRORS,
    parse_prefix,
    read_frame,
    read_header,
    read_prefix,
)
from knee_abnormality.data.sources import SERIES, list_series
from knee_abnormality.provenance import copy_sources, sha256_file

TYPE = ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]


class Settings(CLISettings):
    data_root: Path = Path("data/raw")
    splits: list[str] = Field(default_factory=lambda: ["train", "test"])
    output_dir: Path = Field(description="Fresh output directory for the audit")
    workers: int = 12
    chunk_bytes: int = 8192
    studies: int | None = None


def _chunk(path: Path, header, size: int, chunk_bytes: int) -> tuple[bytes, bool]:
    descriptor = header.descriptor
    if descriptor is not None:
        start = descriptor.offset + (descriptor.length // 2 - chunk_bytes // 2) // 2 * 2
        length = min(chunk_bytes, descriptor.length)
    else:
        start = max(size - chunk_bytes, 0)
        length = size - start
    handle = os.open(path, os.O_RDONLY)
    try:
        return os.pread(handle, length, start), descriptor is not None
    finally:
        os.close(handle)


def scan_series(task: tuple[str, str, str, Path, int]) -> list[dict]:
    split, study, series, directory, chunk_bytes = task
    records = []
    for path in list_series(directory):
        try:
            buffer, info = read_prefix(path)
            header = parse_prefix(path, (buffer, info))
            chunk, native = _chunk(path, header, info.st_size, chunk_bytes)
        except PREPARATION_ERRORS as error:
            records.append(
                {
                    "split": split,
                    STUDY: study,
                    SERIES: series,
                    "file": path.name,
                    "error": repr(error),
                }
            )
            continue
        pose = header.pose
        tags = header.tags
        records.append(
            {
                "split": split,
                STUDY: study,
                SERIES: series,
                "file": path.name,
                "size": info.st_size,
                "native": native,
                "rows": pose.layout.rows if pose else None,
                "columns": pose.layout.columns if pose else None,
                "position": list(pose.position) if pose else None,
                "orientation": list(pose.layout.orientation) if pose else None,
                "spacing": list(pose.layout.spacing) if pose else None,
                "physical_position": header.physical_position,
                "series_number": tags.series_number,
                "description": tags.description,
                "image_type": "\\".join(tags.image_type),
                "repetition_time": tags.repetition_time,
                "echo_time": tags.echo_time,
                "chunk_hash": hashlib.blake2b(chunk, digest_size=16).hexdigest(),
                "chunk_constant": len(set(chunk)) <= 1,
                "error": None,
            }
        )
    return records


def scan(settings: Settings) -> tuple[pl.DataFrame, pl.DataFrame]:
    inventory = []
    for split in settings.splits:
        frame = pl.read_csv(settings.data_root / f"{split}_series.csv").with_columns(
            pl.lit(split).alias("split")
        )
        if settings.studies is not None:
            keep = frame[STUDY].unique().sort().head(settings.studies)
            frame = frame.filter(pl.col(STUDY).is_in(keep.implode()))
        inventory.append(frame)
    inventory = pl.concat(inventory)
    tasks = [
        (
            row["split"],
            row[STUDY],
            row[SERIES],
            settings.data_root / f"{row['split']}_series" / row[STUDY] / row[SERIES],
            settings.chunk_bytes,
        )
        for row in inventory.iter_rows(named=True)
    ]
    rows = []
    with Pool(settings.workers) as pool:
        for index, records in enumerate(
            pool.imap_unordered(scan_series, tasks, chunksize=8), 1
        ):
            rows.extend(records)
            if index % 1000 == 0:
                print(
                    f"scanned {index}/{len(tasks)} series", file=sys.stderr, flush=True
                )
    return inventory, pl.from_dicts(rows, infer_schema_length=None)


def summarize_series(files: pl.DataFrame, inventory: pl.DataFrame) -> pl.DataFrame:
    valid = files.filter(pl.col("error").is_null())
    return (
        valid.group_by(["split", STUDY, SERIES])
        .agg(
            pl.len().alias("files"),
            (~pl.col("chunk_constant")).sum().alias("informative"),
            pl.col("native").all(),
            pl.col("chunk_hash").n_unique().alias("distinct_chunks"),
            pl.col("series_number").mode().first(),
            pl.col("description").mode().first(),
            pl.col("image_type").mode().first(),
            pl.col("repetition_time").mode().first(),
            pl.col("echo_time").mode().first(),
            pl.col("rows").mode().first(),
            pl.col("columns").mode().first(),
            pl.col("position")
            .list.eval(pl.element().round(2).cast(pl.String))
            .list.join(",")
            .sort()
            .str.join(";")
            .hash()
            .alias("geometry_hash"),
        )
        .join(
            files.filter(pl.col("error").is_not_null())
            .group_by(SERIES)
            .agg(pl.len().alias("errors")),
            on=SERIES,
            how="left",
        )
        .with_columns(pl.col("errors").fill_null(0))
        .join(inventory.select([SERIES, *TYPE]), on=SERIES)
    )


def shared_pairs(files: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    informative = (
        files.filter(pl.col("error").is_null() & ~pl.col("chunk_constant"))
        .select([STUDY, SERIES, "chunk_hash"])
        .unique([SERIES, "chunk_hash"])
    )
    left = informative.rename({STUDY: "study_a", SERIES: "series_a"})
    right = informative.rename({STUDY: "study_b", SERIES: "series_b"})
    counts = (
        left.join(right, on="chunk_hash")
        .filter(pl.col("series_a") < pl.col("series_b"))
        .group_by(["study_a", "series_a", "study_b", "series_b"])
        .agg(pl.len().alias("shared"))
    )
    columns = [
        SERIES,
        "distinct_chunks",
        "files",
        "series_number",
        "description",
        "repetition_time",
        "echo_time",
        "geometry_hash",
        *TYPE,
    ]
    distinct = (
        files.filter(pl.col("error").is_null() & ~pl.col("chunk_constant"))
        .group_by(SERIES)
        .agg(pl.col("chunk_hash").n_unique().alias("informative_distinct"))
    )
    side = series.select(columns).join(distinct, on=SERIES)
    return (
        counts.join(
            side.rename(lambda name: f"{name}_a"),
            left_on="series_a",
            right_on=f"{SERIES}_a",
        )
        .join(
            side.rename(lambda name: f"{name}_b"),
            left_on="series_b",
            right_on=f"{SERIES}_b",
        )
        .with_columns(
            (pl.col("study_a") == pl.col("study_b")).alias("same_study"),
            pl.when(
                (pl.col("shared") == pl.col("informative_distinct_a"))
                & (pl.col("shared") == pl.col("informative_distinct_b"))
            )
            .then(pl.lit("identical"))
            .when(
                (pl.col("shared") == pl.col("informative_distinct_a"))
                | (pl.col("shared") == pl.col("informative_distinct_b"))
            )
            .then(pl.lit("subset"))
            .otherwise(pl.lit("partial"))
            .alias("relation"),
            pl.all_horizontal(
                pl.col(f"{name}_a") == pl.col(f"{name}_b") for name in TYPE
            ).alias("same_type"),
        )
        .sort(["same_study", "relation", "shared"], descending=[True, False, True])
    )


def verify(files: pl.DataFrame, pairs: pl.DataFrame, data_root: Path) -> pl.DataFrame:
    located = files.filter(
        pl.col("error").is_null() & ~pl.col("chunk_constant")
    ).select(["split", STUDY, SERIES, "file", "chunk_hash"])
    results = []
    for pair in pairs.iter_rows(named=True):
        a = located.filter(pl.col(SERIES) == pair["series_a"])
        b = located.filter(pl.col(SERIES) == pair["series_b"])
        matches = a.join(b, on="chunk_hash", suffix="_b")
        equal = 0
        compared = 0
        for row in matches.iter_rows(named=True):
            first = (
                data_root
                / f"{row['split']}_series"
                / row[STUDY]
                / row[SERIES]
                / row["file"]
            )
            second = (
                data_root
                / f"{row['split_b']}_series"
                / row[f"{STUDY}_b"]
                / row[f"{SERIES}_b"]
                / row["file_b"]
            )
            left = read_frame(first, read_header(first).descriptor).pixels
            right = read_frame(second, read_header(second).descriptor).pixels
            compared += 1
            equal += int(left.shape == right.shape and np.array_equal(left, right))
        results.append(
            {
                "series_a": pair["series_a"],
                "series_b": pair["series_b"],
                "compared": compared,
                "pixel_equal": equal,
            }
        )
    return pl.from_dicts(
        results,
        schema={
            "series_a": pl.String,
            "series_b": pl.String,
            "compared": pl.Int64,
            "pixel_equal": pl.Int64,
        },
    )


def header_rule(series: pl.DataFrame, pairs: pl.DataFrame) -> dict:
    keys = [
        "geometry_hash",
        "files",
        "rows",
        "columns",
        "repetition_time",
        "echo_time",
    ]
    candidates = (
        series.filter(pl.col("files") > 0)
        .join(series, on=[STUDY, *keys], suffix="_b")
        .filter(pl.col(SERIES) < pl.col(f"{SERIES}_b"))
        .select(
            pl.col(SERIES).alias("series_a"), pl.col(f"{SERIES}_b").alias("series_b")
        )
    )
    identical = pairs.filter(pl.col("same_study") & (pl.col("relation") == "identical"))
    hits = candidates.join(
        identical.select(["series_a", "series_b"]), on=["series_a", "series_b"]
    )
    return {
        "keys": keys,
        "candidate_pairs": candidates.height,
        "pixel_identical_pairs": identical.height,
        "identical_found_by_rule": hits.height,
        "rule_false_positives": candidates.height - hits.height,
    }


def run(settings: Settings) -> dict:
    if settings.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {settings.output_dir}")
    settings.output_dir.mkdir(parents=True)
    started = datetime.now(UTC)
    inventory, files = scan(settings)
    files.write_parquet(settings.output_dir / "files.parquet")
    series = summarize_series(files, inventory)
    series.write_parquet(settings.output_dir / "series.parquet")
    pairs = shared_pairs(files, series)
    checked = verify(files, pairs, settings.data_root)
    pairs = pairs.join(checked, on=["series_a", "series_b"])
    pairs.write_parquet(settings.output_dir / "pairs.parquet")
    within = (
        files.filter(pl.col("error").is_null() & ~pl.col("chunk_constant"))
        .group_by([STUDY, SERIES, "chunk_hash"])
        .agg(pl.len().alias("copies"))
        .filter(pl.col("copies") > 1)
        .group_by([STUDY, SERIES])
        .agg(pl.len().alias("repeated_chunks"), pl.col("copies").sum())
    )
    within.write_parquet(settings.output_dir / "within_series.parquet")
    same_study = pairs.filter(pl.col("same_study"))
    summary = {
        "audited_at": started.isoformat(),
        "elapsed_seconds": round((datetime.now(UTC) - started).total_seconds(), 1),
        "config": json.loads(settings.model_dump_json()),
        "files": files.height,
        "file_errors": int(files["error"].is_not_null().sum()),
        "non_native_files": int((~files["native"].fill_null(True)).sum()),
        "constant_chunks": int(files["chunk_constant"].fill_null(False).sum()),
        "series": series.height,
        "pairs_sharing_chunks": {
            "same_study": same_study.group_by("relation")
            .len()
            .sort("relation")
            .to_dicts(),
            "cross_study": pairs.filter(~pl.col("same_study"))
            .group_by("relation")
            .len()
            .sort("relation")
            .to_dicts(),
        },
        "same_study_identical_by_type": same_study.filter(
            pl.col("relation") == "identical"
        )
        .group_by("same_type")
        .len()
        .to_dicts(),
        "verified_pixel_equal_fraction": float(
            pairs["pixel_equal"].sum() / max(pairs["compared"].sum(), 1)
        ),
        "studies_with_identical_series": same_study.filter(
            pl.col("relation") == "identical"
        )["study_a"].n_unique(),
        "series_with_repeated_slices": within.height,
        "header_rule": header_rule(series, pairs),
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    source = Path(__file__).resolve()
    snapshot = copy_sources([source], settings.output_dir / "source")
    outputs = [
        "files.parquet",
        "series.parquet",
        "pairs.parquet",
        "within_series.parquet",
        "summary.json",
    ]
    manifest = {
        "created_at": summary["audited_at"],
        "inputs": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in [
                *(
                    settings.data_root / f"{split}_series.csv"
                    for split in settings.splits
                ),
                source,
                Path("src/knee_abnormality/data/preprocessing/dicom.py"),
                Path("uv.lock"),
            ]
        },
        "outputs": {
            name: {
                "sha256": sha256_file(settings.output_dir / name),
                "bytes": (settings.output_dir / name).stat().st_size,
            }
            for name in outputs
        },
        "source_snapshot": snapshot,
        "config": summary["config"],
    }
    (settings.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return summary


def main() -> None:
    try:
        summary = run(Settings())
    except (OSError, ValueError, SettingsError) as error:
        print(f"Repeated-series audit error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
