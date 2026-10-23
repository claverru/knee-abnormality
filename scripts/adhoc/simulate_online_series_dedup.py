import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.preprocessing.dicom import PREPARATION_ERRORS, read_frame
from knee_abnormality.data.preprocessing.series import index_series
from knee_abnormality.data.sources import SERIES
from knee_abnormality.data.splits import STUDY
from knee_abnormality.provenance import copy_sources, sha256_file

TYPE = ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]


class Settings(CLISettings):
    data_root: Path = Path("data/raw")
    split: str = "train"
    ground_truth: Path = Path(
        "outputs/analysis/r/repeated-series-20260922/pairs.parquet"
    )
    output_dir: Path = Field(description="Fresh output directory for the simulation")
    workers: int = 12
    studies: int | None = None


def _shape(indexed) -> tuple[int, int] | None:
    for record in indexed.records:
        if record.descriptor is not None:
            return record.descriptor.rows, record.descriptor.columns
        if record.pose is not None:
            return record.pose.layout.rows, record.pose.layout.columns
    return None


def _frame(indexed) -> np.ndarray:
    record = indexed.records[len(indexed.records) // 2]
    return read_frame(record.path, record.descriptor).pixels


def _compare(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        return {"equal": False, "correlation": None, "equal_fraction": None}
    a = left.astype(np.float64).ravel()
    b = right.astype(np.float64).ravel()
    spread = a.std() * b.std()
    return {
        "equal": bool(np.array_equal(left, right)),
        "correlation": float(np.mean((a - a.mean()) * (b - b.mean())) / spread)
        if spread > 0
        else None,
        "equal_fraction": float(np.mean(left == right)),
    }


def simulate_study(task: tuple[str, Path, list[tuple[str, str]]]) -> dict:
    study, directory, members = task
    started = time.perf_counter()
    indexed = {}
    for uid, slot in members:
        try:
            indexed[uid] = (index_series(directory / uid), slot)
        except PREPARATION_ERRORS:
            continue
    indexing = time.perf_counter() - started
    started = time.perf_counter()
    strict = defaultdict(list)
    loose = defaultdict(list)
    for uid, (value, slot) in indexed.items():
        if not value.records:
            continue
        count = len(value.records)
        shape = _shape(value)
        positions = tuple(
            round(record.physical_position, 2) for record in value.records
        )
        strict[(slot, count, shape, positions)].append(uid)
        loose[(count, shape)].append(uid)
    strict_members = {
        uid for group in strict.values() if len(group) > 1 for uid in group
    }
    keying = time.perf_counter() - started
    started = time.perf_counter()
    frames = {uid: _frame(indexed[uid][0]) for uid in sorted(strict_members)}
    strict_reads = time.perf_counter() - started
    strict_pairs = {
        tuple(sorted(pair))
        for group in strict.values()
        for pair in combinations(group, 2)
    }
    pairs = []
    for group in loose.values():
        for a, b in combinations(sorted(group), 2):
            for uid in (a, b):
                if uid not in frames:
                    frames[uid] = _frame(indexed[uid][0])
            pairs.append(
                {
                    STUDY: study,
                    "series_a": a,
                    "series_b": b,
                    "same_slot": indexed[a][1] == indexed[b][1],
                    "strict_candidate": (a, b) in strict_pairs,
                    "series_number_a": indexed[a][0].summary.series_number,
                    "series_number_b": indexed[b][0].summary.series_number,
                    "signature_equal": indexed[a][0].summary.signature
                    == indexed[b][0].summary.signature,
                    **_compare(frames[a], frames[b]),
                }
            )
    return {
        "study": {
            STUDY: study,
            "series": len(indexed),
            "slices": sum(len(value.records) for value, _ in indexed.values()),
            "indexing_seconds": indexing,
            "keying_seconds": keying,
            "strict_frames": len(strict_members),
            "strict_read_seconds": strict_reads,
            "loose_frames": len(
                {uid for group in loose.values() if len(group) > 1 for uid in group}
            ),
        },
        "pairs": pairs,
    }


def run(settings: Settings) -> dict:
    if settings.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {settings.output_dir}")
    settings.output_dir.mkdir(parents=True)
    started = datetime.now(UTC)
    inventory = pl.read_csv(
        settings.data_root / f"{settings.split}_series.csv"
    ).with_columns(pl.concat_str(TYPE, separator="/").alias("slot"))
    studies = inventory[STUDY].unique().sort()
    if settings.studies is not None:
        studies = studies.head(settings.studies)
    tasks = [
        (
            study,
            settings.data_root / f"{settings.split}_series" / study,
            list(group.select(SERIES, "slot").iter_rows()),
        )
        for (study,), group in inventory.filter(
            pl.col(STUDY).is_in(studies.implode())
        ).group_by(STUDY)
    ]
    study_rows, pair_rows = [], []
    with Pool(settings.workers) as pool:
        for index, result in enumerate(pool.imap_unordered(simulate_study, tasks), 1):
            study_rows.append(result["study"])
            pair_rows.extend(result["pairs"])
            if index % 500 == 0:
                print(
                    f"simulated {index}/{len(tasks)} studies",
                    file=sys.stderr,
                    flush=True,
                )
    per_study = pl.from_dicts(study_rows)
    pairs = pl.from_dicts(pair_rows, infer_schema_length=None)
    truth = (
        pl.read_parquet(settings.ground_truth)
        .filter(
            pl.col("same_study")
            & (pl.col("relation") == "identical")
            & pl.col("study_a").is_in(per_study[STUDY].implode())
        )
        .select(
            pl.min_horizontal("series_a", "series_b").alias("series_a"),
            pl.max_horizontal("series_a", "series_b").alias("series_b"),
            pl.lit(True).alias("truth"),
        )
    )
    pairs = pairs.join(truth, on=["series_a", "series_b"], how="left").with_columns(
        pl.col("truth").fill_null(False),
        (pl.col("strict_candidate") & pl.col("equal")).alias("online_duplicate"),
    )
    per_study.write_parquet(settings.output_dir / "studies.parquet")
    pairs.write_parquet(settings.output_dir / "pairs.parquet")
    online = pairs.filter(pl.col("online_duplicate"))
    truth_in_scope = truth.join(
        pairs.select("series_a", "series_b"), on=["series_a", "series_b"], how="semi"
    )
    strict = pairs.filter(pl.col("strict_candidate"))
    loose_not_strict = pairs.filter(~pl.col("strict_candidate"))
    summary = {
        "simulated_at": started.isoformat(),
        "config": json.loads(settings.model_dump_json()),
        "studies": per_study.height,
        "series": int(per_study["series"].sum()),
        "slices": int(per_study["slices"].sum()),
        "accuracy": {
            "ground_truth_pairs": truth.height,
            "ground_truth_pairs_in_loose_buckets": truth_in_scope.height,
            "online_duplicates": online.height,
            "true_positives": int(online["truth"].sum()),
            "false_positives": int((~online["truth"]).sum()),
            "missed": truth.height - int(online["truth"].sum()),
            "equal_outside_strict_key": int(loose_not_strict["equal"].sum()),
        },
        "cost": {
            "studies_needing_reads": int((per_study["strict_frames"] > 0).sum()),
            "strict_frames": int(per_study["strict_frames"].sum()),
            "loose_frames": int(per_study["loose_frames"].sum()),
            "mean_frames_per_study": float(per_study["strict_frames"].mean()),
            "max_frames_per_study": int(per_study["strict_frames"].max()),
            "indexing_seconds_per_study": float(per_study["indexing_seconds"].mean()),
            "keying_ms_per_study": 1000 * float(per_study["keying_seconds"].mean()),
            "strict_read_ms_per_read": 1000
            * float(
                per_study["strict_read_seconds"].sum()
                / max(per_study["strict_frames"].sum(), 1)
            ),
        },
        "strict_non_equal_correlation": strict.filter(~pl.col("equal"))
        .select(
            pl.col("correlation").max().alias("max"),
            pl.col("correlation").quantile(0.99).alias("p99"),
            pl.col("correlation").median().alias("median"),
            pl.col("equal_fraction").max().alias("max_equal_fraction"),
        )
        .to_dicts()[0],
        "strict_non_equal_by_correlation": strict.filter(~pl.col("equal"))
        .with_columns(
            pl.col("correlation")
            .cut([0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
            .alias("band")
        )
        .group_by("band")
        .len()
        .sort("band")
        .with_columns(pl.col("band").cast(pl.String))
        .to_dicts(),
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    source = Path(__file__).resolve()
    snapshot = copy_sources([source], settings.output_dir / "source")
    outputs = ["studies.parquet", "pairs.parquet", "summary.json"]
    manifest = {
        "created_at": summary["simulated_at"],
        "inputs": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in [
                settings.data_root / f"{settings.split}_series.csv",
                settings.ground_truth,
                source,
                Path("src/knee_abnormality/data/preprocessing/series.py"),
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
        print(f"Online series dedup simulation error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
