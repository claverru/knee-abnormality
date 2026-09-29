import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsError
from torch.utils.data import Subset

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import SLICES_PER_INPUT, load_config
from knee_abnormality.data.dataset import (
    KneeStudyDataset,
    build_data_loader,
    collate_studies,
)
from knee_abnormality.data.preprocessing import series as series_module
from knee_abnormality.data.preprocessing.dicom import PREFIX_BYTES, pydicom_read_header
from knee_abnormality.data.splits import STUDY
from knee_abnormality.profiling import (
    finish_profile,
    profile_manifest,
    sample_dataset,
    setup_profile,
    write_profile_manifest,
)

ARMS = ("pydicom", "parser", "threaded")


class PydicomStudyDataset(KneeStudyDataset):
    def _indexed(self, candidate):
        original = series_module.parse_prefix
        series_module.parse_prefix = lambda path, prefix: pydicom_read_header(path)
        try:
            return super()._indexed(candidate)
        finally:
            series_module.parse_prefix = original


class CompareCLI(CLISettings):
    config: CliPositionalArg[Path]
    output_dir: Path
    mode: Literal["loader", "audit"] = "loader"
    implementation: Literal["both", "pydicom", "parser", "threaded"] = "both"
    samples: int = Field(default=256, ge=8, le=1024)
    workers: int | None = Field(default=None, ge=0, le=16)
    batch_size: int | None = Field(default=None, ge=1, le=64)
    threads: int = Field(default=4, ge=1, le=16)
    passes: int = Field(default=2, ge=1, le=3)
    prewarm: bool = True
    main_threads: int = Field(default=16, ge=1, le=16)


def build_arm(
    source: KneeStudyDataset, implementation: str, threads: int
) -> KneeStudyDataset:
    arguments = (
        source.studies,
        source.series_index,
        source.data,
        source.targets,
        source.training_seed,
        source.image_limit_multiplier,
    )
    if implementation == "pydicom":
        dataset = PydicomStudyDataset(*arguments, 1)
    else:
        dataset = KneeStudyDataset(
            *arguments, threads if implementation == "threaded" else 1
        )
    dataset.set_epoch(source.epoch.value)
    return dataset


def series_paths(dataset: KneeStudyDataset) -> list[Path]:
    paths = []
    for uid in dataset.studies[STUDY].to_list():
        for values in dataset.series_index[uid].slots:
            for candidate in values:
                paths.extend(sorted(candidate.directory.glob("*.dcm")))
    return paths


def prewarm(paths: list[Path]) -> dict:
    started = time.perf_counter()
    total = 0
    for path in paths:
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1 << 20):
                    total += len(chunk)
        except OSError:
            continue
    return {
        "files": len(paths),
        "megabytes": total / 1e6,
        "seconds": time.perf_counter() - started,
    }


def page_cache_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("Cached:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def batch_digest(batch) -> dict:
    fields = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            fields[key] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            }
        else:
            fields[key] = value
    return fields


def measure(config, source, args, implementation: str, size: int, order: list) -> dict:
    loader = build_data_loader(
        Subset(build_arm(source, implementation, args.threads), order),
        config.trainer,
        size,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    passes = []
    for repeat in range(args.passes):
        started = time.perf_counter()
        iterator = iter(loader)
        iterator_seconds = time.perf_counter() - started
        rows = []
        for step in range(len(loader)):
            batch_started = time.perf_counter()
            batch = next(iterator)
            rows.append(
                {
                    "batch": step,
                    "seconds": time.perf_counter() - batch_started,
                    "studies": batch["study_uid"],
                    "slices": int(batch["slice_mask"].sum()) * SLICES_PER_INPUT,
                }
            )
        wall = time.perf_counter() - started
        summary = {
            "implementation": implementation,
            "pass": repeat,
            "wall_seconds": wall,
            "iterator_seconds": iterator_seconds,
            "studies_per_second": len(rows) * size / wall,
            "slices_per_second": sum(row["slices"] for row in rows) / wall,
            "first_batch_seconds": rows[0]["seconds"],
            "median_batch_seconds": float(np.median([row["seconds"] for row in rows])),
            "batches": len(rows),
        }
        print(json.dumps(summary), flush=True)
        passes.append({**summary, "batch_rows": rows})
    return {"implementation": implementation, "passes": passes}


def audit(source, args) -> dict:
    torch.set_num_threads(1)
    results = {}
    for implementation in ARMS:
        dataset = build_arm(source, implementation, args.threads)
        started = time.perf_counter()
        digests = [
            batch_digest(collate_studies([dataset[index]]))
            for index in range(len(dataset))
        ]
        results[implementation] = {
            "seconds": time.perf_counter() - started,
            "studies": digests,
        }
        print(
            json.dumps(
                {
                    "implementation": implementation,
                    "seconds": results[implementation]["seconds"],
                }
            ),
            flush=True,
        )
    reference = results["pydicom"]["studies"]
    matches = {name: value["studies"] == reference for name, value in results.items()}
    differing = sorted(name for name, ok in matches.items() if not ok)
    if differing:
        raise RuntimeError(f"Preparation differs from pydicom: {differing}")
    return {
        "audit": {
            "matches_pydicom": matches,
            "studies": len(reference),
            "seconds": {k: v["seconds"] for k, v in results.items()},
        }
    }


def run(args: CompareCLI) -> None:
    config = load_config(args.config)
    allocator, snapshot = setup_profile(
        config,
        args.output_dir,
        args.main_threads,
        [Path(__file__), args.config],
    )
    size = args.batch_size or config.trainer.batch_size
    started = time.perf_counter()
    dataset, labels = sample_dataset(config, args.samples)
    if args.mode == "loader" and args.samples % size:
        raise ValueError(f"--samples must be a multiple of the batch size {size}")
    result = profile_manifest(
        config,
        args.model_dump(mode="json"),
        allocator,
        snapshot,
        labels,
        time.perf_counter() - started,
        {
            "study_ids": dataset.studies[STUDY].to_list(),
            "start_method": multiprocessing.get_start_method(),
            "cpu_count": os.cpu_count(),
            "batch_size": size,
            "workers": args.workers
            if args.workers is not None
            else config.trainer.num_workers,
            "prefix_bytes": PREFIX_BYTES,
            "page_cache_kb": page_cache_kb(),
        },
    )
    write_profile_manifest(args.output_dir, result)
    if args.prewarm:
        result["prewarm"] = prewarm(series_paths(dataset))
        print(json.dumps(result["prewarm"]), flush=True)
    if args.mode == "audit":
        result.update(audit(dataset, args))
    else:
        order = np.random.default_rng(config.seed).permutation(len(dataset)).tolist()
        chosen = ARMS if args.implementation == "both" else (args.implementation,)
        arms = [
            measure(config, dataset, args, implementation, size, order)
            for implementation in chosen
        ]
        result["arms"] = arms
        if len(arms) > 1:
            baseline = arms[0]["passes"][0]["studies_per_second"]
            result["speedup_first_pass"] = {
                arm["implementation"]: arm["passes"][0]["studies_per_second"] / baseline
                for arm in arms
            }
            print(json.dumps({"speedup_first_pass": result["speedup_first_pass"]}))
    finish_profile(args.output_dir, result, started)


def main() -> None:
    try:
        run(CompareCLI())
    except (OSError, RuntimeError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Header comparison error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
