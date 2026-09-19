import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat as repeat_value
from multiprocessing.util import Finalize
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
from knee_abnormality.data.labels import STUDY
from knee_abnormality.data.preprocessing import preparation as preparation_module
from knee_abnormality.data.preprocessing.dicom import (
    DecodedFrame,
    read_frame,
)
from knee_abnormality.data.preprocessing.preparation import decode_slice
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.modeling.model import KneeAbnormalityModel
from knee_abnormality.profiling import (
    finish_profile,
    profile_manifest,
    sample_dataset,
    setup_profile,
    training_step,
    write_profile_manifest,
)
from knee_abnormality.training.lightning import (
    build_optimizer,
    fixed_class_balance,
    validated_class_balance_weights,
)


class ProfileCLI(CLISettings):
    config: CliPositionalArg[Path]
    output_dir: Path
    mode: Literal["audit", "loader", "gpu"] = "loader"
    samples: int = Field(default=256, ge=8, le=1024)
    workers: int = Field(default=8, ge=1, le=16)
    decode_threads: int = Field(default=1, ge=1, le=4)
    read_threads: int = Field(default=1, ge=1, le=16)
    passes: int = Field(default=2, ge=1, le=3)
    warmup_batches: int = Field(default=8, ge=1)
    main_threads: int = Field(default=16, ge=1, le=16)


def read_payload(item):
    path, descriptor = item
    try:
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                (info.st_size, info.st_mtime_ns)
                != (descriptor.file_size, descriptor.modified_ns)
                or descriptor.offset < 0
                or descriptor.length != descriptor.rows * descriptor.columns * 2
                or descriptor.offset + descriptor.length != info.st_size
            ):
                return path, None
            stream.seek(descriptor.offset)
            payload = stream.read(descriptor.length)
    except OSError:
        return path, None
    return path, payload if len(payload) == descriptor.length else None


def frame_from_payload(payload, descriptor):
    pixels = np.frombuffer(payload, dtype="<i2" if descriptor.signed else "<u2")
    shift = 16 - descriptor.bits_stored
    if shift:
        pixels = np.left_shift(pixels, shift)
        np.right_shift(pixels, shift, out=pixels)
    pixels = pixels.reshape(descriptor.rows, descriptor.columns).astype(np.float32)
    return DecodedFrame(
        pixels,
        descriptor.row_spacing,
        descriptor.column_spacing,
        descriptor.slope,
        descriptor.intercept,
    )


class ThreadedStudyDataset(KneeStudyDataset):
    def __init__(self, source, decode_threads):
        super().__init__(
            source.studies,
            source.series_index,
            source.data,
            source.targets,
            source.training_seed,
            source.image_limit_multiplier,
        )
        self.set_epoch(source.epoch.value)
        self.decode_threads = decode_threads
        self.read_threads = 1
        self._pool = None
        self._finalizer = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pool"] = None
        state["_finalizer"] = None
        return state

    def close(self):
        if self._finalizer is not None:
            self._finalizer()
        self._finalizer = None
        self._pool = None

    def _read_threaded(self, records, spacing_mm, image_size):
        wanted = {
            record.path: record.descriptor
            for record in records
            if record.descriptor is not None
        }
        payloads = {}
        if wanted:
            with ThreadPoolExecutor(min(self.read_threads, len(wanted))) as pool:
                for path, payload in pool.map(read_payload, wanted.items()):
                    if payload is not None:
                        payloads[path] = payload

        def patched(path, descriptor):
            payload = payloads.pop(path, None)
            if payload is None or descriptor is None:
                return read_frame(path, descriptor)
            return frame_from_payload(payload, descriptor)

        original = preparation_module.read_frame
        preparation_module.read_frame = patched
        try:
            return KneeStudyDataset._decode_series(
                self, records, spacing_mm, image_size
            )
        finally:
            preparation_module.read_frame = original

    def _decode_series(self, records, spacing_mm, image_size):
        if self.read_threads > 1:
            return self._read_threaded(records, spacing_mm, image_size)
        if self.decode_threads == 1:
            return super()._decode_series(records, spacing_mm, image_size)
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.decode_threads,
                initializer=torch.set_num_threads,
                initargs=(1,),
            )
            self._finalizer = Finalize(
                self, self._pool.shutdown, args=(True,), exitpriority=10
            )
        return tuple(
            self._pool.map(
                decode_slice,
                records,
                repeat_value(spacing_mm),
                repeat_value(image_size),
            )
        )


def page_cache_kb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("Cached:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def batch_digest(batch):
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


def audit(source, args):
    torch.set_num_threads(1)
    rows = []
    reference = None
    for decode_threads, read_threads in ((1, 1), (2, 1), (4, 1), (1, 4), (1, 16)):
        dataset = ThreadedStudyDataset(source, decode_threads)
        dataset.read_threads = read_threads
        started = time.perf_counter()
        digests = []
        try:
            for index in range(len(dataset)):
                digests.append(batch_digest(collate_studies([dataset[index]])))
        finally:
            dataset.close()
        if reference is None:
            reference = digests
        row = {
            "decode_threads": decode_threads,
            "read_threads": read_threads,
            "seconds": time.perf_counter() - started,
            "matches_production": digests == reference,
            "studies": digests,
        }
        rows.append(row)
        (args.output_dir / "audit_progress.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k != "studies"}), flush=True)
        if not row["matches_production"]:
            raise RuntimeError("Threaded preparation differs from production")
    return {"audit": rows}


def profile(config, source, args, class_balance: dict | None):
    size = config.trainer.batch_size
    if args.samples % size or args.samples // size <= args.warmup_batches + 16:
        raise ValueError("Use whole batches and more than 16 measured batches")
    dataset = ThreadedStudyDataset(source, args.decode_threads)
    dataset.read_threads = args.read_threads
    order = np.random.default_rng(config.seed).permutation(len(dataset)).tolist()
    loader = build_data_loader(
        Subset(dataset, order),
        config.trainer,
        size,
        num_workers=args.workers,
        pin_memory=args.mode == "gpu",
        persistent_workers=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    started = time.perf_counter()
    iterator = iter(loader)
    iterator_seconds = time.perf_counter() - started
    if args.mode == "gpu":
        class_balance_weights = validated_class_balance_weights(
            config, dataset.targets, class_balance
        )
        if class_balance_weights is not None:
            class_balance_weights = class_balance_weights.to("cuda")
        model = (
            KneeAbnormalityModel(
                config.model, config.data, len(dataset.targets), load_pretrained=False
            )
            .cuda()
            .train()
        )
        optimizer = build_optimizer(model, config.optimizer)
        torch.cuda.reset_peak_memory_stats()
    passes = []
    for repeat in range(args.passes):
        pass_started = (
            started if repeat == 0 and args.mode == "loader" else time.perf_counter()
        )
        if repeat:
            iterator = iter(loader)
        rows = []
        for step in range(len(loader)):
            if step == args.warmup_batches:
                measured_started = time.perf_counter()
            if args.mode == "gpu":
                torch.cuda.synchronize()
            started = time.perf_counter()
            batch = next(iterator)
            wait = time.perf_counter() - started
            if args.mode == "gpu":
                batch = move_batch(batch, torch.device("cuda"))
                loss = training_step(
                    model, optimizer, batch, config, class_balance_weights
                )
                torch.cuda.synchronize()
                if not math.isfinite(loss):
                    raise RuntimeError("Non-finite profiling loss")
            else:
                loss = None
            duration = time.perf_counter() - started
            rows.append(
                {
                    "batch": step,
                    "seconds": duration,
                    "loader_wait_seconds": wait,
                    "loss": loss,
                    "studies": batch["study_uid"],
                    "inputs": int(batch["slice_mask"].sum()),
                    "slices": int(batch["slice_mask"].sum()) * SLICES_PER_INPUT,
                }
            )
            if (step + 1) % 8 == 0:
                print(
                    json.dumps({"pass": repeat, "completed_batches": step + 1}),
                    flush=True,
                )
        ended = time.perf_counter()
        measured = rows[args.warmup_batches :]
        seconds = ended - measured_started
        waits = [row["loader_wait_seconds"] for row in measured]
        summary = {
            "pass": repeat,
            "studies_per_second": len(measured) * size / seconds,
            "slices_per_second": sum(row["slices"] for row in measured) / seconds,
            "all_studies_per_second": len(dataset) / (ended - pass_started),
            "pass_wall_seconds": ended - pass_started,
            "loader_wait_fraction": sum(waits) / seconds,
            "loader_wait_mean_seconds": float(np.mean(waits)),
            "loader_wait_p95_seconds": float(np.quantile(waits, 0.95)),
            "all_batch_seconds": sum(row["seconds"] for row in rows),
            "measured_seconds": seconds,
            "measured_studies": len(measured) * size,
        }
        if args.mode == "gpu":
            summary["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            summary["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        print(json.dumps(summary), flush=True)
        passes.append({**summary, "batches": rows})
        (args.output_dir / "progress.json").write_text(json.dumps(passes, indent=2))
    return {
        "passes": passes,
        "iterator_seconds": iterator_seconds,
        "order": order,
        "class_balance": class_balance,
    }


def run(args: ProfileCLI) -> None:
    config = load_config(args.config)
    allocator, snapshot = setup_profile(
        config,
        args.output_dir,
        args.main_threads,
        [
            Path(__file__),
            Path("scripts/adhoc/profile_training.py"),
            args.config,
        ],
    )
    started = time.perf_counter()
    dataset, labels = sample_dataset(config, args.samples)
    class_balance = (
        fixed_class_balance(dataset.studies, dataset.targets)
        if config.trainer.class_balanced_bce and args.mode == "gpu"
        else None
    )
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
            "thread_environment": {
                name: os.getenv(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "epoch": 0,
            "page_cache_kb": page_cache_kb(),
            "class_balance": class_balance,
            "weights": "seeded random; no checkpoint saved"
            if args.mode == "gpu"
            else None,
        },
    )
    write_profile_manifest(args.output_dir, result)
    result.update(
        audit(dataset, args)
        if args.mode == "audit"
        else profile(config, dataset, args, class_balance)
    )
    finish_profile(args.output_dir, result, started)


def main() -> None:
    try:
        run(ProfileCLI())
    except (OSError, RuntimeError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Threaded loading profile error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
