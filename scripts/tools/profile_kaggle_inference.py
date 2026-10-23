import json
import random
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from kaggle.api.kaggle_api_extended import KaggleApi
from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliSubCommand
from requests import HTTPError

from knee_abnormality.cli import CLISettings, command, describe
from knee_abnormality.data.splits import STUDY
from knee_abnormality.evaluation.metrics import validation_metrics
from knee_abnormality.inference.bundle import verify_bundle
from knee_abnormality.kaggle import COMPETITION, KaggleId, NotebookRun
from knee_abnormality.provenance import sha256_file
from scripts.pipeline.submit import (
    _dataset_exists,
    _dataset_identity,
    _jsonable,
    _kernel_status,
)

CONDITIONS = {
    "fp32-b4-w4": {
        "kind": "prediction",
        "precision": "fp32",
        "batch_size": 4,
        "num_workers": 4,
    },
    "fp16-b4-w4": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 4,
        "num_workers": 4,
    },
    "fp16-b2-w8": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 2,
        "num_workers": 8,
    },
    "fp16-b2-w8-compiled": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 2,
        "num_workers": 8,
        "environment": {"KNEE_COMPILE_BACKBONE": "1"},
    },
    **{
        f"paired-{first}-{second}": {
            "kind": "paired",
            "precision": "fp16",
            "batch_size": 4,
            "num_workers": 8,
            "arms": [
                {"name": name, "environment": {"KNEE_BACKBONE_MODE": name}}
                for name in (first, second)
            ],
        }
        for first, second in (("graph", "compiled-graph"), ("compiled-graph", "graph"))
    },
    **{
        f"paired-s167-{order}": {
            "kind": "paired",
            "precision": "fp16",
            "batch_size": 4,
            "num_workers": 8,
            "arms": [
                {
                    "submitted": {
                        "name": "submitted",
                        "environment": {},
                        "bundle_slug": "knee-convnext-silver15-share075-ft-f0-x4-tau085",
                    },
                    "branch": {"name": "branch", "environment": {}},
                    "compiled": {
                        "name": "compiled",
                        "environment": {"KNEE_BACKBONE_MODE": "compiled-graph"},
                    },
                }[name]
                for name in names
            ],
        }
        for order, names in (
            ("abc", ("submitted", "branch", "compiled")),
            ("cba", ("compiled", "branch", "submitted")),
        )
    },
    "fp16-b4-w8": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 4,
        "num_workers": 8,
    },
    "fp16-b4-w16": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 4,
        "num_workers": 16,
    },
    "fp16-b4-w12": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 4,
        "num_workers": 12,
    },
    "fp32-b4-w16": {
        "kind": "prediction",
        "precision": "fp32",
        "batch_size": 4,
        "num_workers": 16,
    },
    "fp32-b4-w24": {
        "kind": "prediction",
        "precision": "fp32",
        "batch_size": 4,
        "num_workers": 24,
    },
    "fp32-b4-w16-prefetch": {
        "kind": "prediction",
        "precision": "fp32",
        "batch_size": 4,
        "num_workers": 16,
        "prefetch_studies": 150,
        "warm_packages": True,
        "pycache_prefix": True,
    },
    "fp32-b4-w8-prefetch": {
        "kind": "prediction",
        "precision": "fp32",
        "batch_size": 4,
        "num_workers": 8,
        "prefetch_studies": 150,
        "warm_packages": True,
        "pycache_prefix": True,
    },
    "fp16-b4-w16-chunks": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 4,
        "num_workers": 16,
    },
    "fp16-b4-w16-warmup": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 4,
        "num_workers": 16,
    },
    "fp16-b4-w16-warmup-main": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 4,
        "num_workers": 16,
    },
    "fp32-b4-w16-control": {
        "kind": "prediction",
        "precision": "fp32",
        "batch_size": 4,
        "num_workers": 16,
    },
    "fp32-b4-w16-list": {
        "kind": "prediction",
        "precision": "fp32",
        "batch_size": 4,
        "num_workers": 16,
        "list_series": True,
    },
    "cpu-fp32-w32-tpuv6e8": {
        "kind": "prediction",
        "precision": "fp32",
        "batch_size": 4,
        "num_workers": 32,
        "device": "cpu",
        "accelerator": "TpuV6E8",
    },
    "fp32-b4-w16-highmem": {
        "kind": "prediction",
        "precision": "fp32",
        "batch_size": 4,
        "num_workers": 16,
        "accelerator": "NvidiaTeslaT4Highmem",
    },
    **{
        f"fp16-b4-w16-io{threads}": {
            "kind": "prediction",
            "precision": "fp16",
            "batch_size": 4,
            "num_workers": 16,
            "io_threads": threads,
        }
        for threads in (1, 2, 4, 8, 16)
    },
    "precision-chunks": {"kind": "precision", "studies": 32},
    "loading": {
        "kind": "loading",
        "timed_studies": 100,
        "throughput_groups": [
            ["header", 1],
            ["header", 4],
            ["header", 16],
            ["header", 64],
            ["bytes", 1],
            ["bytes", 16],
        ],
    },
    "loader-lab": {
        "kind": "loader-lab",
        "timed_studies": 100,
        "throughput_groups": [],
        "rounds": 3,
        "pass_timeout_seconds": 600,
        "targets": [
            ["engine", "build_series_index"],
            ["dataset", "index_series"],
            ["series", "parse_prefix"],
            ["dataset", "clean_series"],
            ["dataset", "estimate_reference"],
            ["dataset", "split_knee_geometry"],
            ["dataset", "resolve_knee_side"],
            ["dataset", "build_pool"],
            ["dataset", "select_triplets"],
            ["dataset", "view_position"],
            ["dataset", "fallback_window_center"],
            ["dataset", "prepare_series"],
            ["dataset", "decode_series"],
            ["preparation", "decode_slice"],
            ["preparation", "shared_empty"],
            ["preparation", "mask_stored_bits"],
            ["preparation", "_descriptor_plan"],
            ["preparation", "_read_rows"],
            ["preparation", "read_frame"],
            ["series", "_safe_prefix"],
            ["Dataset", "_index_study_geometry"],
            ["Dataset", "_slot_sources"],
            ["Dataset", "_study_budgets"],
            ["Dataset", "_prepare_slot"],
        ],
    },
    "startup": {"kind": "startup"},
    "cudnn": {"kind": "cudnn"},
    "mainloop": {"kind": "mainloop", "studies": 64},
    "compile-lab": {
        "kind": "compile-lab",
        "studies": 64,
        "runs": [
            ["graph", "graph", "none"],
            ["default-cold", "default", "default"],
            ["default-warm", "default", "default"],
            ["reduce-cold", "reduce-overhead", "reduce"],
            ["reduce-warm", "reduce-overhead", "reduce"],
        ],
    },
    "compile-families": {
        "kind": "compile-lab",
        "studies": 32,
        "runs": [
            run
            for member, family in enumerate(
                (
                    "convnext",
                    "resnet50",
                    "mambavision",
                    "efficientvit",
                    "focalnet",
                    "nextvit",
                )
            )
            for run in (
                [f"{family}-graph", "graph", "none", member, f"{family}-graph"],
                [f"{family}-compiled", "default", family, member, f"{family}-graph"],
            )
        ],
    },
    "compile-ensemble": {
        "kind": "compile-lab",
        "studies": 32,
        "runs": [
            ["ensemble-graph", "graph", "none", list(range(10)), "ensemble-graph"],
            [
                "ensemble-compiled",
                "default",
                "ensemble",
                list(range(10)),
                "ensemble-graph",
            ],
        ],
    },
    "page-lab": {"kind": "page-lab", "accelerator": "NvidiaTeslaT4"},
    **{
        f"variants-{shift}": {
            "kind": "variants",
            "variants": [
                ("base", "fs", "idle")[(shift + item) % 3] for item in range(3)
            ],
        }
        for shift in range(3)
    },
    "parity": {"kind": "parity", "studies": 300},
    **{
        f"census-{part}": {"kind": "census", "part": part, "parts": 5}
        for part in range(5)
    },
    "fp16-b4-w8-pyspy": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 4,
        "num_workers": 8,
        "pyspy": True,
    },
    "fp16-b4-w8-deadline": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 4,
        "num_workers": 8,
        "deadline": 600,
        "internet": True,
    },
    "fp16-b4-w16-pyspy": {
        "kind": "prediction",
        "precision": "fp16",
        "batch_size": 4,
        "num_workers": 16,
        "pyspy": True,
    },
}
LOADING_SEED = 20260915
TERMINAL = ("complete", "error", "fail", "cancel")


class ProfileCommand(BaseModel):
    output_dir: Path = Field(
        description="Profiling directory shared by launch and analyze"
    )
    fold: int = Field(default=0, ge=0, description="Fold whose studies are profiled")
    target_studies: int | None = Field(
        default=None,
        gt=0,
        description="Profile this many studies, topping up from other folds; unset "
        "profiles the whole fold and scores FP16/FP32 parity",
    )
    split_dir: Path = Field(
        default=Path("data/folds/all_5fold_v3"), description="Fold directory"
    )


class Launch(ProfileCommand, NotebookRun):
    model_config = describe(
        "Upload the bundle and run one Kaggle profiling notebook per runtime condition"
    )

    bundle: Path = Field(description="Ready exported bundle directory")
    dataset: KaggleId = Field(description="Private Kaggle dataset for the bundle")
    conditions: list[str] = Field(
        min_length=1,
        description=f"Runtime conditions to run: {', '.join(CONDITIONS)}",
    )
    notebook_suffix: str = Field(
        default="20260915",
        description="Suffix of the knee-profile-<condition> notebooks",
    )
    concurrency: int = Field(default=2, gt=0, description="Notebooks running at once")
    reference_dataset: KaggleId | None = Field(
        default=None, description="Extra Kaggle dataset attached to every notebook"
    )

    def cli_cmd(self) -> None:
        print(json.dumps(launch(self), indent=2))


class Analyze(ProfileCommand):
    model_config = describe(
        "Score parity and break down startup, loader, transfer, forward and DICOM "
        "loading time from the finished notebooks"
    )

    reference_predictions: Path = Field(
        default=Path(
            "outputs/training/reflection-sigma40/fold_0/validation_predictions.csv"
        ),
        description="Local fold predictions used as the parity reference",
    )

    def cli_cmd(self) -> None:
        print(json.dumps(analyze(self), indent=2))


class ProfileCLI(CLISettings):
    model_config = describe(
        "Profile a frozen inference bundle over fold studies in Kaggle notebooks, one "
        "per runtime condition"
    )

    launch: CliSubCommand[Launch] = command(Launch)
    analyze: CliSubCommand[Analyze] = command(Analyze)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def profile_studies(args: ProfileCommand) -> list[str]:
    studies = fold_frame(args)[STUDY].to_list()
    if args.target_studies is None:
        return studies
    if args.target_studies <= len(studies):
        return studies[: args.target_studies]
    extra = (
        pl.read_csv(args.split_dir / "studies.csv")
        .filter(pl.col("fold") != args.fold)
        .sort(STUDY)[STUDY]
        .to_list()
    )
    return studies + extra[: args.target_studies - len(studies)]


def fold_frame(args: ProfileCommand) -> pl.DataFrame:
    return (
        pl.read_csv(args.split_dir / "studies.csv")
        .filter(pl.col("fold") == args.fold)
        .join(pl.read_csv(args.split_dir / "targets.csv"), on=STUDY, validate="1:1")
        .sort(STUDY)
    )


LOADING_CELL = """import random
import resource
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(bundle / "code"))
import torch

torch.set_num_threads(1)
from knee_abnormality.config import load_config
from knee_abnormality.data import dataset as dataset_module
from knee_abnormality.data.preprocessing import dicom as dicom_module
from knee_abnormality.data.preprocessing import laterality as laterality_module
from knee_abnormality.data.preprocessing import preparation as preparation_module
from knee_abnormality.data.preprocessing import series as series_module
from knee_abnormality.data.preprocessing import spatial as spatial_module
from knee_abnormality.inference import engine as engine_module

config = load_config(bundle / "config.json", data_root=data_root)
manifest = json.loads((bundle / "manifest.json").read_text())
read_header = dicom_module.read_header
timings = defaultdict(float)
processor_seconds = defaultdict(float)
counts = defaultdict(int)
thread_seconds = defaultdict(float)
thread_counts = defaultdict(int)
lock = threading.Lock()

def timed(module, name):
    original = getattr(module, name)
    def wrapper(*arguments, **keywords):
        started = time.perf_counter()
        started_cpu = time.process_time()
        try:
            return original(*arguments, **keywords)
        finally:
            wall = time.perf_counter() - started
            processor = time.process_time() - started_cpu
            with lock:
                timings[name] += wall
                processor_seconds[name] += processor
                counts[name] += 1
    setattr(module, name, wrapper)

def timed_on_threads(module, name):
    original = getattr(module, name)
    def wrapper(*arguments, **keywords):
        started = time.perf_counter()
        try:
            return original(*arguments, **keywords)
        finally:
            elapsed = time.perf_counter() - started
            with lock:
                thread_seconds[name] += elapsed
                thread_counts[name] += 1
    setattr(module, name, wrapper)

for module, name in (
    (engine_module, "build_series_index"),
    (dataset_module, "index_series"),
    (series_module, "parse_prefix"),
    (dataset_module, "clean_series"),
    (dataset_module, "estimate_reference"),
    (dataset_module, "split_knee_geometry"),
    (dataset_module, "resolve_knee_side"),
    (dataset_module, "build_pool"),
    (dataset_module, "select_triplets"),
    (dataset_module, "view_position"),
    (dataset_module, "fallback_window_center"),
    (dataset_module, "prepare_series"),
    (dataset_module, "decode_series"),
    (preparation_module, "decode_slice"),
    (preparation_module, "resample_crop"),
    (dataset_module.KneeStudyDataset, "_index_study_geometry"),
    (dataset_module.KneeStudyDataset, "_slot_sources"),
    (dataset_module.KneeStudyDataset, "_study_budgets"),
    (dataset_module.KneeStudyDataset, "_prepare_slot"),
):
    timed(module, name)

for module, name in (
    (series_module, "_safe_prefix"),
    (preparation_module, "_read_rows"),
    (preparation_module, "read_frame"),
    (dicom_module, "_read_pydicom_frame"),
):
    timed_on_threads(module, name)

def reset_process_caches():
    dicom_module._TAGS_CACHE.clear()
    for cached in (spatial_module._full_axis_filter, spatial_module._axis_filter, spatial_module.resample_plan, spatial_module._window_filters, laterality_module.text_sides):
        cached.cache_clear()
    if dicom_module._POOL is not None:
        dicom_module._POOL.shutdown(wait=True)
        dicom_module._POOL = None
        dicom_module._POOL_IDENTITY = None

def run_pass():
    reset_process_caches()
    timings.clear()
    processor_seconds.clear()
    counts.clear()
    thread_seconds.clear()
    thread_counts.clear()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    pass_started = time.perf_counter()
    pass_cpu = time.process_time()
    started = time.perf_counter()
    dataset, _, _ = engine_module.prediction_data(config, "test", 0, image_multiplier=manifest["runtime"]["image_multiplier"], study_ids=STUDIES)
    setup_seconds = time.perf_counter() - started
    studies = []
    for index in range(len(dataset)):
        begin = time.perf_counter()
        item = dataset[index]
        prepared = time.perf_counter()
        batch = dataset_module.collate_studies([item])
        studies.append({
            "study": item["study_uid"],
            "getitem_seconds": prepared - begin,
            "collate_seconds": time.perf_counter() - prepared,
            "images": int(batch["slice_mask"].sum()),
            "failures": len(item["failures"]),
        })
    finished = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "setup_seconds": setup_seconds,
        "pass_seconds": time.perf_counter() - pass_started,
        "pass_cpu_seconds": time.process_time() - pass_cpu,
        "minor_faults": finished.ru_minflt - usage.ru_minflt,
        "major_faults": finished.ru_majflt - usage.ru_majflt,
        "timings": dict(timings),
        "cpu_seconds": dict(processor_seconds),
        "counts": dict(counts),
        "thread_seconds": dict(thread_seconds),
        "thread_counts": dict(thread_counts),
        "studies": studies,
    }

def read_one(operation, path):
    try:
        operation(path)
        return path.stat().st_size
    except Exception:
        return 0

def throughput(uids, threads, operation_name):
    started = time.perf_counter()
    paths = [path for uid in uids for path in sorted((competition / "train_series" / uid).glob("*/*.dcm"))]
    listed = time.perf_counter()
    operation = read_header if operation_name == "header" else Path.read_bytes
    with ThreadPoolExecutor(threads) as pool:
        sizes = list(pool.map(lambda path: read_one(operation, path), paths))
    finished = time.perf_counter()
    return {
        "operation": operation_name,
        "threads": threads,
        "studies": len(uids),
        "files": len(paths),
        "megabytes": sum(sizes) / 2**20,
        "list_seconds": listed - started,
        "read_seconds": finished - listed,
        "files_per_second": len(paths) / (finished - listed),
    }

result = {"cold": run_pass()}
result["warm"] = run_pass()
groups = CONDITION["throughput_groups"]
size = len(THROUGHPUT_STUDIES) // len(groups)
result["throughput"] = [
    throughput(THROUGHPUT_STUDIES[index * size:(index + 1) * size], threads, operation)
    for index, (operation, threads) in enumerate(groups)
]
(profile_dir / "loading.json").write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps({pass_name: {"setup_seconds": value["setup_seconds"], "getitem_seconds": sum(item["getitem_seconds"] for item in value["studies"]), "timings": value["timings"]} for pass_name, value in result.items() if pass_name != "throughput"}, indent=1))
print(json.dumps(result["throughput"], indent=1))
"""


LOADER_LAB_PASS = """import hashlib
import json
import resource
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, spec["code"])
import numpy as np
import torch

torch.set_num_threads(1)
from knee_abnormality.config import load_config
from knee_abnormality.data import dataset as dataset_module
from knee_abnormality.data.preprocessing import device as device_module
from knee_abnormality.data.preprocessing import preparation as preparation_module
from knee_abnormality.data.preprocessing import series as series_module
from knee_abnormality.inference import engine as engine_module

owners = {
    "engine": engine_module,
    "dataset": dataset_module,
    "series": series_module,
    "preparation": preparation_module,
    "Dataset": dataset_module.KneeStudyDataset,
}
records = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
lock = threading.Lock()


def timed(owner, name):
    original = getattr(owner, name, None)
    if original is None:
        return

    def wrapper(*arguments, **keywords):
        usage = resource.getrusage(resource.RUSAGE_THREAD)
        started = time.perf_counter()
        try:
            return original(*arguments, **keywords)
        finally:
            wall = time.perf_counter() - started
            finished = resource.getrusage(resource.RUSAGE_THREAD)
            with lock:
                record = records[name]
                record[0] += 1
                record[1] += wall
                record[2] += finished.ru_utime - usage.ru_utime
                record[3] += finished.ru_stime - usage.ru_stime

    setattr(owner, name, wrapper)


for owner, name in spec["targets"]:
    timed(owners[owner], name)


def digest(value, hasher):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        hasher.update(f"T{value.dtype}{tuple(value.shape)}".encode())
        hasher.update(
            value.view(torch.uint8).numpy().tobytes() if value.numel() else b""
        )
    elif isinstance(value, np.ndarray):
        hasher.update(f"A{value.dtype}{value.shape}".encode())
        hasher.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            if key in {"frame_buffers", "frames"}:
                continue
            hasher.update(f"K{key}".encode())
            digest(value[key], hasher)
    elif isinstance(value, (list, tuple)):
        hasher.update(f"L{len(value)}".encode())
        for item in value:
            digest(item, hasher)
    elif value is None or isinstance(value, (str, int, float, bool)):
        hasher.update(repr(value).encode())
    else:
        hasher.update(type(value).__name__.encode())


def fingerprint(batch):
    keys = {}
    for key in sorted(batch, key=str):
        if key in {"frame_buffers", "frames"}:
            continue
        hasher = hashlib.sha256()
        digest(batch[key], hasher)
        keys[key] = hasher.hexdigest()[:16]
    return keys


def io_counters():
    values = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":")
        values[key] = int(value)
    return values


config = load_config(
    Path(spec["bundle"]) / "config.json", data_root=Path(spec["data_root"])
)
usage = resource.getrusage(resource.RUSAGE_SELF)
io_before = io_counters()
pass_started = time.perf_counter()
dataset, _, _ = engine_module.prediction_data(
    config, "test", 0, image_multiplier=spec["multiplier"], study_ids=spec["studies"]
)
setup_seconds = time.perf_counter() - pass_started
studies = []
for index in range(len(dataset)):
    begin = time.perf_counter()
    item = dataset[index]
    prepared = time.perf_counter()
    batch = dataset_module.collate_studies([item])
    collated = time.perf_counter()
    entry = {
        "study": item["study_uid"],
        "getitem_seconds": prepared - begin,
        "collate_seconds": collated - prepared,
        "images": int(batch["slice_mask"].sum()),
        "failures": len(item["failures"]),
    }
    if spec["fingerprint"]:
        entry["fingerprint"] = fingerprint(device_module.prepare_images(batch))
    studies.append(entry)
pass_seconds = time.perf_counter() - pass_started
finished = resource.getrusage(resource.RUSAGE_SELF)
io_after = io_counters()
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "setup_seconds": setup_seconds,
            "pass_seconds": pass_seconds,
            "user_seconds": finished.ru_utime - usage.ru_utime,
            "system_seconds": finished.ru_stime - usage.ru_stime,
            "minor_faults": finished.ru_minflt - usage.ru_minflt,
            "major_faults": finished.ru_majflt - usage.ru_majflt,
            "voluntary_switches": finished.ru_nvcsw - usage.ru_nvcsw,
            "involuntary_switches": finished.ru_nivcsw - usage.ru_nivcsw,
            "io": {key: io_after[key] - io_before[key] for key in io_after},
            "functions": {
                name: dict(zip(("calls", "wall", "user", "system"), value))
                for name, value in records.items()
            },
            "studies": studies,
        }
    )
    + "\\n"
)
"""


LOADER_LAB_CELL = (
    "LAB_PASS = "
    + json.dumps(LOADER_LAB_PASS)
    + "\n"
    + """labs = sorted({path.parent for pattern in ("datasets/*/*/lab.json", "datasets/*/*/*/lab.json", "*/lab.json", "*/*/lab.json") for path in Path("/kaggle/input").glob(pattern)})
if len(labs) != 1:
    raise RuntimeError(f"Expected one loader lab, found {labs}")
lab = labs[0]
variants = json.loads((lab / "lab.json").read_text())["variants"]
pass_script = Path("/kaggle/working/loader_lab_pass.py")
pass_script.write_text(LAB_PASS)
multiplier = json.loads((bundle / "manifest.json").read_text())["runtime"]["image_multiplier"]
files = [path for uid in STUDIES for path in sorted((competition / "train_series" / uid).glob("*/*.dcm"))]
NFS_PROCEDURES = ("null", "getattr", "setattr", "lookup", "access", "readlink", "read", "write", "create", "mkdir", "symlink", "mknod", "remove", "rmdir", "rename", "link", "readdir", "readdirplus", "fsstat", "fsinfo", "pathconf", "commit")

def evict():
    for path in files:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)

def nfs_calls():
    path = Path("/proc/net/rpc/nfs")
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if line.startswith("proc3 "):
            return dict(zip(NFS_PROCEDURES, map(int, line.split()[2:])))
    return {}

def check_eviction(sample):
    evict()
    started = time.perf_counter()
    for path in sample:
        path.read_bytes()
    cold = time.perf_counter() - started
    started = time.perf_counter()
    for path in sample:
        path.read_bytes()
    return {"files": len(sample), "cold_seconds": cold, "warm_seconds": time.perf_counter() - started}

def run(name, label, fingerprint):
    spec_path = profile_dir / f"spec-{name}-{label}.json"
    output = profile_dir / f"lab-{name}-{label}.json"
    spec_path.write_text(json.dumps({"code": str(lab / "variants" / name), "bundle": str(bundle), "data_root": str(data_root), "studies": STUDIES, "multiplier": multiplier, "fingerprint": fingerprint, "targets": CONDITION["targets"]}))
    before = nfs_calls()
    started = time.perf_counter()
    try:
        completed = subprocess.run([sys.executable, str(pass_script), str(spec_path), str(output)], capture_output=True, text=True, timeout=CONDITION["pass_timeout_seconds"])
    except subprocess.TimeoutExpired:
        print(json.dumps({"variant": name, "label": label, "timed_out": True}), flush=True)
        return {"variant": name, "label": label, "timed_out": True}
    elapsed = time.perf_counter() - started
    after = nfs_calls()
    if completed.returncode:
        print(completed.stdout[-3000:])
        print(completed.stderr[-6000:])
        raise RuntimeError(f"{name} {label} failed")
    result = json.loads(output.read_text())
    result.update(variant=name, label=label, process_seconds=elapsed, nfs_calls={key: after[key] - before.get(key, 0) for key in after})
    print(json.dumps({"variant": name, "label": label, "pass_seconds": result["pass_seconds"], "user": result["user_seconds"], "system": result["system_seconds"], "nfs": {key: value for key, value in result["nfs_calls"].items() if value}}), flush=True)
    return result

lab_results = {"variants": variants, "kernel": os.uname().release, "transparent_hugepages": {name: Path("/sys/kernel/mm/transparent_hugepage", name).read_text().strip() for name in ("enabled", "shmem_enabled") if Path("/sys/kernel/mm/transparent_hugepage", name).exists()}, "eviction": check_eviction(files[:300]), "passes": []}
print(json.dumps({key: lab_results[key] for key in ("kernel", "transparent_hugepages", "eviction")}), flush=True)
for round_index in range(CONDITION["rounds"]):
    shift = round_index % len(variants)
    for name in variants[shift:] + variants[:shift]:
        evict()
        lab_results["passes"].append(run(name, f"cold{round_index}", False))
for name in variants:
    lab_results["passes"].append(run(name, "fingerprint", True))
(profile_dir / "lab.json").write_text(json.dumps(lab_results) + "\\n")
"""
)


PRECISION_CELL = """sys.path.insert(0, str(bundle / "code"))
import torch

torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")
from knee_abnormality.config import load_config
from knee_abnormality.data import dataset as dataset_module
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.data.preprocessing.device import prepare_images
from knee_abnormality.inference.engine import move_batch, prediction_data

config = load_config(bundle / "config.json", data_root=data_root)
manifest = json.loads((bundle / "manifest.json").read_text())
dataset, targets, _ = prediction_data(config, "test", 0, image_multiplier=manifest["runtime"]["image_multiplier"], study_ids=STUDIES)
items = [dataset[index] for index in range(len(dataset))]
batches = [dataset_module.collate_studies(items[index:index + 4]) for index in range(0, len(items), 4)]
device = torch.device("cuda:0")
model = load_checkpoint_model(config, targets, bundle / "model.ckpt", device)
gpu_batches = [prepare_images(move_batch(batch, device)) for batch in batches]
images = [int(batch["slice_mask"].sum()) for batch in batches]
total = sum(images)
original_features = model.backbone.forward_features
layout = {"chunk": None}

def chunked_features(values):
    size = layout["chunk"]
    if size is None:
        return original_features(values)
    count = values.shape[0]
    padded = -(-count // size) * size
    if padded != count:
        values = torch.cat([values, values.new_zeros((padded - count, *values.shape[1:]))])
    if model.channels_last:
        values = values.contiguous(memory_format=torch.channels_last)
    return torch.cat([original_features(values[start:start + size]) for start in range(0, padded, size)])[:count]

model.backbone.forward_features = chunked_features

def set_layout(chunk, channels_last):
    layout["chunk"] = chunk
    model.channels_last = channels_last
    model.backbone.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)

results = []
predictions = {}

def measure(label, autocast, store=None):
    outputs = []
    torch.cuda.synchronize()
    wall = time.perf_counter()
    cpu = time.process_time()
    with torch.inference_mode():
        for batch in gpu_batches:
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                outputs.append(torch.sigmoid(model(batch).to(torch.float32)))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - wall
    record = {"label": label, "seconds": elapsed, "images": total, "seconds_per_1000_images": 1000 * elapsed / total, "cpu_seconds_per_wall_second": (time.process_time() - cpu) / elapsed, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    if store is not None:
        predictions[store] = torch.cat(outputs).cpu()
    results.append(record)
    print(json.dumps(record))

for chunk, channels_last, precision in ((128, False, "fp16"), (128, True, "fp16"), (64, True, "fp16"), (128, False, "fp32"), (128, True, "fp32"), (None, False, "fp32"), (None, False, "fp16")):
    set_layout(chunk, channels_last)
    name = f"{precision} chunk={chunk} {'nhwc' if channels_last else 'nchw'}"
    torch.cuda.reset_peak_memory_stats()
    measure(name + " new shapes", precision == "fp16")
    measure(name + " repeated shapes", precision == "fp16", store=name)

set_layout(128, True)
burners = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(3)]
try:
    measure("fp16 chunk=128 nhwc repeated shapes, 3 CPU burners", True)
    set_layout(None, False)
    measure("fp32 chunk=None nchw repeated shapes, 3 CPU burners", False)
finally:
    for process in burners:
        process.kill()

reference = predictions["fp32 chunk=None nchw"]
differences = {name: {"max_abs": float((value - reference).abs().max()), "mean_abs": float((value - reference).abs().mean())} for name, value in predictions.items()}
print(json.dumps(differences, indent=1))
(profile_dir / "precision.json").write_text(json.dumps({"results": results, "differences_from_fp32": differences, "batch_images": images, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cudnn": torch.backends.cudnn.version()}, indent=2) + "\\n")
"""


CENSUS_CELL = """sys.path.insert(0, str(bundle / "code"))
import torch

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.inference.engine import prediction_data

config = load_config(bundle / "config.json", data_root=data_root)
manifest = json.loads((bundle / "manifest.json").read_text())
dataset, _, _ = prediction_data(config, "test", 0, image_multiplier=manifest["runtime"]["image_multiplier"], study_ids=STUDIES)
loader = build_data_loader(dataset, config.trainer, 1, num_workers=4)
census = {"studies": 0, "series": 0, "failures": []}
started = time.perf_counter()
for batch in loader:
    census["studies"] += len(batch["study_uid"])
    census["series"] += int(batch["slice_mask"].shape[0])
    for uid, values in zip(batch["study_uid"], batch["failures"]):
        census["failures"].extend(f"{uid} | {value}" for value in values)
census["seconds"] = time.perf_counter() - started
(profile_dir / "census.json").write_text(json.dumps(census, indent=1) + "\\n")
print(json.dumps({key: value for key, value in census.items() if key != "failures"}), len(census["failures"]))
"""


BUNDLE_SELECTION = """import os
from pathlib import Path

_selected = sorted(path.parent for pattern in ("*/manifest.json", "*/*/manifest.json", "*/*/*/manifest.json", "*/*/*/*/manifest.json") for path in Path("/kaggle/input").glob(pattern) if "{slug}" in str(path))
if len(_selected) == 1:
    os.environ["KNEE_BUNDLE_DIR"] = str(_selected[0])
"""


PARITY_SCRIPT = """import hashlib, inspect, json, os, sys
from pathlib import Path

import numpy as np
import torch

bundle, data_root, output, studies_path = map(Path, sys.argv[1:5])
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.inference.engine import move_batch, prediction_data

config = load_config(bundle / "config.json", data_root=data_root)
manifest = json.loads((bundle / "manifest.json").read_text())
studies = json.loads(studies_path.read_text())
dataset, _, _ = prediction_data(config, "test", 0, image_multiplier=manifest["runtime"]["image_multiplier"], study_ids=studies)
options = {"prepare_on_loader": False} if os.environ.get("KNEE_PARITY_GPU") and "prepare_on_loader" in inspect.signature(build_data_loader).parameters else {}
loader = build_data_loader(dataset, config.trainer, 1, num_workers=4, **options)
records = {}
saved = 0
for batch in loader:
    constant = None
    if "images" not in batch:
        from knee_abnormality.data.preprocessing.device import prepare_images
        mask = batch["slice_mask"].clone()
        batch = prepare_images(move_batch(batch, torch.device("cuda")))
        constant = batch["constant_series"].cpu().tolist()
    else:
        mask = batch["slice_mask"]
    images = batch["images"].cpu()
    series = []
    for index in range(images.shape[0]):
        count = int(mask[index].sum())
        values = images[index, :count].numpy()
        series.append({
            "images": hashlib.sha256(values.tobytes()).hexdigest(),
            "count": count,
            "sum": int(values.astype(np.int64).sum()),
            "geometry": hashlib.sha256(b"".join(batch[key][index, :count].cpu().numpy().tobytes() for key in ("valid_rects", "slice_positions", "patient_maps", "geometry_valid"))).hexdigest(),
        })
        if saved < 6:
            np.save(output.parent / f"{output.stem}-{batch['study_uid'][0]}-{index}.npy", values)
            saved += 1
    records[batch["study_uid"][0]] = {"series": series, "failures": list(batch["failures"][0]), "constant": constant}
output.write_text(json.dumps(records))
"""


PARITY_CELL = f"""reference = sorted(path.parent for pattern in ("*/manifest.json", "*/*/manifest.json", "*/*/*/manifest.json", "*/*/*/*/manifest.json") for path in Path("/kaggle/input").glob(pattern) if path.parent != bundle)
if len(reference) != 1:
    raise RuntimeError(f"Expected one reference bundle, found {{reference}}")
studies_path = profile_dir / "studies.json"
studies_path.write_text(json.dumps(STUDIES))
parity = {{}}
for name, root, extra in (("reference", reference[0], {{}}), ("candidate", bundle, {{}}), ("candidate_gpu", bundle, {{"KNEE_PARITY_GPU": "1"}})):
    environment_values = os.environ.copy()
    environment_values.update(extra)
    environment_values["PYTHONPATH"] = str(root / "code")
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, "-c", {PARITY_SCRIPT!r}, str(root), str(data_root), str(profile_dir / f"{{name}}.json"), str(studies_path)], env=environment_values, capture_output=True, text=True)
    parity[name] = {{"seconds": time.perf_counter() - started, "returncode": completed.returncode, "stderr": completed.stderr[-6000:]}}
    print(name, parity[name]["seconds"], completed.returncode, completed.stderr[-3000:])
reference_records = json.loads((profile_dir / "reference.json").read_text())
for candidate_name in ("candidate", "candidate_gpu"):
    candidate_records = json.loads((profile_dir / f"{{candidate_name}}.json").read_text())
    mismatches = []
    series_total = 0
    for study, expected in reference_records.items():
        actual = candidate_records.get(study)
        if actual is None or len(actual["series"]) != len(expected["series"]):
            mismatches.append({{"study": study, "reason": "series_count"}})
            continue
        for index, (left, right) in enumerate(zip(expected["series"], actual["series"])):
            series_total += 1
            if left != right:
                mismatches.append({{"study": study, "series": index, "reference": left, "candidate": right}})
    parity[f"{{candidate_name}}_result"] = {{"studies": len(reference_records), "series": series_total, "mismatches": mismatches, "constant": sum(sum(record["constant"] or []) for record in candidate_records.values())}}
(profile_dir / "parity.json").write_text(json.dumps(parity, indent=1) + "\\n")
print(json.dumps({{name: {{key: (len(value) if key == "mismatches" else value) for key, value in parity[name].items()}} for name in ("candidate_result", "candidate_gpu_result")}}, indent=1))
"""


CUDNN_PROBE = """import json, os, sys, time
import numpy as np
import torch
import timm
torch.set_num_threads(1)
if os.environ.get("KNEE_BENCHMARK") == "1":
    torch.backends.cudnn.benchmark = True
backbone = timm.create_model("rexnet_200", pretrained=False, in_chans=3, num_classes=0, global_pool="").cuda().eval().to(memory_format=torch.channels_last)
torch.manual_seed(0)
images = torch.randn((64, 3, 224, 224), generator=torch.Generator().manual_seed(0)).cuda().contiguous(memory_format=torch.channels_last)
def forward():
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        return backbone.forward_features(images).float().cpu().numpy()
started = time.perf_counter(); cpu = time.process_time()
first = forward()
warmup = time.perf_counter() - started; warmup_cpu = time.process_time() - cpu
torch.cuda.synchronize()
started = time.perf_counter()
for _ in range(20):
    output = forward()
steady = (time.perf_counter() - started) / 20
np.save(sys.argv[1], output)
print(json.dumps({"warmup_seconds": warmup, "warmup_cpu_seconds": warmup_cpu, "chunk_seconds": steady}))
"""


CUDNN_CELL = f"""results = {{}}
for label, extra in (("default", {{}}), ("v8_disabled", {{"TORCH_CUDNN_V8_API_DISABLED": "1"}}), ("benchmark", {{"KNEE_BENCHMARK": "1"}}), ("default_repeat", {{}})):
    values = os.environ.copy()
    values.update(extra)
    output = profile_dir / f"{{label}}.npy"
    completed = subprocess.run([sys.executable, "-c", {CUDNN_PROBE!r}, str(output)], env=values, capture_output=True, text=True)
    record = json.loads(completed.stdout.strip().splitlines()[-1]) if completed.returncode == 0 else {{"error": completed.stderr[-3000:]}}
    results[label] = record
    print(label, record)
import numpy as np
reference = np.load(profile_dir / "default.npy")
for label in results:
    if (profile_dir / f"{{label}}.npy").exists():
        other = np.load(profile_dir / f"{{label}}.npy")
        results[label]["max_abs_difference"] = float(np.abs(other - reference).max())
        results[label]["identical"] = bool((other == reference).all())
(profile_dir / "cudnn.json").write_text(json.dumps(results, indent=2) + "\\n")
print(json.dumps(results, indent=1))
"""


MAINLOOP_CELL = """sys.path.insert(0, str(bundle / "code"))
import importlib.util
import traceback
import warnings
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from knee_abnormality.inference.processes import use_blocking_sync

use_blocking_sync()
torch.set_num_threads(1)
sys.modules["wandb"] = None
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.inference import engine
from knee_abnormality.inference.checkpoint import load_models

config = load_config(bundle / "config.json", data_root=data_root)
manifest = json.loads((bundle / "manifest.json").read_text())
runtime = manifest["runtime"]
dataset, targets, _ = engine.prediction_data(config, "test", 0, image_multiplier=runtime["image_multiplier"], study_ids=STUDIES)
loader = build_data_loader(dataset, config.trainer, 4, num_workers=4, pin_memory=False, prepare_on_loader=False)
cuda = torch.device("cuda")
models = load_models(config, targets, bundle / "models" if "checkpoints" in manifest else bundle / "model.ckpt", cuda, attention_temperature=runtime["attention_temperature"])
plain_features = [model._backbone_features for model in models]
middle_only = all(model.repeated_middle_slice for model in models)
batches = list(loader)
wall = defaultdict(float)
cpu = defaultdict(float)
counts = Counter()
synchronized = True

def timed(name, function, *arguments, **keywords):
    if synchronized:
        torch.cuda.synchronize()
    started = time.perf_counter()
    cpu_started = time.process_time()
    value = function(*arguments, **keywords)
    if synchronized:
        torch.cuda.synchronize()
    wall[name] += time.perf_counter() - started
    cpu[name] += time.process_time() - cpu_started
    return value

def capture(index, model):
    engine.use_fixed_backbone_chunks(model, config.data.image_size)
    chunked = model._backbone_features

    def features(images):
        counts[f"images.{index}"] += images.shape[0]
        counts[f"chunks.{index}"] += -(-images.shape[0] // engine.BACKBONE_CHUNK_IMAGES)
        return timed(f"backbone.{index}", chunked, images)

    model._backbone_features = features

original_grid_sample = F.grid_sample
F.grid_sample = lambda *arguments, **keywords: timed("grid_sample", original_grid_sample, *arguments, **keywords)

def run(selected):
    outputs = []
    with torch.inference_mode():
        for batch in selected:
            locked = timed("page_lock", lambda: [buffer for buffer in batch.get("frame_buffers", ()) if engine._page_lock(buffer)])
            moved = timed("transfer", engine.move_batch, batch, cuda)
            prepared = timed("prepare_images", engine.with_images, moved, middle_only)
            with torch.autocast("cuda", dtype=torch.float16):
                probabilities = []
                for index, model in enumerate(models):
                    pose = timed(f"pose.{index}", model.first_look_pose, prepared)
                    probabilities.append(timed(f"read.{index}", lambda: torch.sigmoid(model(prepared, pose).to(torch.float32))))
                outputs.append(timed("to_cpu", lambda: torch.stack(probabilities).mean(dim=0).cpu()))
            timed("unlock", lambda: [torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr()) for buffer in locked])
    return torch.cat(outputs)

def measure():
    global synchronized
    synchronized = True
    run(batches)
    wall.clear()
    cpu.clear()
    counts.clear()
    outputs = run(batches)
    stages = {name: {"wall_seconds_per_batch": wall[name] / len(batches), "cpu_seconds_per_batch": cpu[name] / len(batches)} for name in wall}
    chunks = dict(counts)
    synchronized = False
    torch.cuda.synchronize()
    started = time.perf_counter()
    cpu_started = time.process_time()
    repeated = run(batches)
    torch.cuda.synchronize()
    free = {"wall_seconds_per_batch": (time.perf_counter() - started) / len(batches), "cpu_seconds_per_batch": (time.process_time() - cpu_started) / len(batches)}
    return outputs, {"stages": stages, "counts": chunks, "free_running": free, "repeat_max_abs_difference": float((outputs - repeated).abs().max())}

def fold(module):
    folded = 0
    previous_name, previous = None, None
    for name, child in list(module.named_children()):
        if type(child) is torch.nn.BatchNorm2d and isinstance(previous, torch.nn.Conv2d):
            setattr(module, previous_name, fuse_conv_bn_eval(previous, child))
            setattr(module, name, torch.nn.Identity())
            folded += 1
            previous_name, previous = None, None
        else:
            folded += fold(child)
            previous_name, previous = name, child
    return folded

for index, model in enumerate(models):
    capture(index, model)
reference, result = measure()
result = {"batches": len(batches), "studies": len(STUDIES), "members": [type(model.backbone).__name__ for model in models], "baseline": result}
print(json.dumps(result["baseline"], indent=1))

sites = Counter()

def note(message, category, filename, lineno, file=None, line=None):
    frames = [frame for frame in traceback.extract_stack() if "knee_abnormality" in frame.filename]
    sites[f"{Path(frames[-1].filename).name}:{frames[-1].lineno} {frames[-1].line}" if frames else f"{filename}:{lineno} {message}"] += 1

synchronized = False
torch.cuda.set_sync_debug_mode("warn")
with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.showwarning = note
    run(batches[:2])
torch.cuda.set_sync_debug_mode("default")
result["sync_sites_per_batch"] = {site: count / 2 for site, count in sites.most_common()}
print(json.dumps(result["sync_sites_per_batch"], indent=1))

folded = []
for index, model in enumerate(models):
    model._backbone_features = plain_features[index]
    folded.append(fold(model.backbone))
    capture(index, model)
outputs, measured = measure()
result["folded"] = {**measured, "folded_layers": folded, "max_abs_difference": float((outputs - reference).abs().max()), "mean_abs_difference": float((outputs - reference).abs().mean())}
print(json.dumps(result["folded"], indent=1))

packages = json.loads(subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], capture_output=True, text=True).stdout or "[]")
result["packages"] = {package["name"]: package["version"] for package in packages if any(word in package["name"].lower() for word in ("onnx", "nvidia", "torch", "cudnn", "openvino", "pycuda"))}
result["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}

from knee_abnormality.data.preprocessing import device as device_module

def prepare_all():
    images = []
    seconds = 0.0
    with torch.inference_mode():
        for batch in batches:
            locked = [buffer for buffer in batch.get("frame_buffers", ()) if engine._page_lock(buffer)]
            moved = engine.move_batch(batch, cuda)
            torch.cuda.synchronize()
            started = time.perf_counter()
            prepared = engine.with_images(moved, middle_only)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            images.append(prepared["images"].cpu())
            for buffer in locked:
                torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())
    return images, seconds / len(batches)

matrices = {}

def dense_resample(blocks, plan):
    key = (id(plan), blocks.shape[1], blocks.shape[2])
    if key not in matrices:
        vertical_index, vertical_weight, horizontal_index, horizontal_weight = plan
        matrices[key] = (
            torch.zeros((vertical_index.shape[0], blocks.shape[1]), device=blocks.device, dtype=vertical_weight.dtype).scatter_add_(1, vertical_index, vertical_weight),
            torch.zeros((horizontal_index.shape[0], blocks.shape[2]), device=blocks.device, dtype=horizontal_weight.dtype).scatter_add_(1, horizontal_index, horizontal_weight).T.contiguous(),
        )
    vertical, horizontal = matrices[key]
    return torch.matmul(torch.matmul(vertical.to(blocks.dtype), blocks), horizontal.to(blocks.dtype))

synchronized = False
prepare_all()
baseline_images, baseline_seconds = prepare_all()
before = run(batches)
tap_resample = device_module._resample
device_module._resample = dense_resample
prepare_all()
dense_images, dense_seconds = prepare_all()
changed = 0
total = 0
largest = 0
for expected, actual in zip(baseline_images, dense_images):
    difference = (expected.to(torch.int16) - actual.to(torch.int16)).abs()
    changed += int((difference > 0).sum())
    total += difference.numel()
    largest = max(largest, int(difference.max()))
del dense_images
after = run(batches)
device_module._resample = tap_resample
del baseline_images
result["dense_resample"] = {"tap_prepare_seconds_per_batch": baseline_seconds, "dense_prepare_seconds_per_batch": dense_seconds, "changed_pixels": changed, "pixels": total, "largest_pixel_difference": largest, "max_abs_difference": float((after - before).abs().max()), "mean_abs_difference": float((after - before).abs().mean())}
print(json.dumps(result["dense_resample"], indent=1))

import inspect

def wrap(owner, name, label):
    function = getattr(owner, name)
    setattr(owner, name, lambda *arguments, **keywords: timed(label, function, *arguments, **keywords))

for name, value in list(vars(device_module).items()):
    if inspect.isfunction(value) and value.__module__ == device_module.__name__:
        wrap(device_module, name, f"device.{name}")
import faulthandler

faulthandler.dump_traceback_later(300, exit=True)
for name in ("solve", "inv", "matrix_exp"):
    wrap(torch.linalg, name, f"linalg.{name}")
for index, model in enumerate(models):
    for module_name, module in model.named_modules():
        if module_name.split(".")[0] == "backbone" or not type(module).__module__.startswith("knee_abnormality"):
            continue
        for name, value in vars(type(module)).items():
            if inspect.isfunction(value) and name not in vars(module) and not name.startswith("__") and name != "train":
                wrap(module, name, f"model.{index}.{module_name or 'model'}.{name}")
synchronized = True
wall.clear()
cpu.clear()
functions = Counter()
original_timed = timed

def timed(name, function, *arguments, **keywords):
    functions[name] += 1
    return original_timed(name, function, *arguments, **keywords)

run(batches[:4])
faulthandler.cancel_dump_traceback_later()
result["inclusive_functions"] = {name: {"wall_seconds_per_batch": wall[name] / 4, "cpu_seconds_per_batch": cpu[name] / 4, "calls_per_batch": functions[name] / 4} for name in sorted(wall, key=wall.get, reverse=True)}
print(json.dumps(result["inclusive_functions"], indent=1))
(profile_dir / "mainloop.json").write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps({key: result[key] for key in ("packages", "torch")}, indent=1))
"""


STARTUP_MODEL_LOAD = """import json, os, time
marks = {}
last = time.perf_counter()
def mark(name):
    global last
    now = time.perf_counter()
    marks[name] = now - last
    last = now
import torch
mark("import_torch")
import timm
mark("import_timm")
from pathlib import Path
from knee_abnormality.config import load_config
from knee_abnormality.inference.checkpoint import apply_attention_temperature, read_checkpoint, validate_checkpoint
from knee_abnormality.inference.engine import use_fixed_backbone_chunks
from knee_abnormality.modeling.model import KneeAbnormalityModel
mark("import_package")
torch.set_float32_matmul_precision("high")
torch.set_num_threads(1)
bundle = Path(os.environ["KNEE_BUNDLE"])
data_root = Path(os.environ["KNEE_DATA"])
config = load_config(bundle / "config.json", data_root=data_root)
targets = tuple((data_root / "sample_submission.csv").read_text().splitlines()[0].split(",")[1:])
mark("config")
checkpoint = read_checkpoint(bundle / "model.ckpt")
mark("read_checkpoint")
validate_checkpoint(checkpoint, config, targets)
mark("validate_checkpoint")
backbone = timm.create_model(config.model.backbone, pretrained=False, in_chans=3, num_classes=0, global_pool="")
mark("timm_create_only")
del backbone
model = KneeAbnormalityModel(config.model, config.data, targets, load_pretrained=False)
mark("construct_model")
state = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
model.load_state_dict(state, strict=True)
apply_attention_temperature(model, 0.75)
mark("load_state")
torch.cuda.init()
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
mark("cuda_init")
model.to("cuda").eval()
torch.cuda.synchronize()
mark("to_device")
use_fixed_backbone_chunks(model, config.data.image_size)
mark("chunk_warmup")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    model.backbone.forward_features(torch.zeros((64, 3, config.data.image_size, config.data.image_size), device="cuda"))
torch.cuda.synchronize()
mark("second_chunk")
print(json.dumps({"marks": marks, "process_cpu_seconds": time.process_time(), "torch": torch.__version__, "timm": timm.__version__, "cudnn": torch.backends.cudnn.version(), "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING")}))
"""


STARTUP_SPIN = """import ctypes, json, os, time
mode = os.environ["KNEE_SYNC"]
driver = ctypes.CDLL("libcuda.so.1")
if driver.cuInit(0) != 0:
    raise RuntimeError("cuInit failed")
device = ctypes.c_int()
if driver.cuDeviceGet(ctypes.byref(device), 0) != 0:
    raise RuntimeError("cuDeviceGet failed")
if mode == "blocking":
    result = driver.cuDevicePrimaryCtxSetFlags_v2(device, 4)
    if result != 0:
        raise RuntimeError(f"cuDevicePrimaryCtxSetFlags failed with {result}")
import torch
import timm
torch.set_num_threads(1)
backbone = timm.create_model("rexnet_200", pretrained=False, in_chans=3, num_classes=0, global_pool="").cuda().eval().to(memory_format=torch.channels_last)
images = torch.zeros((448, 3, 224, 224), device="cuda").contiguous(memory_format=torch.channels_last)
def step():
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        return torch.cat([backbone.forward_features(images[start:start + 64]) for start in range(0, 448, 64)]).float().mean().cpu()
for _ in range(3):
    step()
torch.cuda.synchronize()
flags = ctypes.c_uint()
active = ctypes.c_int()
driver.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active))
wall = time.perf_counter()
cpu = time.process_time()
for _ in range(30):
    step()
wall = time.perf_counter() - wall
cpu = time.process_time() - cpu
print(json.dumps({"mode": mode, "flags": flags.value, "active": active.value, "wall_seconds": wall, "cpu_seconds": cpu, "cpu_per_wall": cpu / wall, "images": 30 * 448}))
"""


STARTUP_CELL = f"""import re
code_dir = bundle / "code"
child_environment = os.environ.copy()
child_environment.update(PYTHONPATH=str(code_dir), KNEE_BUNDLE=str(bundle), KNEE_DATA=str(data_root))
startup = {{"runs": []}}

def run_child(label, arguments, extra=None):
    values = dict(child_environment)
    values.update(extra or {{}})
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, *arguments], env=values, cwd=str(code_dir), capture_output=True, text=True)
    record = {{"label": label, "seconds": time.perf_counter() - started, "returncode": completed.returncode, "stdout": completed.stdout[-20000:], "stderr_tail": completed.stderr[-3000:]}}
    (profile_dir / f"{{label}}.stderr").write_text(completed.stderr)
    startup["runs"].append(record)
    print(label, round(record["seconds"], 2), completed.returncode, completed.stdout[-2000:])
    return record

def import_table(record):
    rows = []
    for line in (profile_dir / f"{{record['label']}}.stderr").read_text().splitlines():
        match = re.match(r"import time:\\s+(\\d+) \\|\\s+(\\d+) \\| (\\s*)(\\S+)", line)
        if match:
            rows.append({{"self_us": int(match.group(1)), "cumulative_us": int(match.group(2)), "depth": len(match.group(3)) // 2, "module": match.group(4)}})
    record["import_total_seconds"] = sum(row["cumulative_us"] for row in rows if row["depth"] == 0) / 1e6
    record["import_top"] = sorted((row for row in rows if row["depth"] <= 1), key=lambda row: -row["cumulative_us"])[:40]
    record["import_self_top"] = sorted(rows, key=lambda row: -row["self_us"])[:40]

for label, module in (("coordinator-import-cold", "scripts.pipeline.predict"), ("coordinator-import-warm", "scripts.pipeline.predict"), ("worker-import-warm", "scripts.pipeline.predict_worker")):
    import_table(run_child(label, ["-X", "importtime", "-c", f"import {{module}}"]))
for label in ("model-load", "model-load-repeat"):
    run_child(label, ["-c", {STARTUP_MODEL_LOAD!r}])
for label, mode in (("spin-default", "default"), ("spin-blocking", "blocking")):
    run_child(label, ["-c", {STARTUP_SPIN!r}], {{"KNEE_SYNC": mode}})
burners = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(4)]
try:
    for label, mode in (("spin-default-burners", "default"), ("spin-blocking-burners", "blocking")):
        run_child(label, ["-c", {STARTUP_SPIN!r}], {{"KNEE_SYNC": mode}})
finally:
    for process in burners:
        process.kill()
(profile_dir / "startup.json").write_text(json.dumps(startup, indent=2) + "\\n")
"""


VARIANTS_CELL = """variants = CONDITION["variants"]
parts = [STUDIES[index :: len(variants)] for index in range(len(variants))]
header = (data_root / "sample_submission.csv").read_text().splitlines()[0]
series = (data_root / "test_series.csv").read_text().splitlines()
results = {}
for position, (variant, studies) in enumerate(zip(variants, parts)):
    root = Path(f"/tmp/knee-variant-{position}")
    (root / "test_series").mkdir(parents=True)
    chosen = set(studies)
    (root / "sample_submission.csv").write_text("\\n".join([header, *(f"{uid}," + ",".join(["0.5"] * header.count(",")) for uid in studies)]) + "\\n")
    (root / "test_series.csv").write_text("\\n".join([series[0], *(row for row in series[1:] if row.split(",", 1)[0] in chosen)]) + "\\n")
    for uid in studies:
        (root / "test_series" / uid).symlink_to((data_root / "test_series" / uid).resolve(), target_is_directory=True)
    output = profile_dir / f"{position}-{variant}"
    output.mkdir()
    command = [sys.executable, "-m", "scripts.pipeline.predict", str(bundle / "config.json"), "--bundle", str(bundle), "--data-root", str(root), "--output", str(output / "predictions.csv"), "--device", "cuda", "--precision", "fp16", "--batch-size", "4", "--num-workers", "8"]
    environment_variables = os.environ.copy()
    environment_variables["PYTHONPATH"] = str(bundle / "code")
    environment_variables["KNEE_VARIANT"] = variant
    started = time.time()
    with (output / "predict.log").open("w") as log:
        returncode = subprocess.run(command, env=environment_variables, stdout=log, stderr=subprocess.STDOUT).returncode
    results[f"{position}-{variant}"] = {"seconds": time.time() - started, "returncode": returncode, "studies": len(studies)}
    print(position, variant, results[f"{position}-{variant}"])
    if returncode:
        print("\\n".join((output / "predict.log").read_text(errors="replace").splitlines()[-40:]))
(profile_dir / "variants.json").write_text(json.dumps(results, indent=2) + "\\n")
if any(result["returncode"] for result in results.values()):
    raise RuntimeError("A variant failed")
"""


STARTUP_IMPORTS = r"""import json, os, resource, sys, time
marks = {}
last = time.perf_counter()
def mark(name):
    global last
    now = time.perf_counter()
    marks[name] = now - last
    last = now
import torch
mark("torch")
import scripts.pipeline.predict_worker
mark("worker_module")
sys.modules["wandb"] = None
from knee_abnormality.evaluation.predictions import make_prediction_frame
from knee_abnormality.inference.checkpoint import load_models, member_checkpoints
from knee_abnormality.inference.engine import predict_loader
mark("model_stack")
usage = resource.getrusage(resource.RUSAGE_SELF)
io = dict(line.split(": ") for line in open("/proc/self/io").read().splitlines())
record = {"marks": marks, "total": sum(marks.values()), "cpu_seconds": usage.ru_utime + usage.ru_stime, "major_faults": usage.ru_majflt, "read_bytes": int(io["read_bytes"]), "rchar": int(io["rchar"])}
if len(sys.argv) > 1:
    files = {getattr(module, name, None) for module in list(sys.modules.values()) for name in ("__file__", "__cached__")}
    for line in open("/proc/self/maps").read().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            files.add(parts[5])
    files = sorted(path for path in files if isinstance(path, str) and os.path.isfile(path))
    json.dump(files, open(sys.argv[1], "w"))
    record["files"] = len(files)
print(json.dumps(record))
"""


PAGE_LAB_BODY = r"""import ctypes
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
code_dir = bundle / "code"
lab_dir = Path("/kaggle/working/page-lab")
lab_dir.mkdir(parents=True, exist_ok=True)
(lab_dir / "imports.py").write_text(STARTUP_IMPORTS)
child_environment = os.environ.copy()
child_environment.update(PYTHONPATH=str(code_dir), KNEE_BUNDLE=str(bundle))
lab = {}
file_list = lab_dir / "import-files.json"
page = os.sysconf("SC_PAGE_SIZE")
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

def import_probe(label, extra=(), flags=()):
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, *flags, str(lab_dir / "imports.py"), *extra], env=child_environment, cwd=str(code_dir), capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-3000:])
    record = json.loads(completed.stdout.strip().splitlines()[-1])
    record["process_seconds"] = time.perf_counter() - started
    record["stderr"] = completed.stderr
    print(label, json.dumps({key: value for key, value in record.items() if key != "stderr"}))
    return record

def resident(path):
    size = os.path.getsize(path)
    if not size:
        return []
    descriptor = os.open(path, os.O_RDONLY)
    try:
        address = libc.mmap(None, size, 1, 1, descriptor, 0)
    finally:
        os.close(descriptor)
    if address in (None, ctypes.c_void_p(-1).value):
        return []
    count = (size + page - 1) // page
    vector = (ctypes.c_ubyte * count)()
    libc.mincore(address, size, vector)
    libc.munmap(address, size)
    present = np.frombuffer(vector, dtype=np.uint8) & 1
    edges = np.flatnonzero(np.diff(np.concatenate(([0], present, [0]))))
    return [[int(start) * page, int(end - start) * page] for start, end in zip(edges[::2], edges[1::2])]

def drop_caches():
    try:
        os.sync()
        Path("/proc/sys/vm/drop_caches").write_text("3")
        return True
    except OSError:
        for path in json.loads(file_list.read_text()):
            try:
                descriptor = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
        return False

def read_ranges(item):
    path, ranges = item
    total = 0
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        for offset, length in ranges:
            while length > 0:
                chunk = os.pread(descriptor, min(length, 1 << 23), offset)
                if not chunk:
                    break
                total += len(chunk)
                offset += len(chunk)
                length -= len(chunk)
    finally:
        os.close(descriptor)
    return total

def will_need(item):
    path, ranges = item
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        for offset, length in ranges:
            os.posix_fadvise(descriptor, offset, length, os.POSIX_FADV_WILLNEED)
    finally:
        os.close(descriptor)
    return sum(length for _, length in ranges)

def pre_read(method, threads=16):
    started = time.perf_counter()
    items = sorted(json.loads((lab_dir / "ranges.json").read_text()).items(), key=lambda item: -sum(length for _, length in item[1]))
    with ThreadPoolExecutor(threads) as pool:
        total = sum(pool.map(method, items))
    return {"seconds": time.perf_counter() - started, "gib": total / 2**30, "files": len(items), "threads": threads}

def concurrent(count, method=None):
    started = time.perf_counter()
    with ThreadPoolExecutor(count + 1) as pool:
        warming = pool.submit(pre_read, method) if method else None
        probes = [pool.submit(import_probe, f"concurrent-{index}") for index in range(count)]
        records = [probe.result() for probe in probes]
    for record in records:
        record.pop("stderr")
    return {"wall_seconds": time.perf_counter() - started, "probes": records, "pre_read": warming.result() if warming else None}

def save():
    (profile_dir / "page-lab.json").write_text(json.dumps(lab, indent=1))

lab["cold"] = import_probe("cold", [str(file_list)])
lab["cold"].pop("stderr")
ranges = {path: resident(path) for path in json.loads(file_list.read_text())}
ranges = {path: value for path, value in ranges.items() if value}
(lab_dir / "ranges.json").write_text(json.dumps(ranges))
(profile_dir / "ranges.json").write_text(json.dumps(ranges))
lab["resident"] = {"files": len(ranges), "gib": sum(length for value in ranges.values() for _, length in value) / 2**30, "runs": sum(len(value) for value in ranges.values())}
print(json.dumps(lab["resident"]))
lab["drop_caches"] = drop_caches()
lab["reset_cold"] = import_probe("reset_cold")
lab["reset_cold"].pop("stderr")
save()
for label, method, threads in (("pread16", read_ranges, 16), ("pread32", read_ranges, 32), ("willneed16", will_need, 16)):
    drop_caches()
    lab[label] = {"pre_read": pre_read(method, threads)}
    lab[label]["probe"] = import_probe(label)
    lab[label]["probe"].pop("stderr")
    save()
drop_caches()
lab["two"] = concurrent(2)
save()
drop_caches()
lab["two_pread"] = concurrent(2, read_ranges)
save()
importtime = import_probe("importtime", flags=("-X", "importtime"))
rows = []
for line in importtime.pop("stderr").splitlines():
    match = re.match(r"import time:\s+(\d+) \|\s+(\d+) \| (\s*)(\S+)", line)
    if match:
        rows.append({"self_us": int(match.group(1)), "cumulative_us": int(match.group(2)), "depth": len(match.group(3)) // 2, "module": match.group(4)})
lab["importtime_top"] = sorted((row for row in rows if row["depth"] <= 1), key=lambda row: -row["cumulative_us"])[:60]
lab["importtime_self_top"] = sorted(rows, key=lambda row: -row["self_us"])[:40]
save()
print(json.dumps(lab["importtime_top"][:30], indent=0))
"""

PAGE_LAB_CELL = (
    "STARTUP_IMPORTS = " + json.dumps(STARTUP_IMPORTS) + "\n" + PAGE_LAB_BODY
)

COMPILE_CHILD = r"""import json, os, sys, time
from pathlib import Path
import numpy as np
bundle, data_root, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec = json.loads(sys.argv[4])
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
sys.modules["wandb"] = None
from knee_abnormality.inference.processes import use_blocking_sync
use_blocking_sync()
torch.set_num_threads(1)
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.inference import engine
from knee_abnormality.inference.checkpoint import load_models
config = load_config(bundle / "config.json", data_root=data_root)
runtime = json.loads((bundle / "manifest.json").read_text())["runtime"]
result = {"mode": spec["mode"], "member": spec["member"]}
dataset, targets, _ = engine.prediction_data(config, "test", 0, image_multiplier=runtime["image_multiplier"], study_ids=spec["studies"])
batches = list(build_data_loader(dataset, config.trainer, 4, num_workers=4, pin_memory=False, prepare_on_loader=False))
cuda = torch.device("cuda")
manifest = json.loads((bundle / "manifest.json").read_text())
loaded = load_models(config, targets, bundle / "models" if "checkpoints" in manifest else bundle / "model.ckpt", cuda, attention_temperature=runtime["attention_temperature"])
members = spec["member"] if isinstance(spec["member"], list) else [spec["member"]]
models = [loaded[index] for index in members]
result["backbone"] = [type(model.backbone).__name__ for model in models]
size = engine.BACKBONE_CHUNK_IMAGES
result["compile_seconds"] = []

def compile_backbone(model):
    model.channels_last = True
    model.backbone.to(memory_format=torch.channels_last)
    compiled = torch.compile(model._backbone_features, mode=spec["mode"], dynamic=False)

    def chunked(images):
        maps, extras = [], []
        for start in range(0, images.shape[0], size):
            chunk = images[start : start + size]
            count = chunk.shape[0]
            if count < size:
                chunk = torch.cat((chunk, chunk.new_zeros((size - count, *chunk.shape[1:]))))
            features, extra = compiled(chunk.contiguous(memory_format=torch.channels_last))
            maps.append(features[:count].clone())
            if extra is not None:
                extras.append(extra[:count].clone())
        return torch.cat(maps), (torch.cat(extras) if extras else None)

    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        chunked(torch.zeros((size, *batches[0]["images"].shape[1:]) if "images" in batches[0] else (size, 3, config.data.image_size, config.data.image_size), device=cuda))
    torch.cuda.synchronize()
    result["compile_seconds"].append(time.perf_counter() - started)
    model._backbone_features = chunked

for model in models:
    if spec["mode"] == "graph":
        engine.use_fixed_backbone_chunks(model, config.data.image_size)
    else:
        compile_backbone(model)

def run():
    _, predictions, _ = engine.predict_loader(models, [dict(batch) for batch in batches], cuda, "fp16", False, None)
    return predictions

run()
torch.cuda.synchronize()
started = time.perf_counter()
predictions = run()
torch.cuda.synchronize()
result["seconds_per_batch"] = (time.perf_counter() - started) / len(batches)
result["batches"] = len(batches)
np.save(output.with_suffix(".npy"), predictions.numpy())
output.write_text(json.dumps(result))
print(json.dumps(result), flush=True)
"""


COMPILE_LAB_BODY = r"""code_dir = bundle / "code"
lab_dir = Path("/kaggle/working/compile-lab")
lab_dir.mkdir(parents=True, exist_ok=True)
(lab_dir / "child.py").write_text(COMPILE_CHILD)
results = {}
import numpy as np
for label, mode, cache, *member_reference in CONDITION["runs"]:
    member, reference_label = member_reference or (0, "graph")
    values = os.environ.copy()
    values.update(PYTHONPATH=str(code_dir), TORCHINDUCTOR_CACHE_DIR=f"/kaggle/working/inductor-{cache}", TRITON_CACHE_DIR=f"/kaggle/working/inductor-{cache}/triton")
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, str(lab_dir / "child.py"), str(bundle), str(data_root), str(lab_dir / f"{label}.json"), json.dumps({"mode": mode, "studies": STUDIES, "member": member})], env=values, cwd=str(code_dir), capture_output=True, text=True)
    record = json.loads((lab_dir / f"{label}.json").read_text()) if completed.returncode == 0 else {"error": completed.stderr[-4000:]}
    record["process_seconds"] = time.perf_counter() - started
    if completed.returncode == 0 and label != reference_label and (lab_dir / f"{reference_label}.npy").exists():
        reference = np.load(lab_dir / f"{reference_label}.npy"); values = np.load(lab_dir / f"{label}.npy")
        record["max_abs_difference"] = float(np.abs(values - reference).max())
        record["mean_abs_difference"] = float(np.abs(values - reference).mean())
    results[label] = record
    print(label, json.dumps(record)[:1500], flush=True)
    (profile_dir / "compile-lab.json").write_text(json.dumps(results, indent=1))
"""

COMPILE_LAB_CELL = (
    "COMPILE_CHILD = " + json.dumps(COMPILE_CHILD) + "\n" + COMPILE_LAB_BODY
)


PAIRED_BODY = r"""base_profile = profile_dir
def drop_cached(root):
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                descriptor = os.open(os.path.join(directory, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
primary_bundle = bundle
for arm in CONDITION["arms"]:
    bundle = primary_bundle
    if arm.get("bundle_slug"):
        bundle = next(path.parent for path in Path("/kaggle/input/datasets").glob("*/*/manifest.json") if path.parent.name == arm["bundle_slug"])
    for uid in STUDIES:
        drop_cached(competition / "train_series" / uid)
    drop_cached(bundle)
    profile_dir = base_profile / arm["name"]
    profile_dir.mkdir(parents=True, exist_ok=True)
    CONDITION["environment"] = arm["environment"]
    print("arm", arm["name"], flush=True)
    exec(PREDICTION, globals())
"""


def notebook_source(
    bundle: Path,
    condition: dict,
    studies: list[str],
    throughput_studies: list[str],
    dataset: str,
) -> dict:
    setup = (
        "import time\nnotebook_started_at = time.time()\n"
        + BUNDLE_SELECTION.format(slug=dataset.split("/", 1)[1])
        + "".join(
            json.loads((bundle / "inference.ipynb").read_text())["cells"][0]["source"]
        )
    )
    profile = f"""import csv
import os
import threading

CONDITION = json.loads({json.dumps(json.dumps(condition))})
STUDIES = {json.dumps(studies)}
THROUGHPUT_STUDIES = {json.dumps(throughput_studies)}
input_root = Path("/kaggle/input")
competitions = sorted({{path.parent for pattern in ("*/train_series.csv", "*/*/train_series.csv", "*/*/*/train_series.csv") for path in input_root.glob(pattern)}})
if len(competitions) != 1:
    raise RuntimeError(f"Expected one competition root, found {{competitions}}")
competition = competitions[0]
data_root = Path("/tmp/knee-profile-data")
(data_root / "test_series").mkdir(parents=True)
with (competition / "sample_submission.csv").open(newline="") as stream:
    header = next(csv.reader(stream))
with (data_root / "sample_submission.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows([uid] + ["0.5"] * (len(header) - 1) for uid in STUDIES)
selected = set(STUDIES)
with (competition / "train_series.csv").open(newline="") as source, (data_root / "test_series.csv").open("w", newline="") as target:
    reader = csv.reader(source)
    writer = csv.writer(target)
    writer.writerow(next(reader))
    writer.writerows(row for row in reader if row[0] in selected)
for uid in STUDIES:
    (data_root / "test_series" / uid).symlink_to(competition / "train_series" / uid, target_is_directory=True)

def read_command(arguments):
    try:
        return subprocess.run(arguments, capture_output=True, text=True, check=False).stdout
    except OSError as error:
        return str(error)

profile_dir = Path("/kaggle/working/profile")
profile_dir.mkdir(parents=True, exist_ok=True)
cpuinfo = Path("/proc/cpuinfo").read_text().splitlines()
environment = {{
    "condition": CONDITION,
    "studies": len(STUDIES),
    "competition_root": str(competition),
    "cpu_count": os.cpu_count(),
    "cpu_affinity": len(os.sched_getaffinity(0)),
    "cpu_model": next((line.split(":", 1)[1].strip() for line in cpuinfo if line.startswith("model name")), None),
    "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
    "meminfo": Path("/proc/meminfo").read_text().splitlines()[:3],
    "gpus": read_command(["nvidia-smi", "-L"]),
    "mounts": [line for line in Path("/proc/mounts").read_text().splitlines() if "/kaggle" in line],
    "shm_bytes": shutil.disk_usage("/dev/shm").total,
}}
(profile_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\\n")
print(json.dumps(environment))
"""
    prediction = """stop = threading.Event()

def sample_cpu():
    with (profile_dir / "cpu.csv").open("w") as stream:
        while not stop.is_set():
            fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:9]
            cgroup = Path("/sys/fs/cgroup/cpu.stat")
            usage = cgroup.read_text().split()[1] if cgroup.exists() else "nan"
            stream.write(f"{time.time()},{','.join(fields)},{usage}\\n")
            stream.flush()
            stop.wait(1)

def sample_processes():
    ticks = os.sysconf("SC_CLK_TCK")
    page = os.sysconf("SC_PAGE_SIZE")
    heartbeat = []
    beats = 0
    start = time.time()
    with (profile_dir / "processes.csv").open("w") as stream:
        stream.write("timestamp,pid,ppid,cpu_seconds,rss_mib,command\\n")
        while not stop.is_set():
            now = time.time()
            for name in os.listdir("/proc"):
                if not name.isdigit():
                    continue
                try:
                    stat = Path("/proc", name, "stat").read_text().rsplit(") ", 1)[1].split()
                    command = Path("/proc", name, "cmdline").read_bytes().decode("utf-8", "replace")
                except (OSError, IndexError):
                    continue
                command = command.replace("\\x00", " ").strip()[:160].replace(",", " ")
                if not command:
                    continue
                cpu = (int(stat[11]) + int(stat[12])) / ticks
                rss = int(stat[21]) * page / 2**20
                stream.write(f"{now},{name},{stat[1]},{cpu:.3f},{rss:.1f},{command}\\n")
                if "predict" in command:
                    heartbeat.append((name, cpu, rss))
            stream.flush()
            beats += 1
            if beats % 20 == 0:
                print(f"heartbeat {now - start:.0f}s", " ".join(f"{pid}:{cpu:.0f}s/{rss:.0f}M" for pid, cpu, rss in heartbeat), [log.read_text(errors="replace").strip().splitlines()[-1:] for log in sorted(profile_dir.glob(".*/worker-*.log"))], flush=True)
            heartbeat.clear()
            stop.wait(1)

sampler = threading.Thread(target=sample_cpu, daemon=True)
sampler.start()
process_sampler = threading.Thread(target=sample_processes, daemon=True)
process_sampler.start()
gpu_log = (profile_dir / "gpu.csv").open("w")
gpu_monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "1000"], stdout=gpu_log, stderr=subprocess.STDOUT) if shutil.which("nvidia-smi") else None
command = [
    sys.executable, "-m", "scripts.pipeline.predict", str(bundle / "config.json"),
    "--bundle", str(bundle),
    "--data-root", str(data_root),
    "--output", str(profile_dir / "predictions.csv"),
    "--device", CONDITION.get("device", "cuda"),
    "--precision", CONDITION["precision"],
    "--batch-size", str(CONDITION["batch_size"]),
    "--num-workers", str(CONDITION["num_workers"]),
]
if CONDITION.get("io_threads") is not None:
    command += ["--io-threads", str(CONDITION["io_threads"])]
if CONDITION.get("pyspy") or CONDITION.get("deadline"):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "py-spy"], check=True)
if CONDITION.get("pyspy"):
    command = [shutil.which("py-spy"), "record", "--subprocesses", "--nonblocking", "--rate", "20", "--format", "raw", "--output", str(profile_dir / "pyspy.txt"), "--", *command]
environment_variables = os.environ.copy()
environment_variables["PYTHONPATH"] = str(bundle / "code")
environment_variables.update(CONDITION.get("environment", {}))
record = {"notebook_started_at": notebook_started_at, "cell_started_at": time.time()}
if CONDITION.get("pycache_prefix"):
    environment_variables["PYTHONPYCACHEPREFIX"] = "/tmp/knee-pycache"
if CONDITION.get("warm_packages"):
    import importlib.util
    from concurrent.futures import ThreadPoolExecutor
    warm_started = time.time()
    roots = []
    for name in ("torch", "nvidia", "triton", "timm", "kornia", "kornia_rs", "scipy", "numpy", "polars", "pydicom", "pydantic", "pydantic_core", "pydantic_settings", "sklearn", "pytorch_lightning", "lightning_fabric", "torchmetrics", "yaml", "safetensors", "huggingface_hub", "PIL"):
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.submodule_search_locations:
            roots.extend(spec.submodule_search_locations)
    package_files = [path for root in roots for path in Path(root).rglob("*") if path.is_file()]
    listed_at = time.time()
    def read_file(path):
        total = 0
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1 << 24):
                    total += len(chunk)
        except OSError:
            pass
        return total
    with ThreadPoolExecutor(32) as pool:
        warm_bytes = sum(pool.map(read_file, package_files))
    record["warm_packages"] = {"roots": roots, "files": len(package_files), "gib": warm_bytes / 2**30, "list_seconds": listed_at - warm_started, "read_seconds": time.time() - listed_at}
    print(json.dumps(record["warm_packages"]))
prefetch = {"studies_read": 0, "gib": 0.0, "throttled_seconds": 0.0, "max_lead": 0, "progress": -1}
def loader_progress(own_pid, index_of):
    best = -1
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == own_pid:
            continue
        try:
            for descriptor in (process / "fd").iterdir():
                target = os.readlink(descriptor)
                if target.endswith(".dcm"):
                    best = max(best, index_of.get(target.split("/")[-3], -1))
        except OSError:
            continue
    return best
def list_series():
    from concurrent.futures import ThreadPoolExecutor
    started = time.time()
    with ThreadPoolExecutor(32) as pool:
        series = list(pool.map(lambda uid: sorted((competition / "train_series" / uid).glob("*/*.dcm")), STUDIES))
    prefetch["list_seconds"] = time.time() - started
    return series
def prefetch_dicoms(limit):
    from concurrent.futures import ThreadPoolExecutor
    index_of = {uid: index for index, uid in enumerate(STUDIES)}
    started = time.time()
    series = list_series()
    own_pid = os.getpid()
    total = 0
    with ThreadPoolExecutor(16) as pool:
        for index, paths in enumerate(series):
            while not stop.is_set():
                progress = max(prefetch["progress"], loader_progress(own_pid, index_of))
                prefetch["progress"] = progress
                if index - progress <= limit:
                    break
                prefetch["throttled_seconds"] += 0.25
                time.sleep(0.25)
            if stop.is_set():
                break
            total += sum(pool.map(lambda path: len(path.read_bytes()), paths))
            prefetch["studies_read"] = index + 1
            prefetch["gib"] = total / 2**30
            prefetch["max_lead"] = max(prefetch["max_lead"], index - prefetch["progress"])
    prefetch["finished_after_seconds"] = time.time() - started
launched_at = time.time()
with (profile_dir / "predict.log").open("w") as log:
    process = subprocess.Popen(command, env=environment_variables, stdout=log, stderr=subprocess.STDOUT)
    prefetcher = None
    if CONDITION.get("prefetch_studies"):
        prefetcher = threading.Thread(target=prefetch_dicoms, args=(CONDITION["prefetch_studies"],), daemon=True)
        prefetcher.start()
    elif CONDITION.get("list_series"):
        prefetcher = threading.Thread(target=list_series, daemon=True)
        prefetcher.start()
    try:
        returncode = process.wait(timeout=CONDITION.get("deadline"))
    except subprocess.TimeoutExpired:
        dumps = []
        for name in os.listdir("/proc"):
            try:
                command_line = Path("/proc", name, "cmdline").read_bytes().decode("utf-8", "replace").replace("\\x00", " ")
            except OSError:
                continue
            if "predict" in command_line and name.isdigit():
                dumped = subprocess.run([shutil.which("py-spy"), "dump", "--pid", name, "--locals"], capture_output=True, text=True, timeout=60)
                dumps.append(f"### {name} {command_line[:160]}\\n{dumped.stdout}{dumped.stderr}")
        (profile_dir / "hang.txt").write_text("\\n".join(dumps))
        print("\\n".join(dumps)[-20000:], flush=True)
        process.kill()
        returncode = process.wait()
finished_at = time.time()
stop.set()
if prefetcher is not None:
    prefetcher.join()
    record["prefetch"] = prefetch
sampler.join()
process_sampler.join()
if gpu_monitor is not None:
    gpu_monitor.terminate()
    gpu_monitor.wait()
gpu_log.close()
record.update(launched_at=launched_at, finished_at=finished_at, returncode=returncode)
(profile_dir / "launch.json").write_text(json.dumps(record) + "\\n")
print(json.dumps(record))
lines = (profile_dir / "predict.log").read_text(errors="replace").splitlines()
print("\\n".join(lines[-40:]))
if returncode:
    raise RuntimeError(f"Prediction failed with {returncode}")
"""
    return {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": source,
            }
            for source in (
                setup,
                profile,
                {
                    "loading": LOADING_CELL,
                    "loader-lab": LOADER_LAB_CELL,
                    "precision": PRECISION_CELL,
                    "startup": STARTUP_CELL,
                    "cudnn": CUDNN_CELL,
                    "mainloop": MAINLOOP_CELL,
                    "page-lab": PAGE_LAB_CELL,
                    "compile-lab": COMPILE_LAB_CELL,
                    "variants": VARIANTS_CELL,
                    "parity": PARITY_CELL,
                    "census": CENSUS_CELL,
                    "paired": "PREDICTION = "
                    + json.dumps(prediction)
                    + "\n"
                    + PAIRED_BODY,
                }.get(condition["kind"], prediction),
            )
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def ensure_dataset(api: KaggleApi, bundle: Path, dataset: str, args: Launch) -> dict:
    manifest_sha256 = sha256_file(bundle / "manifest.json")
    description = f"Private inference profiling bundle {manifest_sha256}"
    if _dataset_exists(api, dataset):
        with tempfile.TemporaryDirectory(prefix="knee-profile-") as temporary:
            metadata = json.loads(
                Path(api.dataset_metadata(dataset, temporary)).read_text()
            )
        if (metadata.get("info", {}).get("description") or "").splitlines()[
            -1:
        ] not in (
            [description],
            [f"Private inference bundle {manifest_sha256}"],
        ):
            raise ValueError(f"Dataset {dataset} holds a different bundle")
    else:
        verify_bundle(bundle)
        with tempfile.TemporaryDirectory(prefix="knee-profile-") as temporary:
            directory = Path(temporary)
            shutil.copytree(bundle, directory / "bundle")
            (directory / "dataset-metadata.json").write_text(
                json.dumps(
                    {
                        "id": dataset,
                        "title": dataset.split("/", 1)[1].replace("-", " ").title(),
                        "licenses": [{"name": "other"}],
                        "description": description,
                    },
                    indent=2,
                )
                + "\n"
            )
            response = api.dataset_create_new(
                str(directory), public=False, quiet=False, dir_mode="zip"
            )
            if response is None or getattr(response, "error", None):
                raise RuntimeError(f"Dataset upload failed: {_jsonable(response)}")
    while True:
        try:
            if api.dataset_status(dataset) == "ready":
                return _dataset_identity(api, dataset)
        except (RuntimeError, HTTPError):
            pass
        time.sleep(args.poll_seconds)


def write_status(path: Path, status: dict, owned: list[str]) -> None:
    current = json.loads(path.read_text()) if path.exists() else {}
    runs = current.get("runs", {}) | {
        name: value for name, value in status["runs"].items() if name in owned
    }
    path.write_text(json.dumps(current | status | {"runs": runs}, indent=2) + "\n")


def launch(args: Launch) -> dict:
    unknown = sorted(set(args.conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Choose known conditions; unknown: {unknown}")
    bundle = args.bundle.expanduser().resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if not manifest.get("ready"):
        raise ValueError(f"Bundle is not ready: {manifest.get('unmet_prerequisites')}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.setdefault("runs", {})
    studies = profile_studies(args)
    api = KaggleApi()
    api.authenticate()
    status["dataset"] = ensure_dataset(api, bundle, args.dataset, args)
    status["bundle"] = str(bundle)
    write_status(status_path, status, args.conditions)
    owner = args.dataset.split("/", 1)[0]
    pending = list(args.conditions)
    active = {}
    while pending or active:
        while pending and len(active) < args.concurrency:
            name = pending[0]
            slug = f"knee-profile-{name}-{args.notebook_suffix}"
            notebook = f"{owner}/{slug}"
            directory = output / "notebooks" / name
            directory.mkdir(parents=True, exist_ok=True)
            condition = CONDITIONS[name]
            accelerator = (
                None
                if condition["kind"] in {"loading", "loader-lab", "census"}
                else condition.get("accelerator", args.accelerator)
            )
            tpu = accelerator is not None and accelerator.startswith("Tpu")
            if condition["kind"] in {"prediction", "variants", "paired"}:
                selected, throughput_studies = studies, []
            elif condition["kind"] in {"startup", "cudnn", "page-lab"}:
                selected, throughput_studies = studies[:8], []
            elif condition["kind"] in {
                "mainloop",
                "compile-lab",
                "parity",
            }:
                selected, throughput_studies = studies[: condition["studies"]], []
            elif condition["kind"] == "census":
                everything = (
                    pl.read_csv(args.split_dir / "studies.csv")
                    .sort(STUDY)[STUDY]
                    .to_list()
                )
                selected = everything[condition["part"] :: condition["parts"]]
                throughput_studies = []
            elif condition["kind"] == "precision":
                selected = random.Random(LOADING_SEED).sample(
                    studies, condition["studies"]
                )
                throughput_studies = []
            else:
                sample = random.Random(LOADING_SEED).sample(
                    studies,
                    condition["timed_studies"]
                    + 10 * len(condition["throughput_groups"]),
                )
                selected = sample[: condition["timed_studies"]]
                throughput_studies = sample[condition["timed_studies"] :]
            (directory / "profile.ipynb").write_text(
                json.dumps(
                    notebook_source(
                        bundle, condition, selected, throughput_studies, args.dataset
                    ),
                    indent=1,
                )
                + "\n"
            )
            (directory / "kernel-metadata.json").write_text(
                json.dumps(
                    {
                        "id": notebook,
                        "title": slug.replace("-", " ").title(),
                        "code_file": "profile.ipynb",
                        "language": "python",
                        "kernel_type": "notebook",
                        "is_private": True,
                        "enable_gpu": accelerator is not None and not tpu,
                        "enable_tpu": tpu,
                        "enable_internet": bool(
                            condition.get("pyspy") or condition.get("internet")
                        ),
                        "competition_sources": [COMPETITION],
                        "dataset_sources": [args.dataset]
                        + ([args.reference_dataset] if args.reference_dataset else []),
                    },
                    indent=2,
                )
                + "\n"
            )
            response = api.kernels_push(
                str(directory),
                str(args.notebook_timeout_seconds),
                accelerator,
            )
            error = (
                getattr(response, "error", None) if response is not None else "empty"
            )
            if error and "Maximum batch GPU session" in str(error):
                break
            pending.pop(0)
            run = {
                "notebook": notebook,
                "condition": condition,
                "studies": len(selected),
                "pushed_at": datetime.now(UTC).isoformat(),
                "push_response": _jsonable(response),
            }
            if error:
                run["state"] = "push_failed"
            else:
                run["state"] = "pushed"
                active[name] = {
                    "notebook": notebook,
                    "seen_active": False,
                    "pushed": time.monotonic(),
                }
            status["runs"][name] = run
            write_status(status_path, status, args.conditions)
        time.sleep(args.poll_seconds)
        for name, item in list(active.items()):
            state, response = _kernel_status(api, item["notebook"])
            terminal = any(value in state for value in TERMINAL)
            if not terminal:
                item["seen_active"] = True
                status["runs"][name]["state"] = state
                if "running" in state.lower():
                    status["runs"][name].setdefault(
                        "running_at", datetime.now(UTC).isoformat()
                    )
                write_status(status_path, status, args.conditions)
                continue
            status["runs"][name]["completed_at"] = datetime.now(UTC).isoformat()
            if not item["seen_active"] and time.monotonic() - item["pushed"] < 600:
                continue
            run_dir = output / "runs" / name
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)
            api.kernels_output(
                item["notebook"], path=str(run_dir), force=True, quiet=True
            )
            (run_dir / "notebook.log").write_text(api.kernels_logs(item["notebook"]))
            status["runs"][name].update(
                state=state,
                finished_at=datetime.now(UTC).isoformat(),
                notebook_status=_jsonable(response),
            )
            write_status(status_path, status, args.conditions)
            del active[name]
    return status


def cpu_usage(path: Path, start: float, end: float) -> dict:
    rows = [
        [float(value) for value in line.split(",")]
        for line in path.read_text().splitlines()
        if line
    ]
    window = np.array([row for row in rows if start <= row[0] <= end])
    if len(window) < 2:
        return {"busy": None, "iowait": None, "cgroup_cores": None}
    delta = window[-1] - window[0]
    total = delta[1:9].sum()
    return {
        "busy": float(1 - (delta[4] + delta[5]) / total),
        "iowait": float(delta[5] / total),
        "cgroup_cores": float(delta[9] / 1e6 / delta[0]),
    }


def process_usage(path: Path, start: float, end: float) -> dict:
    if not path.exists():
        return {}
    samples: dict[str, list] = {}
    for line in path.read_text().splitlines()[1:]:
        parts = line.split(",", 5)
        if len(parts) != 6:
            continue
        stamp = float(parts[0])
        if not start <= stamp <= end:
            continue
        samples.setdefault(parts[1], []).append(
            (float(parts[3]), float(parts[4]), parts[2], parts[5])
        )
    predicting = {
        pid for pid, rows in samples.items() if "predict_worker" in rows[0][3]
    }
    mains = {pid for pid in predicting if samples[pid][0][2] not in predicting}
    roles: dict[str, dict] = {}
    for pid, rows in samples.items():
        command = rows[0][3]
        if pid in mains:
            role = "gpu_main"
        elif pid in predicting:
            role = "loader_worker"
        elif "scripts.pipeline.predict" in command:
            role = "coordinator"
        else:
            role = "other"
        entry = roles.setdefault(
            role, {"processes": 0, "cpu_seconds": 0.0, "peak_rss_mib": 0.0}
        )
        entry["processes"] += 1
        entry["cpu_seconds"] += rows[-1][0] - rows[0][0]
        entry["peak_rss_mib"] = max(entry["peak_rss_mib"], max(row[1] for row in rows))
    total = sum(entry["cpu_seconds"] for entry in roles.values())
    for entry in roles.values():
        entry["cpu_share"] = entry["cpu_seconds"] / total if total else None
    return {
        "seconds": end - start,
        "cpu_seconds": total,
        "cores_in_use": total / (end - start),
        "roles": roles,
    }


def gpu_usage(path: Path, start: float, end: float) -> dict:
    if not path.stat().st_size:
        return {"mean_utilization_while_loaded": None, "max_memory_mib": None}
    frame = (
        pl.read_csv(
            path,
            has_header=False,
            new_columns=["timestamp", "index", "utilization", "memory_mib"],
            schema_overrides={"utilization": pl.Float64, "memory_mib": pl.Float64},
            ignore_errors=True,
        )
        .drop_nulls()
        .with_columns(
            pl.col("timestamp")
            .str.strip_chars()
            .str.to_datetime("%Y/%m/%d %H:%M:%S%.3f", time_zone="UTC", strict=False)
            .dt.epoch("s")
            .alias("epoch")
        )
    )
    window = frame.filter(pl.col("epoch").is_between(start, end))
    return {
        "samples": window.height,
        "seconds": end - start,
        "mean_utilization": float(window["utilization"].mean())
        if window.height
        else None,
        "max_memory_mib": float(window["memory_mib"].max()) if window.height else None,
    }


def loading_summary(result: dict) -> dict:
    passes = {}
    for name in ("cold", "warm"):
        value = result[name]
        timing = defaultdict(float, value["timings"])
        processor = defaultdict(float, value["cpu_seconds"])
        count = defaultdict(int, value["counts"])
        thread_timing = defaultdict(float, value["thread_seconds"])
        thread_count = defaultdict(int, value["thread_counts"])
        studies = len(value["studies"])
        getitem = sum(item["getitem_seconds"] for item in value["studies"])
        collate = sum(item["collate_seconds"] for item in value["studies"])

        def split(source):
            sampling = source["build_pool"] + source["select_triplets"]
            views = source["view_position"] + source["fallback_window_center"]
            return {
                "series_listing": source["build_series_index"],
                "header_reads": source["index_series"] - source["parse_prefix"],
                "header_parse": source["parse_prefix"],
                "series_cleaning": source["clean_series"],
                "reference_and_side": source["estimate_reference"]
                + source["split_knee_geometry"]
                + source["resolve_knee_side"],
                "geometry_other": source["_index_study_geometry"]
                + source["_slot_sources"]
                - source["index_series"]
                - source["clean_series"]
                - source["estimate_reference"]
                - source["split_knee_geometry"]
                - source["resolve_knee_side"],
                "sampling": sampling,
                "view_position": views,
                "row_reads": source["decode_series"] - source["decode_slice"],
                "fallback_decode": source["decode_slice"],
                "prepare_other": source["prepare_series"] - source["decode_series"],
                "slot_other": source["_prepare_slot"]
                - sampling
                - views
                - source["prepare_series"],
            }

        stages = {
            **split(timing),
            "setup_other": value["setup_seconds"] - timing["build_series_index"],
            "getitem_other": getitem
            - timing["_index_study_geometry"]
            - timing["_slot_sources"]
            - timing["_study_budgets"]
            - timing["_prepare_slot"],
            "collate": collate,
        }
        total = value["setup_seconds"] + getitem + collate
        passes[name] = {
            "studies": studies,
            "total_seconds": total,
            "seconds_per_study": total / studies,
            "pass_seconds": value["pass_seconds"],
            "pass_cpu_seconds": value["pass_cpu_seconds"],
            "cpu_seconds_per_wall_second": value["pass_cpu_seconds"]
            / value["pass_seconds"],
            "minor_faults": value["minor_faults"],
            "major_faults": value["major_faults"],
            "images": sum(item["images"] for item in value["studies"]),
            "slice_headers": count["parse_prefix"],
            "milliseconds_per_header_parse": 1000
            * timing["parse_prefix"]
            / max(count["parse_prefix"], 1),
            "thread_milliseconds_per_prefix_read": 1000
            * thread_timing["_safe_prefix"]
            / max(thread_count["_safe_prefix"], 1),
            "thread_milliseconds_per_row_read": 1000
            * thread_timing["_read_rows"]
            / max(thread_count["_read_rows"], 1),
            "stages_seconds": stages,
            "stages_fraction": {key: item / total for key, item in stages.items()},
            "stages_cpu_seconds": split(processor),
            "thread_seconds": dict(thread_timing),
            "thread_counts": dict(thread_count),
        }
    return {
        "passes": passes,
        "cold_minus_warm_seconds": {
            key: passes["cold"]["stages_seconds"][key]
            - passes["warm"]["stages_seconds"][key]
            for key in passes["cold"]["stages_seconds"]
        },
        "throughput": result["throughput"],
    }


def lab_summary(result: dict) -> dict:
    variants = {}
    fingerprints = {}
    timed_out = defaultdict(list)
    for value in result["passes"]:
        if value.get("timed_out"):
            timed_out[value["variant"]].append(value["label"])
            continue
        if value["label"] == "fingerprint":
            fingerprints[value["variant"]] = {
                item["study"]: item["fingerprint"] for item in value["studies"]
            }
            continue
        total = (
            value["setup_seconds"]
            + sum(item["getitem_seconds"] for item in value["studies"])
            + sum(item["collate_seconds"] for item in value["studies"])
        )
        variants.setdefault(value["variant"], []).append(
            {
                "total_seconds": total,
                "pass_seconds": value["pass_seconds"],
                "user_seconds": value["user_seconds"],
                "system_seconds": value["system_seconds"],
                "cpu_seconds": value["user_seconds"] + value["system_seconds"],
                "syscalls_read": value["io"].get("syscr", 0),
                "read_megabytes": value["io"].get("rchar", 0) / 2**20,
                "nfs_calls": value["nfs_calls"],
                "functions": value["functions"],
            }
        )
    reference = fingerprints.get(result["variants"][0], {})
    summary = {
        key: result.get(key) for key in ("kernel", "transparent_hugepages", "eviction")
    } | {"variants": {}, "timed_out": dict(timed_out)}
    for name, runs in variants.items():
        mean = {
            key: float(np.mean([run[key] for run in runs]))
            for key in (
                "total_seconds",
                "pass_seconds",
                "user_seconds",
                "system_seconds",
                "cpu_seconds",
                "syscalls_read",
                "read_megabytes",
            )
        }
        functions = defaultdict(lambda: defaultdict(float))
        for run in runs:
            for function, record in run["functions"].items():
                for key, item in record.items():
                    functions[function][key] += item / len(runs)
        nfs = defaultdict(float)
        for run in runs:
            for key, item in run["nfs_calls"].items():
                nfs[key] += item / len(runs)
        own = fingerprints.get(name, {})
        mismatched = {
            study: sorted(
                key
                for key in set(values) | set(reference.get(study, {}))
                if values.get(key) != reference.get(study, {}).get(key)
            )
            for study, values in own.items()
            if values != reference.get(study)
        }
        summary["variants"][name] = {
            "rounds": len(runs),
            "mean": mean,
            "runs": [
                {key: run[key] for key in ("total_seconds", "cpu_seconds")}
                for run in runs
            ],
            "nfs_calls": {key: item for key, item in nfs.items() if item},
            "functions": {
                key: dict(value)
                for key, value in sorted(
                    functions.items(), key=lambda pair: -pair[1]["wall"]
                )
            },
            "fingerprint_studies": len(own),
            "fingerprint_mismatches": mismatched,
        }
    return summary


def analyze(args: Analyze) -> dict:
    output = args.output_dir.expanduser().resolve()
    scoring = args.target_studies is None
    truth_frame = fold_frame(args)
    targets = [
        name
        for name in truth_frame.columns
        if name not in {STUDY, "report_group", "is_gold", "fold"}
    ]
    truth = truth_frame.select(targets).to_numpy().astype(float)
    gold = truth_frame["is_gold"].to_numpy()
    predictions = {}
    reference_values = None
    if scoring:
        reference = pl.read_csv(args.reference_predictions).sort(STUDY)
        if reference[STUDY].to_list() != truth_frame[STUDY].to_list():
            raise ValueError("Reference predictions do not cover the fold")
        reference_values = reference.select(targets).to_numpy().astype(float)
        predictions["local_bf16_reference"] = reference_values
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "studies": truth_frame.height if scoring else args.target_studies,
        "scoring": scoring,
        "runs": {},
    }
    for run_dir in sorted((output / "runs").glob("*")):
        lab = next(iter(run_dir.rglob("lab.json")), None)
        if lab is not None:
            summary["runs"][run_dir.name] = lab_summary(json.loads(lab.read_text()))
            continue
        loading = next(iter(run_dir.rglob("loading.json")), None)
        if loading is not None:
            summary["runs"][run_dir.name] = loading_summary(
                json.loads(loading.read_text())
            )
            continue
        profile = next(iter(run_dir.rglob("predictions.runtime.json")), None)
        if profile is None:
            summary["runs"][run_dir.name] = {"state": "missing_outputs"}
            continue
        profile_dir = profile.parent
        runtime = json.loads(profile.read_text())
        launch_record = json.loads((profile_dir / "launch.json").read_text())
        environment = json.loads((profile_dir / "environment.json").read_text())
        scores = {}
        if scoring:
            frame = pl.read_csv(profile_dir / "predictions.csv").sort(STUDY)
            if frame[STUDY].to_list() != truth_frame[STUDY].to_list():
                raise ValueError(f"{run_dir.name} predictions do not cover the fold")
            values = frame.select(targets).to_numpy().astype(float)
            predictions[run_dir.name] = values
            metrics = validation_metrics(truth, values, targets, gold)
            scores = {
                "macro_auc": metrics["macro_auc"],
                "per_target_auc": metrics["per_target_auc"],
                "max_abs_difference_from_local_bf16": float(
                    np.abs(values - reference_values).max()
                ),
                "mean_abs_difference_from_local_bf16": float(
                    np.abs(values - reference_values).mean()
                ),
            }
        workers = []
        for worker in runtime["workers"]:
            steady = worker["batches"][1:]
            steady_seconds = sum(
                batch["wait_seconds"]
                + batch["transfer_seconds"]
                + batch["forward_seconds"]
                for batch in steady
            )
            workers.append(
                {
                    name: worker[name]
                    for name in (
                        "gpu_name",
                        "studies",
                        "images",
                        "startup_seconds",
                        "data_setup_seconds",
                        "model_load_seconds",
                        "first_batch_wait_seconds",
                        "loader_wait_seconds",
                        "transfer_seconds",
                        "forward_seconds",
                        "prediction_seconds",
                        "peak_allocated_gib",
                    )
                }
                | {
                    "steady_studies_per_second": sum(
                        batch["studies"] for batch in steady
                    )
                    / steady_seconds,
                    "steady_wait_fraction": sum(
                        batch["wait_seconds"] for batch in steady
                    )
                    / steady_seconds,
                    "forward_seconds_per_1000_images": 1000
                    * sum(batch["forward_seconds"] for batch in steady)
                    / sum(batch["images"] for batch in steady),
                }
            )
        spawn = min(worker["spawned_at"] for worker in runtime["workers"])
        predicting = max(
            worker["started_at"]
            + worker.get(
                "setup_wall_seconds",
                worker["data_setup_seconds"] + worker["model_load_seconds"],
            )
            for worker in runtime["workers"]
        )
        end = max(
            worker["started_at"]
            + worker.get(
                "setup_wall_seconds",
                worker["data_setup_seconds"] + worker["model_load_seconds"],
            )
            + worker["prediction_seconds"]
            for worker in runtime["workers"]
        )
        summary["runs"][run_dir.name] = {
            "environment": {
                name: environment[name]
                for name in (
                    "cpu_count",
                    "cpu_affinity",
                    "cpu_model",
                    "cgroup_cpu_max",
                    "gpus",
                )
            },
            "condition": environment["condition"],
            "precision": runtime["precision"],
            **scores,
            "command_seconds": launch_record["finished_at"]
            - launch_record["launched_at"],
            "notebook": {
                key: launch_record[key]
                for key in ("warm_packages", "prefetch")
                if key in launch_record
            }
            | (
                {
                    "notebook_to_launch_seconds": launch_record["launched_at"]
                    - launch_record["notebook_started_at"],
                    "notebook_to_finish_seconds": launch_record["finished_at"]
                    - launch_record["notebook_started_at"],
                }
                if "notebook_started_at" in launch_record
                else {}
            ),
            "coordinator_import_seconds": runtime["started_at"]
            - launch_record["launched_at"],
            "verify_seconds": runtime["verify_seconds"],
            "setup_seconds": runtime["setup_seconds"],
            "worker_wall_seconds": runtime["worker_wall_seconds"],
            "studies_per_second": runtime["studies_per_second"],
            "cpu_during_setup": cpu_usage(profile_dir / "cpu.csv", spawn, predicting),
            "cpu_during_prediction": cpu_usage(
                profile_dir / "cpu.csv", predicting, end
            ),
            "processes_during_setup": process_usage(
                profile_dir / "processes.csv", spawn, predicting
            ),
            "processes_during_prediction": process_usage(
                profile_dir / "processes.csv", predicting, end
            ),
            "gpu": gpu_usage(profile_dir / "gpu.csv", predicting, end),
            "workers": workers,
        }
    if "fp32-b4-w4" in predictions and "fp16-b4-w4" in predictions:
        fp32 = summary["runs"]["fp32-b4-w4"]
        fp16 = summary["runs"]["fp16-b4-w4"]
        summary["fp16_minus_fp32"] = {
            "macro_auc": fp16["macro_auc"] - fp32["macro_auc"],
            "per_target_auc": {
                target: fp16["per_target_auc"][target] - fp32["per_target_auc"][target]
                for target in targets
            },
            "max_abs_prediction_difference": float(
                np.abs(predictions["fp16-b4-w4"] - predictions["fp32-b4-w4"]).max()
            ),
            "mean_abs_prediction_difference": float(
                np.abs(predictions["fp16-b4-w4"] - predictions["fp32-b4-w4"]).mean()
            ),
        }
    if scoring:
        summary["local_bf16_reference_macro_auc"] = validation_metrics(
            truth, reference_values, targets, gold
        )["macro_auc"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    CliApp.run(ProfileCLI)


if __name__ == "__main__":
    main()
