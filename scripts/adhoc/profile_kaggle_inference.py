import json
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from kaggle.api.kaggle_api_extended import KaggleApi
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY
from knee_abnormality.evaluation.metrics import validation_metrics
from knee_abnormality.inference.bundle import verify_bundle
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
}
LOADING_SEED = 20260915
TERMINAL = ("complete", "error", "fail", "cancel")


class ProfileCLI(CLISettings):
    command: CliPositionalArg[Literal["launch", "analyze"]]
    output_dir: Path
    bundle: Path | None = None
    dataset: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
    conditions: list[str] = Field(default_factory=list)
    notebook_suffix: str = "20260915"
    fold: int = Field(default=0, ge=0)
    split_dir: Path = Path("data/folds/all_5fold_v3")
    reference_predictions: Path = Path(
        "outputs/training/reflection-sigma40/fold_0/validation_predictions.csv"
    )
    concurrency: int = Field(default=2, gt=0)
    poll_seconds: int = Field(default=60, gt=0)
    accelerator: str = "NvidiaTeslaT4"
    notebook_timeout_seconds: int = Field(default=32400, gt=0, le=32400)


def fold_frame(args: ProfileCLI) -> pl.DataFrame:
    return (
        pl.read_csv(args.split_dir / "studies.csv")
        .filter(pl.col("fold") == args.fold)
        .join(pl.read_csv(args.split_dir / "targets.csv"), on=STUDY, validate="1:1")
        .sort(STUDY)
    )


LOADING_CELL = """import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(bundle / "code"))
import torch

torch.set_num_threads(1)
from knee_abnormality.config import load_config
from knee_abnormality.data import dataset as dataset_module
from knee_abnormality.data.preprocessing import dicom as dicom_module
from knee_abnormality.data.preprocessing import preparation as preparation_module
from knee_abnormality.data.preprocessing import series as series_module
from knee_abnormality.inference import engine as engine_module

config = load_config(bundle / "config.json", data_root=data_root)
manifest = json.loads((bundle / "manifest.json").read_text())
read_header = dicom_module.read_header
timings = defaultdict(float)
counts = defaultdict(int)

def timed(module, name):
    original = getattr(module, name)
    def wrapper(*arguments, **keywords):
        started = time.perf_counter()
        try:
            return original(*arguments, **keywords)
        finally:
            timings[name] += time.perf_counter() - started
            counts[name] += 1
    setattr(module, name, wrapper)

for module, name in (
    (engine_module, "build_series_index"),
    (dataset_module, "index_series"),
    (series_module, "read_header"),
    (dataset_module, "estimate_reference"),
    (dataset_module, "build_pool"),
    (dataset_module, "select_triplets"),
    (dataset_module, "prepare_series"),
    (preparation_module, "read_frame"),
    (dicom_module, "_read_pydicom_frame"),
    (preparation_module, "resample_crop"),
    (preparation_module, "_linear_quantiles"),
    (preparation_module, "_normalize_pixels"),
):
    timed(module, name)

def run_pass():
    timings.clear()
    counts.clear()
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
    return {"setup_seconds": setup_seconds, "timings": dict(timings), "counts": dict(counts), "studies": studies}

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


PRECISION_CELL = """sys.path.insert(0, str(bundle / "code"))
import torch

torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")
from knee_abnormality.config import load_config
from knee_abnormality.data import dataset as dataset_module
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch, prediction_data

config = load_config(bundle / "config.json", data_root=data_root)
manifest = json.loads((bundle / "manifest.json").read_text())
dataset, targets, _ = prediction_data(config, "test", 0, image_multiplier=manifest["runtime"]["image_multiplier"], study_ids=STUDIES)
items = [dataset[index] for index in range(len(dataset))]
batches = [dataset_module.collate_studies(items[index:index + 4]) for index in range(0, len(items), 4)]
device = torch.device("cuda:0")
model = load_checkpoint_model(config, targets, bundle / "model.ckpt", device)
gpu_batches = [move_batch(batch, device) for batch in batches]
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


def notebook_source(
    bundle: Path,
    condition: dict,
    studies: list[str],
    throughput_studies: list[str],
) -> dict:
    setup = "import time\nnotebook_started_at = time.time()\n" + "".join(
        json.loads((bundle / "inference.ipynb").read_text())["cells"][0]["source"]
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

sampler = threading.Thread(target=sample_cpu, daemon=True)
sampler.start()
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
environment_variables = os.environ.copy()
environment_variables["PYTHONPATH"] = str(bundle / "code")
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
    returncode = process.wait()
finished_at = time.time()
stop.set()
if prefetcher is not None:
    prefetcher.join()
    record["prefetch"] = prefetch
sampler.join()
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
                {"loading": LOADING_CELL, "precision": PRECISION_CELL}.get(
                    condition["kind"], prediction
                ),
            )
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def ensure_dataset(
    api: KaggleApi, bundle: Path, dataset: str, args: ProfileCLI
) -> dict:
    description = (
        f"Private inference profiling bundle {sha256_file(bundle / 'manifest.json')}"
    )
    if _dataset_exists(api, dataset):
        with tempfile.TemporaryDirectory(prefix="knee-profile-") as temporary:
            metadata = json.loads(
                Path(api.dataset_metadata(dataset, temporary)).read_text()
            )
        if metadata.get("info", {}).get("description") != description:
            raise ValueError(f"Dataset {dataset} holds a different bundle")
    else:
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
            identity = _dataset_identity(api, dataset)
            if identity["total_bytes"] > 0:
                return identity
        except RuntimeError:
            pass
        time.sleep(args.poll_seconds)


def write_status(path: Path, status: dict, owned: list[str]) -> None:
    current = json.loads(path.read_text()) if path.exists() else {}
    runs = current.get("runs", {}) | {
        name: value for name, value in status["runs"].items() if name in owned
    }
    path.write_text(json.dumps(current | status | {"runs": runs}, indent=2) + "\n")


def launch(args: ProfileCLI) -> dict:
    if args.bundle is None or args.dataset is None:
        raise ValueError("launch requires --bundle and --dataset")
    unknown = sorted(set(args.conditions) - set(CONDITIONS))
    if unknown or not args.conditions:
        raise ValueError(f"Choose known conditions; unknown: {unknown}")
    bundle = args.bundle.expanduser().resolve()
    manifest = verify_bundle(bundle)
    if not manifest.get("ready"):
        raise ValueError(f"Bundle is not ready: {manifest.get('unmet_prerequisites')}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.setdefault("runs", {})
    studies = fold_frame(args)[STUDY].to_list()
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
                if condition["kind"] == "loading"
                else condition.get("accelerator", args.accelerator)
            )
            tpu = accelerator is not None and accelerator.startswith("Tpu")
            if condition["kind"] == "prediction":
                selected, throughput_studies = studies, []
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
                    notebook_source(bundle, condition, selected, throughput_studies),
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
                        "enable_internet": False,
                        "competition_sources": ["rsna-knee-abnormality-detection"],
                        "dataset_sources": [args.dataset],
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
                write_status(status_path, status, args.conditions)
                continue
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


def gpu_usage(path: Path) -> dict:
    if not path.stat().st_size:
        return {"mean_utilization_while_loaded": None, "max_memory_mib": None}
    frame = pl.read_csv(
        path,
        has_header=False,
        new_columns=["timestamp", "index", "utilization", "memory_mib"],
        schema_overrides={"utilization": pl.Float64, "memory_mib": pl.Float64},
        ignore_errors=True,
    ).drop_nulls()
    active = frame.filter(pl.col("memory_mib") > 1000)
    return {
        "mean_utilization_while_loaded": float(active["utilization"].mean())
        if active.height
        else None,
        "max_memory_mib": float(frame["memory_mib"].max()) if frame.height else None,
    }


def loading_summary(result: dict) -> dict:
    passes = {}
    for name in ("cold", "warm"):
        value = result[name]
        timing = defaultdict(float, value["timings"])
        count = defaultdict(int, value["counts"])
        studies = len(value["studies"])
        getitem = sum(item["getitem_seconds"] for item in value["studies"])
        sampling = (
            timing["estimate_reference"]
            + timing["project_reference"]
            + timing["build_pool"]
            + timing["select_triplets"]
        )
        stages = {
            "series_listing_index": timing["build_series_index"],
            "slice_headers": timing["read_header"],
            "series_index_other": timing["index_series"] - timing["read_header"],
            "geometry_and_sampling": sampling,
            "frame_reads": timing["read_frame"],
            "resample_crop": timing["resample_crop"],
            "quantiles": timing["_linear_quantiles"],
            "normalize": timing["_normalize_pixels"],
            "prepare_other": timing["prepare_series"]
            - timing["read_frame"]
            - timing["resample_crop"]
            - timing["_linear_quantiles"]
            - timing["_normalize_pixels"],
            "getitem_other": getitem
            - timing["index_series"]
            - sampling
            - timing["prepare_series"],
            "collate": sum(item["collate_seconds"] for item in value["studies"]),
        }
        total = value["setup_seconds"] + getitem + stages["collate"]
        passes[name] = {
            "studies": studies,
            "total_seconds": total,
            "seconds_per_study": total / studies,
            "images": sum(item["images"] for item in value["studies"]),
            "slice_headers": count["read_header"],
            "frames": count["read_frame"],
            "pydicom_frames": count["_read_pydicom_frame"],
            "milliseconds_per_header": 1000
            * timing["read_header"]
            / max(count["read_header"], 1),
            "milliseconds_per_frame": 1000
            * timing["read_frame"]
            / max(count["read_frame"], 1),
            "stages_seconds": stages,
            "stages_fraction": {key: item / total for key, item in stages.items()},
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


def analyze(args: ProfileCLI) -> dict:
    output = args.output_dir.expanduser().resolve()
    truth_frame = fold_frame(args)
    targets = [
        name
        for name in truth_frame.columns
        if name not in {STUDY, "report_group", "is_gold", "fold"}
    ]
    truth = truth_frame.select(targets).to_numpy().astype(float)
    gold = truth_frame["is_gold"].to_numpy()
    reference = pl.read_csv(args.reference_predictions).sort(STUDY)
    if reference[STUDY].to_list() != truth_frame[STUDY].to_list():
        raise ValueError("Reference predictions do not cover the fold")
    reference_values = reference.select(targets).to_numpy().astype(float)
    predictions = {"local_bf16_reference": reference_values}
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "studies": truth_frame.height,
        "runs": {},
    }
    for run_dir in sorted((output / "runs").glob("*")):
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
        frame = pl.read_csv(profile_dir / "predictions.csv").sort(STUDY)
        if frame[STUDY].to_list() != truth_frame[STUDY].to_list():
            raise ValueError(f"{run_dir.name} predictions do not cover the fold")
        values = frame.select(targets).to_numpy().astype(float)
        predictions[run_dir.name] = values
        metrics = validation_metrics(truth, values, targets, gold)
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
            "macro_auc": metrics["macro_auc"],
            "per_target_auc": metrics["per_target_auc"],
            "max_abs_difference_from_local_bf16": float(
                np.abs(values - reference_values).max()
            ),
            "mean_abs_difference_from_local_bf16": float(
                np.abs(values - reference_values).mean()
            ),
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
            "cpu_during_prediction": cpu_usage(profile_dir / "cpu.csv", spawn, end),
            "gpu": gpu_usage(profile_dir / "gpu.csv"),
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
    summary["local_bf16_reference_macro_auc"] = validation_metrics(
        truth, reference_values, targets, gold
    )["macro_auc"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    try:
        args = ProfileCLI()
        result = launch(args) if args.command == "launch" else analyze(args)
        print(json.dumps(result, indent=2))
    except (OSError, RuntimeError, TypeError, ValueError, SettingsError) as error:
        print(f"Profiling error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
