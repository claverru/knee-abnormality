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
    "startup": {"kind": "startup"},
    "cudnn": {"kind": "cudnn"},
    "mainloop": {"kind": "mainloop", "studies": 64},
    "parity": {"kind": "parity", "studies": 300},
    **{
        f"census-{part}": {"kind": "census", "part": part, "parts": 5}
        for part in range(5)
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


class ProfileCLI(CLISettings):
    command: CliPositionalArg[Literal["launch", "analyze"]]
    output_dir: Path
    bundle: Path | None = None
    dataset: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
    conditions: list[str] = Field(default_factory=list)
    notebook_suffix: str = "20260915"
    fold: int = Field(default=0, ge=0)
    target_studies: int | None = Field(default=None, gt=0)
    split_dir: Path = Path("data/folds/all_5fold_v3")
    reference_predictions: Path = Path(
        "outputs/training/reflection-sigma40/fold_0/validation_predictions.csv"
    )
    concurrency: int = Field(default=2, gt=0)
    poll_seconds: int = Field(default=60, gt=0)
    accelerator: str = "NvidiaTeslaT4"
    notebook_timeout_seconds: int = Field(default=32400, gt=0, le=32400)
    reference_dataset: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")


def profile_studies(args: ProfileCLI) -> list[str]:
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


def fold_frame(args: ProfileCLI) -> pl.DataFrame:
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
    (dataset_module, "estimate_reference"),
    (dataset_module, "build_pool"),
    (dataset_module, "select_triplets"),
    (dataset_module, "prepare_series"),
    (dataset_module, "decode_series"),
    (preparation_module, "resample_crop"),
):
    timed(module, name)

for module, name in (
    (preparation_module, "read_frame"),
    (preparation_module, "read_payload"),
    (preparation_module, "frame_from_payload"),
    (dicom_module, "_read_pydicom_frame"),
):
    timed_on_threads(module, name)

def reset_process_caches():
    dicom_module._TAGS_CACHE.clear()
    spatial_module._axis_filter.cache_clear()
    spatial_module.resample_plan.cache_clear()
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


PARITY_SCRIPT = """import hashlib, json, sys
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
loader = build_data_loader(dataset, config.trainer, 1, num_workers=4)
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
for name, root in (("reference", reference[0]), ("candidate", bundle)):
    environment_values = os.environ.copy()
    environment_values["PYTHONPATH"] = str(root / "code")
    started = time.perf_counter()
    completed = subprocess.run([sys.executable, "-c", {PARITY_SCRIPT!r}, str(root), str(data_root), str(profile_dir / f"{{name}}.json"), str(studies_path)], env=environment_values, capture_output=True, text=True)
    parity[name] = {{"seconds": time.perf_counter() - started, "returncode": completed.returncode, "stderr": completed.stderr[-6000:]}}
    print(name, parity[name]["seconds"], completed.returncode, completed.stderr[-3000:])
reference_records = json.loads((profile_dir / "reference.json").read_text())
candidate_records = json.loads((profile_dir / "candidate.json").read_text())
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
parity.update(studies=len(reference_records), series=series_total, mismatches=mismatches, constant=sum(sum(record["constant"] or []) for record in candidate_records.values()))
(profile_dir / "parity.json").write_text(json.dumps(parity, indent=1) + "\\n")
print(json.dumps({{key: value for key, value in parity.items() if key != "mismatches"}}, indent=1), len(mismatches))
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
import torch
from collections import defaultdict

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.data.preprocessing import device as device_module
from knee_abnormality.inference import engine
from knee_abnormality.inference.checkpoint import load_models

config = load_config(bundle / "config.json", data_root=data_root)
manifest = json.loads((bundle / "manifest.json").read_text())
dataset, targets, _ = engine.prediction_data(config, "test", 0, image_multiplier=manifest["runtime"]["image_multiplier"], study_ids=STUDIES)
loader = build_data_loader(dataset, config.trainer, 4, num_workers=4, pin_memory=False)
cuda = torch.device("cuda")
models = load_models(config, targets, bundle / "model.ckpt", cuda, attention_temperature=manifest["runtime"]["attention_temperature"])
engine.use_fixed_backbone_chunks(models[0], config.data.image_size)
timings = defaultdict(float)
sizes = defaultdict(int)
original_sort = torch.Tensor.sort

def timed(name, function, *arguments):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = function(*arguments)
    torch.cuda.synchronize()
    timings[name] += time.perf_counter() - started
    return value

def sort(self, *arguments, **keywords):
    return timed("prepare.sort", lambda: original_sort(self, *arguments, **keywords))

batches = list(loader)
with torch.inference_mode():
    for repeat in range(2):
        timings.clear()
        for batch in batches:
            sizes["values"] += sum(count for _, count, _ in batch["series_frames"])
            locked = timed("page_lock", lambda: [buffer for buffer in batch["frame_buffers"] if engine._page_lock(buffer)])
            moved = timed("transfer", engine.move_batch, batch, cuda)
            torch.Tensor.sort = sort
            prepared = timed("prepare_images", device_module.prepare_images, moved)
            torch.Tensor.sort = original_sort
            with torch.autocast("cuda", dtype=torch.float16):
                timed("forward", lambda: torch.sigmoid(models[0](prepared).float()).cpu())
            timed("unlock", lambda: [torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr()) for buffer in locked])
        print(repeat, {key: round(value, 3) for key, value in timings.items()})
result = {"batches": len(batches), "per_batch_seconds": {key: value / len(batches) for key, value in timings.items()}, "bytes_per_batch": sum(buffer.numel() * buffer.element_size() for batch in batches for buffer in batch["frame_buffers"]) / len(batches)}
(profile_dir / "mainloop.json").write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps(result, indent=1))
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
            stream.flush()
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
if CONDITION.get("pyspy"):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "py-spy"], check=True)
    command = [shutil.which("py-spy"), "record", "--subprocesses", "--nonblocking", "--rate", "20", "--format", "raw", "--output", str(profile_dir / "pyspy.txt"), "--", *command]
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
                    "precision": PRECISION_CELL,
                    "startup": STARTUP_CELL,
                    "cudnn": CUDNN_CELL,
                    "mainloop": MAINLOOP_CELL,
                    "parity": PARITY_CELL,
                    "census": CENSUS_CELL,
                }.get(condition["kind"], prediction),
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
                if condition["kind"] in {"loading", "census"}
                else condition.get("accelerator", args.accelerator)
            )
            tpu = accelerator is not None and accelerator.startswith("Tpu")
            if condition["kind"] == "prediction":
                selected, throughput_studies = studies, []
            elif condition["kind"] in {"startup", "cudnn"}:
                selected, throughput_studies = studies[:8], []
            elif condition["kind"] in {"mainloop", "parity"}:
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
                        "enable_internet": bool(condition.get("pyspy")),
                        "competition_sources": ["rsna-knee-abnormality-detection"],
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
        sampling = (
            timing["estimate_reference"]
            + timing["build_pool"]
            + timing["select_triplets"]
        )
        sampling_cpu = (
            processor["estimate_reference"]
            + processor["build_pool"]
            + processor["select_triplets"]
        )
        frames = thread_count["read_frame"] + thread_count["frame_from_payload"]
        stages = {
            "series_listing_index": timing["build_series_index"],
            "setup_other": value["setup_seconds"] - timing["build_series_index"],
            "slice_headers": timing["parse_prefix"],
            "series_index_other": timing["index_series"] - timing["parse_prefix"],
            "geometry_and_sampling": sampling,
            "frame_reads": timing["decode_series"] - timing["resample_crop"],
            "resample_crop": timing["resample_crop"],
            "quantiles": timing["_linear_quantiles"],
            "normalize": timing["_normalize_pixels"],
            "prepare_other": timing["prepare_series"]
            - timing["decode_series"]
            - timing["_linear_quantiles"]
            - timing["_normalize_pixels"],
            "getitem_other": getitem
            - timing["index_series"]
            - sampling
            - timing["prepare_series"],
            "collate": sum(item["collate_seconds"] for item in value["studies"]),
        }
        stages_cpu = {
            "series_listing_index": processor["build_series_index"],
            "slice_headers": processor["parse_prefix"],
            "series_index_other": processor["index_series"] - processor["parse_prefix"],
            "geometry_and_sampling": sampling_cpu,
            "frame_reads": processor["decode_series"] - processor["resample_crop"],
            "resample_crop": processor["resample_crop"],
            "quantiles": processor["_linear_quantiles"],
            "normalize": processor["_normalize_pixels"],
            "prepare_other": processor["prepare_series"]
            - processor["decode_series"]
            - processor["_linear_quantiles"]
            - processor["_normalize_pixels"],
        }
        total = value["setup_seconds"] + getitem + stages["collate"]
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
            "frames": frames,
            "pydicom_frames": thread_count["_read_pydicom_frame"],
            "milliseconds_per_header": 1000
            * timing["parse_prefix"]
            / max(count["parse_prefix"], 1),
            "thread_milliseconds_per_frame": 1000
            * (
                thread_timing["read_frame"]
                + thread_timing["read_payload"]
                + thread_timing["frame_from_payload"]
            )
            / max(frames, 1),
            "stages_seconds": stages,
            "stages_fraction": {key: item / total for key, item in stages.items()},
            "stages_cpu_seconds": stages_cpu,
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


def analyze(args: ProfileCLI) -> dict:
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
    try:
        args = ProfileCLI()
        result = launch(args) if args.command == "launch" else analyze(args)
        print(json.dumps(result, indent=2))
    except (OSError, RuntimeError, TypeError, ValueError, SettingsError) as error:
        print(f"Profiling error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
