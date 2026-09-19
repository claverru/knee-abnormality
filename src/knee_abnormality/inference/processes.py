import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import FrameType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from knee_abnormality.config import (
    INFERENCE_MALLOC_ARENA_MAX,
    INFERENCE_MALLOC_THRESHOLD_BYTES,
    ExperimentConfig,
    Fold,
    PositiveFiniteFloat,
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: ExperimentConfig
    checkpoint: Path
    split: Literal["test", "validation"]
    fold: Fold
    image_multiplier: int = Field(gt=0)
    attention_temperature: PositiveFiniteFloat
    reflection_tta: bool
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    io_threads: int = Field(gt=0)
    device: Literal["cpu", "cuda"]
    precision: Literal["auto", "fp32", "fp16", "bf16"]
    study_ids: list[str] = Field(min_length=1)
    worker_index: int = Field(ge=0)
    output: Path


def _terminate(signum: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def _signal_process(process: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def run_prediction_processes(
    requests: list[PredictionRequest],
    gpu_ids: list[str | None],
    directory: Path,
) -> list[dict]:
    if not requests or len(requests) != len(gpu_ids):
        raise ValueError("Each inference worker requires one device assignment")
    processes = []
    logs = []
    spawned_at = []
    previous_handler = signal.signal(signal.SIGTERM, _terminate)
    try:
        with ExitStack() as stack:
            for request, gpu_id in zip(requests, gpu_ids, strict=True):
                path = directory / f"worker-{request.worker_index}.json"
                path.write_text(request.model_dump_json(indent=2) + "\n")
                log_path = directory / f"worker-{request.worker_index}.log"
                logs.append(log_path)
                log = stack.enter_context(log_path.open("w", encoding="utf-8"))
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = "" if gpu_id is None else gpu_id
                environment.update(
                    MALLOC_MMAP_THRESHOLD_=str(INFERENCE_MALLOC_THRESHOLD_BYTES),
                    MALLOC_TRIM_THRESHOLD_=str(INFERENCE_MALLOC_THRESHOLD_BYTES),
                    MALLOC_ARENA_MAX=str(INFERENCE_MALLOC_ARENA_MAX),
                )
                if gpu_id is not None:
                    environment.update(
                        OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                        POLARS_MAX_THREADS="1",
                    )
                spawned_at.append(time.time())
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "scripts.pipeline.predict_worker",
                            str(path),
                        ],
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
            pending = set(range(len(processes)))
            while pending:
                for index in tuple(pending):
                    status = processes[index].poll()
                    if status is None:
                        continue
                    if status != 0:
                        _signal_process(processes[index], signal.SIGTERM)
                        with logs[index].open("rb") as stream:
                            stream.seek(max(0, logs[index].stat().st_size - 12000))
                            detail = stream.read().decode(errors="replace")
                        raise RuntimeError(
                            f"Inference worker {index} exited with {status}:\n{detail}"
                        )
                    pending.remove(index)
                if pending:
                    time.sleep(0.1)
    finally:
        try:
            for process in processes:
                if process.poll() is None:
                    _signal_process(process, signal.SIGTERM)
            deadline = time.monotonic() + 10
            for process in processes:
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    _signal_process(process, signal.SIGKILL)
            for process in processes:
                process.wait()
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    results = []
    for request, gpu_id, spawned in zip(requests, gpu_ids, spawned_at, strict=True):
        result = json.loads(request.output.with_suffix(".runtime.json").read_text())
        result["visible_gpu_id"] = gpu_id
        result["spawned_at"] = spawned
        result["startup_seconds"] = result["started_at"] - spawned
        results.append(result)
    return results
