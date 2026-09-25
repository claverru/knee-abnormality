import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import Field
from pydantic_settings import CliPositionalArg

from knee_abnormality.cli import CLISettings, Runtime, describe
from knee_abnormality.config import (
    ATTENTION_TEMPERATURE,
    INFERENCE_BATCH_SIZE,
    INFERENCE_IO_THREADS,
    INFERENCE_NUM_WORKERS,
    Fold,
    PositiveFiniteFloat,
    load_config,
)
from knee_abnormality.data.labels import STUDY
from knee_abnormality.data.sources import (
    prediction_image_multiplier,
    prediction_studies,
)
from knee_abnormality.inference.bundle import discover_kaggle_data_root, verify_bundle
from knee_abnormality.inference.processes import (
    PredictionRequest,
    cuda_device_count,
    run_prediction_processes,
)


class PredictCLI(CLISettings, Runtime):
    model_config = describe(
        "Predict complete studies with a repository checkpoint or an exported bundle, "
        "spreading them across independent GPU processes"
    )

    config: CliPositionalArg[Path] = Field(
        description="Experiment YAML or saved effective_config.json file"
    )
    bundle: Path | None = Field(
        default=None,
        description="Exported bundle; config must be its config.json and it supplies the checkpoint",
    )
    checkpoint: Path | None = Field(
        default=None, description="Training checkpoint for repository inference"
    )
    output: Path = Field(description="Predictions CSV to write")
    split: Literal["test", "validation"] = Field(
        default="test", description="Studies to predict"
    )
    fold: Fold | None = Field(
        default=None, description="Validation fold; defaults to the config's first fold"
    )
    data_root: Path | None = Field(
        default=None,
        description="Data location; defaults to the config's, or the Kaggle input for bundles",
    )
    label_manifest: Path | None = Field(
        default=None, description="Label manifest replacing the config's"
    )
    batch_size: int | None = Field(
        default=None,
        gt=0,
        description="Studies per GPU batch; defaults to the bundle or repository value",
    )
    num_workers: int | None = Field(
        default=None,
        ge=0,
        description="Total loader workers across all GPUs; defaults to the bundle or repository value",
    )
    io_threads: int | None = Field(
        default=None,
        gt=0,
        description="Reader threads per loader worker; defaults to the repository value",
    )
    image_multiplier: int | None = Field(
        default=None,
        gt=0,
        description="Test-time image multiplier; defaults to the bundle or repository value",
    )
    attention_temperature: PositiveFiniteFloat | None = Field(
        default=None,
        description="Pooling softmax temperature; defaults to the bundle or repository value",
    )


def main() -> None:
    started_at = time.time()
    started = time.perf_counter()
    logging.basicConfig(level=logging.INFO)
    args = PredictCLI()
    bundle = args.bundle.expanduser().resolve() if args.bundle is not None else None
    manifest = verify_bundle(bundle) if bundle is not None else None
    verified = time.perf_counter()
    if args.checkpoint is not None and bundle is not None:
        raise ValueError("A bundle supplies its own checkpoint")
    config_path = args.config.expanduser().resolve()
    if bundle is not None and config_path != bundle / "config.json":
        raise ValueError("Bundle inference must use the bundle's config.json")
    if bundle is None and args.checkpoint is None:
        raise ValueError("Repository inference requires --checkpoint")
    checkpoint_path = (
        (
            args.checkpoint
            if args.checkpoint is not None
            else (
                bundle / "models"
                if "checkpoints" in manifest
                else bundle / "model.ckpt"
            )
        )
        .expanduser()
        .resolve()
    )
    data_root = args.data_root
    if data_root is None and bundle is not None and Path("/kaggle/input").is_dir():
        data_root = discover_kaggle_data_root()
    config = load_config(
        config_path,
        data_root=data_root,
        label_manifest=args.label_manifest,
    )
    runtime = (
        manifest["runtime"]
        if manifest is not None
        else {
            "image_multiplier": args.image_multiplier,
            "attention_temperature": ATTENTION_TEMPERATURE
            if args.attention_temperature is None
            else args.attention_temperature,
            "reflection_tta": False,
            "batch_size": INFERENCE_BATCH_SIZE,
            "num_workers": INFERENCE_NUM_WORKERS,
        }
    )
    if (
        manifest is not None
        and args.image_multiplier is not None
        and args.image_multiplier != runtime["image_multiplier"]
    ):
        raise ValueError("Export a new bundle to change its image multiplier")
    if (
        manifest is not None
        and args.attention_temperature is not None
        and args.attention_temperature != runtime["attention_temperature"]
    ):
        raise ValueError("Export a new bundle to change its attention temperature")
    fold = args.fold if args.fold is not None else config.validation.folds[0]
    targets, studies, _ = prediction_studies(config, args.split, fold)
    image_multiplier = (
        prediction_image_multiplier(config, args.split)
        if runtime["image_multiplier"] is None
        else runtime["image_multiplier"]
    )
    batch_size = runtime["batch_size"] if args.batch_size is None else args.batch_size
    num_workers = (
        runtime["num_workers"] if args.num_workers is None else args.num_workers
    )
    io_threads = INFERENCE_IO_THREADS if args.io_threads is None else args.io_threads
    count = 0 if args.device == "cpu" else cuda_device_count()
    if args.device == "cuda" and count < 1:
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if count else "cpu"
    gpu_ids: list[str | None] = [None]
    if device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_ids = (
            [value.strip() for value in visible.split(",")][:count]
            if visible is not None
            else [str(index) for index in range(count)]
        )
    gpu_ids = gpu_ids[: studies.height]
    worker_count = len(gpu_ids)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info(
        "Inference uses %s processes, %s studies per device, %s total loader workers, %s io threads",
        worker_count,
        batch_size,
        num_workers,
        io_threads,
    )
    spawning = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}-workers-", dir=output.parent
    ) as temporary:
        directory = Path(temporary)
        study_ids = studies[STUDY].to_list()
        requests = [
            PredictionRequest(
                config=config,
                checkpoint=checkpoint_path,
                split=args.split,
                fold=fold,
                image_multiplier=image_multiplier,
                attention_temperature=runtime["attention_temperature"],
                reflection_tta=runtime["reflection_tta"],
                batch_size=batch_size,
                num_workers=num_workers // worker_count
                + int(index < num_workers % worker_count),
                io_threads=io_threads,
                device=device,
                precision=args.precision,
                study_ids=study_ids[index::worker_count],
                worker_index=index,
                output=directory / f"predictions-{index}.csv",
            )
            for index in range(worker_count)
        ]
        workers = run_prediction_processes(requests, gpu_ids, directory)
        predicted = time.perf_counter()
        frame = pl.concat([pl.read_csv(request.output) for request in requests])
        columns = [STUDY]
        if args.split == "validation":
            columns.append("fold")
        columns.extend(targets)
        if (
            frame.columns != columns
            or frame.height != studies.height
            or frame[STUDY].n_unique() != studies.height
            or set(frame[STUDY]) != set(study_ids)
        ):
            raise ValueError(
                "Worker outputs do not cover every requested study exactly once"
            )
        frame = studies.select(STUDY).join(
            frame,
            on=STUDY,
            how="left",
            validate="1:1",
            maintain_order="left",
        )
        if frame.select(
            pl.any_horizontal(
                pl.col(targets).is_null()
                | ~pl.col(targets).is_finite()
                | ~pl.col(targets).is_between(0, 1)
            ).any()
        ).item():
            raise ValueError("Worker outputs contain missing or invalid predictions")
        if args.split == "validation" and not (frame["fold"] == fold).all():
            raise ValueError("Worker outputs contain the wrong validation fold")
        staged_output = directory / "merged.csv"
        frame.write_csv(staged_output)
        precisions = {worker["precision"] for worker in workers}
        result = {
            "output": str(output),
            "studies": frame.height,
            "device": device,
            "devices": worker_count,
            "precision": next(iter(precisions)) if len(precisions) == 1 else "mixed",
            "image_multiplier": image_multiplier,
            "attention_temperature": runtime["attention_temperature"],
            "reflection_tta": runtime["reflection_tta"],
            "batch_size": batch_size,
            "batch_size_scope": "per_device",
            "num_workers": num_workers,
            "num_workers_scope": "total",
            "started_at": started_at,
            "verify_seconds": verified - started,
            "setup_seconds": spawning - verified,
            "worker_wall_seconds": predicted - spawning,
            "write_seconds": time.perf_counter() - predicted,
            "studies_per_second": frame.height / (predicted - spawning),
            "workers": workers,
        }
        staged_runtime = directory / "runtime.json"
        staged_runtime.write_text(json.dumps(result, indent=2) + "\n")
        staged_runtime.replace(output.with_suffix(".runtime.json"))
        staged_output.replace(output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
