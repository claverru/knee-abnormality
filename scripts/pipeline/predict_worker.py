import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from pydantic_settings import CliPositionalArg, SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.data.sources import prediction_studies, select_studies
from knee_abnormality.evaluation.predictions import make_prediction_frame
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import (
    BACKBONE_CHUNK_IMAGES,
    predict_loader,
    prediction_dataset,
    resolve_device,
    resolve_precision,
    use_fixed_backbone_chunks,
)
from knee_abnormality.inference.processes import PredictionRequest


class PredictionWorkerCLI(CLISettings):
    request: CliPositionalArg[Path]


def predict_request(request: PredictionRequest) -> dict:
    started_at = time.time()
    started = time.perf_counter()
    torch.set_float32_matmul_precision("high")
    device = resolve_device(request.device)
    if device.type == "cuda":
        if torch.cuda.device_count() != 1:
            raise ValueError("A prediction worker must see exactly one GPU")
        torch.set_num_threads(1)
    actual_precision, _ = resolve_precision(request.precision, device)
    targets, studies, metadata_split = prediction_studies(
        request.config, request.split, request.fold
    )
    studies = select_studies(studies, request.study_ids)

    def load_model():
        loading = time.perf_counter()
        model = load_checkpoint_model(
            request.config,
            targets,
            request.checkpoint,
            device,
            expected_fold=request.fold if request.split == "validation" else None,
            attention_temperature=request.attention_temperature,
        )
        if actual_precision == "fp16":
            use_fixed_backbone_chunks(model, request.config.data.image_size)
        return model, time.perf_counter() - loading

    def prepare_data():
        indexing = time.perf_counter()
        dataset = prediction_dataset(
            request.config,
            targets,
            studies,
            metadata_split,
            request.image_multiplier,
        )
        loader = build_data_loader(
            dataset,
            request.config.trainer,
            request.batch_size,
            num_workers=request.num_workers,
        )
        loader.in_order = False
        return loader, time.perf_counter() - indexing

    with ThreadPoolExecutor(1) as pool:
        preparing = pool.submit(prepare_data)
        model, model_load_seconds = load_model()
        loader, data_setup_seconds = preparing.result()
    loaded = time.perf_counter()
    study_ids, predictions, batches = predict_loader(
        model, loader, device, request.precision
    )
    predicted = time.perf_counter()
    frame = make_prediction_frame(
        study_ids,
        predictions,
        targets,
        request.fold if request.split == "validation" else None,
    )
    if frame.height != studies.height or set(study_ids) != set(request.study_ids):
        raise ValueError("Worker predictions do not match its assigned studies")
    frame.write_csv(request.output)
    result = {
        "worker_index": request.worker_index,
        "attention_temperature": request.attention_temperature,
        "studies": frame.height,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "precision": actual_precision,
        "backbone_chunk_images": BACKBONE_CHUNK_IMAGES
        if actual_precision == "fp16"
        else None,
        "batch_size": request.batch_size,
        "num_workers": request.num_workers,
        "started_at": started_at,
        "data_setup_seconds": data_setup_seconds,
        "model_load_seconds": model_load_seconds,
        "setup_wall_seconds": loaded - started,
        "prediction_seconds": predicted - loaded,
        "loader_wait_seconds": sum(batch["wait_seconds"] for batch in batches),
        "first_batch_wait_seconds": batches[0]["wait_seconds"],
        "transfer_seconds": sum(batch["transfer_seconds"] for batch in batches),
        "forward_seconds": sum(batch["forward_seconds"] for batch in batches),
        "images": sum(batch["images"] for batch in batches),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
        if device.type == "cuda"
        else None,
        "batches": batches,
    }
    request.output.with_suffix(".runtime.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        args = PredictionWorkerCLI()
        request = PredictionRequest.model_validate(json.loads(args.request.read_text()))
        print(json.dumps(predict_request(request)))
    except (OSError, RuntimeError, TypeError, ValueError, SettingsError) as error:
        print(f"Prediction worker error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
