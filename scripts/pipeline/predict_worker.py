import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import Field
from pydantic_settings import CliPositionalArg

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import INFERENCE_LOADER_NICENESS
from knee_abnormality.data.sources import (
    StudySeries,
    build_series_index,
    prediction_studies,
    select_studies,
)
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.processes import PredictionRequest, use_blocking_sync


class PredictionWorkerCLI(CLISettings):
    model_config = describe(
        "Predict one study partition on one device for the prediction coordinator"
    )

    request: CliPositionalArg[Path] = Field(
        description="Prediction request JSON written by the coordinator"
    )


def lower_priority(worker_id: int) -> None:
    os.nice(INFERENCE_LOADER_NICENESS)


def index_studies(
    request: PredictionRequest, study_ids: list[str], metadata_split: str
) -> tuple[dict[str, StudySeries], float]:
    indexing = time.perf_counter()
    index = build_series_index(
        request.config.paths.data_root,
        request.config.data,
        metadata_split,
        study_ids,
    )
    return index, time.perf_counter() - indexing


def predict_request(request: PredictionRequest) -> dict:
    started_at = time.time()
    started = time.perf_counter()
    targets, studies, metadata_split = prediction_studies(
        request.config, request.split, request.fold
    )
    studies = select_studies(studies, request.study_ids)
    with ThreadPoolExecutor(1) as pool:
        indexing = pool.submit(
            index_studies, request, studies[STUDY].to_list(), metadata_split
        )
        import torch

        from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader

        imported = time.perf_counter()
        series_index, data_setup_seconds = indexing.result()
    indexed = time.perf_counter()
    if request.device == "cuda":
        use_blocking_sync()
        torch.set_num_threads(1)
    loader = build_data_loader(
        KneeStudyDataset(
            studies,
            series_index,
            request.config.data,
            targets,
            None,
            request.image_multiplier,
            request.io_threads,
        ),
        request.config.trainer,
        request.batch_size,
        num_workers=request.num_workers,
        pin_memory=False,
        prepare_on_loader=False,
    )
    loader.in_order = False
    loader.worker_init_fn = lower_priority
    batches = iter(loader)
    forked = time.perf_counter()
    sys.modules["wandb"] = None
    from knee_abnormality.evaluation.predictions import make_prediction_frame
    from knee_abnormality.inference.checkpoint import load_models
    from knee_abnormality.inference.engine import (
        BACKBONE_CHUNK_IMAGES,
        predict_loader,
        resolve_device,
        resolve_precision,
        use_fixed_backbone_chunks,
    )

    loading = time.perf_counter()
    torch.set_float32_matmul_precision("high")
    device = resolve_device(request.device)
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("A prediction worker must see exactly one GPU")
    actual_precision, _ = resolve_precision(request.precision, device)
    models = load_models(
        request.config,
        targets,
        request.checkpoint,
        device,
        expected_fold=request.fold if request.split == "validation" else None,
        attention_temperature=request.attention_temperature,
    )
    if actual_precision == "fp16":
        for model in models:
            use_fixed_backbone_chunks(model, request.config.data.image_size)
    loaded = time.perf_counter()
    study_ids, predictions, records = predict_loader(
        models, batches, device, request.precision, request.reflection_tta
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
        "reflection_tta": request.reflection_tta,
        "member_count": len(models),
        "studies": frame.height,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "precision": actual_precision,
        "backbone_chunk_images": BACKBONE_CHUNK_IMAGES
        if actual_precision == "fp16"
        else None,
        "batch_size": request.batch_size,
        "num_workers": request.num_workers,
        "io_threads": request.io_threads,
        "started_at": started_at,
        "import_seconds": imported - started,
        "data_setup_seconds": data_setup_seconds,
        "loader_start_seconds": forked - indexed,
        "model_import_seconds": loading - forked,
        "model_load_seconds": loaded - loading,
        "setup_wall_seconds": loaded - started,
        "prediction_seconds": predicted - loaded,
        "loader_wait_seconds": sum(record["wait_seconds"] for record in records),
        "first_batch_wait_seconds": records[0]["wait_seconds"],
        "transfer_seconds": sum(record["transfer_seconds"] for record in records),
        "forward_seconds": sum(record["forward_seconds"] for record in records),
        "images": sum(record["images"] for record in records),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
        if device.type == "cuda"
        else None,
        "batches": records,
    }
    request.output.with_suffix(".runtime.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = PredictionWorkerCLI()
    request = PredictionRequest.model_validate(json.loads(args.request.read_text()))
    print(json.dumps(predict_request(request)))


if __name__ == "__main__":
    main()
