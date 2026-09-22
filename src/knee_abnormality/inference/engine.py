import logging
import time
from collections.abc import Iterable
from typing import Literal

import polars as pl
import torch

from knee_abnormality.config import (
    INFERENCE_IO_THREADS,
    SLICES_PER_INPUT,
    ExperimentConfig,
)
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.labels import STUDY
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import (
    build_series_index,
    prediction_image_multiplier,
    prediction_studies,
    select_studies,
)
from knee_abnormality.inference.reflection import reflect_batch, sagittal_types
from knee_abnormality.modeling.model import KneeAbnormalityModel

BACKBONE_CHUNK_IMAGES = 64


def use_fixed_backbone_chunks(model: KneeAbnormalityModel, image_size: int) -> None:
    features = model.backbone.forward_features
    model.channels_last = True
    model.backbone.to(memory_format=torch.channels_last)
    device = next(model.parameters()).device
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device)
    with torch.inference_mode():
        static_input = torch.zeros(
            (BACKBONE_CHUNK_IMAGES, SLICES_PER_INPUT, image_size, image_size),
            device=device,
        ).contiguous(memory_format=torch.channels_last)
        stream.wait_stream(torch.cuda.current_stream(device))
        with (
            torch.cuda.stream(stream),
            torch.autocast("cuda", dtype=torch.float16, cache_enabled=False),
        ):
            for _ in range(2):
                features(static_input)
        torch.cuda.current_stream(device).wait_stream(stream)
        with (
            torch.autocast("cuda", dtype=torch.float16, cache_enabled=False),
            torch.cuda.graph(graph),
        ):
            static_output = features(static_input)
    torch.cuda.synchronize(device)

    def chunked_features(images: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, images.shape[0], BACKBONE_CHUNK_IMAGES):
            chunk = images[start : start + BACKBONE_CHUNK_IMAGES]
            static_input[: len(chunk)].copy_(chunk)
            static_input[len(chunk) :].zero_()
            graph.replay()
            outputs.append(static_output[: len(chunk)].clone())
        return torch.cat(outputs)

    model.backbone.forward_features = chunked_features


RuntimePrecision = Literal["auto", "fp32", "fp16", "bf16"]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def resolve_precision(
    requested: RuntimePrecision, device: torch.device
) -> tuple[str, torch.dtype | None]:
    if requested == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported(including_emulation=False):
                return "bf16", torch.bfloat16
            return "fp16", torch.float16
        return "fp32", None
    if requested == "fp32":
        return requested, None
    if requested == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 inference is supported only on CUDA")
        return requested, torch.float16
    if requested == "bf16":
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("BF16 inference requires a CPU or CUDA device")
        return requested, torch.bfloat16
    raise ValueError(f"Unsupported inference precision: {requested}")


MODEL_BATCH_KEYS = frozenset(
    {
        "images",
        "frame_buffers",
        "image_positions",
        "image_sources",
        "reflection_eligible",
        "series_presence",
        "series_study_index",
        "series_type_index",
        "slice_mask",
        "slice_positions",
        "valid_rects",
        "patient_maps",
        "geometry_valid",
        "knee_side_sign",
        "knee_center_x_mm",
    }
)


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return [item.to(device, non_blocking=True) for item in value]


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: _to_device(value, device) if key in MODEL_BATCH_KEYS else value
        for key, value in batch.items()
    }


def _page_lock(tensor: torch.Tensor) -> bool:
    runtime = torch.cuda.cudart()
    return (
        tensor.numel() > 0
        and not tensor.is_pinned()
        and runtime.cudaHostRegister(
            tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0
        )
        == runtime.cudaError.success
    )


def predict_loader(
    models: list[KneeAbnormalityModel],
    loader: Iterable[dict],
    device: torch.device,
    precision: RuntimePrecision = "auto",
    reflection_tta: bool = False,
) -> tuple[list[str], torch.Tensor, list[dict]]:
    study_ids = []
    outputs = []
    batches = []
    actual_precision, autocast_dtype = resolve_precision(precision, device)
    sagittal = (
        sagittal_types(models[0].augmentation_settings.series_types)
        if reflection_tta
        else None
    )
    logging.getLogger(__name__).info(
        "Inference runtime device=%s precision=%s", device, actual_precision
    )
    with torch.inference_mode():
        requested = time.perf_counter()
        for batch in loader:
            received = time.perf_counter()
            study_ids.extend(batch["study_uid"])
            failures = [failure for values in batch["failures"] for failure in values]
            if failures:
                logging.getLogger(__name__).warning(
                    "Preparation conditions for studies=%s details=%s",
                    batch["study_uid"],
                    failures,
                )
            counts = {
                "studies": len(batch["study_uid"]),
                "series": int(batch["slice_mask"].shape[0]),
                "images": int(batch["slice_mask"].sum()),
                "padded_images": int(batch["slice_mask"].numel()),
            }
            locked = (
                [
                    buffer
                    for buffer in batch.get("frame_buffers", ())
                    if _page_lock(buffer)
                ]
                if device.type == "cuda"
                else []
            )
            copy_events = (
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                if device.type == "cuda"
                else None
            )
            if copy_events is not None:
                copy_events[0].record()
            batch = move_batch(batch, device)
            if copy_events is not None:
                copy_events[1].record()
            batch = with_images(batch)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                probabilities = torch.stack(
                    [torch.sigmoid(model(batch).to(torch.float32)) for model in models]
                ).mean(dim=0)
                if reflection_tta:
                    mirrored = reflect_batch(batch, sagittal)
                    reflected_probabilities = torch.stack(
                        [
                            torch.sigmoid(model(mirrored).to(torch.float32))
                            for model in models
                        ]
                    ).mean(dim=0)
                    eligible = batch["reflection_eligible"][:, None]
                    probabilities = torch.where(
                        eligible,
                        (probabilities + reflected_probabilities) / 2,
                        probabilities,
                    )
                outputs.append(probabilities.cpu())
            for value in locked:
                torch.cuda.cudart().cudaHostUnregister(value.data_ptr())
            finished = time.perf_counter()
            transfer_seconds = (
                copy_events[0].elapsed_time(copy_events[1]) / 1000
                if copy_events is not None
                else 0.0
            )
            batches.append(
                {
                    **counts,
                    "wait_seconds": received - requested,
                    "transfer_seconds": transfer_seconds,
                    "forward_seconds": finished - received,
                    "constant_series": int(batch["constant_series"].sum()),
                }
            )
            requested = time.perf_counter()
    return study_ids, torch.cat(outputs), batches


def prediction_dataset(
    config: ExperimentConfig,
    targets: tuple[str, ...],
    studies: pl.DataFrame,
    metadata_split: str,
    image_multiplier: int,
    io_threads: int = INFERENCE_IO_THREADS,
) -> KneeStudyDataset:
    return KneeStudyDataset(
        studies,
        build_series_index(
            config.paths.data_root,
            config.data,
            metadata_split,
            studies[STUDY].to_list(),
        ),
        config.data,
        targets,
        None,
        image_multiplier,
        io_threads,
    )


def prediction_data(
    config: ExperimentConfig,
    split: str,
    fold: int,
    *,
    image_multiplier: int | None = None,
    study_ids: list[str] | None = None,
) -> tuple[KneeStudyDataset, tuple[str, ...], pl.DataFrame]:
    targets, studies, metadata_split = prediction_studies(config, split, fold)
    if study_ids is not None:
        studies = select_studies(studies, study_ids)
    dataset = prediction_dataset(
        config,
        targets,
        studies,
        metadata_split,
        prediction_image_multiplier(config, split)
        if image_multiplier is None
        else image_multiplier,
    )
    return dataset, targets, studies
