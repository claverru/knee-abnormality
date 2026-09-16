import logging
import time
from typing import Literal

import polars as pl
import torch
from torch.utils.data import DataLoader

from knee_abnormality.config import (
    SLICES_PER_INPUT,
    TEST_IMAGE_MULTIPLIER,
    ExperimentConfig,
)
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.sources import (
    build_series_index,
    prediction_studies,
    select_studies,
)
from knee_abnormality.modeling.model import KneeAbnormalityModel

BACKBONE_CHUNK_IMAGES = 64


def use_fixed_backbone_chunks(model: KneeAbnormalityModel, image_size: int) -> None:
    features = model.backbone.forward_features

    def chunked_features(images: torch.Tensor) -> torch.Tensor:
        count = images.shape[0]
        padded = -(-count // BACKBONE_CHUNK_IMAGES) * BACKBONE_CHUNK_IMAGES
        if padded != count:
            images = torch.cat(
                (images, images.new_zeros((padded - count, *images.shape[1:])))
            )
        images = images.contiguous(memory_format=torch.channels_last)
        return torch.cat(
            [
                features(images[start : start + BACKBONE_CHUNK_IMAGES])
                for start in range(0, padded, BACKBONE_CHUNK_IMAGES)
            ]
        )[:count]

    model.channels_last = True
    model.backbone.to(memory_format=torch.channels_last)
    model.backbone.forward_features = chunked_features
    device = next(model.parameters()).device
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        chunked_features(
            torch.zeros(
                (BACKBONE_CHUNK_IMAGES, SLICES_PER_INPUT, image_size, image_size),
                device=device,
            )
        )
    torch.cuda.synchronize(device)


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


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_loader(
    model: KneeAbnormalityModel,
    loader: DataLoader,
    device: torch.device,
    precision: RuntimePrecision = "auto",
) -> tuple[list[str], torch.Tensor, list[dict]]:
    study_ids = []
    outputs = []
    batches = []
    actual_precision, autocast_dtype = resolve_precision(precision, device)
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
            batch = move_batch(batch, device)
            synchronize(device)
            transferred = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                outputs.append(torch.sigmoid(model(batch).to(torch.float32)).cpu())
            finished = time.perf_counter()
            batches.append(
                {
                    **counts,
                    "wait_seconds": received - requested,
                    "transfer_seconds": transferred - received,
                    "forward_seconds": finished - transferred,
                }
            )
            requested = time.perf_counter()
    return study_ids, torch.cat(outputs), batches


def prediction_image_multiplier(config: ExperimentConfig, split: str) -> int:
    return (
        config.data.validation_image_multiplier
        if split == "validation"
        else TEST_IMAGE_MULTIPLIER
    )


def prediction_dataset(
    config: ExperimentConfig,
    targets: tuple[str, ...],
    studies: pl.DataFrame,
    metadata_split: str,
    image_multiplier: int,
) -> KneeStudyDataset:
    return KneeStudyDataset(
        studies,
        build_series_index(config.paths.data_root, config.data, metadata_split),
        config.data,
        targets,
        False,
        config.seed,
        image_multiplier,
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
