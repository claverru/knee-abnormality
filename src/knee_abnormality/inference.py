import logging
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from knee_abnormality.checkpoints import validate_checkpoint
from knee_abnormality.config import ExperimentConfig
from knee_abnormality.dataset import (
    KneeStudyDataset,
    build_data_loader,
    build_series_index,
    resolve_raw_dir,
)
from knee_abnormality.labels import load_fold_labels
from knee_abnormality.model import KneeAbnormalityModel
from knee_abnormality.runtime import move_batch


def load_checkpoint_model(
    config: ExperimentConfig,
    targets: tuple[str, ...],
    checkpoint_path: Path,
    device: torch.device,
    *,
    expected_fold: int | None = None,
) -> KneeAbnormalityModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, config, targets, expected_fold)
    state = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.") for key in state):
        state = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
    model = KneeAbnormalityModel(
        config.model, config.data, len(targets), load_pretrained=False
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


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
        if device.type == "cuda" and torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            return "bf16", torch.bfloat16
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


def predict_loader(
    model: KneeAbnormalityModel,
    loader: DataLoader,
    device: torch.device,
    precision: RuntimePrecision = "auto",
) -> tuple[list[str], torch.Tensor]:
    study_ids = []
    outputs = []
    actual_precision, autocast_dtype = resolve_precision(precision, device)
    logging.getLogger(__name__).info(
        "Inference runtime device=%s precision=%s", device, actual_precision
    )
    with torch.inference_mode():
        for batch in loader:
            study_ids.extend(batch["study_uid"])
            failures = [failure for values in batch["failures"] for failure in values]
            if failures:
                logging.getLogger(__name__).warning(
                    "Preparation conditions for studies=%s details=%s",
                    batch["study_uid"],
                    failures,
                )
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                outputs.append(torch.sigmoid(model(batch).to(torch.float32)).cpu())
    return study_ids, torch.cat(outputs)


def prediction_data(
    config: ExperimentConfig,
    split: str,
    fold: int,
    *,
    image_multiplier: int | None = None,
) -> tuple[KneeStudyDataset, tuple[str, ...], pl.DataFrame]:
    image_limit_multiplier = image_multiplier
    if image_limit_multiplier is None:
        image_limit_multiplier = (
            config.data.validation_image_multiplier if split == "validation" else 2
        )
    if type(image_limit_multiplier) is not int or image_limit_multiplier < 1:
        raise ValueError("Image multiplier must be a positive integer")
    if split == "validation":
        labels = load_fold_labels(config, fold)
        studies = labels.validation
        targets = labels.targets
        metadata_split = "train"
    elif split == "test":
        raw_dir = resolve_raw_dir(config.paths.data_root, "test")
        template = pl.read_csv(raw_dir / "sample_submission.csv")
        targets = tuple(template.columns[1:])
        if template.columns[0] != "StudyInstanceUID":
            raise ValueError("Submission template must start with StudyInstanceUID")
        studies = template.select("StudyInstanceUID")
        metadata_studies = (
            pl.read_csv(raw_dir / "test_series.csv").select("StudyInstanceUID").unique()
        )
        if set(studies["StudyInstanceUID"]) != set(
            metadata_studies["StudyInstanceUID"]
        ):
            raise ValueError(
                "Mounted test series membership differs from sample_submission.csv"
            )
        metadata_split = "test"
    else:
        raise ValueError(f"Unsupported prediction split: {split}")
    index = build_series_index(config.paths.data_root, config.data, metadata_split)
    dataset = KneeStudyDataset(
        studies,
        index,
        config.data,
        targets,
        False,
        config.seed,
        image_limit_multiplier,
    )
    return dataset, targets, studies


def prediction_loader(
    config: ExperimentConfig,
    dataset: KneeStudyDataset,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    actual_batch_size = config.trainer.batch_size if batch_size is None else batch_size
    actual_num_workers = (
        config.trainer.num_workers if num_workers is None else num_workers
    )
    if actual_batch_size < 1 or actual_num_workers < 0:
        raise ValueError("Batch size must be positive and worker count nonnegative")
    return build_data_loader(
        dataset,
        config.trainer,
        batch_size=actual_batch_size,
        num_workers=actual_num_workers,
    )


def make_prediction_frame(
    study_ids: list[str],
    prediction: torch.Tensor | np.ndarray,
    targets: tuple[str, ...],
    fold: int | None = None,
) -> pl.DataFrame:
    if prediction.ndim != 2 or tuple(prediction.shape) != (
        len(study_ids),
        len(targets),
    ):
        raise ValueError("Prediction shape does not match studies and targets")
    if len(set(study_ids)) != len(study_ids) or any(
        not isinstance(uid, str) or not uid.strip() for uid in study_ids
    ):
        raise ValueError("Prediction study IDs must be unique nonempty strings")
    values = (
        prediction.detach().cpu().numpy()
        if isinstance(prediction, torch.Tensor)
        else np.asarray(prediction)
    )
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    columns = {"StudyInstanceUID": study_ids}
    if fold is not None:
        columns["fold"] = [fold] * len(study_ids)
    columns.update({target: values[:, index] for index, target in enumerate(targets)})
    return pl.DataFrame(columns)
