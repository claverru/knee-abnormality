from pathlib import Path

import numpy as np
import polars as pl
import torch

from knee_abnormality.data.labels import STUDY
from knee_abnormality.evaluation.metrics import validation_metrics


def fold_prediction_macro_auc(
    split_dir: Path, fold: int, targets: tuple[str, ...], path: Path
) -> float:
    frame = (
        pl.read_csv(split_dir / "studies.csv")
        .filter(pl.col("fold") == fold)
        .join(pl.read_csv(split_dir / "targets.csv"), on=STUDY, validate="1:1")
        .sort(STUDY)
    )
    prediction = pl.read_csv(path).sort(STUDY)
    if prediction[STUDY].to_list() != frame[STUDY].to_list():
        raise ValueError(f"Prediction membership differs from the fold: {path}")
    return validation_metrics(
        frame.select(targets).to_numpy().astype(float),
        prediction.select(targets).to_numpy().astype(float),
        targets,
        frame["is_gold"].to_numpy(),
    )["macro_auc"]


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
    columns = {STUDY: study_ids}
    if fold is not None:
        columns["fold"] = [fold] * len(study_ids)
    columns.update({target: values[:, index] for index, target in enumerate(targets)})
    return pl.DataFrame(columns)
