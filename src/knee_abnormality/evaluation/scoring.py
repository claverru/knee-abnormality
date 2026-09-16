from pathlib import Path

import numpy as np
import polars as pl

from knee_abnormality.data.labels import STUDY, load_split, require_unique_ids
from knee_abnormality.evaluation.metrics import validation_metrics
from knee_abnormality.provenance import sha256_file


def score_predictions(
    reference: pl.DataFrame,
    predictions: pl.DataFrame,
    targets: list[str],
    n_splits: int,
) -> dict:
    require_unique_ids(reference, "Reference table", reject_blank=True)
    require_unique_ids(predictions, "Predictions", reject_blank=True)
    if set(predictions.columns) != {STUDY, "fold", *targets}:
        raise ValueError(
            "Predictions must contain study ID, producing fold, and exactly the target columns"
        )
    if predictions.schema["fold"] not in (pl.Int32, pl.Int64):
        raise ValueError("Prediction fold must be an integer")
    reference = reference.sort(STUDY)
    predictions = predictions.sort(STUDY)
    if not reference[STUDY].equals(predictions[STUDY]):
        raise ValueError("Predictions must match all and only the reference study IDs")
    if predictions["fold"].null_count() or not np.array_equal(
        reference["fold"].to_numpy(), predictions["fold"].to_numpy()
    ):
        raise ValueError("Prediction folds do not match the frozen assignments")
    if set(reference["fold"].to_list()) != set(range(n_splits)):
        raise ValueError("Reference table must cover every fold")
    if any(not predictions.schema[target].is_numeric() for target in targets):
        raise ValueError("Predictions must be numeric")
    y = reference.select(targets).to_numpy().astype(float)
    prediction = predictions.select(targets).to_numpy().astype(float)
    if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    gold = reference["is_gold"].to_numpy()
    pooled = validation_metrics(y, prediction, targets, gold)
    folds = []
    for fold in range(n_splits):
        mask = reference["fold"].to_numpy() == fold
        folds.append(
            {
                "fold": fold,
                **validation_metrics(y[mask], prediction[mask], targets, gold[mask]),
            }
        )
    return {
        "validation_studies": reference.height,
        "reference_studies": int(gold.sum()),
        "pooled_oof": pooled,
        "folds": folds,
        "mean_fold_macro_auc": float(np.mean([row["macro_auc"] for row in folds])),
    }


def score(raw_dir: Path, split_dir: Path, predictions_path: Path) -> dict:
    split = load_split(raw_dir, split_dir)
    protocol = split.protocol
    reference = split.membership.select(STUDY, "fold", "is_gold").join(
        split.evaluation, on=STUDY, how="left", validate="1:1"
    )
    predictions = pl.read_csv(predictions_path, infer_schema_length=None)
    result = score_predictions(
        reference, predictions, protocol["targets"], protocol["n_splits"]
    )
    return {
        "manifest_sha256": protocol["manifest_sha256"],
        "validation_scope": protocol["validation_scope"],
        "predictions_sha256": sha256_file(predictions_path),
        **result,
    }
