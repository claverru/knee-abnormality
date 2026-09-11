import numpy as np
from sklearn.metrics import roc_auc_score

TARGET_THRESHOLD = 0.5


def aucs(
    y: np.ndarray,
    prediction: np.ndarray,
    targets: list[str] | tuple[str, ...],
    *,
    strict: bool = True,
) -> dict:
    result = {}
    for index, target in enumerate(targets):
        truth = y[:, index]
        values = prediction[:, index]
        valid = np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        defined = np.array_equal(np.unique(truth), [0, 1])
        if not valid or not defined:
            if strict:
                reason = (
                    "finite probabilities in [0, 1]"
                    if not valid
                    else "both target classes"
                )
                raise ValueError(f"{target}: ROC-AUC requires {reason}")
            result[target] = None
            continue
        result[target] = float(roc_auc_score(truth, values))
    undefined = [target for target, value in result.items() if value is None]
    return {
        "macro_auc": None if undefined else float(np.mean(list(result.values()))),
        "per_target_auc": result,
        **({"undefined_targets": undefined} if not strict else {}),
    }


def validation_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    targets: list[str] | tuple[str, ...],
    is_gold: np.ndarray,
    threshold: float = TARGET_THRESHOLD,
) -> dict:
    if y.shape != prediction.shape or y.ndim != 2 or y.shape[1] != len(targets):
        raise ValueError("Prediction and target shapes must match the target schema")
    if len(is_gold) != len(y):
        raise ValueError("Gold indicators must align with validation studies")
    for name, values in (("targets", y), ("predictions", prediction)):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Validation {name} must be finite probabilities")
    hard = (y >= threshold).astype(np.int8)
    probabilities = np.clip(prediction.astype(np.float64), 1e-7, 1 - 1e-7)
    result = {
        "studies": len(y),
        "auc_target_threshold": threshold,
        **aucs(hard, prediction, targets),
        "soft_target_bce": float(
            -(y * np.log(probabilities) + (1 - y) * np.log1p(-probabilities)).mean()
        ),
    }
    gold = np.asarray(is_gold, dtype=bool)
    for name, mask in (("gold", gold), ("report_derived", ~gold)):
        result[name] = {
            "studies": int(mask.sum()),
            **aucs(hard[mask], prediction[mask], targets, strict=False),
        }
    return result
