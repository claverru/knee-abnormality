import numpy as np
from sklearn.metrics import roc_auc_score

from knee_abnormality.data.labels import TARGET_THRESHOLD

COMPARTMENT_PAIRS = {
    "meniscus": ("Medial Meniscus", "Lateral Meniscus"),
    "oa": ("Medial OA", "Lateral OA"),
}


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


def soft_auc(target: np.ndarray, score: np.ndarray) -> float:
    values, inverse = np.unique(score, return_inverse=True)
    negative = 1 - target
    per_value = np.bincount(inverse, weights=negative, minlength=len(values))
    below = np.cumsum(per_value) - per_value
    tied = per_value[inverse] - negative
    numerator = (target * (below[inverse] + 0.5 * tied)).sum()
    denominator = target.sum() * negative.sum() - (target * negative).sum()
    return float(numerator / denominator)


def soft_aucs(
    y: np.ndarray, prediction: np.ndarray, targets: list[str] | tuple[str, ...]
) -> dict:
    result = {
        target: soft_auc(
            y[:, index].astype(np.float64), prediction[:, index].astype(np.float64)
        )
        for index, target in enumerate(targets)
    }
    return {
        "soft_macro_auc": float(np.mean(list(result.values()))),
        "per_target_soft_auc": result,
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
        **soft_aucs(y, prediction, targets),
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


def bootstrap_auc(
    truth: np.ndarray, scores: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(counts))
    for target in range(truth.shape[1]):
        positive = truth[:, target] == 1
        pos = counts[:, positive]
        neg = counts[:, ~positive]
        difference = scores[positive, target, None] - scores[~positive, target]
        wins = (difference > 0).astype(float) + 0.5 * (difference == 0)
        result += (pos * (neg @ wins.T)).sum(axis=1) / (
            pos.sum(axis=1) * neg.sum(axis=1)
        )
    return result / truth.shape[1]


def logit_margin(probabilities: np.ndarray, medial: int, lateral: int) -> np.ndarray:
    values = np.clip(
        probabilities[:, [medial, lateral]].astype(np.float64), 1e-7, 1 - 1e-7
    )
    logits = np.log(values) - np.log1p(-values)
    return logits[:, 0] - logits[:, 1]


def margin_auc(
    hard: np.ndarray, margin: np.ndarray, medial: int, lateral: int
) -> float | None:
    exclusive = hard[:, medial] != hard[:, lateral]
    medial_only = hard[exclusive, medial]
    if len(np.unique(medial_only)) != 2:
        return None
    return float(roc_auc_score(medial_only, margin[exclusive]))


def compartment_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    targets: list[str] | tuple[str, ...],
    threshold: float = TARGET_THRESHOLD,
) -> dict:
    hard = y >= threshold
    result = {}
    for family, pair in COMPARTMENT_PAIRS.items():
        medial, lateral = (list(targets).index(name) for name in pair)
        exclusive = hard[:, medial] != hard[:, lateral]
        positives = int(hard[:, lateral].sum())
        ranked = np.argsort(-prediction[:, lateral], kind="stable")
        flagged = np.zeros(len(hard), dtype=bool)
        flagged[ranked[:positives]] = True
        negatives = ~hard[:, lateral]
        false_positives = flagged & negatives
        crossed = false_positives & hard[:, medial]
        medial_negatives = negatives & hard[:, medial]
        result[family] = {
            "exclusive_studies": int(exclusive.sum()),
            "medial_only": int((exclusive & hard[:, medial]).sum()),
            "lateral_only": int((exclusive & hard[:, lateral]).sum()),
            "margin_auc": margin_auc(
                hard, logit_margin(prediction, medial, lateral), medial, lateral
            ),
            "lateral_positives": positives,
            "lateral_false_positives": int(false_positives.sum()),
            "medial_positive_false_positives": int(crossed.sum()),
            "lateral_cross_talk": float(crossed.sum() / false_positives.sum())
            if false_positives.any()
            else None,
            "lateral_negatives": int(negatives.sum()),
            "medial_positive_lateral_negatives": int(medial_negatives.sum()),
            "medial_positive_base_rate": float(medial_negatives.sum() / negatives.sum())
            if negatives.any()
            else None,
        }
    return result
