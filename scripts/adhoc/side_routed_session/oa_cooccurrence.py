from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from knee_abnormality.data.splits import STUDY

RUNS = {
    "reference": "outputs/autoresearch/fixes-and-removals/experiments/slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc/training",
    "first": "outputs/training/20261005-181012-side-routed-heads",
    "second": "outputs/training/20261005-185535-side-routed-knee-frame",
}
FAMILIES = {
    "OA": ("Medial OA", "Lateral OA", "PF OA"),
    "Meniscus": ("Medial Meniscus", "Lateral Meniscus"),
}
split = Path("data/folds/all_5fold_scanner_v2")
frame = (
    pl.read_csv(split / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split / "targets.csv"), on=STUDY)
    .sort(STUDY)
)
predictions = {
    name: pl.read_csv(Path(path) / "fold_1" / "validation_predictions.csv").sort(STUDY)
    for name, path in RUNS.items()
}
for prediction in predictions.values():
    assert prediction[STUDY].to_list() == frame[STUDY].to_list()


def auc(positive, negative, score):
    if positive.sum() < 5 or negative.sum() < 5:
        return None
    keep = positive | negative
    return roc_auc_score(positive[keep], score[keep])


for family, members in FAMILIES.items():
    truth = {name: frame[name].to_numpy() >= 0.5 for name in members}
    any_positive = np.any(list(truth.values()), axis=0)
    count = np.sum(list(truth.values()), axis=0)
    print(f"\n{family}: studies {len(frame)}, none {int((count == 0).sum())}, one {int((count == 1).sum())}, several {int((count > 1).sum())}")
    for label in members:
        positive = truth[label]
        cases = {
            "alone vs family-free": (positive & (count == 1), ~any_positive),
            "with others vs family-free": (positive & (count > 1), ~any_positive),
            "positive vs other-compartment-only": (positive, ~positive & any_positive),
            "all vs all": (positive, ~positive),
        }
        for case, (pos, neg) in cases.items():
            values = [
                auc(pos, neg, predictions[name][label].to_numpy()) for name in RUNS
            ]
            if values[0] is None:
                print(f"  {label:17s} {case:36s} n+ {int(pos.sum()):3d} n- {int(neg.sum()):3d}  too few")
                continue
            ref, first, second = values
            print(
                f"  {label:17s} {case:36s} n+ {int(pos.sum()):3d} n- {int(neg.sum()):3d}  "
                f"ref {ref:.4f}  first {first - ref:+.4f}  second {second - ref:+.4f}"
            )
