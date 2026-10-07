from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

LABEL = "MCL"
E = Path("outputs/autoresearch/routing-and-shifts/experiments")
RUNS = {
    "incumbent": E / "height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402",
    "spatial maps": E / "spot-shift-locator-keeps-spatial-maps-k-20261007T043843Z-632ceac9",
    "spots, first": E / "shift-per-study-from-the-spots-unit-output-20261007T031154Z-b1230765",
    "rerun: series c": E / "shift-per-series-type-light-port-c-20261007T003634Z-50765893",
    "rerun: unit": E / "shift-per-series-type-unit-parameter-20261007T015236Z-f9c09c16",
    "label drift: one point": E / "shift-per-study-one-point-unit-output-20261007T021934Z-90f8e11c",
}
split = Path("data/folds/all_5fold_scanner_v2")
frame = (
    pl.read_csv(split / "studies.csv").filter(pl.col("fold") == 1)
    .join(pl.read_csv(split / "targets.csv"), on="StudyInstanceUID").sort("StudyInstanceUID")
)
truth = frame[LABEL].to_numpy() >= 0.5
scores = {}
for name, run in RUNS.items():
    prediction = pl.read_csv(run / "training" / "fold_1" / "validation_predictions.csv").sort("StudyInstanceUID")
    assert prediction["StudyInstanceUID"].to_list() == frame["StudyInstanceUID"].to_list()
    scores[name] = prediction[LABEL].to_numpy()
negatives = (~truth).sum()
print(f"{LABEL}: {truth.sum()} positives, {negatives} negatives, {truth.sum() * negatives} pairs")


def beaten(values):
    return np.array([(values[~truth] > value).sum() + 0.5 * (values[~truth] == value).sum() for value in values[truth]])


base = beaten(scores["incumbent"])
print(f"{'run':24}{'AUC':>8}{'pairs gained':>14}{'positives better':>18}{'worse':>7}{'gain from the 3 biggest movers':>32}")
for name, values in scores.items():
    lost = beaten(values)
    change = base - lost
    order = np.sort(change)[::-1]
    print(f"{name:24}{roc_auc_score(truth, values):8.4f}{change.sum():14.0f}{int((change > 5).sum()):18d}{int((change < -5).sum()):7d}{order[:3].sum():22.0f} of {change.sum():.0f}")
for name in ("spatial maps", "spots, first"):
    change = base - beaten(scores[name])
    print(f"\n{name}: change in negatives ranked above each positive (positive = better), all {truth.sum()} positives sorted")
    print(np.sort(change.astype(int))[::-1].tolist())
a = base - beaten(scores["spatial maps"])
b = base - beaten(scores["spots, first"])
c = base - beaten(scores["rerun: series c"])
print("\ncorrelation of the per-positive change, spatial maps against spots-first:", round(float(np.corrcoef(a, b)[0, 1]), 2), "| against a rerun:", round(float(np.corrcoef(a, c)[0, 1]), 2))
hard = np.argsort(base)[::-1][:10]
print("the ten positives the incumbent ranks worst (negatives above them): incumbent", base[hard].astype(int).tolist())
print("                                                         spatial maps", beaten(scores["spatial maps"])[hard].astype(int).tolist())
print("                                                         spots, first", beaten(scores["spots, first"])[hard].astype(int).tolist())
print("                                                         rerun series c", beaten(scores["rerun: series c"])[hard].astype(int).tolist())
