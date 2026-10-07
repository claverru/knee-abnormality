from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from knee_abnormality.data.splits import STUDY

split = Path("data/folds/all_5fold_scanner_v2")
frame = (
    pl.read_csv(split / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split / "targets.csv"), on=STUDY)
    .sort(STUDY)
)
root = Path("outputs/autoresearch/fixes-and-removals/experiments")
reference = "slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc"
runs = {
    "fifth": Path("outputs/training/20261005-205251-side-routed-run5"),
    "fourth": Path("outputs/training/20261005-202001-side-routed-run4"),
    "third": Path("outputs/training/20261005-194836-side-routed-attention-prior"),
    "second": Path("outputs/training/20261005-185535-side-routed-knee-frame"),
    "first": Path("outputs/training/20261005-181012-side-routed-heads"),
}
siblings = {}
for path in sorted(root.glob("*/training/fold_1/validation_predictions.csv")):
    siblings[path.parents[2].name] = path


def scores(path):
    prediction = pl.read_csv(path).sort(STUDY)
    if prediction[STUDY].to_list() != frame[STUDY].to_list():
        return None
    return {
        label: roc_auc_score(frame[label].to_numpy() >= 0.5, prediction[label].to_numpy())
        for label in prediction.columns[2:]
    }


base = scores(siblings.pop(reference))
sibling_scores = [s for s in (scores(path) for path in siblings.values()) if s]
print("siblings", len(sibling_scores))
ours = {name: scores(path / "fold_1" / "validation_predictions.csv") for name, path in runs.items()}
print(f"{'label':18s} sib mean   sd     min     max  | fifth  fourth third  second  first  | fifth below siblings")
for label in base:
    deltas = np.array([s[label] - base[label] for s in sibling_scores])
    mine = {name: value[label] - base[label] for name, value in ours.items()}
    below = int((deltas > mine["fifth"]).sum())
    print(
        f"{label:18s} {deltas.mean():+.4f} {deltas.std():.4f} {deltas.min():+.4f} {deltas.max():+.4f} | "
        f"{mine['fifth']:+.4f} {mine['fourth']:+.4f} {mine['third']:+.4f} {mine['second']:+.4f} {mine['first']:+.4f} | below {below} of {len(deltas)}"
    )
macro = np.array([np.mean(list(s.values())) for s in sibling_scores]) - np.mean(list(base.values()))
print("macro siblings", macro.mean().round(4), macro.std().round(4), macro.min().round(4), macro.max().round(4))

for name, value in ours.items():
    print(name, round(float(np.mean(list(value.values()))), 6), "routed five sum", round(sum(value[k] - base[k] for k in ("Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA")), 4))
