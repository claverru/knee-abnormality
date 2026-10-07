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
families = {
    "Medial Meniscus": ("Medial Meniscus", "Lateral Meniscus"),
    "Lateral Meniscus": ("Medial Meniscus", "Lateral Meniscus"),
    "Medial OA": ("Medial OA", "Lateral OA", "PF OA"),
    "Lateral OA": ("Medial OA", "Lateral OA", "PF OA"),
    "PF OA": ("Medial OA", "Lateral OA", "PF OA"),
}
truth = {name: frame[name].to_numpy() >= 0.5 for name in families}


def local(path):
    table = pl.read_csv(path).sort(STUDY)
    if table[STUDY].to_list() != frame[STUDY].to_list():
        return None
    result = {}
    for name, members in families.items():
        anywhere = np.any([truth[m] for m in members], axis=0)
        result[name] = roc_auc_score(truth[name][anywhere], table[name].to_numpy()[anywhere])
    return result


old = Path("outputs/autoresearch/fixes-and-removals/experiments")
siblings = [
    local(path)
    for path in sorted(old.glob("*/training/fold_1/validation_predictions.csv"))
]
siblings = [s for s in siblings if s]
runs = {
    "run 1": "outputs/training/20261005-181012-side-routed-heads",
    "run 2": "outputs/training/20261005-185535-side-routed-knee-frame",
    "run 3": "outputs/training/20261005-194836-side-routed-attention-prior",
    "run 4": "outputs/training/20261005-202001-side-routed-run4",
    "run 5": "outputs/training/20261005-205251-side-routed-run5",
}
new = Path("outputs/autoresearch/side-routing/experiments")
for path in sorted(new.glob("*/training/fold_1/validation_predictions.csv")):
    runs[path.parents[2].name[:34]] = str(path.parents[1])
print(f"{'':36s}" + "".join(f"{name[:11]:>12s}" for name in families))
values = np.array([[s[name] for name in families] for s in siblings])
print(f"{'old recipe, 13 runs: mean':36s}" + "".join(f"{v:12.4f}" for v in values.mean(0)))
print(f"{'old recipe: sd':36s}" + "".join(f"{v:12.4f}" for v in values.std(0)))
print(f"{'old recipe: max':36s}" + "".join(f"{v:12.4f}" for v in values.max(0)))
for label, run in runs.items():
    result = local(Path(run) / "fold_1" / "validation_predictions.csv")
    print(f"{label:36s}" + "".join(f"{result[name]:12.4f}" for name in families))
