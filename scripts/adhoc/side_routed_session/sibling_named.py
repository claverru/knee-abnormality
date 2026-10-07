from pathlib import Path

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
labels = ("ACL", "MCL", "Synovitis", "Baker's", "Fracture", "Effusion", "Contusion")


def scores(path):
    prediction = pl.read_csv(path).sort(STUDY)
    if prediction[STUDY].to_list() != frame[STUDY].to_list():
        return None
    return {
        label: roc_auc_score(frame[label].to_numpy() >= 0.5, prediction[label].to_numpy())
        for label in prediction.columns[2:]
    }


base = scores(root / reference / "training/fold_1/validation_predictions.csv")
print(f"{'run':52s}" + "".join(f"{label[:9]:>10s}" for label in labels) + "     macro")
rows = {
    path.parents[2].name: path
    for path in sorted(root.glob("*/training/fold_1/validation_predictions.csv"))
}
rows["RUN 4"] = Path("outputs/training/20261005-202001-side-routed-run4/fold_1/validation_predictions.csv")
for name, path in rows.items():
    value = scores(path)
    if value is None or name == reference:
        continue
    macro = sum(value.values()) / 12 - sum(base.values()) / 12
    print(
        f"{name[:50]:52s}"
        + "".join(f"{value[label] - base[label]:+10.4f}" for label in labels)
        + f"{macro:+10.4f}"
    )
