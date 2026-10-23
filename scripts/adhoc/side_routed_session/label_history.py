import sys
from pathlib import Path

import polars as pl
from sklearn.metrics import roc_auc_score

names = sys.argv[1:]
split = Path("data/folds/all_5fold_scanner_v2")
frame = (
    pl.read_csv(split / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split / "targets.csv"), on="StudyInstanceUID")
    .sort("StudyInstanceUID")
)
runs = [
    Path(
        "outputs/autoresearch/fixes-and-removals/experiments/slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc/training"
    )
] + sorted(
    path
    for path in Path("outputs/training").glob("2026100[56]-*side-routed*")
    if "rn50" not in path.name
)
print(f"{'run':52}" + "".join(f"{name[:11]:>12}" for name in names) + f"{'macro':>10}")
for run in runs:
    file = run / "fold_1" / "validation_predictions.csv"
    if not file.exists():
        continue
    prediction = pl.read_csv(file).sort("StudyInstanceUID")
    targets = prediction.columns[2:]
    scores = {
        name: roc_auc_score(frame[name].to_numpy() >= 0.5, prediction[name].to_numpy())
        for name in targets
    }
    label = "old recipe" if "autoresearch" in str(run) else run.name[16:]
    print(
        f"{label:52}"
        + "".join(f"{scores[name]:12.4f}" for name in names)
        + f"{sum(scores.values()) / len(scores):10.4f}"
    )
