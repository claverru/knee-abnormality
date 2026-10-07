import json
import sys
from pathlib import Path

import polars as pl

from knee_abnormality.data.splits import STUDY
from knee_abnormality.evaluation.metrics import validation_metrics

runs = [Path(path) for path in sys.argv[1:]]
config = json.loads((runs[0] / "effective_config.json").read_text())
print({k: v for k, v in config["validation"].items()})
print(config["paths"])
split_dir = Path(sys.argv[1]).parents[1] / "splits"
candidates = list(Path("data").glob(f"**/{config['validation'].get('split', '')}/studies.csv"))
print(candidates)
split_dir = candidates[0].parent
first = pl.read_csv(runs[0] / "fold_1" / "validation_predictions.csv")
targets = tuple(first.columns[2:])
frame = (
    pl.read_csv(split_dir / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split_dir / "targets.csv"), on=STUDY)
    .sort(STUDY)
)
truth = frame.select(targets).to_numpy().astype(float)
results = []
for run in runs:
    prediction = pl.read_csv(run / "fold_1" / "validation_predictions.csv").sort(STUDY)
    assert prediction[STUDY].to_list() == frame[STUDY].to_list()
    results.append(validation_metrics(truth, prediction.select(targets).to_numpy().astype(float), targets, frame['is_gold'].to_numpy()))
print("macro", [round(r["macro_auc"], 6) for r in results], "bce", [round(r["soft_target_bce"], 4) for r in results])
for index, target in enumerate(targets):
    a, b = (r["per_target_auc"][target] for r in results)
    print(f"{target:18s} {a:.4f} {b:.4f} {a - b:+.4f}  positives {int((truth[:, index] >= 0.5).sum())}")
