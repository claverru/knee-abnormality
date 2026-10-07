from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("outputs/analysis/routing-and-shifts-frames-20261007/mcl-spatial-maps")
data = np.load(OUT / "pattern-directions-coronal-x4.npz", allow_pickle=True)
labels = list(data["labels"])
names = pl.DataFrame({"StudyInstanceUID": list(data["names"])})
truth = names.join(pl.read_csv("data/folds/all_5fold_scanner_v2/targets.csv"), on="StudyInstanceUID", how="left")
PASSES = ("anchor", "all_3_lateral", "all_3_medial", "all_3_up", "all_3_down")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


stack = np.stack([logit(data[name]) for name in PASSES])
rows = []
for i, label in enumerate(labels):
    y = truth[label].to_numpy() >= 0.5
    values = stack[:, :, i]
    within = values.std(0)
    mean = values.mean(0)
    gap = mean[y].mean() - mean[~y].mean()
    pooled = np.sqrt((mean[y].var() + mean[~y].var()) / 2)
    order = np.argsort(mean)
    border = np.abs(mean - np.quantile(mean, 1 - y.mean())) < 0.5 * pooled
    directions = {name: np.abs(stack[k, :, i] - stack[0, :, i]).mean() for k, name in enumerate(PASSES) if k}
    rows.append(
        (label, int(y.sum()), within.mean(), within[y].mean(), within[~y].mean(), gap, within.mean() / gap,
         within[border].mean() / gap, *directions.values())
    )
print(f"{'label':18}{'pos':>5}{'spread':>8}{'on pos':>8}{'on neg':>8}{'pos-neg gap':>12}{'spread/gap':>11}{'near cut':>9}{'lateral':>9}{'medial':>8}{'up':>7}{'down':>7}")
for row in sorted(rows, key=lambda r: -r[6]):
    print(f"{row[0]:18}{row[1]:5d}{row[2]:8.3f}{row[3]:8.3f}{row[4]:8.3f}{row[5]:12.2f}{row[6]:11.4f}{row[7]:9.4f}{row[8]:9.3f}{row[9]:8.3f}{row[10]:7.3f}{row[11]:7.3f}")
ratio = np.array([r[6] for r in rows])
count = np.array([r[1] for r in rows])
print("correlation of spread/gap with log positives:", round(float(np.corrcoef(ratio, np.log(count))[0, 1]), 2))
print("correlation of raw spread with log positives:", round(float(np.corrcoef([r[2] for r in rows], np.log(count))[0, 1]), 2))
