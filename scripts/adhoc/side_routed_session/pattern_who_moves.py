from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("outputs/analysis/r/routing-and-shifts-frames-20261007/mcl-spatial-maps")
data = np.load(next(OUT.glob("pattern-content-or-position-*-x4.npz")), allow_pickle=True)
labels = list(data["labels"])
names = pl.DataFrame({"StudyInstanceUID": list(data["names"])})
truth = names.join(pl.read_csv("data/folds/all_5fold_scanner_v2/targets.csv"), on="StudyInstanceUID", how="left")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


print(f"{'label':18}{'pos shift':>10}{'neg shift':>10}{'pos sd':>8}{'neg sd':>8}{'|change|':>9}{'top-decile neg shift':>22}")
for label in labels:
    i = labels.index(label)
    y = truth[label].to_numpy() >= 0.5
    a, b = logit(data["anchor"][:, i]), logit(data["pattern"][:, i])
    change = b - a
    middle = (a + b) / 2
    high = middle >= np.quantile(middle[~y], 0.9)
    print(f"{label:18}{change[y].mean():+10.3f}{change[~y].mean():+10.3f}{change[y].std():8.3f}{change[~y].std():8.3f}{np.abs(change).mean():9.3f}{change[~y & high].mean():+22.3f}{change[y & high].mean():+10.3f}{change[~y & ~high].mean():+10.3f}")
