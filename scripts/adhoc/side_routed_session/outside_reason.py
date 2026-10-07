from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("outputs/analysis/side-routing-old-vs-reference-20261006")
old = np.load(ROOT / "old.npz", allow_pickle=True)
new = np.load(ROOT / "reference.npz", allow_pickle=True)
targets = [str(name) for name in new["targets"]]
split = Path("data/folds/all_5fold_scanner_v2")
labels = pl.read_csv(split / "studies.csv").filter(pl.col("fold") == 1).join(pl.read_csv(split / "targets.csv"), on="StudyInstanceUID").sort("StudyInstanceUID")
truth = np.stack([labels[name].to_numpy() >= 0.5 for name in targets], 1)[new["study"]]
for name in ("ACL", "Synovitis", "Effusion", "PF OA", "MCL"):
    number = targets.index(name)
    total = below = 0.0
    heights = []
    for knee in np.flatnonzero(truth[:, number]):
        frame = new["frame"][knee]
        region = np.unpackbits(new["region"][knee], axis=1)[:, : len(frame)].astype(bool)[number]
        index, weight = old["index"][knee, number, :10], old["weight"][knee, number, :10]
        weight = weight / weight.sum()
        out = ~region[index]
        total += float((weight * out).sum())
        below += float((weight * out * (frame[index, 2] < 3.0)).sum())
        heights.append(float((weight * frame[index, 2]).sum()))
    count = int(truth[:, number].sum())
    print(f"{name:10} old top-10 outside the reference's region {100 * total / count:5.1f}%, of which below the lower cut {100 * below / max(total, 1e-9):5.1f}%; old tokens' mean height in the reference frame {np.mean(heights):+.1f} mm")
