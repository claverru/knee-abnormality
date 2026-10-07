from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path("outputs/analysis/side-routing-old-vs-reference-20261006")
old = np.load(ROOT / "old.npz", allow_pickle=True)
new = np.load(ROOT / "reference.npz", allow_pickle=True)
targets = [str(name) for name in new["targets"]]
assert [str(name) for name in old["targets"]] == targets
split = Path("data/folds/all_5fold_scanner_v2")
labels = (
    pl.read_csv(split / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split / "targets.csv"), on="StudyInstanceUID")
    .sort("StudyInstanceUID")
)
truth = np.stack([labels[name].to_numpy() >= 0.5 for name in targets], 1)[new["study"]]
knees = len(new["study"])
TOP = (3, 10, 20)
print(f"{knees} knees. Where the old recipe's most attended tokens fall, in the reference's geometric frame (medial, front, up; mm), on positive knees")
print(f"{'label':18}{'old spot':>22}{'reference spot':>22}   distance  | share of the old model's top tokens that the reference may not read (top 3 / 10 / 20)  | same tokens in both top 10")
rows = {}
for number, name in enumerate(targets):
    spots = {"old": [], "new": []}
    outside = {k: [] for k in TOP}
    shared = []
    for knee in np.flatnonzero(truth[:, number]):
        position = new["position"][knee]
        region = np.unpackbits(new["region"][knee], axis=1)[:, : len(position)].astype(bool)
        for label, data in (("old", old), ("new", new)):
            index, weight = data["index"][knee, number, :10], data["weight"][knee, number, :10]
            spots[label].append((weight[:, None] * position[index]).sum(0) / weight.sum())
        for k in TOP:
            index, weight = old["index"][knee, number, :k], old["weight"][knee, number, :k]
            outside[k].append(float((weight * ~region[number][index]).sum() / weight.sum()))
        shared.append(len(set(old["index"][knee, number, :10]) & set(new["index"][knee, number, :10])) / 10)
    a, b = np.array(spots["old"]).mean(0), np.array(spots["new"]).mean(0)
    rows[name] = (a, b, {k: float(np.mean(v)) for k, v in outside.items()}, float(np.mean(shared)))
    print(
        f"{name:18}{np.array2string(a.round(1)):>22}{np.array2string(b.round(1)):>22}   {np.linalg.norm(a - b):6.1f}    | "
        + "  ".join(f"{100 * np.mean(outside[k]):5.1f}%" for k in TOP)
        + f"   | {100 * np.mean(shared):4.0f}%"
    )
print("\nPer label: AUC at x1 of each model, and of positives where the old model looks outside the reference's region")
for number, name in enumerate(targets):
    a = roc_auc_score(truth[:, number], old["logit"][:, number])
    b = roc_auc_score(truth[:, number], new["logit"][:, number])
    print(f"{name:18} old {a:.4f}  reference {b:.4f}  {b - a:+.4f}")
EDGE, CELL = 54.0, 6.0
bins = np.arange(-EDGE, EDGE + CELL, CELL)
shown = ("ACL", "Medial Meniscus", "Contusion", "Lateral Meniscus")
figure, axes = plt.subplots(len(shown), 6, figsize=(17, 2.9 * len(shown)), squeeze=False)
for line, name in enumerate(shown):
    number = targets.index(name)
    for half, (label, data) in enumerate((("old recipe", old), ("reference", new))):
        points, weights = [], []
        for knee in np.flatnonzero(truth[:, number]):
            position = new["position"][knee]
            index, weight = data["index"][knee, number, :10], data["weight"][knee, number, :10]
            points.append(position[index])
            weights.append(weight / weight.sum())
        points, weights = np.concatenate(points), np.concatenate(weights)
        for view, (x, y, xlabel, ylabel) in enumerate(((0, 1, "lateral <- medial", "back <- front"), (0, 2, "lateral <- medial", "down <- up"), (1, 2, "back <- front", "down <- up"))):
            panel = axes[line][half * 3 + view]
            histogram, _, _ = np.histogram2d(points[:, x], points[:, y], bins=(bins, bins), weights=weights)
            panel.imshow(histogram.T, origin="lower", cmap="magma", extent=(-EDGE, EDGE, -EDGE, EDGE), interpolation="nearest")
            panel.axhline(0, color="white", linewidth=0.4)
            panel.axvline(0, color="white", linewidth=0.4)
            panel.set_xlabel(xlabel, fontsize=7)
            panel.set_ylabel(ylabel, fontsize=7)
            panel.tick_params(labelsize=6)
            if view == 0:
                panel.set_title(f"{name}: {label}", fontsize=9, loc="left")
figure.suptitle("Where the ten most attended tokens fall on positive knees: the old recipe (left three panels) and the reference (right three)", fontsize=11)
figure.tight_layout(rect=(0, 0, 1, 0.97))
figure.savefig(ROOT / "old-vs-reference.png", dpi=85)
print(ROOT / "old-vs-reference.png")
