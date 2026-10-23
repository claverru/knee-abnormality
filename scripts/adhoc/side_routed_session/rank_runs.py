import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

NUMBERS = {
    "side-routed-heads": 1,
    "knee-frame": 2,
    "attention-prior": 3,
    "run4": 4,
    "run5": 5,
    "hard-octants": 6,
    "soft-gradient-own-routes": 7,
    "side-routed-study-frames": 8,
    "fine-landmarks": 9,
    "fine-fused-routed-reports": 10,
    "study-frames-landmarks": 11,
    "eighteen-regions": 12,
    "regions-v31-reports": 13,
    "regions-acl-unrouted": 14,
    "regions-shared-frame": 15,
    "regions-mirrored-position": 16,
    "regions-soft-gradient-knee-level": 18,
    "regions-core-kept-by-sides": 19,
    "regions-wider-table": 20,
    "regions-nod-per-knee": 21,
    "regions-kneecap-above-lower-cut": 22,
    "meniscus-spot-frame-eight-sections": 23,
    "spot-frame-ring-sectors": 24,
    "four-anchor-frame": 25,
    "acl-all-heights": 26,
    "midway-cuts": 28,
    "held-cuts-balanced": 29,
    "mirrored-held-division": 30,
}
NOISE = {
    "ACL": 0.0014,
    "MCL": 0.0029,
    "Medial Meniscus": 0.0007,
    "Lateral Meniscus": 0.0024,
    "Medial OA": 0.0008,
    "Lateral OA": 0.0034,
    "PF OA": 0.0005,
    "Effusion": 0.0015,
    "Synovitis": 0.0022,
    "Baker's": 0.0015,
    "Contusion": 0.0013,
    "Fracture": 0.0044,
}
split = Path("data/folds/all_5fold_scanner_v2")
frame = (
    pl.read_csv(split / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split / "targets.csv"), on="StudyInstanceUID")
    .sort("StudyInstanceUID")
)
runs = [
    (
        "old",
        Path(
            "outputs/autoresearch/fixes-and-removals/experiments/slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc/training"
        ),
    )
]
for path in sorted(Path("outputs/training").glob("2026100[56]-*side-routed*")):
    number = next(
        (
            n
            for key, n in NUMBERS.items()
            if path.name[16:] == key or path.name.endswith(key)
        ),
        None,
    )
    if number is not None and (path / "fold_1" / "validation_predictions.csv").exists():
        runs.append((str(number), path))
rows, labels = [], None
for name, path in runs:
    prediction = pl.read_csv(path / "fold_1" / "validation_predictions.csv").sort(
        "StudyInstanceUID"
    )
    labels = prediction.columns[2:]
    scores = [
        roc_auc_score(frame[label].to_numpy() >= 0.5, prediction[label].to_numpy())
        for label in labels
    ]
    selected = json.loads((path / "fold_1" / "selected_checkpoint.json").read_text())
    rows.append((name, selected["score"], float(np.mean(scores)), scores))
names = [row[0] for row in rows]
x1 = np.array([row[1] for row in rows])
x4 = np.array([row[2] for row in rows])
table = np.array([row[3] for row in rows])
both = (x1 - x1.mean()) / x1.std() + (x4 - x4.mean()) / x4.std()
order = np.argsort(-both)
print("sorted by x1 and x4 together (sum of the two standardised scores)")
print(
    f"{'run':>5}{'x1':>9}{'x4':>9}  " + "".join(f"{label[:8]:>9}" for label in labels)
)
for i in order:
    print(
        f"{names[i]:>5}{x1[i]:9.4f}{x4[i]:9.4f}  "
        + "".join(f"{value:9.4f}" for value in table[i])
    )
routed = np.array([name != "old" for name in names])
print(
    "\nper label over the routed runs: SD across runs, its sibling noise, ratio; correlation with the run's macro (x4) and with x1; share of the macro's variance it carries"
)
macro = table[routed].mean(1)
centred = table[routed] - table[routed].mean(0)
for column, label in enumerate(labels):
    spread = table[routed][:, column].std()
    share = float(
        (centred[:, column] * (macro - macro.mean())).sum()
        / (len(labels) * ((macro - macro.mean()) ** 2).sum())
    )
    print(
        f"{label:18} SD {spread:.4f}  noise {NOISE[label]:.4f}  ratio {spread / NOISE[label]:4.1f}   "
        f"r(x4) {np.corrcoef(table[routed][:, column], macro)[0, 1]:+.2f}  r(x1) {np.corrcoef(table[routed][:, column], x1[routed])[0, 1]:+.2f}  share {share:+.2f}"
    )
top, bottom = order[:6], order[-6:]
print("\ntop six", [names[i] for i in top], "bottom six", [names[i] for i in bottom])
print("mean of the top six minus mean of the bottom six, per label:")
for column, label in enumerate(labels):
    gap = table[top][:, column].mean() - table[bottom][:, column].mean()
    print(f"  {label:18} {gap:+.4f}  ({gap / NOISE[label]:+.1f} of its noise)")
print(
    f"x1 against x4 across runs: r = {np.corrcoef(x1[routed], x4[routed])[0, 1]:.2f}; SD x1 {x1[routed].std():.4f}, SD x4 {x4[routed].std():.4f}"
)
