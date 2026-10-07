import csv
import json
import sys
from pathlib import Path


def metrics(run):
    path = next(Path(run).glob("fold_*/logs/version_0/metrics.csv"))
    rows = list(csv.DictReader(path.open()))
    return rows


def series(rows, key):
    return [float(row[key]) for row in rows if row.get(key) not in ("", None)]


def labels(run):
    files = sorted(
        next(Path(run).glob("fold_*")).joinpath("epochs").glob("metrics_epoch_*.json")
    )
    result = []
    for path in files:
        entry = json.loads(path.read_text())
        if "per_target_auc" not in entry:
            entry = next(
                v for v in entry.values() if isinstance(v, dict) and "per_target_auc" in v
            )
        result.append(entry["per_target_auc"])
    return result


new, old, reference = sys.argv[1:4]
rows = {name: metrics(path) for name, path in (("new", new), ("old", old), ("ref", reference))}
for key in ("val_macro_auc", "val_soft_target_bce"):
    print(key)
    values = {name: series(r, key) for name, r in rows.items()}
    for epoch in range(len(values["new"])):
        n, o, r = (values[name][epoch] for name in ("new", "old", "ref"))
        print(f"  {epoch + 1:2d}  new {n:.4f}  old {o:.4f}  ref {r:.4f}  new-ref {n - r:+.4f}  old-ref {o - r:+.4f}")
per = {name: labels(path) for name, path in (("new", new), ("old", old), ("ref", reference))}
last = len(per["new"]) - 1
print("per label at epoch", last + 1)
for label in per["new"][last]:
    n, o, r = (per[name][last][label] for name in ("new", "old", "ref"))
    print(f"  {label:18s} new {n:.4f}  new-ref {n - r:+.4f}  old-ref {o - r:+.4f}")
for key in (
    "centre_offset_lateral_mm",
    "centre_offset_front_mm",
    "centre_offset_up_mm",
    "val_centre_shift_front_mean_mm",
    "val_centre_shift_front_std_mm",
    "routing_width_medial_mm",
    "routing_width_front_mm",
    "val_knee_rotation_abs_deg",
    "val_side_head_accuracy",
):
    print(key, [round(v, 2) for v in series(rows["new"], key)])
