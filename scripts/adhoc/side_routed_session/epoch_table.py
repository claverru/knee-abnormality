import csv
import sys
from pathlib import Path


def load(run):
    path = next(Path(run).glob("fold_*/logs/version_0/metrics.csv"))
    rows = list(csv.DictReader(path.open()))
    return rows, list(rows[0].keys())


def series(rows, key):
    return [float(row[key]) for row in rows if row.get(key) not in ("", None)]


current, keys = load(sys.argv[1])
reference, reference_keys = load(sys.argv[2])
print([k for k in keys if k.startswith("val_") and "attention" not in k and "view" not in k][:40])
for key in sys.argv[3:]:
    a = series(current, key) if key in keys else []
    b = series(reference, key) if key in reference_keys else []
    print(key)
    for epoch in range(max(len(a), len(b))):
        x = a[epoch] if epoch < len(a) else None
        y = b[epoch] if epoch < len(b) else None
        delta = f"{x - y:+.4f}" if x is not None and y is not None else ""
        print(f"  {epoch + 1:2d}  {x if x is None else round(x, 4)}  {y if y is None else round(y, 4)}  {delta}")
