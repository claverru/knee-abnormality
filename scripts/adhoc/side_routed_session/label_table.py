import json
import sys
from pathlib import Path


def epochs(run):
    files = sorted(next(Path(run).glob("fold_*")).joinpath("epochs").glob("metrics_epoch_*.json"))
    return [json.loads(path.read_text()) for path in files]


def per_label(entry):
    for key in ("per_target_auc", "target_auc", "aucs"):
        if key in entry:
            return entry[key]
    for value in entry.values():
        if isinstance(value, dict) and "per_target_auc" in value:
            return value["per_target_auc"]
    return None


current = epochs(sys.argv[1])
reference = epochs(sys.argv[2])
if per_label(current[0]) is None:
    print(list(current[0].keys())[:40])
    sys.exit()
labels = list(per_label(current[0]))
last = len(current) - 1
print("epochs done", len(current))
print(f"{'label':18s}" + "".join(f"  e{e + 1:<6d}" for e in range(len(current))))
for label in labels:
    row = "".join(
        f"  {per_label(current[e])[label] - per_label(reference[e])[label]:+.4f}"
        for e in range(len(current))
    )
    print(f"{label:18s}{row}   now {per_label(current[last])[label]:.4f}")
