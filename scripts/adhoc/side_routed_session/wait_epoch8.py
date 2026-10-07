import csv
import sys
import time
from pathlib import Path

experiment = sys.argv[1]
base = Path(sys.argv[2])
root = Path("outputs/autoresearch/side-routing/experiments")


def curve(path):
    values = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("val_macro_auc") not in ("", None):
                values.append(float(row["val_macro_auc"]))
    return values


reference = curve(next(base.glob("fold_*/logs/version_*/metrics.csv")))
deadline = time.time() + 3000
while time.time() < deadline:
    matches = sorted(root.glob(f"{experiment}-*/training/fold_*/logs/version_*/metrics.csv"))
    if matches:
        try:
            values = curve(matches[-1])
        except OSError:
            values = []
        if len(values) >= 8:
            for epoch, value in enumerate(values, 1):
                print(f"epoch {epoch}: {value:.4f} vs base {reference[epoch - 1]:.4f} ({value - reference[epoch - 1]:+.4f})")
            ahead = values[7] >= reference[7]
            print("AHEAD_AT_EPOCH_8" if ahead else "BEHIND_AT_EPOCH_8", matches[-1].parents[4].name)
            break
    time.sleep(20)
else:
    print("TIMEOUT waiting for epoch 8", experiment)
