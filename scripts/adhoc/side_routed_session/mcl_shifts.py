import json
import sys
from pathlib import Path

E = Path("outputs/autoresearch/routing-and-shifts/experiments").resolve()
RUN = E / "spot-shift-locator-keeps-spatial-maps-k-20261007T043843Z-632ceac9"
BASE = E / "height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402"
sys.path[:0] = [str(RUN / "source"), str(RUN / "source" / "src")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch

OUT = Path("outputs/analysis/routing-and-shifts-frames-20261007/mcl-spatial-maps")
OUT.mkdir(parents=True, exist_ok=True)
config = load_config(RUN / "training" / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
names = studies[STUDY].to_list()


def scores(run):
    table = pl.read_csv(run / "training" / "fold_1" / "validation_predictions.csv").sort(STUDY)
    assert table[STUDY].to_list() == names
    return table["MCL"].to_numpy()


labels = pl.read_csv(Path("data/folds/all_5fold_scanner_v2") / "targets.csv")
truth = studies.select(STUDY).join(labels, on=STUDY, how="left")["MCL"].to_numpy() >= 0.5
old, new = scores(BASE), scores(RUN)


def beaten(values):
    return np.array([(values[~truth] > v).sum() for v in values])


change = beaten(old) - beaten(new)
positives = np.flatnonzero(truth)
rng = np.random.default_rng(0)
negatives = rng.choice(np.flatnonzero(~truth), 150, replace=False)
rows = np.concatenate((positives, negatives))
index = build_series_index(config.paths.data_root, config.data, split, names)
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
selected = json.loads((RUN / "training" / "fold_1" / "selected_checkpoint.json").read_text())
gpu = torch.device("cuda")
model = load_checkpoint_model(config, targets, Path(selected["path"]), gpu, expected_fold=1).eval()
planes = [kind.plane for kind in config.data.series_types]
fluid = [kind.fluid_sensitive for kind in config.data.series_types]
fields = [view.field_mm for _, view in config.data.views]
movers = positives[np.argsort(change[positives])[::-1][:5]]
records, drawings = [], {}
for row in rows:
    batch = with_images(move_batch(collate_studies([dataset[int(row)]]), gpu))
    with torch.inference_mode():
        delta = model.shift_locator(batch)["delta"]
        after = model.view_windows(batch, delta)
        before = model.view_windows(batch, torch.zeros_like(delta))
    sign = -1.0 if bool(batch["knee_side_left"][0]) else 1.0
    d = delta[0].float().cpu().numpy()
    records.append((int(row), bool(truth[row]), int(change[row]), sign * d[0], d[1], d[2]))
    if row in movers:
        kinds = batch["series_type_index"].cpu().numpy()
        views = batch["view_index"].cpu().numpy()
        mask = batch["slice_mask"].cpu().numpy()
        applied = after["applied_view_shift_mm"].float().cpu().numpy()
        panels = []
        for plane in ("Coronal", "Axial"):
            members = [i for i, k in enumerate(kinds) if planes[k] == plane and mask[i].any()]
            members.sort(key=lambda i: (not fluid[kinds[i]], -fields[views[i]]))
            if not members:
                continue
            i = members[0]
            t = np.flatnonzero(mask[i])[len(np.flatnonzero(mask[i])) // 2]
            panels.append(
                (
                    plane,
                    fields[views[i]],
                    before["images"][i, t, 0].float().cpu().numpy(),
                    after["images"][i, t, 0].float().cpu().numpy(),
                    applied[i, t],
                )
            )
        drawings[int(row)] = (sign, d, panels)
table = pl.DataFrame(records, schema=["row", "positive", "gain", "medial", "front", "up"], orient="row")
table.write_csv(OUT / "shifts.csv")
for name, part in (("MCL positives", table.filter(pl.col("positive"))), ("negatives (150)", table.filter(~pl.col("positive")))):
    print(f"{name:18} n {len(part):4d} | medial mean {part['medial'].mean():+.2f} sd {part['medial'].std():.2f} | front mean {part['front'].mean():+.2f} sd {part['front'].std():.2f} | up mean {part['up'].mean():+.2f} sd {part['up'].std():.2f}")
pos = table.filter(pl.col("positive"))
for key in ("medial", "front", "up"):
    print(f"positives: correlation of gain with {key} shift {np.corrcoef(pos['gain'].to_numpy(), pos[key].to_numpy())[0, 1]:+.2f}; with its size {np.corrcoef(pos['gain'].to_numpy(), np.abs(pos[key].to_numpy()))[0, 1]:+.2f}")
print(pos.sort("gain", descending=True).head(8))
print(pos.sort("gain").head(5))
figure, axes = plt.subplots(len(movers), 4, figsize=(14, 3.6 * len(movers)))
for line, row in enumerate(movers):
    sign, d, panels = drawings[int(row)]
    for column in range(4):
        axes[line][column].axis("off")
    for p, (plane, field, first, second, applied) in enumerate(panels):
        for q, (image, name) in enumerate(((first, "centred as before"), (second, "moved by the locator"))):
            panel = axes[line][2 * p + q]
            panel.imshow(image, cmap="gray", vmin=0, vmax=255)
            panel.axhline(111.5, color="cyan", linewidth=0.6)
            panel.axvline(111.5, color="cyan", linewidth=0.6)
            extra = f"\nmoved {applied[1]:+.1f} mm across, {applied[0]:+.1f} mm down" if q else ""
            panel.set_title(f"{plane.lower()}, {field:.0f} mm, {name}{extra}", fontsize=8.5)
    axes[line][0].text(
        -0.02, 0.5,
        f"row {row}\nMCL rank +{int(change[row])}\nmedial {sign * d[0]:+.1f}\nfront {d[1]:+.1f}\nup {d[2]:+.1f} mm",
        transform=axes[line][0].transAxes, ha="right", va="center", fontsize=9,
    )
figure.suptitle("The five MCL knees that gained most: the same slice cut at the old centre and at the locator's point", fontsize=12)
figure.tight_layout(rect=(0.06, 0, 1, 0.97))
figure.savefig(OUT / "top-movers-crops.png", dpi=80)
print("saved", OUT / "top-movers-crops.png")
