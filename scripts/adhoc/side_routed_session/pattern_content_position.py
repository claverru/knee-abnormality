import json
import sys
from pathlib import Path

RUN = Path("outputs/training/20261007-081431-side-routed-crop-offsets-3mm").resolve()
WEIGHTS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "outputs/autoresearch/routing-and-shifts/experiments/height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402/training"
)
MULTIPLIER = int(sys.argv[1])

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import checkpoint_model, read_checkpoint
from knee_abnormality.inference.engine import move_batch

OUT = Path("outputs/analysis/r/routing-and-shifts-frames-20261007/mcl-spatial-maps")
config = load_config(RUN / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
names = studies[STUDY].to_list()
index = build_series_index(config.paths.data_root, config.data, split, names)
dataset = KneeStudyDataset(studies, index, config.data, targets, None, MULTIPLIER)
selected = json.loads((WEIGHTS / "fold_1" / "selected_checkpoint.json").read_text())
gpu = torch.device("cuda")
model = checkpoint_model(read_checkpoint(Path(selected["path"])), config, targets, gpu).eval()
windows = model.view_windows
original_offsets = windows.offsets
original_forward = windows.forward
state = {"images": 0.0, "maps": 0.0}


def offsets(active):
    return original_offsets(active) * state["scale"]


def forward(batch, middle_only=False):
    state["scale"] = state["images"]
    pictures = original_forward(batch, middle_only)
    if state["maps"] == state["images"]:
        return pictures
    state["scale"] = state["maps"]
    places = original_forward(batch, middle_only)
    return {**pictures, "patient_maps": places["patient_maps"]}


windows.offsets = offsets
windows.forward = forward
PASSES = ("anchor", "pattern", "pattern pictures, anchor positions", "anchor pictures, pattern positions")
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate_studies)
outputs = {name: [] for name in PASSES}
for batch in loader:
    batch = with_images(move_batch(batch, gpu))
    for name, (images, maps) in zip(PASSES, ((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))):
        state["images"], state["maps"] = images, maps
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs[name].append(torch.sigmoid(model(batch).float()).cpu().numpy())
outputs = {name: np.concatenate(values) for name, values in outputs.items()}
labels = list(targets)
truth = studies.select(STUDY).join(
    pl.read_csv(Path("data/folds/all_5fold_scanner_v2") / "targets.csv"), on=STUDY, how="left"
)
np.savez(OUT / f"pattern-content-or-position-{WEIGHTS.parent.name[:30]}-x{MULTIPLIER}.npz", names=np.array(names), labels=np.array(labels), **{k.replace(" ", "_").replace(",", ""): v for k, v in outputs.items()})
print(f"{'label':17}{'anchor':>8}{'pattern':>10}{'pat pict':>11}{'anc pict':>11}")
print(f"{'':17}{'':>8}{'':>10}{'anc posit':>11}{'pat posit':>11}")
rows = []
for i, label in enumerate(labels):
    y = truth[label].to_numpy() >= 0.5
    values = [roc_auc_score(y, outputs[name][:, i]) for name in PASSES]
    rows.append(values)
    print(f"{label:17}{values[0]:8.4f}{values[1] - values[0]:+10.4f}{values[2] - values[0]:+11.4f}{values[3] - values[0]:+11.4f}")
values = np.mean(rows, 0)
print(f"{'macro':17}{values[0]:8.4f}{values[1] - values[0]:+10.4f}{values[2] - values[0]:+11.4f}{values[3] - values[0]:+11.4f}")
