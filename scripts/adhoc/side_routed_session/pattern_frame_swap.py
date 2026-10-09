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
original_frames = model.study_frames
state = {"scale": 0.0, "replay": None, "last": None}


def offsets(active):
    return original_offsets(active) * state["scale"]


def study_frames(*args, **kwargs):
    out = original_frames(*args, **kwargs)
    state["last"] = out
    return out if state["replay"] is None else state["replay"]


windows.offsets = offsets
model.study_frames = study_frames
PASSES = ("anchor", "pattern", "pattern crops, anchor frame", "anchor crops, pattern frame")
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate_studies)
outputs = {name: [] for name in PASSES}
moves = []


def run(batch, scale, replay):
    state["scale"], state["replay"] = scale, replay
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = torch.sigmoid(model(batch).float()).cpu().numpy()
    return out, state["last"]


for batch in loader:
    batch = with_images(move_batch(batch, gpu))
    a, frame_a = run(batch, 0.0, None)
    b, frame_b = run(batch, 1.0, None)
    c, _ = run(batch, 1.0, frame_a)
    d, _ = run(batch, 0.0, frame_b)
    for name, value in zip(PASSES, (a, b, c, d)):
        outputs[name].append(value)
    centre = (frame_b[0] - frame_a[0]).float().norm(dim=-1)
    relative = frame_b[1].float() @ frame_a[1].float().transpose(-1, -2)
    angle = torch.rad2deg(torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
    spots = (frame_b[2] - frame_a[2]).float().norm(dim=-1)
    sizes = (frame_b[3] - frame_a[3]).float().abs()
    moves.append(torch.cat((centre[:, None], angle[:, None], spots, sizes), -1).cpu().numpy())
outputs = {name: np.concatenate(values) for name, values in outputs.items()}
moves = np.concatenate(moves)
labels = list(targets)
truth = studies.select(STUDY).join(
    pl.read_csv(Path("data/folds/all_5fold_scanner_v2") / "targets.csv"), on=STUDY, how="left"
)
np.savez(OUT / f"pattern-frame-swap-{WEIGHTS.parent.name[:30]}-x{MULTIPLIER}.npz", names=np.array(names), labels=np.array(labels), moves=moves, **{k.replace(" ", "_").replace(",", ""): v for k, v in outputs.items()})
print(f"{'label':17}{'anchor':>8}{'pattern':>10}{'pat crops':>11}{'anc crops':>11}")
print(f"{'':17}{'':>8}{'':>10}{'anc frame':>11}{'pat frame':>11}")
rows = []
for i, label in enumerate(labels):
    y = truth[label].to_numpy() >= 0.5
    values = [roc_auc_score(y, outputs[name][:, i]) for name in PASSES]
    rows.append(values)
    print(f"{label:17}{values[0]:8.4f}{values[1] - values[0]:+10.4f}{values[2] - values[0]:+11.4f}{values[3] - values[0]:+11.4f}")
values = np.mean(rows, 0)
print(f"{'macro':17}{values[0]:8.4f}{values[1] - values[0]:+10.4f}{values[2] - values[0]:+11.4f}{values[3] - values[0]:+11.4f}")
columns = ("centre mm", "rotation deg", "medial spot mm", "lateral spot mm", "upper cut mm", "lower cut mm", "core mm", "ring mm")
print("how far the frame moves between anchor and pattern, per knee: median, 90th percentile")
for i, name in enumerate(columns):
    print(f"  {name:16}{np.nanmedian(moves[:, i]):7.2f}{np.nanpercentile(moves[:, i], 90):7.2f}")
for label in ("Lateral Meniscus", "Lateral OA", "MCL"):
    i = labels.index(label)
    change = np.abs(np.log(outputs["pattern"][:, i] / (1 - outputs["pattern"][:, i])) - np.log(outputs["anchor"][:, i] / (1 - outputs["anchor"][:, i])))
    print(label, "correlation of |logit change| with:", {name: round(float(np.corrcoef(change, moves[:, k])[0, 1]), 2) for k, name in enumerate(columns)})
