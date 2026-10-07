import json
import math
import sys
from pathlib import Path

RUN = Path("outputs/training/20261007-081431-side-routed-crop-offsets-3mm").resolve()
WEIGHTS = Path(
    "outputs/autoresearch/routing-and-shifts/experiments/height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402/training"
)
MULTIPLIER = int(sys.argv[1])
PLANE = sys.argv[2]

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

OUT = Path("outputs/analysis/routing-and-shifts-frames-20261007/mcl-spatial-maps")
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
planes = [kind.plane for kind in config.data.series_types]
GOLDEN = math.pi * (3 - math.sqrt(5))
R = 3.0
state = {}


def sunflower(active, radius):
    order = (active.cumsum(1) - 1).clamp_min(0).float()
    count = active.sum(1, keepdim=True).float()
    r = torch.where(count > 1, ((order + 0.5) / count.clamp_min(1)).sqrt(), 0.0)
    angle = order * GOLDEN
    return radius * r[..., None] * torch.stack((angle.sin(), angle.cos()), -1), order


def offsets(active):
    mode = state["mode"]
    full, order = sunflower(active, R)
    zero = torch.zeros_like(full)
    lateral = state["lateral"]
    alternate = 1 - 2 * (order % 2)
    if mode == "anchor":
        out = zero
    elif mode == "sunflower 3":
        out = full
    elif mode == "sunflower 1.5":
        out = full / 2
    elif mode == "across only":
        out = torch.stack((zero[..., 0], full[..., 1]), -1)
    elif mode == "down only":
        out = torch.stack((full[..., 0], zero[..., 1]), -1)
    elif mode == "all 3 lateral":
        out = torch.stack((zero[..., 0], R * lateral), -1)
    elif mode == "all 3 medial":
        out = torch.stack((zero[..., 0], -R * lateral), -1)
    elif mode == "two views across":
        out = torch.stack((zero[..., 0], R * alternate), -1)
    elif mode == "two views down":
        out = torch.stack((R * alternate, zero[..., 1]), -1)
    elif mode == "all 3 up":
        out = torch.stack((zero[..., 0] - R, zero[..., 1]), -1)
    elif mode == "all 3 down":
        out = torch.stack((zero[..., 0] + R, zero[..., 1]), -1)
    return out * state["keep"][:, None, None]


windows.offsets = offsets
MODES = (
    "anchor", "sunflower 3", "sunflower 1.5", "across only", "down only", "two views across",
    "two views down", "all 3 lateral", "all 3 medial", "all 3 up", "all 3 down",
)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate_studies)
outputs = {name: [] for name in MODES}
for batch in loader:
    batch = with_images(move_batch(batch, gpu))
    kinds = batch["series_type_index"].tolist()
    state["keep"] = torch.tensor([planes[k] == PLANE for k in kinds], device=gpu).float()
    column = batch["patient_maps"][:, :, 1, 0, 0].float()
    column = torch.where(torch.isfinite(column), column, 0.0).sign()
    left = batch["knee_side_left"][batch["series_study_index"]].float()
    state["lateral"] = column * (2 * left - 1)[:, None]
    for name in MODES:
        state["mode"] = name
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs[name].append(torch.sigmoid(model(batch).float()).cpu().numpy())
outputs = {name: np.concatenate(values) for name, values in outputs.items()}
labels = list(targets)
truth = studies.select(STUDY).join(
    pl.read_csv(Path("data/folds/all_5fold_scanner_v2") / "targets.csv"), on=STUDY, how="left"
)
np.savez(OUT / f"pattern-directions-{PLANE.lower()}-x{MULTIPLIER}.npz", names=np.array(names), labels=np.array(labels), **{k.replace(" ", "_"): v for k, v in outputs.items()})
print(f"{'label':17}{'anchor':>8}" + "".join(f"{name[:11]:>12}" for name in MODES[1:]))
rows = []
for i, label in enumerate(labels):
    y = truth[label].to_numpy() >= 0.5
    values = [roc_auc_score(y, outputs[name][:, i]) for name in MODES]
    rows.append(values)
    print(f"{label:17}{values[0]:8.4f}" + "".join(f"{v - values[0]:+12.4f}" for v in values[1:]))
values = np.mean(rows, 0)
print(f"{'macro':17}{values[0]:8.4f}" + "".join(f"{v - values[0]:+12.4f}" for v in values[1:]))
