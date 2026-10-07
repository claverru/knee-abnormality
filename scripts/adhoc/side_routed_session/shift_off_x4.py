import json
import sys
from pathlib import Path

E = Path("outputs/autoresearch/routing-and-shifts/experiments").resolve()
RUN = E / "spot-shift-locator-keeps-spatial-maps-k-20261007T043843Z-632ceac9"
sys.path[:0] = [str(RUN / "source"), str(RUN / "source" / "src")]

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
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch

OUT = Path("outputs/analysis/routing-and-shifts-frames-20261007/mcl-spatial-maps")
config = load_config(RUN / "training" / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
names = studies[STUDY].to_list()
index = build_series_index(config.paths.data_root, config.data, split, names)
dataset = KneeStudyDataset(studies, index, config.data, targets, None, config.validation.prediction_image_multiplier)
print('multiplier', config.validation.prediction_image_multiplier)
selected = json.loads((RUN / "training" / "fold_1" / "selected_checkpoint.json").read_text())
gpu = torch.device("cuda")
model = load_checkpoint_model(config, targets, Path(selected["path"]), gpu, expected_fold=1).eval()
locator = model.shift_locator


class Switch(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.enabled = True

    def forward(self, batch):
        out = self.inner(batch)
        return out if self.enabled else {**out, "delta": torch.zeros_like(out["delta"])}


switch = Switch(locator)
model.shift_locator = switch
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate_studies)
on, off = [], []
for batch in loader:
    batch = with_images(move_batch(batch, gpu))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        switch.enabled = True
        on.append(torch.sigmoid(model(batch).float()).cpu().numpy())
        switch.enabled = False
        off.append(torch.sigmoid(model(batch).float()).cpu().numpy())
on, off = np.concatenate(on), np.concatenate(off)
saved = pl.read_csv(RUN / "training" / "fold_1" / "validation_predictions.csv").sort(STUDY)
labels = [c for c in saved.columns if c not in (STUDY, "fold")]
assert on.shape[1] == len(labels), (on.shape, labels)
truth = studies.select(STUDY).join(pl.read_csv(Path("data/folds/all_5fold_scanner_v2") / "targets.csv"), on=STUDY, how="left")
np.savez(OUT / "shift-on-off-x4.npz", on=on, off=off, names=np.array(names), labels=np.array(labels))
print(f"{'label':28}{'shift on':>10}{'shift off':>11}{'on - off':>10}")
total = []
for i, label in enumerate(labels):
    y = truth[label].to_numpy() >= 0.5
    a, b = roc_auc_score(y, on[:, i]), roc_auc_score(y, off[:, i])
    total.append((a, b))
    print(f"{label:28}{a:10.4f}{b:11.4f}{a - b:+10.4f}")
a, b = np.mean(total, 0)
print(f"{'macro':28}{a:10.4f}{b:11.4f}{a - b:+10.4f}")
print("check, saved x? MCL:", roc_auc_score(truth["MCL"].to_numpy() >= 0.5, saved["MCL"].to_numpy()))
