import json
import sys
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()
MULTIPLIER = int(sys.argv[2])
FRACTIONS = [float(v) for v in sys.argv[3].split(",")]

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

OUT = Path("outputs/analysis/routing-and-shifts-frames-20261007/frame-crops")
config = load_config(RUN / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
names = studies[STUDY].to_list()
index = build_series_index(config.paths.data_root, config.data, split, names)
dataset = KneeStudyDataset(studies, index, config.data, targets, None, MULTIPLIER)
selected = json.loads((RUN / "fold_1" / "selected_checkpoint.json").read_text())
gpu = torch.device("cuda")
model = checkpoint_model(read_checkpoint(Path(selected["path"])), config, targets, gpu).eval()
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate_studies)
outputs = {fraction: [] for fraction in FRACTIONS}
for batch in loader:
    batch = with_images(move_batch(batch, gpu))
    for fraction in FRACTIONS:
        model.view_frame_rotation_fraction = fraction
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs[fraction].append(torch.sigmoid(model(batch).float()).cpu().numpy())
outputs = {fraction: np.concatenate(values) for fraction, values in outputs.items()}
labels = list(targets)
truth = studies.select(STUDY).join(
    pl.read_csv(Path("data/folds/all_5fold_scanner_v2") / "targets.csv"), on=STUDY, how="left"
)
np.savez(OUT / f"rotation-check-{RUN.name[16:]}-x{MULTIPLIER}.npz", names=np.array(names), labels=np.array(labels), **{f"fraction_{f}": v for f, v in outputs.items()})
print(f"{'label':18}" + "".join(f"{'angle x ' + str(f):>13}" for f in FRACTIONS))
rows = []
for i, label in enumerate(labels):
    y = truth[label].to_numpy() >= 0.5
    values = [roc_auc_score(y, outputs[f][:, i]) for f in FRACTIONS]
    rows.append(values)
    print(f"{label:18}{values[0]:13.4f}" + "".join(f"{v - values[0]:+13.4f}" for v in values[1:]))
values = np.mean(rows, 0)
print(f"{'macro':18}{values[0]:13.4f}" + "".join(f"{v - values[0]:+13.4f}" for v in values[1:]))
print("macro absolute", [round(float(v), 6) for v in values], "recorded", selected.get("prediction_macro_auc"), selected.get("score"))
