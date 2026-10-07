import json
import sys
from pathlib import Path

experiment = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(experiment / "source"), str(experiment / "source" / "src")]

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import apply_attention_temperature, load_checkpoint_model
from knee_abnormality.inference.engine import move_batch

TAUS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25)
MULTIPLIER = int(sys.argv[2])
run = experiment / "training"
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
device = torch.device("cuda")
checkpoint = Path(json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())["path"])
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
dataset = KneeStudyDataset(studies, index, config.data, targets, None, MULTIPLIER)
models = {}
for tau in TAUS:
    model = load_checkpoint_model(config, targets, checkpoint, device, expected_fold=1)
    apply_attention_temperature(model, tau)
    models[tau] = model
logits = {tau: [] for tau in TAUS}
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for row in range(studies.height):
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        for tau, model in models.items():
            logits[tau].append(model(batch).float().cpu()[0])
split_dir = config.paths.data_root / "folds" / config.validation.split
frame = (
    pl.read_csv(split_dir / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split_dir / "targets.csv"), on=STUDY)
    .sort(STUDY)
)
assert frame[STUDY].to_list() == studies[STUDY].to_list()
print(f"image multiplier {MULTIPLIER}")
print(f"{'tau':>6}{'macro':>9}" + "".join(f"{name[:9]:>10}" for name in targets))
for tau in TAUS:
    scores = torch.stack(logits[tau]).numpy()
    aucs = [float(roc_auc_score(frame[name].to_numpy() >= 0.5, scores[:, number])) for number, name in enumerate(targets)]
    print(f"{tau:6.2f}{np.mean(aucs):9.5f}" + "".join(f"{value:10.4f}" for value in aucs))
