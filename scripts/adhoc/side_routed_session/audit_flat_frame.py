import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch

ROUTED = ("Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA")
run = Path("outputs/training/20261006-092326-side-routed-fine-landmarks")
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
device = torch.device("cuda")
selected = json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())
model = load_checkpoint_model(config, targets, Path(selected["path"]), device, expected_fold=1)
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
learned = model.frame_attention.weight.detach().clone()
logits = {"learned frame": [], "frame before learning": []}
with torch.inference_mode():
    for row in range(studies.height):
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        for name, weight in (("learned frame", learned), ("frame before learning", torch.zeros_like(learned))):
            model.frame_attention.weight.copy_(weight)
            logits[name].append(model.pool_encoded(model.encode_batch(batch)).float().cpu()[0])
split_dir = config.paths.data_root / "folds" / config.validation.split
frame = (
    pl.read_csv(split_dir / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split_dir / "targets.csv"), on=STUDY)
    .sort(STUDY)
)
assert frame[STUDY].to_list() == studies[STUDY].to_list()
for name, values in logits.items():
    scores = torch.stack(values).numpy()
    aucs = {
        target: float(roc_auc_score(frame[target].to_numpy() >= 0.5, scores[:, number]))
        for number, target in enumerate(targets)
    }
    print(
        f"{name:24} macro {np.mean(list(aucs.values())):.5f}  routed five {sum(aucs[t] for t in ROUTED):.4f}  "
        + "  ".join(f"{t} {aucs[t]:.4f}" for t in (*ROUTED, "MCL", "Effusion", "Baker's"))
    )
