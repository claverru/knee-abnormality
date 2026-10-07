import json
import sys
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

run = Path(sys.argv[1])
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
device = torch.device("cuda")
selected = json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())
model = load_checkpoint_model(config, targets, Path(selected["path"]), device, expected_fold=1)
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
identity = torch.eye(3, device=device)
shifts, angles = [], []
logits = {"learned frame": [], "starting frame": [], "starting centre, learned angles": [], "learned centre, starting angles": []}
with torch.inference_mode():
    for row in range(studies.height):
        encoded = model.encode_batch(with_images(move_batch(collate_studies([dataset[row]]), device)))
        if encoded["routable"].any():
            shifts.append(encoded["study_frame_shift"][0].float().cpu().numpy())
            angles.append(model.rotation_angles(encoded["study_frame_rotation"][0]).cpu().numpy())
        sign = (1 - 2 * encoded["left_probability"])[:, None]
        zero = torch.zeros_like(encoded["frame_shift"])
        flat = identity.expand_as(encoded["frame_rotation"])
        for name, shift, rotation in (
            ("learned frame", encoded["frame_shift"], encoded["frame_rotation"]),
            ("starting frame", zero, flat),
            ("starting centre, learned angles", zero, encoded["frame_rotation"]),
            ("learned centre, starting angles", encoded["frame_shift"], flat),
        ):
            moved = {**encoded, "frame_shift": shift, "frame_rotation": rotation}
            with torch.autocast(device_type="cuda", enabled=False):
                moved["position"] = model.frame(moved, sign)
            logits[name].append(model.pool_encoded(moved).float().cpu()[0])
shifts, angles = np.array(shifts), np.array(angles)
for name, values, unit in (("shift medial/front/up", shifts, "mm"), ("angle nod/roll/turn", angles, "deg")):
    print(f"{name:24} mean {values.mean(0).round(2)}  SD across knees {values.std(0).round(2)}  min {values.min(0).round(1)}  max {values.max(0).round(1)} {unit}")
split_dir = config.paths.data_root / "folds" / config.validation.split
frame = (
    pl.read_csv(split_dir / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split_dir / "targets.csv"), on=STUDY)
    .sort(STUDY)
)
assert frame[STUDY].to_list() == studies[STUDY].to_list()
print(f"{'':34}" + "".join(f"{t[:9]:>10}" for t in ("macro", *targets)))
for name, values in logits.items():
    scores = torch.stack(values).numpy()
    aucs = [float(roc_auc_score(frame[t].to_numpy() >= 0.5, scores[:, n])) for n, t in enumerate(targets)]
    print(f"{name:34}" + f"{np.mean(aucs):10.5f}" + "".join(f"{a:10.4f}" for a in aucs))
