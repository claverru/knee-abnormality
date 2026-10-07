import json
import sys
from pathlib import Path

import numpy as np
import torch

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
studies = studies.sort(STUDY).head(200)
device = torch.device("cuda")
selected = json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())
model = load_checkpoint_model(config, targets, Path(selected["path"]), device, expected_fold=1)
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
regression = model.frame_output.weight.detach().clone()
landmark = model.landmark_attention.weight.detach().clone()
parts = {"both": (regression, landmark), "regression only": (regression, 0 * landmark), "landmarks only": (0 * regression, landmark)}
angles = {name: [] for name in parts}
shifts = {name: [] for name in parts}
sizes = []
header = []
with torch.inference_mode():
    for row in range(studies.height):
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        for name, (a, b) in parts.items():
            model.frame_output.weight.copy_(a)
            model.landmark_attention.weight.copy_(b)
            encoded = model.encode_batch(batch)
            if not encoded["routable"].any():
                continue
            angles[name].append(model.rotation_angles(encoded["study_frame_rotation"][0]).cpu().numpy())
            shifts[name].append(encoded["study_frame_shift"][0].float().cpu().numpy())
            if name == "both":
                sizes.append(encoded["study_frame_sizes"][0].float().cpu().numpy())
                axis = encoded["study_axis"][0].float().cpu().numpy()
                header.append(np.degrees(np.arctan2(axis[1], axis[0])))
print("shared angles deg", model.rotation_angles(model.frame_rotation().detach()).cpu().numpy().round(2), "shared shift pitches", model.frame_shift.detach().cpu().numpy().round(3), "shared sizes pitches", model.frame_size.detach().cpu().numpy().round(3))
for name in parts:
    a, s = np.array(angles[name]), np.array(shifts[name])
    print(f"{name:16} nod/roll/turn mean {a.mean(0).round(1)} SD {a.std(0).round(1)} min {a.min(0).round(0)} max {a.max(0).round(0)} | shift mean {s.mean(0).round(1)} SD {s.std(0).round(1)}")
sizes = np.array(sizes)
print("sizes top/bottom/radius mean", sizes.mean(0).round(1), "SD", sizes.std(0).round(1), "min", sizes.min(0).round(0), "max", sizes.max(0).round(0))
both = np.array(angles["both"])
header = np.array(header)
print("correlation of inferred turn with the header rotation of the knee:", round(float(np.corrcoef(both[:, 2], header)[0, 1]), 2), " header rotation SD", header.std().round(1))
print("share of knees with |roll| > 10:", round(float((np.abs(both[:, 1]) > 10).mean()), 2), " |nod| > 15:", round(float((np.abs(both[:, 0]) > 15).mean()), 2), " |turn| > 10:", round(float((np.abs(both[:, 2]) > 10).mean()), 2))
