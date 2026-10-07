import json
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

import sys

run = Path(sys.argv[1])
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY).head(150)
device = torch.device("cuda")
selected = json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())
model = load_checkpoint_model(config, targets, Path(selected["path"]), device, expected_fold=1)
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
print("landmark attention weight norm per landmark", model.frame_attention.weight.float().norm(dim=1).tolist())
print("shared angles deg", model.rotation_angles(model.frame_rotation()).tolist())
shifts, angles, effective, counts, spans, places = [], [], [], [], [], []
with torch.inference_mode():
    for row in range(studies.height):
        encoded = model.encode_batch(with_images(move_batch(collate_studies([dataset[row]]), device)))
        if not encoded["routable"].any():
            continue
        shifts.append(encoded["study_frame_shift"][0].float().cpu().numpy())
        angles.append(model.rotation_angles(encoded["study_frame_rotation"][0]).cpu().numpy())
        fine = model.fine(encoded)
        sign = (1 - 2 * encoded["left_probability"])[:, None]
        raw = torch.stack(
            (sign * fine["knee_x"], fine["anterior_axis"], fine["superior_axis"]), -1
        ).float()
        valid = fine["visual_valid"] & encoded["routable"][:, None]
        tokens = model.fine_norm(model._base_tokens(fine))
        scores = model.frame_attention(
            model.frame_head(tokens.float() + model.position_code(raw, valid.any(-1)).float())
        )[valid]
        attention = scores.softmax(0)
        entropy = -(attention * attention.clamp_min(1e-30).log()).sum(0)
        effective.append((entropy.exp() / len(scores)).cpu().numpy())
        counts.append(len(scores))
        spans.append((scores.max(0).values - scores.min(0).values).cpu().numpy())
        places.append((attention.T @ raw[valid]).cpu().numpy())
shifts, angles, effective, spans, places = map(np.array, (shifts, angles, effective, spans, places))
print("studies", len(shifts), "valid fine tokens per study, median", int(np.median(counts)))
for name, values, unit in (("shift medial/front/up", shifts, "mm"), ("angle nod/roll/turn", angles, "deg")):
    print(f"{name:24} mean {values.mean(0).round(2)}  SD across knees {values.std(0).round(2)}  min {values.min(0).round(2)}  max {values.max(0).round(2)} {unit}")
print("share of tokens a landmark effectively uses (1 = flat map), centre/front/up: mean", effective.mean(0).round(3), "min", effective.min(0).round(3))
print("score range within a study (logits), centre/front/up: mean", spans.mean(0).round(2))
print("landmark position mean (medial, front, up) mm:", places.mean(0).round(1).tolist())
print("landmark position SD across knees mm:", places.std(0).round(1).tolist())
