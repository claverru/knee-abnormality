import json
from pathlib import Path

import numpy as np
import torch

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import read_checkpoint
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.modeling.model import FRAME_SIZES, KneeAbnormalityModel

run = Path("outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors")
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
device = torch.device("cuda")
checkpoint = read_checkpoint(Path(json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())["path"]))
state = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
learned = state.pop("frame_size")
state["frame_output.weight"] = state["frame_output.weight"][:6]
model = KneeAbnormalityModel(config.model, config.data, targets, load_pretrained=False)
print(model.load_state_dict(state, strict=False))
model.to(device).eval()
rows = list(range(0, studies.height, 6))
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
sizes, spots = [], []
with torch.inference_mode():
    for row in rows:
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        encoded = model.encode_batch(batch)
        sizes.append(encoded["study_frame_sizes"].float().cpu().numpy()[0])
        spots.append(encoded["study_landmarks"].float().cpu().numpy()[0])
sizes, spots = np.array(sizes), np.array(spots)
print(len(rows), "knees. Cuts placed midway (mm, in each knee's frame): mean, SD, 5th to 95th percentile")
for number, name in enumerate(FRAME_SIZES):
    column = sizes[:, number]
    print(f"{name:14} {column.mean():7.1f} {column.std():6.1f}   {np.percentile(column, 5):6.1f} to {np.percentile(column, 95):6.1f}")
band = sizes[:, 0] - sizes[:, 1]
print(f"joint band thickness {band.mean():.1f} mm (SD {band.std():.1f}); zero-thickness knees {int((band < 1).sum())}; ring equal to core {int((sizes[:, 3] - sizes[:, 2] < 1).sum())}")
print("spots in the fitted frame (medial, front, up), mean and SD:")
for number, name in enumerate(("medial meniscus", "lateral meniscus", "notch", "kneecap")):
    print(f"{name:17} {spots[:, number].mean(0).round(1)}  SD {spots[:, number].std(0).round(1)}")
