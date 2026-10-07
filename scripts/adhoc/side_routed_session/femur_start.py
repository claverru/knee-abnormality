import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import read_checkpoint
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.modeling.model import KneeAbnormalityModel

RUN = Path("outputs/autoresearch/routing-and-shifts/experiments/height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402/training")
config = load_config(RUN / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 4)
selected = json.loads((RUN / "fold_1" / "selected_checkpoint.json").read_text())
checkpoint = read_checkpoint(Path(selected["path"]))
state = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
model = KneeAbnormalityModel(config.model, config.data, targets, load_pretrained=False)
print(model.load_state_dict(state, strict=False))
gpu = torch.device("cuda")
model.to(gpu).eval()
if len(sys.argv) > 1:
    with torch.no_grad():
        model.femur_offset.fill_(float(sys.argv[1]))
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate_studies)
rows = []
for batch in loader:
    batch = with_images(move_batch(batch, gpu))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        encoded = model.encode_batch(batch)
    pitch = torch.zeros(encoded["batch_size"], device=gpu).index_add(0, encoded["study"], encoded["token_pitch"].float()) / torch.zeros(encoded["batch_size"], device=gpu).index_add(0, encoded["study"], torch.ones_like(encoded["token_pitch"].float()))
    place = encoded["study_kneecap_place"].float()
    sizes = encoded["study_frame_sizes"].float()
    rows.append(torch.stack((place[:, 1] / pitch, (place[:, 2] - sizes[:, 0]) / pitch, torch.rad2deg(encoded["study_femur_lean"].float()), encoded["height_cuts_from_spots_applied"].float(), pitch, place[:, 1], place[:, 2], sizes[:, 0]), -1).cpu().numpy())
rows = np.concatenate(rows)
found = rows[:, 3] > 0
print("knees", len(rows), "with the kneecap spot", int(found.sum()))
for i, name in enumerate(("kneecap front, pitches", "kneecap above the upper cut, pitches", "lean deg")):
    v = rows[found, i]
    print(f"{name:38} mean {v.mean():+7.2f} sd {v.std():6.2f} p5 {np.percentile(v, 5):+7.2f} median {np.median(v):+7.2f} p95 {np.percentile(v, 95):+7.2f}")
print("pitch mm mean", rows[:, 4].mean(), "kneecap front mm", rows[found, 5].mean(), "height mm", rows[found, 6].mean(), "upper cut mm", rows[found, 7].mean())
for label, chosen in (("bent by eye", (532, 200, 227, 153, 72)), ("straight by eye", (370, 374, 284, 391, 80))):
    print(label, {row: round(float(rows[row, 2]), 1) for row in chosen}, "kneecap front mm", {row: round(float(rows[row, 5]), 1) for row in chosen})
np.save("outputs/analysis/routing-and-shifts-frames-20261007/knee-bend/femur-start.npy", rows)
