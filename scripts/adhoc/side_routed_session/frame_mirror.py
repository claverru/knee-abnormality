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
from knee_abnormality.inference.reflection import reflect_batch, sagittal_types

plain, mirrored = [], []
kinds = sagittal_types(config.data.series_types)
with torch.inference_mode():
    for row in range(studies.height):
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        first = model.encode_batch(batch)
        if not first["routable"].any() or not bool(batch["knee_side_certain"][0]):
            continue
        second = model.encode_batch(reflect_batch(batch, kinds, model.repeated_middle_slice))
        for store, encoded in ((plain, first), (mirrored, second)):
            store.append(
                np.concatenate(
                    (
                        model.rotation_angles(encoded["study_frame_rotation"][0]).cpu().numpy(),
                        encoded["study_frame_shift"][0].float().cpu().numpy(),
                        encoded["study_frame_sizes"][0].float().cpu().numpy(),
                    )
                )
            )
plain, mirrored = np.array(plain), np.array(mirrored)
names = ("nod", "roll", "turn", "shift medial", "shift front", "shift up", "upper cut", "lower cut", "core radius")
print(len(plain), "side-certain knees: the frame inferred from the knee and from its mirror image")
for number, name in enumerate(names):
    difference = plain[:, number] - mirrored[:, number]
    print(f"{name:14} SD across knees {plain[:, number].std():5.2f}   correlation with the mirrored pass {np.corrcoef(plain[:, number], mirrored[:, number])[0, 1]:+.2f}   SD of the difference {difference.std():5.2f}")
