import json
from pathlib import Path

import torch

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch

run = Path("outputs/training/20261005-185535-side-routed-knee-frame")
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY).head(40)
device = torch.device("cuda")
selected = json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())
model = load_checkpoint_model(
    config, targets, Path(selected["path"]), device, expected_fold=1
)
index = build_series_index(
    config.paths.data_root, config.data, split, studies[STUDY].to_list()
)
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
base_norm, code_norm, shared_norm, change, flips = [], [], [], [], []
with torch.inference_mode():
    for row in range(studies.height):
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        encoded = model.encode_batch(batch)
        valid = encoded["visual_valid"] & encoded["routable"][:, None]
        base = model._base_tokens(encoded).float()
        own = model.position_code(encoded["position"], encoded["routable"], False)
        shared = model.position_code(encoded["position"], encoded["routable"], True)
        base_norm.append(base[valid].norm(dim=-1))
        code_norm.append(own[valid].norm(dim=-1))
        shared_norm.append(shared[valid].norm(dim=-1))
        logits = model.pool_encoded(encoded).float()
        zeroed = {**encoded, "routable": torch.zeros_like(encoded["routable"])}
        without = model._attention_pool(
            model.mil_norm(base.flatten(0, 1)),
            model.mil_norm(base.flatten(0, 1)),
            encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            encoded["image_type"],
            encoded["visual_valid"].flatten(),
            encoded["batch_size"],
            model.routing_weights(
                encoded["knee_x"],
                encoded["anterior_axis"],
                encoded["left_probability"][:, None],
                encoded["routable"][:, None],
                encoded["centre_shift"][:, None],
            ).flatten(1),
            encoded["view_index"][:, None].expand_as(encoded["visual_valid"]).flatten(),
        ).float()
        change.append((logits - without).abs()[0])
base_norm, code_norm, shared_norm = (torch.cat(v) for v in (base_norm, code_norm, shared_norm))
print("token norm before position", float(base_norm.mean()))
print("position code norm, own heads", float(code_norm.mean()), "ratio", float((code_norm / base_norm).mean()))
print("position code norm, shared heads", float(shared_norm.mean()), "ratio", float((shared_norm / base_norm).mean()))
print("unknown position norm", float(model.unknown_position.norm()))
change = torch.stack(change).mean(0)
for target, value in zip(targets, change.tolist(), strict=True):
    print(f"mean |logit change| when the position code is removed: {target:18s} {value:.3f}")
