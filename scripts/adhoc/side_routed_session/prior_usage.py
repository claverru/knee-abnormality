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

run = Path("outputs/training/20261005-194836-side-routed-attention-prior")
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY).head(60)
device = torch.device("cuda")
selected = json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())
model = load_checkpoint_model(
    config, targets, Path(selected["path"]), device, expected_fold=1
)
index = build_series_index(
    config.paths.data_root, config.data, split, studies[STUDY].to_list()
)
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
spread, change, centre = [], [], []
with torch.inference_mode():
    for row in range(studies.height):
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        encoded = model.encode_batch(batch)
        valid = encoded["visual_valid"] & encoded["routable"][:, None]
        prior = model.position_priors(encoded["position"], encoded["routable"])
        spread.append(torch.stack([p[valid].std() for p in prior]))
        best = torch.stack([p.masked_fill(~valid, -torch.inf).flatten().argmax() for p in prior])
        centre.append(encoded["position"].flatten(0, 1)[best])
        logits = model.pool_encoded(encoded).float()
        without = model.pool_encoded(
            {**encoded, "routable": encoded["routable"]}
            | {"position": encoded["position"] * 0 + 1e6}
        ).float()
        weight, bias = model.position_prior.weight.clone(), model.position_prior.bias.clone()
        model.position_prior.weight.zero_()
        model.position_prior.bias.zero_()
        without = model.pool_encoded(encoded).float()
        model.position_prior.weight.copy_(weight)
        model.position_prior.bias.copy_(bias)
        change.append((logits - without).abs()[0])
spread = torch.stack(spread).mean(0)
change = torch.stack(change).mean(0)
centre = torch.stack(centre).median(0).values
print("prior weight norm", float(model.position_prior.weight.norm()))
print(f"{'target':18s} prior sd over tokens | mean abs logit change without prior | where the prior peaks, median mm (medial or own-side depth, front of boundary, up)")
for number, target in enumerate(targets):
    print(
        f"{target:18s} {spread[number]:.2f} | {change[number]:.3f} | "
        f"({centre[number, 0]:+.0f}, {centre[number, 1]:+.0f}, {centre[number, 2]:+.0f})"
    )
print("offsets", model.centre_offsets().tolist(), "widths", model.routing_widths().tolist())
