import json
from itertools import product
from pathlib import Path

import torch

from knee_abnormality.config import load_config
from knee_abnormality.data.sources import prediction_studies
from knee_abnormality.inference.checkpoint import load_checkpoint_model

run = Path("outputs/training/20261006-073158-side-routed-hard-octants")
config = load_config(run / "effective_config.json")
targets, _, _ = prediction_studies(config, "validation", 1)
selected = json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())
model = load_checkpoint_model(config, targets, Path(selected["path"]), torch.device("cpu"), expected_fold=1)
axis = torch.linspace(-45, 45, 31)
grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 1, 3)
with torch.no_grad():
    prior = model.position_priors(grid, torch.ones(grid.shape[0], dtype=torch.bool))[..., 0]
points = grid[:, 0]
print("octant (medial/lateral, front/back, up/down): mean attention prior minus the label's mean, in logits")
names = []
masks = []
for m, f, u in product((1, -1), repeat=3):
    names.append(("M" if m > 0 else "L") + ("F" if f > 0 else "B") + ("U" if u > 0 else "D"))
    masks.append((points[:, 0] * m > 0) & (points[:, 1] * f > 0) & (points[:, 2] * u > 0))
print(f"{'':18}" + "".join(f"{n:>7}" for n in names) + "   spread")
for number, name in enumerate(targets):
    values = prior[number]
    row = [float(values[mask].mean() - values.mean()) for mask in masks]
    print(f"{name:18}" + "".join(f"{v:+7.2f}" for v in row) + f"   {max(row) - min(row):.2f}")
