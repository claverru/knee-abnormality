import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.modeling.model import ROUTING_MINIMUM_WEIGHT

run = Path(sys.argv[1])
SPOTS = ("Medial Meniscus", "Lateral Meniscus", "ACL", "PF OA", "Medial OA", "Lateral OA", "Effusion")
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY).head(400)
device = torch.device("cuda")
selected = json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())
model = load_checkpoint_model(config, targets, Path(selected["path"]), device, expected_fold=1)
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
split_dir = config.paths.data_root / "folds" / config.validation.split
labels = (
    pl.read_csv(split_dir / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split_dir / "targets.csv"), on=STUDY)
    .sort(STUDY)
    .head(400)
)
truth = np.stack([labels[name].to_numpy() >= 0.5 for name in targets], 1)
numbers = [targets.index(name) for name in SPOTS]
spots = {name: [] for name in SPOTS}
held = {name: [] for name in SPOTS}
positive = {name: [] for name in SPOTS}
with torch.inference_mode():
    for row in range(studies.height):
        encoded = model.encode_batch(with_images(move_batch(collate_studies([dataset[row]]), device)))
        if not encoded["routable"].all():
            continue
        valid = encoded["visual_valid"].flatten()
        own = model.attention(model.study_tokens(encoded)).transpose(0, 1).float()
        shared = model.attention(model.study_tokens(encoded, shared=True)).transpose(0, 1).float()
        scores = torch.where(model.shared_channel[:, None], shared, own)
        view = encoded["view_index"][:, None].expand_as(encoded["visual_valid"]).flatten()
        if model.view_label_bias is not None:
            scores = scores + model.view_label_bias.float()[view].transpose(0, 1)
        prior = torch.where(
            model.shared_channel[model.target_channel][:, None, None],
            model.position_priors(encoded["position"], encoded["routable"]),
            model.position_priors(encoded["fitted_position"], encoded["routable"]),
        ).flatten(1).float()
        routing = model.routing_weights(encoded).flatten(1).float()
        weights = torch.cat((routing, torch.ones_like(routing[:1])))
        routed = (valid.float()[None] @ weights.T >= 1)[0, model.target_route]
        log_weights = weights.clamp_min(ROUTING_MINIMUM_WEIGHT).log() / model.routing_temperature
        attention = (
            scores[model.target_channel]
            + prior
            + torch.where(routed[:, None], log_weights[model.target_route], 0.0)
        ).masked_fill(~valid[None], -torch.inf).softmax(-1)
        points = encoded["fitted_position"].float().flatten(0, 1)
        for name, number in zip(SPOTS, numbers, strict=True):
            chosen = attention[number].topk(10)
            share = chosen.values / chosen.values.sum()
            spots[name].append((share[:, None] * points[chosen.indices]).sum(0).cpu().numpy())
            held[name].append(float(chosen.values.sum()))
            positive[name].append(bool(truth[row, number]))
print("spot = attention-weighted position of each label's ten highest-attention tokens, in the geometric frame (medial, front, up; mm)")
print(f"{'':18}{'knees':>6}  {'mean position':>24}   {'SD across knees':>22}   attention in those ten tokens")
out = {}
for name in SPOTS:
    values, flags, mass = np.array(spots[name]), np.array(positive[name]), np.array(held[name])
    for group, mask in (("positive", flags), ("negative", ~flags)):
        v = values[mask]
        print(f"{name + ' ' + group:26}{len(v):5}  {np.array2string(v.mean(0).round(1)):>24}   {np.array2string(v.std(0).round(1)):>22}   {100 * mass[mask].mean():.0f}%")
    out[name] = {"positive": values[flags].tolist(), "negative": values[~flags].tolist()}
medial, lateral = np.array(spots["Medial Meniscus"]), np.array(spots["Lateral Meniscus"])
span = np.linalg.norm(medial - lateral, axis=1)
middle = (medial + lateral) / 2
line = medial - lateral
turn = np.degrees(np.arctan2(line[:, 1], line[:, 0]))
roll = np.degrees(np.arctan2(line[:, 2], line[:, 0]))
print(f"\nfrom the two meniscus spots, over {len(span)} knees: distance mean {span.mean():.1f} mm SD {span.std():.1f}; midpoint mean {middle.mean(0).round(1)} SD {middle.std(0).round(1)}")
print(f"turn of the line between them mean {turn.mean():.1f} deg SD {turn.std():.1f}; roll mean {roll.mean():.1f} deg SD {roll.std():.1f}")
Path("outputs/analysis/s/side-routing-loop-20261006").mkdir(parents=True, exist_ok=True)
Path(f"outputs/analysis/s/side-routing-loop-20261006/spots-{run.name[16:]}.json").write_text(json.dumps(out))
