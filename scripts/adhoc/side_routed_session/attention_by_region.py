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
JOINT, BAND, CORE = -14.0, 10.0, 14.0
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
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
)
assert labels[STUDY].to_list() == studies[STUDY].to_list()
truth = np.stack([labels[name].to_numpy() >= 0.5 for name in targets], 1)
REGIONS = ("core", "MF", "MB", "LF", "LB")
BANDS = ("femur", "joint", "tibia")
mass = np.zeros((len(targets), len(BANDS), len(REGIONS)))
share = np.zeros((len(BANDS), len(REGIONS)))
counted = np.zeros(len(targets))
knees = 0
with torch.inference_mode():
    for row in range(studies.height):
        encoded = model.encode_batch(with_images(move_batch(collate_studies([dataset[row]]), device)))
        if not encoded["routable"].all():
            continue
        valid = encoded["visual_valid"].flatten()
        scores = torch.where(
            model.shared_channel[:, None],
            model.attention(model.study_tokens(encoded, shared=True)).transpose(0, 1),
            model.attention(model.study_tokens(encoded)).transpose(0, 1),
        ).float()
        view = encoded["view_index"][:, None].expand_as(encoded["visual_valid"]).flatten()
        if model.view_label_bias is not None:
            scores = scores + model.view_label_bias.float()[view].transpose(0, 1)
        prior = model.position_priors(encoded["position"], encoded["routable"]).flatten(1).float()
        routing = model.routing_weights(encoded).flatten(1).float()
        weights = torch.cat((routing, torch.ones_like(routing[:1])))
        routed = (valid.float()[None] @ weights.T >= 1)[0, model.target_route]
        log_weights = weights.clamp_min(ROUTING_MINIMUM_WEIGHT).log() / model.routing_temperature
        attention = (
            scores[model.target_channel]
            + prior
            + torch.where(routed[:, None], log_weights[model.target_route], 0.0)
        ).masked_fill(~valid[None], -torch.inf).softmax(-1)
        sign = (1 - 2 * encoded["left_probability"])[:, None]
        m = (sign * encoded["knee_x"]).flatten()
        f = encoded["anterior_axis"].flatten()
        h = encoded["superior_axis"].flatten()
        band = torch.where(h > JOINT + BAND, 0, torch.where(h < JOINT - BAND, 2, 1))
        region = torch.where(
            torch.hypot(m, f) < CORE, 0, 1 + 2 * (m < 0).long() + (f < 0).long()
        )
        cell = (band * len(REGIONS) + region)[valid]
        size = len(BANDS) * len(REGIONS)
        share += (
            torch.bincount(cell, minlength=size).float() / valid.sum()
        ).reshape(len(BANDS), len(REGIONS)).cpu().numpy()
        knees += 1
        for number in np.flatnonzero(truth[row]):
            mass[number] += (
                torch.bincount(cell, weights=attention[number][valid], minlength=size)
                .reshape(len(BANDS), len(REGIONS))
                .cpu()
                .numpy()
            )
            counted[number] += 1
share /= knees
print(f"sizes used: joint line {JOINT} mm from the geometric centre, band +-{BAND} mm, core radius {CORE} mm; {knees} knees")
print("share of all tokens per region (rows femur, joint, tibia; columns core, medial front, medial back, lateral front, lateral back)")
print((100 * share).round(1))
out = {"share": share.tolist(), "targets": {}}
for number, name in enumerate(targets):
    values = mass[number] / max(counted[number], 1)
    out["targets"][name] = values.tolist()
    print(f"\n{name} (positive knees {int(counted[number])}): attention % per region, and in brackets the ratio to the token share")
    for line, band in enumerate(BANDS):
        print(
            f"  {band:6}"
            + "".join(
                f"  {REGIONS[c]:>4} {100 * values[line, c]:5.1f} ({values[line, c] / max(share[line, c], 1e-9):4.1f})"
                for c in range(len(REGIONS))
            )
        )
Path("outputs/analysis/side-routing-attention-by-region-20261006").mkdir(parents=True, exist_ok=True)
Path("outputs/analysis/side-routing-attention-by-region-20261006/attention.json").write_text(json.dumps(out))
