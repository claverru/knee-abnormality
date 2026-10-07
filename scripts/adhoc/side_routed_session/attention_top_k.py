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
TOPS = (3, 10, 30)
top = np.zeros((len(TOPS), len(targets), len(BANDS), len(REGIONS)))
held = np.zeros((len(TOPS), len(targets)))
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
            ordered = attention[number][valid].sort(descending=True)
            for slot, k in enumerate(TOPS):
                chosen = cell[ordered.indices[:k]]
                top[slot, number] += (
                    torch.bincount(chosen, minlength=size).float() / k
                ).reshape(len(BANDS), len(REGIONS)).cpu().numpy()
                held[slot, number] += float(ordered.values[:k].sum())
share /= knees
print(f"sizes: joint line {JOINT} mm from the geometric centre, band +-{BAND} mm, core radius {CORE} mm; {knees} knees; median valid tokens per knee about {int(1 / max(share.min(), 1e-9) * 0)}")
print("token share by band (femur, joint, tibia):", (100 * share.sum(1)).round(1), " core:", round(100 * share[:, 0].sum(), 1))
for slot, k in enumerate(("all tokens", *TOPS)):
    print(f"\n{'attention over all tokens' if slot == 0 else f'the {k} highest-attention tokens per knee'}")
    print(f"{'':18}{'femur':>7}{'joint':>7}{'tibia':>7}{'core':>7}   joint band: core / MF / MB / LF / LB        attention held")
    for number, name in enumerate(targets):
        values = (mass[number] if slot == 0 else top[slot - 1, number]) / max(counted[number], 1)
        kept = 1.0 if slot == 0 else held[slot - 1, number] / max(counted[number], 1)
        print(
            f"{name:18}"
            + "".join(f"{100 * v:7.0f}" for v in values.sum(1))
            + f"{100 * values[:, 0].sum():7.0f}"
            + "   "
            + " / ".join(f"{100 * v:4.0f}" for v in values[1])
            + f"            {100 * kept:5.0f}%"
        )
