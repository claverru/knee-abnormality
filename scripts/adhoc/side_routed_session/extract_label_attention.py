import json
import sys
from pathlib import Path

experiment = Path(sys.argv[1]).resolve()
kind = sys.argv[2]
out = Path(sys.argv[3])
sys.path[:0] = [str(experiment / "source"), str(experiment / "source" / "src")]

import numpy as np
import torch

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch

TOP = 20
run = experiment / "training"
text = (
    (run / "effective_config.json")
    .read_text()
    .replace("/home/claudio/src/knee-abnormality-fixes-and-removals", str(Path.cwd()))
)
local = out.parent / f"{kind}-effective_config.json"
local.parent.mkdir(parents=True, exist_ok=True)
local.write_text(text)
config = load_config(local)
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
device = torch.device("cuda")
checkpoint = Path(
    json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())["path"]
)
if kind == "old":
    model = load_checkpoint_model(
        config, targets, checkpoint, device, expected_fold=1, attention_temperature=1.0
    )
else:
    from knee_abnormality.modeling.model import ROUTING_MINIMUM_WEIGHT

    model = load_checkpoint_model(config, targets, checkpoint, device, expected_fold=1)
index = build_series_index(
    config.paths.data_root, config.data, split, studies[STUDY].to_list()
)
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
store = {
    key: []
    for key in (
        "region",
        "index",
        "weight",
        "tokens",
        "study",
        "axis_a",
        "axis_b",
        "position",
        "frame",
        "valid_count",
        "image_sum",
        "logit",
    )
}
with torch.inference_mode():
    for row in range(studies.height):
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        encoded = model.encode_batch(batch)
        valid = encoded["visual_valid"].flatten()
        if kind == "old":
            tokens, _ = model._pooling_tokens(encoded)
            scores = model.attention(tokens).transpose(0, 1).float()
            view = (
                encoded["view_index"][:, None]
                .expand_as(encoded["visual_valid"])
                .flatten()
            )
            if model.view_label_bias is not None:
                scores = scores + model.view_label_bias.float()[view].transpose(0, 1)
            medial_axis = encoded["medial_axis"].flatten().float()
            signs = model.meniscus_mask_signs[:, None]
            own_side = (signs == 0) | (signs * medial_axis[None] > -10.0)
            restricted = valid[None] & own_side
            allowed = torch.where(
                restricted.any(-1, keepdim=True),
                restricted,
                valid[None].expand_as(restricted),
            )
            attention = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
            axis_a = medial_axis
            axis_b = encoded["anterior_axis"].flatten().float()
            position = torch.zeros(len(valid), 3, device=device)
            frame = position
            region = allowed
        else:
            own = model.attention(model.study_tokens(encoded)).transpose(0, 1).float()
            shared = (
                model.attention(model.study_tokens(encoded, shared=True))
                .transpose(0, 1)
                .float()
            )
            scores = torch.where(model.shared_channel[:, None], shared, own)
            view = (
                encoded["view_index"][:, None]
                .expand_as(encoded["visual_valid"])
                .flatten()
            )
            if model.view_label_bias is not None:
                scores = scores + model.view_label_bias.float()[view].transpose(0, 1)
            prior = (
                torch.where(
                    model.shared_channel[model.target_channel][:, None, None],
                    model.position_priors(encoded["position"], encoded["routable"]),
                    model.position_priors(
                        encoded["fitted_position"], encoded["routable"]
                    ),
                )
                .flatten(1)
                .float()
            )
            routing = model.routing_weights(encoded).flatten(1).float()
            weights = torch.cat((routing, torch.ones_like(routing[:1])))
            routed = (valid.float()[None] @ weights.T >= 1)[0, model.target_route]
            log_weights = (
                weights.clamp_min(ROUTING_MINIMUM_WEIGHT).log()
                / model.routing_temperature
            )
            attention = (
                (
                    scores[model.target_channel]
                    + prior
                    + torch.where(routed[:, None], log_weights[model.target_route], 0.0)
                )
                .masked_fill(~valid[None], -torch.inf)
                .softmax(-1)
            )
            position = encoded["fitted_position"].float().flatten(0, 1)
            frame = encoded["position"].float().flatten(0, 1)
            axis_a = position[:, 0]
            axis_b = position[:, 1]
            region = (
                torch.where(routed[:, None], weights[model.target_route] > 0.5, True)
                & valid[None]
            )
        chosen = attention.topk(TOP, dim=-1)
        store["index"].append(chosen.indices.cpu().numpy())
        store["weight"].append(chosen.values.cpu().numpy())
        store["tokens"].append(len(valid))
        store["region"].append(np.packbits(region.cpu().numpy(), axis=1))
        store["valid_count"].append(int(valid.sum()))
        store["study"].append(row)
        store["axis_a"].append(axis_a.cpu().numpy().astype(np.float32))
        store["axis_b"].append(axis_b.cpu().numpy().astype(np.float32))
        store["position"].append(position.cpu().numpy().astype(np.float32))
        store["frame"].append(frame.cpu().numpy().astype(np.float32))
        store["image_sum"].append(float(batch["images"].float().sum()))
        store["logit"].append(model.pool_encoded(encoded).float().cpu().numpy()[0])


def ragged(values):
    result = np.empty(len(values), dtype=object)
    for number, value in enumerate(values):
        result[number] = value
    return result


out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out,
    targets=np.array(targets),
    index=np.array(store["index"]),
    weight=np.array(store["weight"]),
    tokens=np.array(store["tokens"]),
    valid_count=np.array(store["valid_count"]),
    study=np.array(store["study"]),
    image_sum=np.array(store["image_sum"]),
    logit=np.array(store["logit"]),
    region=ragged(store["region"]),
    axis_a=ragged(store["axis_a"]),
    axis_b=ragged(store["axis_b"]),
    position=ragged(store["position"]),
    frame=ragged(store["frame"]),
    allow_pickle=True,
)
print("saved", out, len(store["study"]), "knees")
