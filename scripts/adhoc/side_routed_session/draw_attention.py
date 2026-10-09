import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

RUN = Path(sys.argv[1])
COUNT = int(sys.argv[2])
OUT = Path("outputs/analysis/s/side-routing-loop-20261006") / RUN.name[16:]
OUT.mkdir(parents=True, exist_ok=True)
config = load_config(RUN / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY).head(120)
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
selected = json.loads((RUN / "fold_1" / "selected_checkpoint.json").read_text())
device = torch.device("cuda")
model = load_checkpoint_model(config, targets, Path(selected["path"]), device, expected_fold=1)
split_dir = config.paths.data_root / "folds" / config.validation.split
labels = (
    pl.read_csv(split_dir / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split_dir / "targets.csv"), on=STUDY)
    .sort(STUDY)
    .head(120)
)
truth = np.stack([labels[name].to_numpy() >= 0.5 for name in targets], 1)
rows = np.argsort(-truth.sum(1), kind="stable")[:COUNT]
origin_y, origin_x, stride_y, stride_x = model.feature_mapping
series_names = [f"{kind.plane}{' fluid' if kind.fluid_sensitive else ''}" for kind in config.data.series_types]


def token_attention(encoded):
    valid = encoded["visual_valid"].flatten()
    own = model.attention(model.study_tokens(encoded)).transpose(0, 1).float()
    shared = model.study_tokens(encoded, shared=True)
    shared = model.attention(shared).transpose(0, 1).float()
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
    return attention.reshape(len(targets), *encoded["visual_valid"].shape)


for row in rows:
    batch = with_images(move_batch(collate_studies([dataset[int(row)]]), device))
    with torch.inference_mode():
        encoded = model.encode_batch(batch)
        attention = token_attention(encoded).cpu().numpy()
        probability = torch.sigmoid(model.pool_encoded(encoded).float())[0].cpu().numpy()
    images = batch["images"].flatten(0, 1)[batch["slice_mask"].flatten()][:, 1].float().cpu().numpy() / 255.0
    kinds = encoded["image_type"].cpu().numpy()
    positive = [number for number in range(len(targets)) if truth[row, number]]
    if not positive:
        continue
    grid = int(attention.shape[-1] ** 0.5)
    size = images.shape[-1]
    extent = (
        origin_x - stride_x / 2,
        origin_x + (grid - 0.5) * stride_x,
        origin_y + (grid - 0.5) * stride_y,
        origin_y - stride_y / 2,
    )
    figure, axes = plt.subplots(len(positive), 3, figsize=(10.5, 3.6 * len(positive)), squeeze=False)
    for line, number in enumerate(positive):
        per_slice = attention[number].sum(-1)
        order = np.argsort(-per_slice)[:3]
        peak = attention[number].max()
        for column, item in enumerate(order):
            panel = axes[line][column]
            panel.imshow(images[item], cmap="gray", vmin=0, vmax=1)
            panel.imshow(
                attention[number, item].reshape(grid, grid),
                cmap="inferno",
                alpha=0.5,
                vmin=0,
                vmax=peak,
                extent=extent,
                interpolation="nearest",
            )
            panel.set_xlim(0, size)
            panel.set_ylim(size, 0)
            panel.set_xticks([])
            panel.set_yticks([])
            panel.set_title(
                f"{targets[number]} (p {probability[number]:.2f}): {series_names[kinds[item]]}, "
                f"{100 * per_slice[item]:.0f}% of its attention",
                fontsize=8.5,
            )
    figure.suptitle(
        f"Validation knee {int(row)}: for each label that is positive in this knee, the three slices it attends to most,\n"
        "with its attention per token over the image the model sees (brighter is more).",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(OUT / f"attention-{int(row):03d}.png", dpi=80)
    plt.close(figure)
    print("saved", OUT / f"attention-{int(row):03d}.png", [targets[n] for n in positive])
