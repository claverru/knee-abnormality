import json
import sys
from pathlib import Path

experiment = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(experiment / "source"), str(experiment / "source" / "src")]

import numpy as np
import polars as pl
import torch

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.report_attributes import load_report_attribute_targets
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.modeling.model import ROUTING_MINIMUM_WEIGHT
from knee_abnormality.training.report_attributes import ReportAttributeHeads

TOP = 10
NEGATIVE = {
    "not_mentioned", "normal", "none", "no_tear", "no_loss", "no_damage", "absent", "intact",
    "no_injury", "no_cyst", "no", "no_osteoarthritis", "other_cyst_only", "only_elsewhere",
    "suprapatellar_only", "soft_tissue_edema_only",
}
run = experiment / "training"
out = Path(sys.argv[2])
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
device = torch.device("cuda")
checkpoint = Path(json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())["path"])
model = load_checkpoint_model(config, targets, checkpoint, device, expected_fold=1)
attributes = load_report_attribute_targets(config.paths.report_attribute_manifest, config.paths.data_root)
heads = ReportAttributeHeads(attributes)
state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
heads.load_state_dict(
    {key.removeprefix("report_attribute_heads."): value for key, value in state.items() if key.startswith("report_attribute_heads.")},
    strict=False,
)
heads = heads.to(device).eval()
questions = [question for question in attributes.questions if question.id in heads.question_ids]
assert [question.id for question in questions] == list(heads.question_ids)
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
names = [*targets, *(f"report:{question.id}" for question in questions)]
rows = {key: [] for key in ("fitted", "frame", "weight", "plane", "truth", "abnormal", "pitch", "study")}
background = []
with torch.inference_mode():
    for row in range(studies.height):
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        encoded = model.encode_batch(batch)
        if not encoded["routable"].all():
            continue
        valid = encoded["visual_valid"].flatten()
        tokens = model.study_tokens(encoded)
        own = model.attention(tokens).transpose(0, 1).float()
        shared = model.attention(model.study_tokens(encoded, shared=True)).transpose(0, 1).float()
        scores = torch.where(model.shared_channel[:, None], shared, own)
        view = encoded["view_index"][:, None].expand_as(encoded["visual_valid"]).flatten()
        if model.view_label_bias is not None:
            scores = scores + model.view_label_bias.float()[view].transpose(0, 1)
        prior = model.position_priors(encoded["position"], encoded["routable"]).flatten(1).float()
        routing = model.routing_weights(encoded).flatten(1).float()
        weights = torch.cat((routing, torch.ones_like(routing[:1])))
        routed = (valid.float()[None] @ weights.T >= 1)[0, model.target_route]
        log_weights = weights.clamp_min(ROUTING_MINIMUM_WEIGHT).log() / model.routing_temperature
        label_attention = (
            scores[model.target_channel]
            + prior
            + torch.where(routed[:, None], log_weights[model.target_route], 0.0)
        ).masked_fill(~valid[None], -torch.inf).softmax(-1)
        report_attention = (
            heads.attention(tokens.float()).transpose(0, 1).masked_fill(~valid[None], -torch.inf).softmax(-1)
        )
        attention = torch.cat((label_attention, report_attention))
        sign = (1 - 2 * encoded["left_probability"])[:, None]
        fitted = torch.stack(
            (sign * encoded["knee_x"], encoded["anterior_axis"], encoded["superior_axis"]), -1
        ).float().flatten(0, 1)
        frame = encoded["position"].float().flatten(0, 1)
        plane = encoded["image_plane"][:, None].expand_as(encoded["visual_valid"]).flatten()
        chosen = attention.topk(TOP, dim=-1)
        rows["fitted"].append(fitted[chosen.indices].cpu().numpy())
        rows["frame"].append(frame[chosen.indices].cpu().numpy())
        rows["weight"].append(chosen.values.cpu().numpy())
        rows["plane"].append(plane[chosen.indices].cpu().numpy())
        rows["truth"].append(np.array([labels[name][row] >= 0.5 for name in targets]))
        table, _, observed = heads.targets(batch["study_uid"])
        abnormal = []
        offset = 0
        for question in questions:
            values = table[0, offset : offset + len(question.options)]
            offset += len(question.options)
            keep = torch.tensor([option not in NEGATIVE for option in question.options], device=device)
            abnormal.append(float(values[keep].sum()) if bool(observed[0]) else float("nan"))
        rows["abnormal"].append(np.array(abnormal))
        rows["pitch"].append(float(encoded["token_pitch"].float().mean()))
        rows["study"].append(row)
        background.append(fitted[valid][:: 6].cpu().numpy())
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out,
    names=np.array(names),
    labels=np.array(targets),
    questions=np.array([question.id for question in questions]),
    background=np.concatenate(background),
    **{key: np.array(value) for key, value in rows.items()},
)
print("saved", out, len(rows["study"]), "knees", len(names), "heads")
