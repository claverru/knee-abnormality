import json
import math
import sys
from pathlib import Path

RUN = Path("outputs/training/20261007-081431-side-routed-crop-offsets-3mm").resolve()
WEIGHTS = Path(
    "outputs/autoresearch/routing-and-shifts/experiments/height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402/training"
)
MULTIPLIER = int(sys.argv[1])
LIMIT = 15.0
MAX_ANGLE = math.radians(15.0)

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import checkpoint_model, read_checkpoint
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.modeling.model import crop_rectangles

OUT = Path("outputs/analysis/r/routing-and-shifts-frames-20261007/frame-crops")
OUT.mkdir(parents=True, exist_ok=True)
config = load_config(RUN / "effective_config.json")
config = config.model_copy(
    update={"data": config.data.model_copy(update={"view_shift_limit_mm": LIMIT})}
)
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
names = studies[STUDY].to_list()
index = build_series_index(config.paths.data_root, config.data, split, names)
selected = json.loads((WEIGHTS / "fold_1" / "selected_checkpoint.json").read_text())
gpu = torch.device("cuda")
model = checkpoint_model(
    read_checkpoint(Path(selected["path"])), config, targets, gpu
).eval()
windows = model.view_windows
windows.radius = 0.0
planes = [kind.plane for kind in config.data.series_types]
original_forward = windows.forward
original_encode = model.encode_batch
state = {"move": None, "encoded": None, "stats": []}
SIZE = config.data.image_size


def encode(batch):
    out = original_encode(batch)
    state["encoded"] = out
    return out


def forward(batch, middle_only=False):
    if state["move"] is None:
        return original_forward(batch, middle_only)
    shift, angle = state["move"]
    with torch.autocast(device_type="cuda", enabled=False):
        canvas = batch["images"]
        maps = batch["patient_maps"].float()
        margin = windows.margin[batch["view_index"]][:, None]
        movable = margin > 0
        shift = torch.where(movable[..., None], shift, 0.0)
        angle = torch.where(movable, angle, 0.0)
        limit = (0.9 * margin)[..., None]
        shift = torch.maximum(torch.minimum(shift, limit), -limit)
        cosine, sine = angle.cos(), angle.sin()
        half = (SIZE - 1) / 2
        centre = margin[..., None] + half + shift
        affine = torch.zeros((*angle.shape, 3, 3), device=canvas.device)
        affine[..., 0, 0], affine[..., 0, 1] = cosine, -sine
        affine[..., 1, 0], affine[..., 1, 1] = sine, cosine
        affine[..., 0, 2] = centre[..., 0] - half * (cosine - sine)
        affine[..., 1, 2] = centre[..., 1] - half * (sine + cosine)
        affine[..., 2, 2] = 1.0
        lines = torch.arange(SIZE, device=canvas.device, dtype=torch.float32)
        rows, columns = torch.meshgrid(lines, lines, indexing="ij")
        points = torch.stack((columns, rows, torch.ones_like(rows)), -1)
        placed = torch.einsum("ntij,hwj->nthwi", affine[..., :2, :], points)
        height, width = canvas.shape[-2:]
        grid = torch.stack(
            (
                (2 * placed[..., 0] + 1) / width - 1,
                (2 * placed[..., 1] + 1) / height - 1,
            ),
            -1,
        ).flatten(0, 1)
        pictures = canvas[:, :, 1:2] if middle_only else canvas
        crops = F.grid_sample(
            pictures.flatten(0, 1).float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).view(*pictures.shape[:3], SIZE, SIZE)
        origin = (margin[..., None] + shift).flip(-1)
        return {
            **batch,
            "images": crops.expand(-1, -1, 3, -1, -1),
            "valid_rects": crop_rectangles(
                batch["valid_rects"], origin[:, :, None], SIZE
            ),
            "patient_maps": maps @ affine[:, :, None],
            "applied_view_shift_mm": torch.zeros_like(shift),
        }


windows.forward = forward
model.encode_batch = encode


def knee_frame(batch):
    state["move"] = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        prediction = torch.sigmoid(model(batch).float()).cpu().numpy()
    encoded = state["encoded"]
    return prediction, encoded


def patient(vector, sign, axis):
    cosine, sine = axis[:, 0], axis[:, 1]
    return torch.stack(
        (
            sign * vector[:, 0] * cosine + vector[:, 1] * sine,
            sign * vector[:, 0] * sine - vector[:, 1] * cosine,
            vector[:, 2],
        ),
        -1,
    )


def moves(batch, encoded, mean_shift, maps=None):
    study = batch["series_study_index"]
    sign = (1 - 2 * encoded["study_left_probability"].float())[study]
    axis = encoded["study_axis"].float()[study]
    rotation = encoded["study_frame_rotation"].float()[study]
    deviation = (encoded["study_frame_shift"].float() - mean_shift)[study]
    maps = batch["patient_maps"].float() if maps is None else maps
    linear = maps[:, :, 1, :, :2]
    linear = torch.where(torch.isfinite(linear), linear, 0.0)
    length = linear.square().sum(-2).clamp_min(1e-12)
    usable = (length > 1e-8).all(-1)
    displacement = patient(deviation, sign, axis)
    shift = torch.einsum("ntpa,np->nta", linear, displacement) / length
    kinds = batch["series_type_index"].tolist()
    coronal = torch.tensor([planes[k] == "Coronal" for k in kinds], device=gpu)
    chosen = torch.where(coronal[:, None], rotation[:, 0], rotation[:, 2])
    direction = torch.einsum(
        "ntpa,np->nta",
        linear / length.sqrt()[..., None, :],
        patient(chosen, sign, axis),
    )
    across, down = direction[..., 0], direction[..., 1]
    angle = torch.where(
        coronal[:, None],
        torch.atan(down / torch.where(across.abs() < 1e-6, 1e-6, across)),
        torch.atan(-across / torch.where(down.abs() < 1e-6, 1e-6, down)),
    )
    angle = torch.where(usable, angle, 0.0).clamp(-MAX_ANGLE, MAX_ANGLE)
    shift = torch.where(usable[..., None], shift, 0.0)
    return shift, angle


def loader_for(multiplier):
    dataset = KneeStudyDataset(studies, index, config.data, targets, None, multiplier)
    return DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=8, collate_fn=collate_studies
    )


shifts = []
for batch in loader_for(1):
    batch = with_images(move_batch(batch, gpu))
    _, encoded = knee_frame(batch)
    shifts.append(encoded["study_frame_shift"].float().cpu())
shifts = torch.cat(shifts)
mean_shift = shifts.mean(0).to(gpu)
print(
    "frame centre over the fold, mean (medial, front, up) mm",
    [round(v, 1) for v in shifts.mean(0).tolist()],
    "sd",
    [round(v, 1) for v in shifts.std(0).tolist()],
)
PASSES = ("anchor", "rotated", "wrong way", "half", "coronal only", "other plane only")
outputs = {name: [] for name in PASSES}
stats = []
for batch in loader_for(MULTIPLIER):
    batch = with_images(move_batch(batch, gpu))
    prediction, encoded = knee_frame(batch)
    outputs["anchor"].append(prediction)
    shift, angle = moves(batch, encoded, mean_shift)
    pitch = windows.pitch[batch["view_index"]][:, None]
    coronal_rows = torch.tensor(
        [planes[k] == "Coronal" for k in batch["series_type_index"].tolist()],
        device=gpu,
    ).float()[:, None]
    movable = (windows.margin[batch["view_index"]] > 0)[:, None] & batch["slice_mask"]
    swapped = angle.flip(0) if len(batch["series_presence"]) > 1 else -angle
    other = batch["series_study_index"]
    order = torch.argsort(other, stable=True)
    for name, move in (
        ("rotated", (torch.zeros_like(shift), angle)),
        ("wrong way", (torch.zeros_like(shift), -angle)),
        ("half", (torch.zeros_like(shift), angle / 2)),
        ("coronal only", (torch.zeros_like(shift), angle * coronal_rows)),
        ("other plane only", (torch.zeros_like(shift), angle * (1 - coronal_rows))),
    ):
        state["move"] = move
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs[name].append(torch.sigmoid(model(batch).float()).cpu().numpy())
    kinds = batch["series_type_index"].tolist()
    coronal = torch.tensor([planes[k] == "Coronal" for k in kinds], device=gpu)[
        :, None
    ].expand_as(movable)
    after = moves(
        batch, encoded, mean_shift, windows.forward(batch)["patient_maps"].float()
    )[1]
    for mask, name in (
        (movable & coronal, "coronal"),
        (movable & ~coronal, "other detail plane"),
    ):
        if mask.any():
            stats.append(
                (
                    name,
                    (shift.norm(dim=-1) * pitch)[mask].cpu().numpy(),
                    torch.rad2deg(angle.abs())[mask].cpu().numpy(),
                    torch.rad2deg(after.abs())[mask].cpu().numpy(),
                    (
                        (
                            shift.abs()
                            > 0.9 * windows.margin[batch["view_index"]][:, None, None]
                        ).any(-1)
                    )[mask]
                    .cpu()
                    .numpy(),
                )
            )
outputs = {name: np.concatenate(values) for name, values in outputs.items()}
labels = list(targets)
truth = studies.select(STUDY).join(
    pl.read_csv(Path("data/folds/all_5fold_scanner_v2") / "targets.csv"),
    on=STUDY,
    how="left",
)
np.savez(
    OUT / f"frame-rotation-controls-x{MULTIPLIER}.npz",
    names=np.array(names),
    labels=np.array(labels),
    **{k.replace(" ", "_"): v for k, v in outputs.items()},
)
for name in ("coronal", "other detail plane"):
    parts = [s for s in stats if s[0] == name]
    move, turn, left, clamped = (
        np.concatenate([p[k] for p in parts]) for k in (1, 2, 3, 4)
    )
    print(
        f"{name}: shift mm median {np.median(move):.1f} p90 {np.percentile(move, 90):.1f}; rotation deg median {np.median(turn):.1f} p90 {np.percentile(turn, 90):.1f}; residual angle after the re-cut median {np.median(left):.2f}; clamped {clamped.mean():.1%}"
    )
print(
    f"{'label':18}{'anchor':>8}{'rotated':>10}{'wrong way':>10}{'half':>10}{'coronal':>10}{'other pl':>10}"
)
rows = []
for i, label in enumerate(labels):
    y = truth[label].to_numpy() >= 0.5
    values = [roc_auc_score(y, outputs[name][:, i]) for name in PASSES]
    rows.append(values)
    print(
        f"{label:18}{values[0]:8.4f}"
        + "".join(f"{v - values[0]:+10.4f}" for v in values[1:])
    )
values = np.mean(rows, 0)
print(
    f"{'macro':18}{values[0]:8.4f}"
    + "".join(f"{v - values[0]:+10.4f}" for v in values[1:])
)
print("macro absolute", [round(float(v), 6) for v in values])
