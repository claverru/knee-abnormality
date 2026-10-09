import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from knee_abnormality.config import DataConfig, load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import checkpoint_model, read_checkpoint
from knee_abnormality.inference.engine import move_batch

RUN = Path("outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors")
OUT = Path("outputs/analysis/r/routing-and-shifts-frames-20261007/knee-bend")
OUT.mkdir(parents=True, exist_ok=True)
BINS = np.arange(-60.0, 75.0, 5.0)
UPPER, LOWER = (30.0, 65.0), (-50.0, -15.0)


def wide_data(data: DataConfig) -> DataConfig:
    values = data.model_dump(mode="json")
    for kind in values["series_types"]:
        for view in kind["views"]:
            view["field_mm"] = values["reference_field_mm"]
            view["horizontal"] = {"anchor": "image", "offset_mm": 0.0}
            view["vertical"] = {"anchor": "image", "offset_mm": 0.0}
    return DataConfig.model_validate(values)


config = load_config(RUN / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
names = studies[STUDY].to_list()
index = build_series_index(config.paths.data_root, config.data, split, names)
wide = wide_data(config.data)
planes = [kind.plane for kind in wide.series_types]
gpu = torch.device("cuda")
selected = json.loads((RUN / "fold_1" / "selected_checkpoint.json").read_text())
model = checkpoint_model(read_checkpoint(Path(selected["path"])), config, targets, gpu).eval()
model.view_frame_rotation_fraction = 0.0
captured = {}
original = model.encode_batch


def encode(batch, view_angle=None):
    out = original(batch, view_angle)
    captured["encoded"] = out
    return out


model.encode_batch = encode


def loader(data):
    return DataLoader(
        KneeStudyDataset(studies, index, data, targets, None, 1),
        batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_studies,
    )


def slope(heights, values, weights, span):
    chosen = (heights >= span[0]) & (heights <= span[1]) & (weights > 200)
    if chosen.sum() < 4:
        return np.nan, np.nan
    fit = np.polyfit(heights[chosen], values[chosen], 1, w=np.sqrt(weights[chosen]))
    return float(np.degrees(np.arctan(fit[0]))), float(np.polyval(fit, 0.0))


rows = []
profiles = {"Sagittal": [], "Coronal": []}
for model_batch, wide_batch in zip(loader(config.data), loader(wide)):
    batch = with_images(move_batch(model_batch, gpu))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        model(batch)
    encoded = captured["encoded"]
    angles = model.rotation_angles(encoded["study_frame_rotation"][0]).cpu().numpy()
    shift = encoded["study_frame_shift"][0].float().cpu().numpy()
    cosine, sine = encoded["study_axis"][0].float().cpu().numpy()
    sign = float(1 - 2 * encoded["study_left_probability"][0].float().cpu())
    w = with_images(move_batch(wide_batch, torch.device("cpu")))
    mask = w["slice_mask"]
    centre = np.array([float(w[key][torch.isfinite(w[key])].mean()) if torch.isfinite(w[key]).any() else np.nan for key in ("knee_center_x_mm", "knee_center_y_mm", "knee_center_z_mm")])
    record = {"study": model_batch["study_uid"][0], "nod": angles[0], "roll": angles[1], "turn": angles[2], "shift_up": shift[2]}
    if not np.isfinite(centre).all():
        rows.append(record)
        continue
    size = w["images"].shape[-1]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    grid = np.stack((xs, ys, np.ones_like(xs)), -1)
    kinds = w["series_type_index"].numpy()
    for plane in ("Sagittal", "Coronal"):
        total = np.zeros(len(BINS) - 1)
        moment = np.zeros(len(BINS) - 1)
        for n in range(len(kinds)):
            if planes[kinds[n]] != plane:
                continue
            for t in np.flatnonzero(mask[n].numpy()):
                maps = w["patient_maps"][n, t, 1].float().numpy()
                if not np.isfinite(maps).all():
                    continue
                image = w["images"][n, t, 1].float().numpy()
                place = np.einsum("ij,hwj->hwi", maps, grid) - centre
                medial = sign * (place[..., 0] * cosine + place[..., 1] * sine)
                front = place[..., 0] * sine - place[..., 1] * cosine
                height = place[..., 2] - shift[2]
                inside = (image > 30) & ((np.abs(medial) < 30) if plane == "Sagittal" else (np.abs(front) < 25))
                value = front if plane == "Sagittal" else medial
                which = np.digitize(height[inside], BINS) - 1
                good = (which >= 0) & (which < len(total))
                total += np.bincount(which[good], minlength=len(total))
                moment += np.bincount(which[good], weights=value[inside][good], minlength=len(total))
        middle = (BINS[:-1] + BINS[1:]) / 2
        profile = np.where(total > 200, moment / np.maximum(total, 1), np.nan)
        profiles[plane].append(profile)
        key = "sag" if plane == "Sagittal" else "cor"
        record[f"{key}_upper"], _ = slope(middle, np.nan_to_num(profile), total, UPPER)
        record[f"{key}_lower"], _ = slope(middle, np.nan_to_num(profile), total, LOWER)
    rows.append(record)
table = pl.DataFrame(rows)
table = table.with_columns(
    (pl.col("sag_upper") - pl.col("sag_lower")).alias("sag_bend"),
    (pl.col("cor_upper") - pl.col("cor_lower")).alias("cor_bend"),
)
table.write_csv(OUT / "bend.csv")
np.savez(OUT / "profiles.npz", bins=BINS, sagittal=np.array(profiles["Sagittal"]), coronal=np.array(profiles["Coronal"]))
print("knees", len(table))
for name in ("sag_upper", "sag_lower", "sag_bend", "cor_upper", "cor_lower", "cor_bend", "nod", "roll"):
    values = table[name].drop_nulls().drop_nans().to_numpy()
    print(f"{name:10} n {len(values):4d} mean {values.mean():+6.1f} sd {values.std():5.1f} p5 {np.percentile(values, 5):+6.1f} median {np.median(values):+6.1f} p95 {np.percentile(values, 95):+6.1f}")
clean = table.drop_nulls().filter(pl.all_horizontal(pl.col(pl.Float64, pl.Float32).is_finite()))
print("complete knees", len(clean))
for a, b in (("nod", "sag_upper"), ("nod", "sag_lower"), ("nod", "sag_bend"), ("roll", "cor_upper"), ("roll", "cor_lower"), ("roll", "cor_bend"), ("sag_upper", "sag_lower"), ("cor_upper", "cor_lower")):
    print(f"correlation {a} ~ {b}: {np.corrcoef(clean[a].to_numpy(), clean[b].to_numpy())[0, 1]:+.2f}")
middle = (BINS[:-1] + BINS[1:]) / 2
for plane in ("Sagittal", "Coronal"):
    stack = np.array(profiles[plane])
    print(plane, "mean limb centre by height (mm):", {int(h): round(float(v), 1) for h, v in zip(middle, np.nanmean(stack, 0)) if np.isfinite(v)})
