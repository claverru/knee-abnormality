import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from knee_abnormality.config import DataConfig, load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.modeling.model import FRAME_SIZE_START_PITCHES

RUN = Path(sys.argv[1])
ROWS = [int(v) for v in sys.argv[2].split(",")]
OUT = Path("outputs/analysis/side-routing-loop-20261006") / RUN.name[16:]
OUT.mkdir(parents=True, exist_ok=True)
QUADRANTS = {
    (1, 1): (0.89, 0.47, 0.20),
    (1, 0): (0.80, 0.25, 0.33),
    (0, 1): (0.30, 0.69, 0.60),
    (0, 0): (0.22, 0.42, 0.69),
}
SLICES = (
    ("Sagittal", 0, 20.0, "sagittal, medial"),
    ("Sagittal", 0, -20.0, "sagittal, lateral"),
    ("Coronal", 1, -8.0, "coronal, behind the centre"),
    ("Coronal", 1, 22.0, "coronal, in front"),
    ("Axial", 2, 18.0, "axial, above the centre"),
    ("Axial", 2, -14.0, "axial, at the joint line"),
)


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
studies = studies.sort(STUDY).head(60)
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
data = wide_data(config.data)
dataset = KneeStudyDataset(studies, index, data, targets, None, 1)
model_dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
selected = json.loads((RUN / "fold_1" / "selected_checkpoint.json").read_text())
gpu = torch.device("cuda")
model = load_checkpoint_model(config, targets, Path(selected["path"]), gpu, expected_fold=1)
planes = [kind.plane for kind in data.series_types]
fluid = [kind.fluid_sensitive for kind in data.series_types]
for row in ROWS:
    batch = with_images(move_batch(collate_studies([dataset[row]]), torch.device("cpu")))
    mask = batch["slice_mask"]
    series_of = mask.nonzero()[:, 0]
    images = batch["images"].flatten(0, 1)[mask.flatten()][:, 1].float().numpy()
    maps = batch["patient_maps"].flatten(0, 1)[mask.flatten()][:, 1].float().numpy()
    kinds = batch["series_type_index"][series_of].numpy()
    centre = np.stack(
        [batch[key][series_of].numpy() for key in ("knee_center_x_mm", "knee_center_y_mm", "knee_center_z_mm")], -1
    )
    left = bool(batch["knee_side_left"][0])
    sign = -1.0 if left else 1.0
    directions = []
    for item, kind in enumerate(kinds):
        if planes[kind] != "Sagittal" and np.isfinite(maps[item]).all():
            vector = maps[item][:2, 0]
            vector = -vector if vector[0] < 0 else vector
            directions.append(vector / np.linalg.norm(vector))
    axis = np.mean(directions, 0)
    axis = axis / np.linalg.norm(axis)
    with torch.inference_mode():
        encoded = model.encode_batch(with_images(move_batch(collate_studies([model_dataset[row]]), gpu)))
        shift = encoded["study_frame_shift"][0].float().cpu().numpy()
        rotation = encoded["study_frame_rotation"][0].float().cpu().numpy()
        angles = model.rotation_angles(encoded["study_frame_rotation"][0]).cpu().numpy()
        sizes = encoded["study_frame_sizes"][0].float().cpu().numpy()
        pitch = float(encoded["token_pitch"][encoded["routable"]].float().mean())
    start = np.array(FRAME_SIZE_START_PITCHES) * pitch
    size = images.shape[-1]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    grid = np.stack((xs, ys, np.ones_like(xs)), -1)
    middle = np.array([size / 2, size / 2, 1.0], dtype=np.float32)

    def knee(points, item):
        offset = points - centre[item]
        u = offset[..., 0] * axis[0] + offset[..., 1] * axis[1]
        a = offset[..., 0] * axis[1] - offset[..., 1] * axis[0]
        return np.stack((sign * u, a, offset[..., 2]), -1)

    figure, axes = plt.subplots(2, len(SLICES), figsize=(3.5 * len(SLICES), 8.6))
    for column, (plane, component, wanted, title) in enumerate(SLICES):
        members = [
            i for i, kind in enumerate(kinds)
            if planes[kind] == plane and not fluid[kind] and np.isfinite(maps[i]).all()
        ] or [i for i, kind in enumerate(kinds) if planes[kind] == plane and np.isfinite(maps[i]).all()]
        if not members:
            for line in range(2):
                axes[line][column].axis("off")
            continue
        item = min(members, key=lambda i: abs(knee(maps[i] @ middle, i)[component] - wanted))
        raw = knee(np.einsum("ij,hwj->hwi", maps[item], grid), item)
        image = images[item] / 255.0
        for line, (frame, moved, cuts, name) in enumerate(
            ((np.eye(3), np.zeros(3), start, "Start"), (rotation, shift, sizes, "Inferred"))
        ):
            position = (raw - moved) @ frame.T
            m, f, h = position[..., 0], position[..., 1], position[..., 2]
            top, bottom, radius = cuts[:3]
            ring = cuts[3] if len(cuts) > 3 else None
            colour = np.zeros((*m.shape, 3))
            for (medial, front), shade in QUADRANTS.items():
                colour[((m > 0) == bool(medial)) & ((f > 0) == bool(front))] = shade
            distance = np.hypot(m, f)
            colour[distance < radius] = (0.93, 0.93, 0.93)
            colour[h > top] = 0.55 * colour[h > top] + 0.45
            colour[h < bottom] = 0.5 * colour[h < bottom]
            panel = axes[line][column]
            panel.imshow(np.clip(0.58 * image[..., None] + 0.42 * colour, 0, 1))
            for values, level, shade in (
                (m, 0.0, "white"),
                (f, 0.0, "yellow"),
                (h, top, "cyan"),
                (h, bottom, "cyan"),
                (distance, radius, "magenta"),
                *(((distance, ring, "magenta"),) if ring is not None else ()),
                *(
                    (
                        (f * np.cos(np.radians(a)) - m * np.sin(np.radians(a)), 0.0, "white")
                        for a in (22.5, 67.5, 112.5, 157.5)
                    )
                    if ring is not None
                    else ()
                ),
            ):
                if values.min() < level < values.max():
                    panel.contour(values, [level], colors=shade, linewidths=1.2)
            panel.set_title(f"{name}: {title}", fontsize=8.5)
            panel.set_xticks([])
            panel.set_yticks([])
    figure.suptitle(
        f"Study row {row} ({'left' if left else 'right'} knee). Top: the frame at the start of training. Bottom: the frame inferred for this knee.\n"
        f"Centre shift (medial, front, up) {shift.round(1).tolist()} mm; nod, roll, turn {angles.round(1).tolist()} deg; "
        f"upper cut {sizes[0]:+.0f} mm, lower cut {sizes[1]:+.0f} mm, core radius {sizes[2]:.0f} mm.\n"
        "White midline, yellow front cut, cyan the two height cuts (femur band lighter, tibia band darker), magenta the core.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.subplots_adjust(hspace=0.14)
    figure.savefig(OUT / f"regions-{row:03d}.png", dpi=85)
    plt.close(figure)
    print("saved", OUT / f"regions-{row:03d}.png")
