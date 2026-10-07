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
from knee_abnormality.inference.engine import move_batch

OUT = Path("outputs/analysis/octant-sketches-20261006")
ROWS = [int(v) for v in sys.argv[1].split(",")]
CORE_MM, PERIPHERY_MM, JOINT_MM, BAND_MM = 14.0, 38.0, -14.0, 8.0
QUADRANTS = {
    (1, 1): ("medial front", (0.89, 0.47, 0.20)),
    (1, 0): ("medial back", (0.80, 0.25, 0.33)),
    (0, 1): ("lateral front", (0.30, 0.69, 0.60)),
    (0, 0): ("lateral back", (0.22, 0.42, 0.69)),
}
SECTORS = (
    ("medial", (0.84, 0.19, 0.15)),
    ("anteromedial", (0.96, 0.55, 0.20)),
    ("anterior", (0.95, 0.80, 0.25)),
    ("anterolateral", (0.55, 0.78, 0.35)),
    ("lateral", (0.20, 0.63, 0.55)),
    ("posterolateral", (0.20, 0.47, 0.75)),
    ("posterior", (0.42, 0.32, 0.70)),
    ("posteromedial", (0.72, 0.30, 0.62)),
)
SLICES = (
    ("Axial", 2, JOINT_MM, "Axial, at the joint line"),
    ("Coronal", 1, 0.0, "Coronal, through the centre"),
    ("Sagittal", 0, 22.0, "Sagittal, medial side"),
)


def wide_data(data: DataConfig) -> DataConfig:
    values = data.model_dump(mode="json")
    for kind in values["series_types"]:
        for view in kind["views"]:
            view["field_mm"] = values["reference_field_mm"]
            view["horizontal"] = {"anchor": "image", "offset_mm": 0.0}
            view["vertical"] = {"anchor": "image", "offset_mm": 0.0}
    return DataConfig.model_validate(values)


config = load_config(Path("configs/training.yaml"))
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY).head(60)
index = build_series_index(
    config.paths.data_root, config.data, split, studies[STUDY].to_list()
)
data = wide_data(config.data)
dataset = KneeStudyDataset(studies, index, data, targets, None, 1)
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
        [
            batch[key][series_of].numpy()
            for key in ("knee_center_x_mm", "knee_center_y_mm", "knee_center_z_mm")
        ],
        -1,
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
    size = images.shape[-1]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    grid = np.stack((xs, ys, np.ones_like(xs)), -1)
    middle = np.array([size / 2, size / 2, 1.0], dtype=np.float32)

    def knee(points, item):
        offset = points - centre[item]
        u = offset[..., 0] * axis[0] + offset[..., 1] * axis[1]
        a = offset[..., 0] * axis[1] - offset[..., 1] * axis[0]
        return np.stack((sign * u, a, offset[..., 2]), -1)

    figure, axes = plt.subplots(2, len(SLICES), figsize=(5.2 * len(SLICES), 11.8))
    for column, (plane, component, wanted, title) in enumerate(SLICES):
        members = [
            i
            for i, kind in enumerate(kinds)
            if planes[kind] == plane and not fluid[kind] and np.isfinite(maps[i]).all()
        ] or [
            i
            for i, kind in enumerate(kinds)
            if planes[kind] == plane and np.isfinite(maps[i]).all()
        ]
        item = min(
            members, key=lambda i: abs(knee(maps[i] @ middle, i)[component] - wanted)
        )
        position = knee(np.einsum("ij,hwj->hwi", maps[item], grid), item)
        m, f, h = position[..., 0], position[..., 1], position[..., 2]
        image = images[item] / 255.0
        radius = np.hypot(m, f)
        sector = np.round(np.degrees(np.arctan2(f, m)) / 45.0).astype(int) % 8
        here = float(knee(maps[item] @ middle, item)[component])

        def label(panel, text, at_m, at_f, at_h):
            distance = (m - at_m) ** 2 + (f - at_f) ** 2 + (h - at_h) ** 2
            y, x = np.unravel_index(np.argmin(distance), distance.shape)
            if distance[y, x] < 60 and 16 < x < size - 16 and 10 < y < size - 10:
                panel.text(
                    x,
                    y,
                    text,
                    color="white",
                    fontsize=8.5,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 1.5, "edgecolor": "none"},
                )

        panel = axes[0][column]
        colour = np.zeros((*m.shape, 3))
        for (medial, front), (_, shade) in QUADRANTS.items():
            colour[((m > 0) == bool(medial)) & ((f > 0) == bool(front))] = shade
        panel.imshow(np.clip(0.55 * image[..., None] + 0.45 * colour, 0, 1))
        for values, shade in ((m, "white"), (f, "yellow"), (h, "cyan")):
            if values.min() < 0 < values.max():
                panel.contour(values, [0.0], colors=shade, linewidths=1.6)
        for (medial, front), (name, _) in QUADRANTS.items():
            for up in (1, -1):
                at = [
                    28.0 * (1 if medial else -1),
                    28.0 * (1 if front else -1),
                    22.0 * up,
                ]
                if abs(at[component] - here) > 40.0 or (at[component] > 0) != (here > 0):
                    continue
                at[component] = here
                label(panel, f"{name}\n{'up' if up > 0 else 'down'}", *at)
        panel.set_title(f"Octants. {title}", fontsize=11)

        panel = axes[1][column]
        colour = np.array([shade for _, shade in SECTORS])[sector]
        colour[radius < CORE_MM] = (0.92, 0.92, 0.92)
        colour[radius > PERIPHERY_MM] = colour[radius > PERIPHERY_MM] * 0.78
        panel.imshow(np.clip(0.55 * image[..., None] + 0.45 * colour, 0, 1))
        for level in (CORE_MM, PERIPHERY_MM):
            if radius.min() < level < radius.max():
                panel.contour(radius, [level], colors="white", linewidths=1.4)
        for level in (JOINT_MM - BAND_MM, JOINT_MM + BAND_MM):
            if h.min() < level < h.max():
                panel.contour(h, [level], colors="cyan", linewidths=1.6)
        if component == 2:
            for number, (name, _) in enumerate(SECTORS):
                angle = np.radians(45.0 * number)
                label(panel, name, 25.0 * np.cos(angle), 25.0 * np.sin(angle), here)
            label(panel, "core", 0.0, 0.0, here)
        else:
            for at_h, band in (
                (JOINT_MM + 28.0, "femur"),
                (JOINT_MM, "joint line"),
                (JOINT_MM - 26.0, "tibia"),
            ):
                for offset in (-25.0, 25.0):
                    at = [offset, offset, at_h]
                    at[component] = here
                    angle = np.degrees(np.arctan2(at[1], at[0]))
                    name = SECTORS[int(np.round(angle / 45.0)) % 8][0]
                    label(panel, f"{name}\n{band}", *at)
        panel.set_title(f"Cylindrical regions. {title}", fontsize=11)
        for line in range(2):
            axes[line][column].set_xticks([])
            axes[line][column].set_yticks([])
    figure.suptitle(
        f"Study row {row} ({'left' if left else 'right'} knee), the same three slices. "
        "Top: the eight octants (white midline, yellow front cut, cyan height cut).\n"
        "Bottom: eight 45° sectors around the leg axis, a core (pale), a periphery (darker, outside the outer white ring) "
        "and three height bands (femur, joint line, tibia; cyan lines).\nBand sizes in this sketch are illustrative.",
        fontsize=11.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.subplots_adjust(hspace=0.09)
    figure.savefig(OUT / f"octants-vs-cylinder-{row:03d}.png", dpi=80)
    plt.close(figure)
    print("saved", OUT / f"octants-vs-cylinder-{row:03d}.png")
