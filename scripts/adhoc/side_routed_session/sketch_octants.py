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

OUT = Path(sys.argv[1])
ROWS = [int(v) for v in sys.argv[2].split(",")]
CONFIGS = [
    tuple(float(v) for v in item.split(":")) for item in sys.argv[3].split(",")
]
OUT.mkdir(parents=True, exist_ok=True)


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
names = [
    f"{kind.plane} {'fluid' if kind.fluid_sensitive else 'T1/PD'}"
    for kind in data.series_types
]
planes = [kind.plane for kind in data.series_types]
device = torch.device("cpu")
for row in ROWS:
    batch = with_images(move_batch(collate_studies([dataset[row]]), device))
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
    rows_xy = []
    for item, kind in enumerate(kinds):
        if planes[kind] != "Sagittal" and np.isfinite(maps[item]).all():
            vector = maps[item][:2, 0]
            vector = -vector if vector[0] < 0 else vector
            rows_xy.append(vector / np.linalg.norm(vector))
    axis = np.mean(rows_xy, 0) if rows_xy else np.array([1.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    present = sorted(set(kinds.tolist()))
    wanted = [
        kind
        for plane in ("Sagittal", "Coronal", "Axial")
        for kind in present
        if planes[kind] == plane
    ][::2][:3] or present[:3]
    size = images.shape[-1]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    figure, axes = plt.subplots(
        len(CONFIGS), len(wanted), figsize=(4.2 * len(wanted), 4.2 * len(CONFIGS)), squeeze=False
    )
    for line, (lateral, front, up, tilt) in enumerate(CONFIGS):
        angle = np.radians(tilt)
        for column, kind in enumerate(wanted):
            members = np.nonzero(kinds == kind)[0]
            middle = np.array([size / 2, size / 2, 1.0], dtype=np.float32)
            offsets = np.stack(
                [maps[member] @ middle - centre[member] for member in members]
            )
            distance = {
                "Sagittal": np.abs(offsets[:, 0] * axis[0] + offsets[:, 1] * axis[1]),
                "Coronal": np.abs(offsets[:, 0] * axis[1] - offsets[:, 1] * axis[0]),
                "Axial": np.abs(offsets[:, 2] - 15.0),
            }[planes[kind]]
            item = members[int(distance.argmin())]
            position = np.einsum(
                "ij,hwj->hwi", maps[item], np.stack((xs, ys, np.ones_like(xs)), -1)
            ) - centre[item]
            u = position[..., 0] * axis[0] + position[..., 1] * axis[1]
            a = position[..., 0] * axis[1] - position[..., 1] * axis[0]
            z = position[..., 2]
            m = sign * u + lateral
            a0, z0 = a - front, z - up
            f = a0 * np.cos(angle) + z0 * np.sin(angle)
            h = -a0 * np.sin(angle) + z0 * np.cos(angle)
            pf = (f > 0) & (h > 0)
            overlay = np.zeros((size, size, 4), dtype=np.float32)
            overlay[pf] = (1.0, 0.85, 0.0, 0.38)
            overlay[~pf & (m > 0)] = (0.9, 0.1, 0.1, 0.22)
            overlay[~pf & (m <= 0)] = (0.1, 0.3, 0.9, 0.22)
            panel = axes[line][column]
            panel.imshow(images[item], cmap="gray", vmin=0, vmax=255)
            panel.imshow(overlay)
            for values, colour in ((m, "white"), (f, "yellow"), (h, "cyan")):
                if values.min() < 0 < values.max():
                    panel.contour(values, [0.0], colors=colour, linewidths=1.2)
            origin = np.hypot(np.hypot(m, f), h)
            if origin.min() < 4:
                y0, x0 = np.unravel_index(origin.argmin(), origin.shape)
                panel.plot(x0, y0, "o", color="lime", markersize=5)
            panel.set_title(
                f"{names[kind]} | lateral {lateral:+.0f}, front {front:+.0f}, up {up:+.0f} mm, tilt {tilt:.0f} deg",
                fontsize=8,
            )
            panel.set_xticks([])
            panel.set_yticks([])
    figure.suptitle(
        f"study row {row} ({'left' if left else 'right'} knee): yellow = front and up (PF OA), "
        "red = rest of medial side, blue = rest of lateral side; lines: white midline, "
        "yellow front cut, cyan height cut",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(OUT / f"octants-{row:03d}.png", dpi=90)
    plt.close(figure)
    print(row, "left" if left else "right", [names[k] for k in wanted])
