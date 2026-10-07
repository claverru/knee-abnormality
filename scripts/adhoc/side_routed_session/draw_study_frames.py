import math
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
import json

OUT = Path("outputs/analysis/octant-sketches-20261006")
OUT.mkdir(parents=True, exist_ok=True)
ROWS = [int(v) for v in sys.argv[1].split(",")]
RUN = Path(sys.argv[2])
WIDTHS = {"medial": 4.72, "front": 5.55, "up": 5.14}
COLOURS = {
    "Medial Meniscus": (0.82, 0.29, 0.36),
    "Lateral Meniscus": (0.23, 0.49, 0.65),
    "Medial OA": (0.88, 0.48, 0.37),
    "Lateral OA": (0.36, 0.56, 0.73),
    "PF OA": (0.95, 0.76, 0.31),
}
SLICES = (
    ("Sagittal", 0, 20.0, "sagittal, 20 mm medial"),
    ("Sagittal", 0, -20.0, "sagittal, 20 mm lateral"),
    ("Coronal", 1, -10.0, "coronal, 10 mm behind the centre"),
    ("Coronal", 1, 25.0, "coronal, 25 mm in front"),
    ("Axial", 2, 20.0, "axial, 20 mm above the centre"),
    ("Axial", 2, -15.0, "axial, 15 mm below"),
)


def rotation(logit: torch.Tensor) -> np.ndarray:
    angles = (math.pi / 2) * torch.tanh(logit.float())
    zero = angles.new_zeros(())
    return torch.linalg.matrix_exp(
        torch.stack(
            (
                torch.stack((zero, -angles[2], angles[1])),
                torch.stack((angles[2], zero, -angles[0])),
                torch.stack((-angles[1], angles[0], zero)),
            )
        )
    ).numpy()


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


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
model_dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
selected = json.loads((RUN / "fold_1" / "selected_checkpoint.json").read_text())
gpu = torch.device("cuda")
model = load_checkpoint_model(config, targets, Path(selected["path"]), gpu, expected_fold=1)
planes = [kind.plane for kind in data.series_types]
fluid = [kind.fluid_sensitive for kind in data.series_types]
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
    directions = []
    for item, kind in enumerate(kinds):
        if planes[kind] != "Sagittal" and np.isfinite(maps[item]).all():
            vector = maps[item][:2, 0]
            vector = -vector if vector[0] < 0 else vector
            directions.append(vector / np.linalg.norm(vector))
    axis = np.mean(directions, 0) if directions else np.array([1.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    size = images.shape[-1]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    grid = np.stack((xs, ys, np.ones_like(xs)), -1)
    middle = np.array([size / 2, size / 2, 1.0], dtype=np.float32)

    def knee(points):
        offset = points - centre[item]
        u = offset[..., 0] * axis[0] + offset[..., 1] * axis[1]
        a = offset[..., 0] * axis[1] - offset[..., 1] * axis[0]
        return np.stack((sign * u, a, offset[..., 2]), -1)

    with torch.inference_mode():
        encoded = model.encode_batch(
            with_images(move_batch(collate_studies([model_dataset[row]]), gpu))
        )
        shift = encoded["study_frame_shift"][0].float().cpu().numpy()
        inferred = encoded["study_frame_rotation"][0].float().cpu().numpy()
        angles = model.rotation_angles(encoded["study_frame_rotation"][0]).cpu().numpy()
    print(row, "left" if left else "right", "shift mm", shift.round(1), "angles deg", angles.round(1))
    figure, axes = plt.subplots(2, len(SLICES), figsize=(3.5 * len(SLICES), 8.4))
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
        if not members:
            for line in range(2):
                axes[line][column].axis("off")
            continue
        distances = []
        for item in members:
            distances.append(abs(knee(maps[item] @ middle)[component] - wanted))
        item = members[int(np.argmin(distances))]
        raw = knee(np.einsum("ij,hwj->hwi", maps[item], grid))
        actual = float(knee(maps[item] @ middle)[component])
        image = images[item] / 255.0
        for line, (frame, moved, name) in enumerate(
            (
                (np.eye(3), np.zeros(3), "Before learning"),
                (inferred, shift, "Inferred"),
            )
        ):
            coordinates = (raw - moved) @ frame.T
            medial = (coordinates[..., 0] > 0).astype(np.float32)
            front = (coordinates[..., 1] > 0).astype(np.float32)
            up = (coordinates[..., 2] > 0).astype(np.float32)
            patellar = front * up
            weights = {
                "PF OA": patellar,
                "Medial OA": medial * (1 - patellar),
                "Lateral OA": (1 - medial) * (1 - patellar),
            }
            cuts = (
                (coordinates[..., 0], "white"),
                (coordinates[..., 1], "yellow"),
                (coordinates[..., 2], "cyan"),
            )
            colour = sum(
                weight[..., None] * np.array(COLOURS[label])
                for label, weight in weights.items()
            )
            blended = 0.58 * image[..., None] + 0.42 * colour
            panel = axes[line][column]
            panel.imshow(np.clip(blended, 0, 1))
            for values, shade in cuts:
                if values.min() < 0 < values.max():
                    panel.contour(values, [0.0], colors=shade, linewidths=1.3)
            panel.set_title(
                f"{name}: {title}\n(slice at {actual:+.0f} mm)", fontsize=8.5
            )
            panel.set_xticks([])
            panel.set_yticks([])
    shown = ("Medial OA", "Lateral OA", "PF OA")
    figure.legend(
        [plt.Rectangle((0, 0), 1, 1, facecolor=COLOURS[k], edgecolor="#333333") for k in shown],
        ["medial side (not front-up)", "lateral side (not front-up)", "front and up"],
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=False,
    )
    figure.suptitle(
        f"Study row {row} ({'left' if left else 'right'} knee). Top: the frame before learning (geometric centre, header alignment). "
        f"Bottom: the frame the model infers for this knee.\n"
        f"Inferred centre shift: medial {shift[0]:+.1f}, front {shift[1]:+.1f}, up {shift[2]:+.1f} mm. "
        f"Inferred angles: nod {angles[0]:+.1f}, roll {angles[1]:+.1f}, turn {angles[2]:+.1f} degrees. "
        "Lines: white midline, yellow front cut, cyan height cut.",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.94))
    figure.subplots_adjust(hspace=0.28)
    figure.savefig(OUT / f"{RUN.name[16:]}-{row:03d}.png", dpi=85)
    plt.close(figure)
    print("saved", OUT / f"{RUN.name[16:]}-{row:03d}.png")
