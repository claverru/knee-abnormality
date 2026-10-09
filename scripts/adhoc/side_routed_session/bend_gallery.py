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

RUN = Path("outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors")
OUT = Path("outputs/analysis/r/routing-and-shifts-frames-20261007/knee-bend")


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
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
wide = wide_data(config.data)
planes = [kind.plane for kind in wide.series_types]
fluid = [kind.fluid_sensitive for kind in wide.series_types]
dataset = KneeStudyDataset(studies, index, wide, targets, None, 4)
rows = np.random.default_rng(3).choice(len(studies), 30, replace=False)
figure, axes = plt.subplots(3, 6, figsize=(21, 11))
axes = axes.ravel()
drawn = 0
for row in rows:
    if drawn == len(axes):
        break
    w = with_images(move_batch(collate_studies([dataset[int(row)]]), torch.device("cpu")))
    kinds = w["series_type_index"].numpy()
    best = None
    for n in range(len(kinds)):
        if planes[kinds[n]] != "Sagittal":
            continue
        centre = np.array([float(w[key][n]) for key in ("knee_center_x_mm", "knee_center_y_mm")])
        for t in np.flatnonzero(w["slice_mask"][n].numpy()):
            maps = w["patient_maps"][n, t, 1].float().numpy()
            if not np.isfinite(maps).all() or not np.isfinite(centre).all():
                continue
            middle = maps @ np.array([111.5, 111.5, 1.0])
            distance = abs(np.hypot(*(middle[:2] - centre)) - 18.0) + (20.0 if fluid[kinds[n]] else 0.0)
            if best is None or distance < best[0]:
                best = (distance, n, t)
    if best is None:
        continue
    _, n, t = best
    panel = axes[drawn]
    panel.imshow(w["images"][n, t, 1].numpy(), cmap="gray", vmin=0, vmax=255)
    for line in range(0, 224, 28):
        panel.axvline(line, color="cyan", linewidth=0.3, alpha=0.5)
    panel.set_title(f"row {row}", fontsize=9)
    panel.set_xticks([])
    panel.set_yticks([])
    drawn += 1
figure.suptitle("Sagittal slices through a condyle, 120 mm field, vertical guide lines every 15 mm", fontsize=12)
figure.tight_layout(rect=(0, 0, 1, 0.97))
figure.savefig(OUT / "sagittal-gallery.png", dpi=75)
print("saved", OUT / "sagittal-gallery.png", drawn)
