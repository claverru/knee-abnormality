import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import move_batch

OFFSETS = (0.0, 1.0, 2.0, 3.0, 5.0, -2.0)
run = Path("outputs/training/20261005-205251-side-routed-run5")
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
device = torch.device("cuda")
selected = json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())
model = load_checkpoint_model(
    config, targets, Path(selected["path"]), device, expected_fold=1
)
index = build_series_index(
    config.paths.data_root, config.data, split, studies[STUDY].to_list()
)
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
original = model.frame
state = {"offset": 0.0}


def shifted(knee_x, anterior_axis, superior_axis, medial_sign):
    coordinates = original(knee_x, anterior_axis, superior_axis, medial_sign)
    shift = torch.zeros(3, device=coordinates.device)
    shift[0] = state["offset"]
    return coordinates + shift


model.frame = shifted
logits = {offset: [] for offset in OFFSETS}
with torch.inference_mode():
    for row in range(studies.height):
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        state["offset"] = 0.0
        encoded = model.encode_batch(batch)
        for offset in OFFSETS:
            state["offset"] = offset
            with torch.autocast(device_type="cuda", enabled=False):
                position = shifted(
                    encoded["knee_x"],
                    encoded["anterior_axis"],
                    encoded["superior_axis"],
                    (1 - 2 * encoded["left_probability"])[:, None],
                )
            moved = {**encoded, "position": position}
            logits[offset].append(model.pool_encoded(moved).float().cpu()[0])
split_dir = config.paths.data_root / "folds" / config.validation.split
frame = (
    pl.read_csv(split_dir / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split_dir / "targets.csv"), on=STUDY)
    .sort(STUDY)
)
assert frame[STUDY].to_list() == studies[STUDY].to_list()
truth = {name: frame[name].to_numpy() >= 0.5 for name in targets}
families = {
    "Medial Meniscus": ("Medial Meniscus", "Lateral Meniscus"),
    "Lateral Meniscus": ("Medial Meniscus", "Lateral Meniscus"),
    "Medial OA": ("Medial OA", "Lateral OA", "PF OA"),
    "Lateral OA": ("Medial OA", "Lateral OA", "PF OA"),
}
rows = []
for offset in OFFSETS:
    scores = torch.stack(logits[offset]).numpy()
    row = {"lateral_offset_mm": offset}
    for name, members in families.items():
        column = scores[:, targets.index(name)]
        anywhere = np.any([truth[m] for m in members], axis=0)
        row[f"{name} auc"] = round(float(roc_auc_score(truth[name], column)), 4)
        row[f"{name} loc"] = round(
            float(roc_auc_score(truth[name][anywhere], column[anywhere])), 4
        )
    rows.append(row)
table = pl.DataFrame(rows)
out = Path("outputs/analysis/s/side-routing-offset-probe-20261006")
out.mkdir(parents=True, exist_ok=True)
table.write_csv(out / "offsets.csv")
with pl.Config(tbl_cols=-1, tbl_width_chars=220):
    print(table)
