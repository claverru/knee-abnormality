import json
import sys
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

AXES = ("medial", "front", "up")
DELTAS = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
SHOWN = ("PF OA", "Medial OA", "Lateral OA", "Medial Meniscus", "Lateral Meniscus", "Effusion", "Baker's", "MCL")
ROUTED = SHOWN[:5]
run = Path(sys.argv[1])
config = load_config(run / "effective_config.json")
targets, studies, split = prediction_studies(config, "validation", 1)
studies = studies.sort(STUDY)
device = torch.device("cuda")
selected = json.loads((run / "fold_1" / "selected_checkpoint.json").read_text())
model = load_checkpoint_model(config, targets, Path(selected["path"]), device, expected_fold=1)
index = build_series_index(config.paths.data_root, config.data, split, studies[STUDY].to_list())
dataset = KneeStudyDataset(studies, index, config.data, targets, None, 1)
settings = [(axis, delta) for axis in (2, 1) for delta in DELTAS]
logits = {setting: [] for setting in settings}
with torch.inference_mode():
    for row in range(studies.height):
        encoded = model.encode_batch(with_images(move_batch(collate_studies([dataset[row]]), device)))
        sign = (1 - 2 * encoded["left_probability"])[:, None]
        for axis, delta in settings:
            shift = encoded["frame_shift"].clone()
            shift[:, axis] += delta
            moved = {**encoded, "frame_shift": shift}
            with torch.autocast(device_type="cuda", enabled=False):
                moved["position"] = model.frame(moved, sign)
                moved["fine_position"] = model.frame(
                    {**model.fine(moved), "frame_shift": shift}, sign
                )
            logits[(axis, delta)].append(model.pool_encoded(moved).float().cpu()[0])
split_dir = config.paths.data_root / "folds" / config.validation.split
frame = (
    pl.read_csv(split_dir / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split_dir / "targets.csv"), on=STUDY)
    .sort(STUDY)
)
assert frame[STUDY].to_list() == studies[STUDY].to_list()
rows = []
for (axis, delta), values in logits.items():
    scores = torch.stack(values).numpy()
    aucs = {
        name: float(roc_auc_score(frame[name].to_numpy() >= 0.5, scores[:, number]))
        for number, name in enumerate(targets)
    }
    rows.append(
        {
            "centre moved": AXES[axis],
            "mm": delta,
            "macro": round(float(np.mean(list(aucs.values()))), 5),
            "routed five": round(sum(aucs[name] for name in ROUTED), 4),
            **{name: round(aucs[name], 4) for name in SHOWN},
        }
    )
table = pl.DataFrame(rows)
out = Path("outputs/analysis/s/side-routing-centre-sweep-20261006")
out.mkdir(parents=True, exist_ok=True)
table.write_csv(out / f"{run.name}.csv")
with pl.Config(tbl_rows=40, tbl_cols=-1, tbl_width_chars=220):
    print(table)
