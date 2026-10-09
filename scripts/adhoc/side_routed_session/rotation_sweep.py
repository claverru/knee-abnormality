import json
import math
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

AXES = ("nod", "roll", "turn")
DELTAS = (-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0)
ROUTED = ("Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA")
run = Path("outputs/training/20261006-073158-side-routed-hard-octants")
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
learned_logit = model.frame_rotation_logit.detach().clone()
learned = (math.pi / 2) * torch.tanh(learned_logit)
print("learned angles deg", torch.rad2deg(learned).tolist())
settings = [(axis, delta) for axis in range(3) for delta in DELTAS]
logits = {(mode, setting): [] for mode in ("all", "cuts") for setting in settings}
with torch.inference_mode():
    for row in range(studies.height):
        batch = with_images(move_batch(collate_studies([dataset[row]]), device))
        model.frame_rotation_logit.copy_(learned_logit)
        encoded = model.encode_batch(batch)
        sign = (1 - 2 * encoded["left_probability"])[:, None]
        for setting in settings:
            axis, delta = setting
            angles = learned.clone()
            angles[axis] += math.radians(delta)
            model.frame_rotation_logit.copy_(
                torch.atanh((angles / (math.pi / 2)).clamp(-0.999, 0.999))
            )
            logits[("cuts", setting)].append(model.pool_encoded(encoded).float().cpu()[0])
            with torch.autocast(device_type="cuda", enabled=False):
                position = model.frame(
                    encoded["knee_x"], encoded["anterior_axis"], encoded["superior_axis"], sign
                )
            logits[("all", setting)].append(
                model.pool_encoded({**encoded, "position": position}).float().cpu()[0]
            )
split_dir = config.paths.data_root / "folds" / config.validation.split
frame = (
    pl.read_csv(split_dir / "studies.csv")
    .filter(pl.col("fold") == 1)
    .join(pl.read_csv(split_dir / "targets.csv"), on=STUDY)
    .sort(STUDY)
)
assert frame[STUDY].to_list() == studies[STUDY].to_list()
truth = {name: frame[name].to_numpy() >= 0.5 for name in targets}
rows = []
for (mode, (axis, delta)), values in logits.items():
    scores = torch.stack(values).numpy()
    aucs = {
        name: float(roc_auc_score(truth[name], scores[:, number]))
        for number, name in enumerate(targets)
    }
    rows.append(
        {
            "mode": mode,
            "axis": AXES[axis],
            "delta_deg": delta,
            "macro": round(float(np.mean(list(aucs.values()))), 5),
            "routed_sum": round(sum(aucs[name] for name in ROUTED), 4),
            "own_sum": round(sum(v for k, v in aucs.items() if k not in ROUTED), 4),
            **{name: round(aucs[name], 4) for name in ROUTED},
        }
    )
table = pl.DataFrame(rows)
out = Path("outputs/analysis/s/side-routing-rotation-sweep-20261006")
out.mkdir(parents=True, exist_ok=True)
table.write_csv(out / "sweep.csv")
with pl.Config(tbl_rows=60, tbl_cols=-1, tbl_width_chars=230):
    print(table.select("mode", "axis", "delta_deg", "macro", "routed_sum", "own_sum", *ROUTED))
