import hashlib
import json
import sys
from pathlib import Path

import polars as pl
import torch

from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import load_models
from knee_abnormality.inference.engine import predict_loader

RUN = Path(sys.argv[1])
FOLD = int(sys.argv[2])
OUT = Path(sys.argv[3])
MULTIPLIER = 4
TEMPERATURE = 0.85
config = load_config(RUN / "effective_config.json")
checkpoint = next((RUN / f"fold_{FOLD}").glob("best-*.ckpt"))
targets, studies, metadata_split = prediction_studies(config, "validation", FOLD)
index = build_series_index(
    config.paths.data_root, config.data, metadata_split, studies[STUDY].to_list()
)
device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
models = load_models(
    config,
    targets,
    checkpoint,
    device,
    expected_fold=FOLD,
    attention_temperature=TEMPERATURE,
)
dataset = KneeStudyDataset(studies, index, config.data, targets, None, MULTIPLIER, 4)
loader = build_data_loader(
    dataset, config.trainer, 2, num_workers=8, pin_memory=False, prepare_on_loader=False
)
ids, predictions, _ = predict_loader(models, loader, device, "auto", True)
frame = (
    pl.DataFrame({STUDY: ids, "fold": [FOLD] * len(ids)})
    .hstack(pl.DataFrame(predictions.float().cpu().numpy(), schema=list(targets)))
    .sort(STUDY)
)
OUT.parent.mkdir(parents=True, exist_ok=True)
frame.write_parquet(OUT)
OUT.with_suffix(".json").write_text(
    json.dumps(
        {
            "run": str(RUN),
            "fold": FOLD,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "image_multiplier": MULTIPLIER,
            "attention_temperature": TEMPERATURE,
            "reflection_tta": True,
            "studies": frame.height,
            "value": "probabilities on the run's own validation fold",
        },
        indent=2,
    )
    + "\n"
)
