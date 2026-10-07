import json
import sys
from pathlib import Path

experiment = Path(sys.argv[1]).resolve()
frozen = experiment / "source"
sys.path[:0] = [str(frozen), str(frozen / "src")]

import polars as pl  # noqa: E402
import torch  # noqa: E402

from knee_abnormality.config import (  # noqa: E402
    INFERENCE_IO_THREADS,
    INFERENCE_NUM_WORKERS,
    load_config,
)
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader  # noqa: E402
from knee_abnormality.data.preprocessing.device import with_images  # noqa: E402
from knee_abnormality.data.sources import build_series_index  # noqa: E402
from knee_abnormality.data.splits import STUDY  # noqa: E402
from knee_abnormality.inference.checkpoint import load_checkpoint_model  # noqa: E402
from knee_abnormality.inference.engine import move_batch, use_fixed_backbone_chunks  # noqa: E402

limit = int(sys.argv[2])
output = Path(sys.argv[3])
run = experiment / "training"
config = load_config(run / "effective_config.json")
fold = config.validation.folds[0]
checkpoint = next((run / f"fold_{fold}").glob("best-*.ckpt"))
membership = pl.read_csv(config.split_dir / "studies.csv").sort(
    pl.col("fold") != fold, STUDY
)
studies = membership.select(STUDY).head(limit)
targets = tuple(json.loads((config.split_dir / "protocol.json").read_text())["targets"])
device = torch.device("cuda")
model = load_checkpoint_model(
    config, targets, checkpoint, device, attention_temperature=0.9
)
use_fixed_backbone_chunks(model, config.data.image_size)
chunked = model.backbone.forward_features
peak = {"value": 0.0}


def recording(images):
    features = chunked(images)
    finite = features[torch.isfinite(features)]
    peak["value"] = float(finite.abs().max()) if finite.numel() else float("nan")
    peak["nonfinite"] = int((~torch.isfinite(features)).sum())
    return features


model.backbone.forward_features = recording
loader = build_data_loader(
    KneeStudyDataset(
        studies,
        build_series_index(
            config.paths.data_root, config.data, "train", studies[STUDY].to_list()
        ),
        config.data,
        targets,
        None,
        4,
        INFERENCE_IO_THREADS,
    ),
    config.trainer,
    4,
    num_workers=INFERENCE_NUM_WORKERS,
    pin_memory=False,
    prepare_on_loader=False,
)
rows = []
with torch.inference_mode():
    for batch in loader:
        ids = batch["study_uid"]
        moved = with_images(move_batch(batch, device))
        error = None
        logit_peak = None
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(moved).to(torch.float32)
            logit_peak = float(logits.abs().max())
        except (RuntimeError, ValueError) as failure:
            error = f"{type(failure).__name__}: {failure}"
        rows.append(
            {
                "studies": ",".join(ids),
                "feature_peak": peak["value"],
                "feature_nonfinite": peak.get("nonfinite", 0),
                "logit_peak": logit_peak,
                "error": error,
            }
        )
frame = pl.DataFrame(rows, infer_schema_length=None)
frame.write_parquet(output)
print(
    json.dumps(
        {
            "experiment": experiment.name,
            "batches": frame.height,
            "errors": frame.filter(pl.col("error").is_not_null()).height,
            "first_errors": frame.filter(pl.col("error").is_not_null())["error"]
            .head(3)
            .to_list(),
            "nonfinite_feature_batches": int((frame["feature_nonfinite"] > 0).sum()),
            "feature_peak_max": frame["feature_peak"].max(),
            "feature_peak_p99": frame["feature_peak"].quantile(0.99),
            "logit_peak_max": frame["logit_peak"].max(),
        }
    )
)
