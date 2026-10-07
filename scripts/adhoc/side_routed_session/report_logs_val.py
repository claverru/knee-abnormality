import math
import sys
from pathlib import Path

import polars as pl

from knee_abnormality.config import load_config
from knee_abnormality.data.report_attributes import load_report_attribute_targets
from knee_abnormality.training.report_attributes import (
    QUESTION_ROUTES,
    REGION_IDS,
    ReportAttributeHeads,
)

run = Path(sys.argv[1])
frame = pl.read_csv(next(run.glob("fold_*/logs/version_0/metrics.csv")), infer_schema_length=None)


def series(name):
    if name not in frame.columns:
        return []
    return [round(float(v), 4) for v in frame[name].drop_nulls().to_list()]


config = load_config(Path("configs/training.yaml"))
heads = ReportAttributeHeads(
    load_report_attribute_targets(config.paths.report_attribute_manifest, config.paths.data_root)
)
print("report_attribute_loss per epoch", series("report_attribute_loss"))
print()
print("region targets: weighted BCE per epoch, against the BCE of always predicting the base rate")
for number, name in enumerate(REGION_IDS):
    target = heads.region_table[:, number]
    weight = heads.region_weights[:, number]
    rate = float((target * weight).sum() / weight.sum())
    base = -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))
    values = series(f"report_attribute_bce_{name}")
    held = series(f"val_report_attribute_bce_{name}")
    last = values[-1] if values else float("nan")
    print(f"{name:44} base {base:.3f}  train {last:.3f} ({1 - last / base:+.0%})  validation first {held[0]:.3f} last {held[-1]:.3f} ({1 - held[-1] / base:+.0%})")
