from pathlib import Path

from knee_abnormality.config import load_config
from knee_abnormality.data.report_attributes import load_report_attribute_targets
from knee_abnormality.training.report_attributes import (
    QUESTION_ROUTES,
    REGION_IDS,
    ReportAttributeHeads,
)

config = load_config(Path("configs/training.yaml"))
heads = ReportAttributeHeads(
    load_report_attribute_targets(
        config.paths.report_attribute_manifest, config.paths.data_root
    )
)
print("question heads", len(heads.question_ids), "routed and shared", len(QUESTION_ROUTES), "attention channels", heads.attention[-1].out_features)
for number, name in enumerate(REGION_IDS):
    target = heads.region_table[:, number]
    weight = heads.region_weights[:, number]
    print(f"{name:40} studies with target > 0.5: {int((target > 0.5).sum()):5}   mean weight {float(weight.mean()):.3f}   weighted positives {float((target * weight).sum()):7.1f}")
