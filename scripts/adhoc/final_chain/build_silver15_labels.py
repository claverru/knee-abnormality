import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

STUDY = "StudyInstanceUID"
PREDICTIONS = Path("/workspace/predictions/silver15")
EXCLUDED = Path("/workspace/data/external/excluded_from_training.csv")
OUTPUT = Path("/workspace/data/labels/prepared/silver15-avg-x4-tau085-tta")
TEACHERS = [
    *(f"resnet50-silver-fold{fold}" for fold in range(4)),
    "rn50-silver-fold-4",
    "convnext-silver-fold-0",
    "convnext-silver-fold-1",
    "convnext-silver-only-imagenet-fold-2",
    "convnext-silver-fold-3",
    "convnext-silver-fold-4",
    *(f"mamba-silver-fold-{fold}" for fold in range(5)),
]


def main() -> None:
    frames, records = [], {}
    targets = None
    for teacher in TEACHERS:
        record = json.loads((PREDICTIONS / teacher / "record.json").read_text())
        if record["image_multiplier"] != 4 or record["attention_temperature"] != 0.85 or not record["reflection_tta"]:
            raise ValueError(f"{teacher}: not x4, tau 0.85, mirror TTA")
        records[teacher] = record
        frame = pl.concat([pl.read_parquet(PREDICTIONS / teacher / f"{name}.parquet") for name in ("fastmri", "oai")]).sort(STUDY)
        columns = [column for column in frame.columns if column != STUDY]
        if targets is None:
            targets = columns
        if columns != targets:
            raise ValueError(f"{teacher}: target columns differ")
        frames.append(frame)
    studies = frames[0][STUDY]
    for teacher, frame in zip(TEACHERS, frames):
        if not frame[STUDY].equals(studies):
            raise ValueError(f"{teacher}: studies differ")
    stack = np.stack([frame.select(targets).to_numpy() for frame in frames])
    if not np.isfinite(stack).all():
        raise ValueError("non-finite predictions")
    mean = stack.mean(0)
    checks = {}
    for teacher, values in zip(TEACHERS, stack):
        correlation = [float(np.corrcoef(values[:, column], mean[:, column])[0, 1]) for column in range(len(targets))]
        checks[teacher] = {"min_correlation_with_mean": round(min(correlation), 4), "mean_prediction": round(float(values.mean()), 4)}
        print(teacher, checks[teacher], flush=True)
        if min(correlation) < 0.7:
            raise ValueError(f"{teacher}: correlation with the mean {min(correlation):.3f}, broken teacher?")
    excluded = set(pl.read_csv(EXCLUDED)[STUDY].to_list())
    table = pl.DataFrame({STUDY: studies, **{target: mean[:, column] for column, target in enumerate(targets)}}).filter(~pl.col(STUDY).is_in(excluded))
    expanded = pl.concat([table.with_columns(pl.lit(fold, dtype=pl.Int64).alias("fold")) for fold in range(5)]).select(STUDY, "fold", *targets).sort(STUDY, "fold")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expanded.write_parquet(OUTPUT / "external_training_table.parquet")
    summary = {
        "teachers": {teacher: {"checkpoint": record["checkpoint"], "sha256": record["sha256"], "fold": record["fold"], "backbone": record["backbone"]} for teacher, record in records.items()},
        "inference": "x4, tau 0.85, mirror TTA, probability mean of the 15 teachers",
        "studies": table.height,
        "fastmri": table.filter(pl.col(STUDY).str.starts_with("fastmri-")).height,
        "oai": table.filter(pl.col(STUDY).str.starts_with("oai-")).height,
        "excluded": len(excluded & set(studies.to_list())),
        "prevalence": {target: round(float(table[target].mean()), 4) for target in targets},
        "teacher_checks": checks,
    }
    (OUTPUT / "tables.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("studies", "fastmri", "oai", "excluded", "prevalence")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
