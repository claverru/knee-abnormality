import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

STUDY = "StudyInstanceUID"
PREDICTIONS = Path("/workspace/predictions/rebuild-external")
EXCLUDED = Path("/workspace/data/external/excluded_from_training.csv")
OAI_DROPPED = ["ACL", "MCL", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main() -> None:
    teachers = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    output = Path(sys.argv[2])
    frames, records, targets = [], {}, None
    for teacher in teachers:
        record = json.loads((PREDICTIONS / teacher / "record.json").read_text())
        if record["image_multiplier"] != 4 or record["attention_temperature"] != 0.85 or not record["reflection_tta"]:
            raise ValueError(f"{teacher}: not x4, tau 0.85, mirror TTA")
        records[teacher] = record
        frame = pl.concat([pl.read_parquet(PREDICTIONS / teacher / f"{name}.parquet") for name in ("fastmri", "oai")]).sort(STUDY)
        columns = [column for column in frame.columns if column != STUDY]
        targets = targets or columns
        if columns != targets:
            raise ValueError(f"{teacher}: target columns differ")
        frames.append(frame)
    studies = frames[0][STUDY]
    for teacher, frame in zip(teachers, frames):
        if not frame[STUDY].equals(studies):
            raise ValueError(f"{teacher}: studies differ")
    stack = np.stack([frame.select(targets).to_numpy() for frame in frames])
    if not np.isfinite(stack).all():
        raise ValueError("non-finite predictions")
    mean = stack.mean(0)
    checks = {}
    for teacher, values in zip(teachers, stack):
        correlation = [float(np.corrcoef(values[:, column], mean[:, column])[0, 1]) for column in range(len(targets))]
        checks[teacher] = round(min(correlation), 4)
        if min(correlation) < 0.7:
            raise ValueError(f"{teacher}: correlation with the mean {min(correlation):.3f}")
    excluded = set(pl.read_csv(EXCLUDED)[STUDY].to_list())
    oai = pl.col(STUDY).str.starts_with("oai-")
    table = (
        pl.DataFrame({STUDY: studies, **{target: mean[:, column] for column, target in enumerate(targets)}})
        .filter(~pl.col(STUDY).is_in(excluded))
        .with_columns(*(pl.when(oai).then(None).otherwise(pl.col(name)).alias(name) for name in OAI_DROPPED))
    )
    expanded = pl.concat([table.with_columns(pl.lit(fold, dtype=pl.Int64).alias("fold")) for fold in range(5)]).select(STUDY, "fold", *targets).sort(STUDY, "fold")
    output.mkdir(parents=True, exist_ok=True)
    expanded.write_parquet(output / "external_training_table.parquet")
    summary = {
        "teachers": {teacher: {"checkpoint": record["checkpoint"], "sha256": record["sha256"], "fold": record["fold"], "backbone": record["backbone"]} for teacher, record in records.items()},
        "inference": "x4, tau 0.85, mirror TTA, probability mean of the teachers",
        "oai_labels_dropped_null": OAI_DROPPED,
        "studies": table.height,
        "fastmri": table.filter(pl.col(STUDY).str.starts_with("fastmri-")).height,
        "oai": table.filter(oai).height,
        "teacher_min_correlation_with_mean": checks,
    }
    (output / "tables.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("studies", "fastmri", "oai")}), len(teachers), "teachers", flush=True)


if __name__ == "__main__":
    main()
