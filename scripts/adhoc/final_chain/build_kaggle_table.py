import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

STUDY = "StudyInstanceUID"


def group_mean(runs, predictions):
    by_family = {}
    for run in runs:
        frame = pl.read_parquet(predictions / f"{run}.parquet")
        record = json.loads((predictions / f"{run}.json").read_text())
        if record["image_multiplier"] != 4 or record["attention_temperature"] != 0.85 or not record["reflection_tta"]:
            raise ValueError(f"{run}: not x4, tau 0.85, mirror TTA")
        by_family.setdefault(run.split("-")[0], []).append(frame)
    families = {}
    for family, frames in by_family.items():
        frame = pl.concat(frames).sort(STUDY)
        if frame[STUDY].n_unique() != frame.height:
            raise ValueError(f"{family}: a study is predicted twice")
        families[family] = frame
    studies = None
    for family, frame in families.items():
        studies = frame[STUDY] if studies is None else studies
        if not frame[STUDY].equals(studies):
            raise ValueError(f"{family}: does not cover the same studies")
    targets = [c for c in next(iter(families.values())).columns if c not in (STUDY, "fold")]
    stack = np.stack([f.select(targets).to_numpy() for f in families.values()])
    return studies, targets, stack.mean(0), sorted(families), families


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2])
    predictions = Path(spec.get("predictions", "/workspace/predictions/rebuild"))
    total, studies, targets, summary, fold = None, None, None, {}, None
    for group in spec["groups"]:
        s, t, mean, families, frames = group_mean(group["runs"], predictions)
        if studies is None:
            studies, targets = s, t
            fold = frames[families[0]]["fold"].cast(pl.Int32)
        elif not s.equals(studies) or t != targets:
            raise ValueError("groups differ in studies or targets")
        total = mean * group["weight"] if total is None else total + mean * group["weight"]
        summary[group["name"]] = {"weight": group["weight"], "families": families, "runs": group["runs"]}
    weight = sum(group["weight"] for group in spec["groups"])
    probability = np.clip(total / weight, 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability)).astype(np.float32)
    table = pl.DataFrame({STUDY: studies, "fold": fold, "reading": ["joint"] * len(studies), **{t: logits[:, i] for i, t in enumerate(targets)}})
    if not np.isfinite(logits).all() or table.height != 4407:
        raise ValueError(f"table check failed: {table.height} rows")
    output.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output / "kaggle_teacher_table.parquet")
    (output / "tables.json").write_text(json.dumps({"kaggle_teacher_table": {"rows": table.height, "value": "logit of the weighted mean of the groups' probabilities; each group is the mean over its families, each family's runs predicting their own validation folds at x4, tau 0.85, mirror TTA", "groups": summary}}, indent=2) + "\n")
    print(table.height, "rows", {k: v["families"] for k, v in summary.items()})


if __name__ == "__main__":
    main()
