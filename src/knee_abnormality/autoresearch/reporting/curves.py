from pathlib import Path

from knee_abnormality.autoresearch.campaign.gates import (
    LOGISTICAL_KEYS,
    config_differences,
    flatten,
)
from knee_abnormality.autoresearch.runtime.layout import campaign_fold, fold_dir
from knee_abnormality.autoresearch.runtime.metrics import metrics_path, read_metrics
from knee_abnormality.autoresearch.runtime.storage import read_json

DECIMALS = 4
STANDARD = ("val_macro_auc", "train_loss_epoch")


def rounded(value):
    return round(value, DECIMALS)


def epoch_columns(rows):
    step = {
        name
        for row in rows
        if any(key.endswith("_step") for key in row)
        for name in row
    }
    columns = {}
    for row in rows:
        if any(key.endswith("_step") for key in row):
            continue
        for name, value in row.items():
            if name not in step and name not in {"epoch", "step"}:
                columns.setdefault(name, {})[int(row["epoch"])] = rounded(value)
    return columns


def label_curves(fold: Path):
    labels = {}
    for path in sorted((fold / "epochs").glob("metrics_epoch_*.json")):
        epoch = int(path.stem.removeprefix("metrics_epoch_"))
        for label, value in read_json(path)["per_target_auc"].items():
            if value is not None:
                labels.setdefault(label, {})[epoch] = rounded(value)
    return labels


def series(values, epochs):
    return [values.get(epoch) for epoch in epochs]


def run_curves(fold: Path):
    columns = epoch_columns(read_metrics(fold))
    labels = label_curves(fold)
    epochs = sorted({epoch for values in columns.values() for epoch in values})
    timing = fold / "timing.json"
    path = metrics_path(fold)
    return {
        "metrics": str(path) if path else None,
        "epoch_metrics": str(fold / "epochs"),
        "epochs": epochs,
        "training_seconds": read_json(timing)["training_seconds"]
        if timing.is_file()
        else None,
        **{name: series(columns.get(name, {}), epochs) for name in STANDARD},
        "per_label": {
            label: series(values, epochs) for label, values in sorted(labels.items())
        },
        "other": {
            name: series(values, epochs)
            for name, values in sorted(columns.items())
            if name not in STANDARD and len(set(values.values())) > 1
        },
    }


def config_changes(recipe, config):
    left, right = flatten(recipe), flatten(config)
    return {
        key: [left.get(key), right.get(key)]
        for key in config_differences(recipe, config)
        if key not in LOGISTICAL_KEYS
    }


def campaign_curves(state):
    fold = campaign_fold(state)
    runs = []
    for exp in state["experiments"].values():
        if not exp.get("decision"):
            continue
        runs.append(
            {
                "experiment": exp["id"],
                "idea": exp["idea"],
                "decision": exp["decision"],
                "incumbent": exp["id"] == state["incumbent"]["id"],
                "config_changes": config_changes(state["config"], exp["config"])
                if "config" in exp
                else None,
                "curves": run_curves(
                    fold_dir(Path(exp["directory"]) / "training", fold)
                ),
            }
        )
    return {
        "reference": {
            "run": state["reference"],
            "curves": run_curves(fold_dir(state["reference"], fold)),
        },
        "runs": runs,
    }


INLINE_RECENT_RUNS = 8
INLINE_DETAILED_RUNS = 4


def compact_curves(curves, detailed, path):
    recent = {run["experiment"] for run in curves["runs"][-INLINE_RECENT_RUNS:]}
    full = {
        run["experiment"]
        for run in [run for run in curves["runs"] if run["experiment"] in detailed][
            -INLINE_DETAILED_RUNS:
        ]
    }
    runs = []
    for run in curves["runs"]:
        if run["incumbent"] or run["experiment"] in full:
            runs.append(run)
        elif run["experiment"] in recent:
            runs.append(
                {
                    **run,
                    "curves": {
                        key: value
                        for key, value in run["curves"].items()
                        if key != "other"
                    },
                }
            )
        else:
            runs.append({key: value for key, value in run.items() if key != "curves"})
    return {
        "full_detail": str(path),
        "inline": "every curve for the reference, the incumbent and the last "
        f"{INLINE_DETAILED_RUNS} new_results; the standard curves for the last "
        f"{INLINE_RECENT_RUNS} other decided runs; configuration changes only for older runs, "
        "whose curves are in full_detail",
        "reference": curves["reference"],
        "runs": runs,
    }
