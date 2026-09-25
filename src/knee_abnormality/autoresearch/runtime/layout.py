from pathlib import Path

from knee_abnormality.autoresearch.runtime.storage import read_json, sha256

PREDICTIONS = "validation_predictions.csv"


def campaign_fold(state):
    return state["config"]["validation"]["folds"][0]


def fold_dir(run, fold):
    return Path(run) / f"fold_{fold}"


def job_experiment(job):
    payload = job["payload"]
    return payload.get("experiment") or payload.get("event", {}).get("experiment")


def comparison_path(exp):
    return Path(exp["directory"]) / "comparison/comparison.json"


def diagnostics_path(comparison):
    return Path(comparison).with_name("per_label.json")


def read_diagnostics(comparison):
    comparison = Path(comparison)
    path = diagnostics_path(comparison)
    if not path.is_file():
        return None
    report = read_json(path)
    if report["comparison"]["sha256"] != sha256(comparison):
        raise ValueError(f"Label diagnostics refer to a changed comparison: {path}")
    return report


def experiment_diagnostics(exp):
    return read_diagnostics(comparison_path(exp))
