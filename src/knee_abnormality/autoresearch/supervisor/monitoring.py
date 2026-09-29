import operator
import time
from pathlib import Path

from knee_abnormality.autoresearch.runtime.layout import campaign_fold, fold_dir
from knee_abnormality.autoresearch.runtime.metrics import metrics_path, read_metrics
from knee_abnormality.autoresearch.runtime.storage import (
    read_json,
    terminate,
    write_json,
)

COMPARISONS = {">": operator.gt, ">=": operator.ge}
PROJECTION_TOLERANCE = 1.02


def activation_check(state, exp, fold, rows, final):
    signature = exp["launch_binding"]["activation"]
    epoch = signature["epoch"]
    if not final and not any(row["epoch"] > epoch for row in rows):
        return None
    metric = signature["metric"]
    values = [row[metric] for row in rows if row["epoch"] <= epoch and metric in row]
    observed = max(values, default=None)
    passed = observed is not None and COMPARISONS[signature["comparison"]](
        observed, signature["threshold"]
    )
    seen = (
        "was never logged" if observed is None else f"reached a maximum of {observed!r}"
    )
    return {
        **signature,
        "observed_maximum": observed,
        "logged_values": len(values),
        "passed": passed,
        "reason": None
        if passed
        else (
            f"Activation signature failed: {metric} {seen} through epoch {epoch}; "
            f"the signature requires {signature['comparison']} {signature['threshold']!r}"
        ),
    }


def training_started(fold: Path):
    marker = fold / "backbone_identity.json"
    return marker.stat().st_mtime if marker.is_file() else None


def projected_training_seconds(exp, fold, started):
    ends = sorted(
        path.stat().st_mtime for path in (fold / "epochs").glob("metrics_epoch_*.json")
    )
    if len(ends) < 2:
        return None
    rate = (ends[-1] - ends[0]) / (len(ends) - 1)
    remaining = exp["config"]["trainer"]["max_epochs"] - len(ends)
    return ends[-1] - started + rate * remaining


def budget_check(state, exp, fold, rows, final):
    if state["budget"]["mode"] != "range":
        return None
    limit = state["limits"]["max_training_seconds"]
    projected = None
    if final:
        timing = fold / "timing.json"
        if not timing.is_file():
            return None
        seconds = read_json(timing)["training_seconds"]
    else:
        ledger = fold / "triplet_accounting.json"
        started = training_started(fold)
        if started is None or (ledger.is_file() and read_json(ledger)["complete"]):
            return None
        seconds = time.time() - started
        projected = projected_training_seconds(exp, fold, started)
        if seconds <= limit and (
            projected is None or projected <= limit * PROJECTION_TOLERANCE
        ):
            return None
    passed = seconds <= limit and (
        projected is None or projected <= limit * PROJECTION_TOLERANCE
    )
    return {
        "max_training_seconds": limit,
        "training_seconds": seconds,
        "projected_training_seconds": projected,
        "passed": passed,
        "reason": None
        if passed
        else (
            f"Training time {seconds:.0f} s exceeded the campaign's upper bound of "
            f"{limit:.0f} s"
            if seconds > limit
            else f"Training time projected from the epochs after the first, "
            f"{projected:.0f} s, exceeds the campaign's upper bound of {limit:.0f} s "
            f"by more than {PROJECTION_TOLERANCE - 1:.0%}"
        ),
    }


RUN_CHECKS = (("activation", activation_check), ("budget", budget_check))


def run_failure(exp):
    return next(
        (
            outcome
            for outcome in exp.get("run_checks", {}).values()
            if not outcome["passed"]
        ),
        None,
    )


class MonitorMixin:
    def check_run(self, exp, final):
        recorded = exp.get("run_checks", {})
        pending = [(name, check) for name, check in RUN_CHECKS if name not in recorded]
        if not pending:
            return False
        fold = fold_dir(Path(exp["directory"]) / "training", campaign_fold(self.state))
        rows = read_metrics(fold)
        path = metrics_path(fold)
        outcomes = {
            name: {
                **outcome,
                "check": name,
                "metrics": str(path) if path else None,
                "evaluated_at": time.time(),
                "at_training_end": final,
            }
            for name, check in pending
            for outcome in [check(self.state, exp, fold, rows, final)]
            if outcome is not None
        }
        if not outcomes:
            return False
        exp["run_checks"] = {**recorded, **outcomes}
        write_json(Path(exp["directory"]) / "run-checks.json", exp["run_checks"])
        return True

    def watch_training(self, job):
        exp = self.state["experiments"][job["payload"]["experiment"]]
        if not exp["launched"]:
            return
        if self.check_run(exp, final=False):
            self.save()
        if run_failure(exp):
            owner = Path(job["path"]) / "owner.json"
            if owner.exists():
                terminate(read_json(owner))
