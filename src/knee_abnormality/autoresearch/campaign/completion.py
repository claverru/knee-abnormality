import math
from pathlib import Path

from knee_abnormality.autoresearch.campaign.gates import evaluate, verdict
from knee_abnormality.autoresearch.runtime.storage import read_json
from knee_abnormality.autoresearch.scoring.diagnostics import read_diagnostics


def completion_blockers(state, event):
    if event["kind"] != "completion":
        return ["Only completion events can use the reporter"]
    if event["eligible_decision"] not in {"reject", "inconclusive"}:
        return ["Wins and invalid runs require the orchestrator"]
    exp = state["experiments"][event["experiment"]]
    approval = exp.get("implementation_review")
    if not approval or approval["decision"] != "approve":
        return ["No recorded implementation approval"]
    if approval["concerns"] or exp.get("concerns"):
        return ["Implementation review has unresolved concerns"]
    if (
        exp["stage"] != "awaiting_decision"
        or state["running"] != exp["id"]
        or exp.get("eligible_decision") != event["eligible_decision"]
        or exp.get("reference") != state["incumbent"]
    ):
        return ["Completion or incumbent state changed"]
    if any(other["id"] != event["id"] for other in state["events"]):
        return ["Other pending events may require research or queue decisions"]
    idea = state["ideas"][exp["idea"]]
    for other in state["ideas"].values():
        if other["id"] == idea["id"] or other["status"] in {"completed", "retired"}:
            continue
        if (
            other["mechanism_id"] == idea["mechanism_id"]
            or other["parent_idea_id"] == idea["id"]
            or idea["id"] in other["dependencies"]
            or other["id"] == idea["parent_idea_id"]
        ):
            return ["Related outstanding ideas require orchestrator interpretation"]
    directory = Path(exp["directory"])
    try:
        seconds = exp["measured_compute_seconds"]
        if not math.isfinite(seconds) or seconds < 0:
            return ["Measured compute is unavailable or invalid"]
        reasons = evaluate(
            directory, state["limits"], {**exp["training_receipt"], "seconds": seconds}
        )
        if reasons:
            return reasons
        evidence = event["evidence"]
        comparison = directory / "comparison/comparison.json"
        if Path(evidence["comparison"]) != comparison:
            return ["Comparison artifact path does not match the attempt"]
        report = read_json(comparison)
        diagnostics = read_diagnostics(comparison)
        macro_bootstrap = (diagnostics or {}).get("macro_bootstrap")
        scores = report["family_means"]["all"]
        bootstrap = report["paired_report_group_bootstrap"]["intervals"]["all"]
        if (
            verdict(report, macro_bootstrap) != event["eligible_decision"]
            or macro_bootstrap != evidence.get("adoption_bootstrap")
            or scores != evidence["macro_auc"]
            or bootstrap != evidence["bootstrap"]
            or bootstrap["valid_draws"] != state["limits"]["bootstrap_draws"]
            or not all(math.isfinite(value) for value in scores.values())
        ):
            return [
                "Comparison evidence is inconsistent or has incomplete bootstrap draws"
            ]
        if len(report["per_target"]) != len(report["targets"]) or any(
            not all(
                math.isfinite(row[key]) for key in ("reference", "candidate", "delta")
            )
            for row in report["per_target"]
        ):
            return ["Per-target results are incomplete or invalid"]
        if diagnostics is None or len(diagnostics["per_target"]) != len(
            report["targets"]
        ):
            return ["Per-label bootstrap diagnostics are missing or incomplete"]
        for row, original in zip(
            diagnostics["per_target"], report["per_target"], strict=True
        ):
            interval = row["bootstrap"]
            bounds = interval["delta_percentile_95"]
            if (
                any(row[key] != value for key, value in original.items())
                or interval["valid_draws"] != bootstrap["valid_draws"]
                or not isinstance(bounds, list)
                or len(bounds) != 2
                or not all(math.isfinite(value) for value in bounds)
            ):
                return [
                    "Per-label bootstrap diagnostics are inconsistent or incomplete"
                ]
        required = (
            "changes.diff",
            "manifest.json",
            "config.yaml",
            "evaluation.json",
            "training/effective_config.json",
            "training/fold_0/input_identity.json",
        )
        if any(not (directory / name).is_file() for name in required):
            return ["Required completion artifacts are missing"]
    except (OSError, KeyError, TypeError, ValueError) as error:
        return [f"Completion evidence cannot be verified: {error}"]
    return []
