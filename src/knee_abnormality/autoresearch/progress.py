import math
import re
from pathlib import Path

from knee_abnormality.autoresearch.storage import read_json


def number(value):
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        else None
    )


def score_text(value, signed=False):
    value = number(value)
    if value is None:
        return "unavailable"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def difference(candidate, reference):
    candidate, reference = number(candidate), number(reference)
    return (
        candidate - reference
        if candidate is not None and reference is not None
        else None
    )


def saved_auc(root, run):
    if not run:
        return None
    fold = root / run / "fold_0"
    try:
        selected = read_json(fold / "selected_checkpoint.json")
        epoch = re.search(r"(?:^|-)epoch=(\d+)(?:-|$)", Path(selected["path"]).name)
        if epoch is None:
            return None
        metrics = read_json(fold / "epochs" / f"metrics_epoch_{int(epoch[1]):03d}.json")
        return number(metrics["macro_auc"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def manual_decisions(state, output):
    events = {entry["event"] for entry in state["findings"]}
    records = []
    for path in sorted((output / "decisions").glob("*/decision.json")):
        try:
            record = read_json(path)
        except (OSError, ValueError):
            continue
        if (
            record.get("event") in events
            and record.get("authority") == "explicit_user_instruction"
            and record.get("previous_incumbent")
            and record.get("incumbent")
        ):
            records.append(record)
    return sorted(records, key=lambda record: record["revision_after"])


def retained_lineage(state, decisions):
    reference = state["incumbent"]
    retained = []
    manual_roots = {
        record["incumbent"]["id"]
        for record in decisions
        if record.get("kind") == "incumbent_replacement"
    }
    while (
        reference["id"] in state["experiments"] and reference["id"] not in manual_roots
    ):
        exp = state["experiments"][reference["id"]]
        if (
            exp["id"] in retained
            or exp.get("decision") != "provisional_accept"
            or not exp.get("reference")
        ):
            return None, []
        retained.append(exp["id"])
        reference = exp["reference"]
    return reference, retained


def decision_text(exp):
    decision = exp.get("decision", "pending")
    original = exp.get("eligible_decision")
    if original == "provisional_accept" and decision not in {
        "provisional_accept",
        "pending",
    }:
        return f"{decision} (acceptance withdrawn)"
    if original and original != decision and decision != "pending":
        return f"{decision} (original verdict: {original})"
    return decision


def metric_line(exp):
    evidence = exp.get("evidence", {})
    scores = evidence.get("macro_auc", {})
    reference, candidate = scores.get("reference"), scores.get("candidate")
    bounds = evidence.get("bootstrap", {}).get("delta_percentile_95")
    interval = (
        f"[{score_text(bounds[0], True)}, {score_text(bounds[1], True)}]"
        if isinstance(bounds, list)
        and len(bounds) == 2
        and all(number(v) is not None for v in bounds)
        else "unavailable"
    )
    return (
        f"AUC: {score_text(reference)} -> {score_text(candidate)}; "
        f"delta {score_text(difference(candidate, reference), True)}; 95% CI {interval}"
    )


def progress_lines(state, root, output):
    decisions = manual_decisions(state, output)
    baseline = saved_auc(root, state["reference"])
    incumbent = state["incumbent"]
    current = saved_auc(root, incumbent["run"])
    anchor, retained = retained_lineage(state, decisions)
    lines = [
        f"Starting reference AUC: {score_text(baseline)} (historical-control)",
        f"Incumbent: {incumbent['id']} | AUC {score_text(current)}",
    ]
    changed_reference = anchor is not None and (
        (root / anchor["run"]).resolve() != (root / state["reference"]).resolve()
        or anchor["predictions_sha256"]
        != state["limits"]["reference_predictions_sha256"]
    )
    lines.append(
        f"AUC change since start: {score_text(difference(current, baseline), True)}"
        + (
            " (includes reference changes)"
            if changed_reference or anchor is None
            else ""
        )
    )
    if anchor is None:
        lines.append(
            "Retained experimental gain: unavailable (accepted lineage is incomplete)"
        )
    else:
        anchor_score = saved_auc(root, anchor["run"])
        gain = difference(current, anchor_score) if retained else 0.0
        lines.append(
            f"Retained experimental gain: {score_text(gain, True)} "
            f"({len(retained)} retained acceptances; vs {anchor['id']})"
        )
        if changed_reference:
            lines.append(
                f"Reference changed: gains measured from {anchor['id']} "
                f"at AUC {score_text(anchor_score)}; reference change excluded from gain."
            )
    withdrawn = sum(
        exp.get("eligible_decision") == "provisional_accept"
        and exp.get("decision") not in {None, "provisional_accept"}
        for exp in state["experiments"].values()
    )
    if withdrawn or decisions:
        lines.append(
            f"History: {withdrawn} withdrawn acceptances; {len(decisions)} manual reference changes "
            f"(see results {state['campaign']})."
        )
    return lines


def results_summary(state, root, output, campaign):
    if state is None:
        return f"{campaign} | NOT_STARTED\nNo experiment results yet."
    lines = [f"{campaign} | {state['status'].upper()} | experiment results"]
    lines.extend(progress_lines(state, root, output))
    completed = [exp for exp in state["experiments"].values() if exp.get("decision")]
    if not completed:
        lines.extend(["", "No decided experiments yet."])
    else:
        lines.extend(["", "Decided experiments (recorded order):"])
        for index, exp in enumerate(completed, 1):
            reference = exp.get("reference", {}).get("id", "unavailable")
            lines.extend(
                [
                    f"{index}. {exp['id']} — {decision_text(exp)}",
                    f"   Idea: {exp['idea']} | Reference: {reference}",
                    f"   {metric_line(exp)}",
                ]
            )
        lines.extend(
            [
                "",
                "Delta is candidate minus its recorded reference, not the bootstrap mean.",
                "Intervals are paired bootstrap 95% intervals; positive inconclusive deltas are not retained gains.",
                "Withdrawn acceptances keep their original scores and intervals; the displayed decision is current.",
            ]
        )
    pending = [
        exp
        for exp in state["experiments"].values()
        if not exp.get("decision") and exp.get("training_receipt")
    ]
    if pending:
        lines.extend(["", "Training attempts without a final decision:"])
        lines.extend(f"  {exp['id']} — {exp['stage']}" for exp in pending)
    decisions = manual_decisions(state, output)
    if decisions:
        lines.extend(
            ["", "Manual reference changes (excluded from experimental gain):"]
        )
        for record in decisions:
            previous, current = record["previous_incumbent"], record["incumbent"]
            lines.extend(
                [
                    f"  Revision {record['revision_after']}: {previous['id']} -> {current['id']}",
                    (
                        f"    AUC {score_text(saved_auc(root, previous['run']))} -> "
                        f"{score_text(saved_auc(root, current['run']))}"
                    ),
                    f"    Record: {output / 'decisions' / record['event'] / 'decision.json'}",
                ]
            )
    lines.extend(["", f"Artifacts: {output / 'experiments'}"])
    return "\n".join(lines)
