import math
from pathlib import Path

from knee_abnormality.autoresearch.runtime.layout import (
    campaign_fold,
    experiment_diagnostics,
    fold_dir,
)
from knee_abnormality.autoresearch.runtime.storage import read_json


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


def saved_auc(root, run, fold, key="prediction_macro_auc"):
    if not run:
        return None
    try:
        selected = read_json(fold_dir(root / run, fold) / "selected_checkpoint.json")
        return number(selected[key])
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


def interval_text(bounds):
    return (
        f"[{score_text(bounds[0], True)}, {score_text(bounds[1], True)}]"
        if isinstance(bounds, list)
        and len(bounds) == 2
        and all(number(v) is not None for v in bounds)
        else "unavailable"
    )


def metric_line(exp):
    evidence = exp.get("evidence", {})
    scores = evidence.get("macro_auc", {})
    reference, candidate = scores.get("reference"), scores.get("candidate")
    bounds = evidence.get("bootstrap", {}).get("delta_percentile_95")
    adoption_bounds = (evidence.get("adoption_bootstrap") or {}).get(
        "delta_percentile_80"
    )
    return (
        f"AUC: {score_text(reference)} -> {score_text(candidate)}; "
        f"delta {score_text(difference(candidate, reference), True)}; 95% CI {interval_text(bounds)}"
        + (
            f"; adoption 80% CI {interval_text(adoption_bounds)}"
            if adoption_bounds is not None
            else ""
        )
    )


def per_label_lines(exp):
    diagnostics = experiment_diagnostics(exp)
    if diagnostics is None:
        return ["   Per-label diagnostics: unavailable"]
    lines = [
        "   Per-label AUC (exploratory; unadjusted paired 95% CI for delta):",
        f"   {'Label':<22} {'Reference':>9} {'Candidate':>9} {'Delta':>10}  {'95% CI':<24} {'Pos/Neg':>9} {'Draws':>10}",
    ]
    soft = diagnostics.get("soft_macro")
    if soft is not None:
        lines.insert(
            0,
            f"   Soft macro AUC (diagnostic): {score_text(soft['reference'])} -> "
            f"{score_text(soft['candidate'])}; delta {score_text(soft['delta'], True)}",
        )
        lines[-1] += f" {'Soft delta':>10}"
    for row in diagnostics["per_target"]:
        bootstrap = row["bootstrap"]
        lines.append(
            f"   {row['target']:<22} {score_text(row['reference']):>9} "
            f"{score_text(row['candidate']):>9} {score_text(row['delta'], True):>10}  "
            f"{interval_text(bootstrap['delta_percentile_95']):<24} "
            f"{row['positives']:>4}/{row['negatives']:<4} "
            f"{bootstrap['valid_draws']:>4}/{diagnostics['bootstrap']['draws']:<4}"
            + (
                f" {score_text(row['soft']['delta'], True):>10}"
                if "soft" in row
                else ""
            )
        )
    return lines


def adoption_result(exp):
    evidence = exp.get("evidence", {})
    scores = evidence.get("macro_auc", {})
    delta = difference(scores.get("candidate"), scores.get("reference"))
    bounds = (evidence.get("adoption_bootstrap") or {}).get("delta_percentile_80")
    return delta, bounds


def findings_summary(state, campaign):
    log = Path(campaign["artifacts_root"]) / "findings.md"
    lines = [
        "# Campaign findings",
        "",
        f"Decision summary. The full orchestrator log is `{log}` (not tracked).",
        "",
        f"Incumbent: `{state['incumbent']['id']}`",
    ]
    decided = [exp for exp in state["experiments"].values() if exp.get("decision")]
    if not decided:
        return "\n".join([*lines, "", "No decided experiments yet.", ""])
    lines.append("")
    for exp in decided:
        delta, bounds = adoption_result(exp)
        soft = exp.get("evidence", {}).get("soft_macro_auc")
        lines.append(
            f"- `{exp['idea']}`: {decision_text(exp)}, macro AUC delta "
            f"{score_text(delta, True)}, adoption 80% CI {interval_text(bounds)}"
            + (f", soft AUC delta {score_text(soft['delta'], True)}" if soft else "")
        )
    return "\n".join([*lines, ""])


def progress_lines(state, root, output):
    fold = campaign_fold(state)
    decisions = manual_decisions(state, output)
    baseline = saved_auc(root, state["reference"], fold)
    incumbent = state["incumbent"]
    current = saved_auc(root, incumbent["run"], fold)
    anchor, retained = retained_lineage(state, decisions)
    lines = [
        f"Starting reference AUC: {score_text(baseline)} (historical-control)",
        (
            f"Incumbent: {incumbent['id']} | AUC {score_text(current)} "
            f"(x1 {score_text(saved_auc(root, incumbent['run'], fold, 'score'))})"
        ),
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
        anchor_score = saved_auc(root, anchor["run"], fold)
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
            lines.extend(per_label_lines(exp))
        lines.extend(
            [
                "",
                "Delta is candidate minus its recorded reference, not the bootstrap mean.",
                "Intervals are paired bootstrap 95% intervals; positive rejected deltas are not retained gains.",
                "Per-label intervals are unadjusted diagnostics, not simultaneous intervals or acceptance criteria.",
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
    fold = campaign_fold(state)
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
                        f"    AUC {score_text(saved_auc(root, previous['run'], fold))} -> "
                        f"{score_text(saved_auc(root, current['run'], fold))}"
                    ),
                    f"    Record: {output / 'decisions' / record['event'] / 'decision.json'}",
                ]
            )
    lines.extend(["", f"Artifacts: {output / 'experiments'}"])
    return "\n".join(lines)
