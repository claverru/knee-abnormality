import time
from pathlib import Path

from knee_abnormality.autoresearch.campaign.rules import (
    CALIBRATION_EPOCHS,
    CALIBRATION_VALIDATION_PASSES,
    SEED_TERMINAL,
)
from knee_abnormality.autoresearch.reporting.progress import (
    adoption_result,
    decision_text,
    interval_text,
    progress_lines,
    score_text,
)
from knee_abnormality.autoresearch.runtime.layout import (
    campaign_fold,
    fold_dir,
    job_experiment,
)
from knee_abnormality.autoresearch.runtime.storage import (
    alive,
    optional_json,
    read_json,
)
from knee_abnormality.autoresearch.runtime.usage import campaign_usage


def status(supervisor):
    state = optional_json(supervisor.state_path)
    owner = optional_json(supervisor.output / "supervisor.json")
    return {
        "campaign": supervisor.campaign["id"],
        "status": state["status"] if state else "not_started",
        "supervisor_alive": alive(owner),
        "provider": state["provider"] if state else None,
        "reason": state.get("reason") if state else None,
        "retry_after": state.get("retry_after") if state else None,
        "queue": state["queue"] if state else [],
        "claimed": state["claimed"] if state else None,
        "running": state["running"] if state else None,
        "incumbent": state["incumbent"]["id"] if state else None,
        "limits": state["limits"] if state else None,
        "artifacts": str(supervisor.output),
    }


def duration_text(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s"


def job_line(job, state):
    directory = Path(job["path"])
    receipt = directory / "receipt.json"
    owner = directory / "owner.json"
    job_status = (
        "result application failed; fix the cause and resume"
        if job.get("application_error")
        else "finished, awaiting processing"
        if receipt.exists()
        else (
            "running"
            if owner.exists() and alive(read_json(owner))
            else "starting or awaiting recovery"
        )
    )
    exp = state["experiments"].get(job_experiment(job))
    detail = exp["idea"] if exp else ""
    if exp and job["kind"] == "training":
        fold = fold_dir(Path(exp["directory"]) / "training", campaign_fold(state))
        ledger = fold / "triplet_accounting.json"
        if ledger.exists():
            detail += (
                f", epoch {len(read_json(ledger)['train'])}/{state['limits']['epochs']}"
            )
        metrics = sorted((fold / "epochs").glob("metrics_epoch_*.json"))
        if metrics:
            try:
                detail += f", val AUC {read_json(metrics[-1])['macro_auc']:.6f}"
            except (OSError, ValueError, KeyError):
                pass
    text = (
        detail
        if detail and job_status == "running"
        else f"{detail} ({job_status})"
        if detail
        else job_status
    )
    elapsed = duration_text(time.time() - job["created_at"])
    return f"  {job['kind']:<13}{elapsed:>8}  {text}"


def result_rows(completed):
    width = max(len(exp["idea"]) for exp in completed)
    rows = [f"  {'idea':<{width}}  {'delta':>9}  {'adoption 80% CI':<24}  verdict"]
    for exp in completed:
        delta, bounds = adoption_result(exp)
        rows.append(
            f"  {exp['idea']:<{width}}  {score_text(delta, True):>9}  "
            f"{interval_text(bounds):<24}  {decision_text(exp)}"
        )
    return rows


def status_summary(supervisor):
    current = status(supervisor)
    supervisor_text = (
        "supervisor alive" if current["supervisor_alive"] else "supervisor not running"
    )
    header = (
        f"{current['campaign']} · {current['provider'] or 'no provider'} · "
        f"{current['status'].upper()} · {supervisor_text}"
    )
    lines = [header]
    if supervisor.state is None:
        return "\n".join(lines)
    state = supervisor.state
    limits = state["limits"]
    lines.append(
        f"Budget {duration_text(limits['max_run_seconds'])} per experiment · "
        f"{limits['epochs']} epochs · fold {campaign_fold(state)}"
    )
    warnings = []
    if current["reason"]:
        warnings.append(f"Reason: {current['reason']}")
    retry_after = state.get("retry_after", 0)
    if current["status"] == "running" and retry_after > time.time():
        warnings.append(
            "Provider wait: no agent jobs until "
            f"{time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(retry_after))} "
            f"({duration_text(retry_after - time.time())} left); training and comparison continue."
        )
    inbox = supervisor.output / "inbox"
    pending = len(list(inbox.glob("*.json")))
    rejected = len(list((inbox / "rejected").glob("*.json")))
    if pending:
        warnings.append(
            f"Inbox: {pending} request(s) waiting for the supervisor's next safe point."
        )
    if rejected:
        warnings.append(f"Inbox: {rejected} rejected request(s) in {inbox / 'rejected'}.")
    if (supervisor.output / "stop.json").exists() and current["status"] != "stopped":
        warnings.append(
            "Stop requested; waiting for the supervisor to finish stopping."
        )
    active = [job for job in state["jobs"].values() if not job["processed"]]
    queue = state["queue"]
    if (
        current["status"] == "running"
        and not active
        and not queue
        and not state["events"]
    ):
        warnings.append(
            "Idle: backlog exhausted; waiting for a new research direction or resume."
        )
    if warnings:
        lines.extend(["", *warnings])

    lines.extend(["", "Now"])
    if current["status"] == "calibrating":
        path = supervisor.output / "baseline/triplet_accounting.json"
        ledger = read_json(path) if path.exists() else {"train": [], "validation": []}
        lines.append(
            f"  calibration  {len(ledger['train'])}/{min(CALIBRATION_EPOCHS, limits['epochs'])} training passes, "
            f"{len(ledger['validation'])}/{CALIBRATION_VALIDATION_PASSES} validation passes (CPU only)"
        )
    lines.extend(job_line(job, state) for job in active)
    if not active:
        lines.append("  no active jobs")

    backlog = (
        f"Backlog  {len(queue)} queued (refill below {state['refill_below']}, "
        f"target {state['refill_to']})"
    )
    if state["seed_ideas"]:
        remaining = [
            idea_id
            for idea_id in state["seed_ideas"]
            if state["ideas"][idea_id]["status"] not in SEED_TERMINAL
        ]
        backlog += (
            f"; forced seeds {len(state['seed_ideas']) - len(remaining)}/"
            f"{len(state['seed_ideas'])} done, research waits for the rest"
            if remaining
            else "; forced seeds done"
        )
    lines.extend(["", backlog])
    for index, idea_id in enumerate(queue, 1):
        idea = state["ideas"][idea_id]
        lines.append(f"  {index}. {idea_id} [{idea['status']}]")
    claimed = state["experiments"].get(state["claimed"])
    lines.append(
        f"  next candidate: {claimed['idea']} ({claimed['stage']})"
        if claimed
        else "  next candidate: none"
    )

    lines.extend(["", "Score"])
    lines.extend(
        f"  {line}"
        for line in progress_lines(state, supervisor.root, supervisor.output)
    )
    completed = [exp for exp in state["experiments"].values() if exp.get("decision")]
    accepted = sum(exp["decision"] == "provisional_accept" for exp in completed)
    lines.extend(
        ["", f"Results  {len(completed)} decided, {accepted} provisionally accepted"]
    )
    if completed:
        lines.extend(result_rows(completed))

    usage = campaign_usage(state)
    totals = usage["total"]
    if totals.get("jobs"):
        lines.extend(["", "Cost (API-equivalent, not subscription quota)"])
        roles = sorted(
            usage["by_role_event"].items(),
            key=lambda item: -(item[1].get("recorded_cost_usd") or 0),
        )
        width = max(len(name) for name, _ in roles)
        for name, values in [*roles, ("total", totals)]:
            cost = (
                f"${values['recorded_cost_usd']:.2f}"
                if values.get("cost_known_jobs")
                else "n/a"
            )
            lines.append(f"  {name:<{width}}  {values['jobs']:>3} jobs  {cost:>8}")
        per_valid = usage["recorded_cost_per_valid_experiment_usd"]
        if per_valid is not None:
            lines.append(
                f"  {'per valid experiment':<{width}}  {'':>8}  {f'${per_valid:.2f}':>8}"
            )
        gaps = {
            "partial": totals.get("partial_jobs", 0),
            "unavailable": totals.get("unavailable_jobs", 0),
            "failed": totals.get("failed_jobs", 0),
        }
        if not usage["cost_coverage_complete"] or any(gaps.values()):
            lines.append(
                "  usage incomplete: "
                + ", ".join(f"{count} {name}" for name, count in gaps.items() if count)
            )
        lines.append(
            f"  tokens: {(totals.get('input', 0) + totals.get('cache_write', 0)) / 1e6:.2f}M new input, "
            f"{totals.get('cache_read', 0) / 1e6:.2f}M cached, "
            f"{totals.get('output', 0) / 1e6:.2f}M output"
        )
        reporting = usage["completion_reporting"]
        if reporting:
            counts = {
                "reported": reporting.get("reported", 0),
                "escalated": reporting.get("escalated", 0),
                "direct to orchestrator": reporting.get("orchestrator", 0),
                "pending": reporting.get("assigned", 0),
            }
            lines.append(
                "  completion reports: "
                + ", ".join(
                    f"{count} {name}" for name, count in counts.items() if count
                )
            )
    lines.extend(["", f"Artifacts: {supervisor.output}"])
    return "\n".join(lines)
