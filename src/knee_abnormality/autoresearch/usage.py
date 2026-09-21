import json
from collections import Counter, defaultdict
from pathlib import Path

from knee_abnormality.autoresearch.models import REPORTS
from knee_abnormality.autoresearch.storage import read_json, write_json

TOKEN_FIELDS = ("input", "cache_write", "cache_read", "output", "thinking")
USAGE_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError)


def events(path):
    if not path.is_file():
        return
    with path.open() as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                yield value


def claude_tokens(usage, model=False):
    keys = (
        (
            "inputTokens",
            "cacheCreationInputTokens",
            "cacheReadInputTokens",
            "outputTokens",
            "thinkingTokens",
        )
        if model
        else (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
            "thinking_tokens",
        )
    )
    result = {
        field: usage.get(key, 0) for field, key in zip(TOKEN_FIELDS, keys, strict=True)
    }
    if not model:
        result["thinking"] = (usage.get("output_tokens_details") or {}).get(
            "thinking_tokens", 0
        )
    return result


def job_usage(provider, job):
    assignment = read_json(job / "assignment.json")
    context = assignment["context"]
    task = context["assignment"]
    total = Counter({key: 0 for key in TOKEN_FIELDS})
    models = {}
    completed = 0
    final = None
    messages = {}
    for event in events(job / "stdout.log"):
        if provider == "claude":
            if event.get("type") == "result":
                final = event
            elif event.get("type") == "assistant":
                message = event.get("message", {})
                if message.get("id") and message.get("usage"):
                    tokens = claude_tokens(message["usage"])
                    previous = messages.get(message["id"], {})
                    messages[message["id"]] = {
                        key: max(value, previous.get(key, 0))
                        for key, value in tokens.items()
                    }
        elif event.get("type") == "turn.completed" and event.get("usage"):
            usage = event["usage"]
            cached = usage.get("cached_input_tokens", 0)
            total.update(
                {
                    "input": usage.get("input_tokens", 0) - cached,
                    "cache_read": cached,
                    "output": usage.get("output_tokens", 0),
                }
            )
            completed += 1
    if final is not None:
        for model, value in final.get("modelUsage", {}).items():
            tokens = claude_tokens(value, model=True)
            models[model] = {"tokens": tokens, "cost_usd": value.get("costUSD")}
            total.update(tokens)
        if not models:
            total.update(claude_tokens(final.get("usage", {})))
        measurement = "complete" if "usage" in final else "unavailable"
    elif provider == "claude":
        for value in messages.values():
            total.update(value)
        measurement = "partial" if messages else "unavailable"
    else:
        receipt_path = job / "receipt.json"
        receipt = read_json(receipt_path) if receipt_path.is_file() else {}
        measurement = (
            "complete"
            if completed and receipt.get("exit_code") == 0 and not receipt.get("error")
            else "partial"
            if completed
            else "unavailable"
        )
    report_path = job / "report.json"
    report_bytes = report_path.stat().st_size if report_path.is_file() else None
    if report_bytes is None and final is not None and "structured_output" in final:
        report_bytes = len(json.dumps(final["structured_output"]).encode())
    return {
        "provider": provider,
        "role": context["role"],
        **assignment["profile"],
        "event": task.get("event", {}).get("kind"),
        "experiment": task.get("experiment", {}).get("id")
        or task.get("event", {}).get("experiment"),
        "measurement": measurement,
        "provider_error": bool(final and final.get("is_error")),
        "tokens": dict(total),
        "models": models,
        "cost_usd": final.get("total_cost_usd") if final is not None else None,
        "turns": final.get("num_turns")
        if final is not None
        else completed
        if provider == "codex"
        else None,
        "prompt_bytes": (job / "stdin.txt").stat().st_size,
        "report_bytes": report_bytes,
        "report_available": report_path.is_file(),
    }


def capture_usage(provider, job):
    try:
        value = job_usage(provider, job)
    except USAGE_ERRORS as error:
        value = {
            "provider": provider,
            "measurement": "unavailable",
            "error": f"{type(error).__name__}: {error}",
        }
    write_json(job / "usage.json", value)
    return value


def campaign_usage(state):
    groups = defaultdict(Counter)
    experiments = defaultdict(Counter)
    total = Counter()
    records = []
    for job in state["jobs"].values():
        if job["kind"] not in REPORTS:
            continue
        directory = Path(job["path"])
        if not (directory / "assignment.json").is_file():
            continue
        try:
            value = job_usage(state["provider"], directory)
        except USAGE_ERRORS as error:
            total.update({"jobs": 1, "unavailable_jobs": 1, "usage_errors": 1})
            records.append(
                {
                    "job": job["id"],
                    "role": job["kind"],
                    "measurement": "unavailable",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        receipt_path = directory / "receipt.json"
        receipt = read_json(receipt_path) if receipt_path.is_file() else {}
        failed = bool(
            value["provider_error"]
            or receipt
            and (receipt.get("exit_code") != 0 or receipt.get("error"))
        )
        rejected_report = bool(
            job["processed"]
            and receipt
            and not failed
            and not value["report_available"]
        )
        counters = Counter(
            {
                **value["tokens"],
                "jobs": 1,
                "failed_jobs": int(failed),
                "rejected_reports": int(rejected_report),
                "cancelled_jobs": int(bool(receipt.get("cancelled"))),
                f"{value['measurement']}_jobs": 1,
                "cost_known_jobs": int(value["cost_usd"] is not None),
                "recorded_cost_usd": value["cost_usd"] or 0,
                "prompt_bytes": value["prompt_bytes"],
                "report_bytes": value["report_bytes"] or 0,
                "turns": value["turns"] or 0,
            }
        )
        total.update(counters)
        groups["/".join(filter(None, (value["role"], value["event"])))].update(counters)
        if value["experiment"]:
            experiments[value["experiment"]].update(counters)
        records.append(
            {
                "job": job["id"],
                **value,
                "failed": failed,
                "rejected_report": rejected_report,
            }
        )
    valid = sum(
        exp.get("decision") in {"provisional_accept", "reject", "inconclusive"}
        for exp in state["experiments"].values()
    )
    experiment_records = {
        exp_id: {
            **dict(experiments[exp_id]),
            "idea": exp["idea"],
            "stage": exp["stage"],
            "decision": exp.get("decision"),
            "repair_of": exp["repair_of"],
            "repair_count": exp["repair_count"],
            "preparation_seconds": exp.get("preparation_seconds"),
            "training_seconds": (exp.get("training_receipt") or {}).get("seconds"),
            "compute_seconds": exp.get("measured_compute_seconds"),
            "completion_reporting": exp.get("completion_reporting"),
        }
        for exp_id, exp in state["experiments"].items()
    }
    return {
        "total": dict(total),
        "by_role_event": {key: dict(value) for key, value in sorted(groups.items())},
        "by_experiment": experiment_records,
        "jobs": records,
        "valid_completed_experiments": valid,
        "completion_reporting": dict(
            Counter(
                exp["completion_reporting"]["status"]
                for exp in state["experiments"].values()
                if exp.get("completion_reporting")
            )
        ),
        "recorded_cost_per_valid_experiment_usd": total["recorded_cost_usd"] / valid
        if valid and total["cost_known_jobs"]
        else None,
        "cost_coverage_complete": total["jobs"] > 0
        and total["jobs"] == total["cost_known_jobs"]
        and not total["partial_jobs"]
        and not total["unavailable_jobs"],
    }
