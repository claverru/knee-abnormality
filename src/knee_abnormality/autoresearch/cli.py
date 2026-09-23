import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import CliImplicitFlag, CliPositionalArg, SettingsError

from knee_abnormality.autoresearch.gates import initial_limits, validate_config
from knee_abnormality.autoresearch.isolation import environment, sandbox
from knee_abnormality.autoresearch.models import ResearchReport
from knee_abnormality.autoresearch.progress import (
    decision_text,
    difference,
    interval_text,
    progress_lines,
    results_summary,
    score_text,
)
from knee_abnormality.autoresearch.providers import auth_environment
from knee_abnormality.autoresearch.storage import (
    alive,
    lock,
    process_identity,
    read_json,
    write_json,
)
from knee_abnormality.autoresearch.supervisor import Supervisor, seed_files
from knee_abnormality.autoresearch.usage import campaign_usage
from knee_abnormality.cli import CLISettings
from knee_abnormality.config import load_config


class AutoresearchCLI(CLISettings):
    action: CliPositionalArg[
        Literal[
            "start",
            "validate",
            "status",
            "results",
            "backfill-labels",
            "usage",
            "stop",
            "resume",
            "import-ideas",
            "supervise",
            "job",
            "calibrate",
            "compare",
        ]
    ]
    campaign: CliPositionalArg[str] = Field(
        default="sampling", pattern=r"^[a-z][a-z0-9-]*$"
    )
    provider: Literal["codex", "claude"] | None = None
    gpu: str = "0"
    abort: CliImplicitFlag[bool] = False
    root: Path = Path(__file__).resolve().parents[3]
    job_dir: Path | None = None
    config: Path | None = None
    destination: Path | None = None
    ideas_file: Path | None = None


def preflight(provider: str, root: Path):
    for executable in (provider, "bwrap", "nvidia-smi", "rtk"):
        if shutil.which(executable) is None:
            raise ValueError(f"Required command is unavailable: {executable}")
    command = (
        ["codex", "login", "status"]
        if provider == "codex"
        else ["claude", "auth", "status"]
    )
    authentication_environment = auth_environment()
    if provider == "claude":
        authentication_environment["CLAUDE_CONFIG_DIR"] = str(Path.home() / ".claude")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        env=authentication_environment,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"{provider} subscription login is unavailable; log in with its CLI first"
        )
    if provider == "codex":
        if "using ChatGPT" not in result.stdout + result.stderr:
            raise ValueError("Codex must use ChatGPT subscription authentication")
    else:
        info = json.loads(result.stdout)
        if (
            not info.get("loggedIn")
            or info.get("authMethod") != "claude.ai"
            or info.get("apiProvider") != "firstParty"
        ):
            raise ValueError("Claude must use its first-party subscription login")
    result = subprocess.run(
        sandbox(["true"], root, []),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"Required Linux isolation is unavailable: {result.stderr.strip()}"
        )


def launch(supervisor: Supervisor):
    output = supervisor.output
    with (output / "supervisor.log").open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "knee_abnormality.autoresearch.cli",
                "supervise",
                supervisor.state["campaign"],
                "--root",
                str(supervisor.root),
            ],
            cwd=supervisor.root,
            env=environment(supervisor.root),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    write_json(output / "supervisor.json", process_identity(process.pid))


def status(supervisor: Supervisor):
    state = read_json(supervisor.state_path) if supervisor.state_path.exists() else None
    owner_path = supervisor.output / "supervisor.json"
    owner = read_json(owner_path) if owner_path.exists() else None
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
    payload = job["payload"] if isinstance(job["payload"], dict) else {}
    exp = state["experiments"].get(payload.get("experiment"))
    detail = exp["idea"] if exp else ""
    if exp and job["kind"] == "training":
        fold = Path(exp["directory"]) / "training/fold_0"
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
        evidence = exp.get("evidence", {})
        scores = evidence.get("macro_auc", {})
        delta = difference(scores.get("candidate"), scores.get("reference"))
        bounds = (evidence.get("adoption_bootstrap") or {}).get("delta_percentile_80")
        rows.append(
            f"  {exp['idea']:<{width}}  {score_text(delta, True):>9}  "
            f"{interval_text(bounds):<24}  {decision_text(exp)}"
        )
    return rows


def status_summary(supervisor: Supervisor):
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
        f"{limits['epochs']} epochs · fold 0"
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
            f"  calibration  {len(ledger['train'])}/{limits['epochs']} training passes, "
            f"{len(ledger['validation'])}/2 validation passes (CPU only)"
        )
    lines.extend(job_line(job, state) for job in active)
    if not active:
        lines.append("  no active jobs")

    backlog = (
        f"Backlog  {len(queue)} queued (refill below {state['refill_below']}, "
        f"target {state['refill_to']})"
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


def validate_campaign(supervisor: Supervisor):
    recipe = load_config(supervisor.root / supervisor.campaign["recipe"], folds=[0])
    config = recipe.model_dump(mode="json")
    reference = supervisor.root / supervisor.campaign["reference_run"]
    limits = initial_limits(
        reference,
        config,
        supervisor.campaign["reference_equivalences"],
        supervisor.campaign["reference_config_exceptions"],
        supervisor.campaign["reference_identity_exceptions"],
    )
    validate_config(config, config, supervisor.policy)
    for name in supervisor.policy.editable_files:
        if not (supervisor.root / name).is_file():
            raise ValueError(f"Missing editable file: {name}")
    contract = supervisor.docs / "contract.md"
    if not contract.read_text().strip():
        raise ValueError("Campaign contract is empty")
    ideas = [
        idea
        for path in seed_files(supervisor.docs)
        for idea in ResearchReport.model_validate(
            yaml.safe_load(path.read_text())
        ).ideas
    ]
    if len({idea.id for idea in ideas}) != len(ideas):
        raise ValueError("Seed idea IDs must be unique")
    print(
        json.dumps(
            {
                "campaign": supervisor.campaign["id"],
                "valid": True,
                "seed_ideas": len(ideas),
                "limits_from_reference": limits,
                "policy_sealed": supervisor.state is not None,
            },
            indent=2,
        )
    )


def main():
    try:
        args = AutoresearchCLI()
        if args.action == "job":
            from knee_abnormality.autoresearch.jobs import run_job

            if args.job_dir is None:
                raise ValueError("job requires --job-dir")
            run_job(args.job_dir)
            return
        if args.action == "calibrate":
            from knee_abnormality.autoresearch.accounting import calibrate

            if args.config is None or args.destination is None:
                raise ValueError("calibrate requires --config and --destination")
            calibrate(args.config, args.destination)
            return
        if args.action == "compare":
            from knee_abnormality.autoresearch.diagnostics import compare

            if args.config is None:
                raise ValueError(
                    "compare requires --config with the frozen comparison command"
                )
            compare(args.config, args.root.resolve())
            return
        supervisor = Supervisor(args.root.resolve(), args.campaign)
        if args.action == "validate":
            validate_campaign(supervisor)
            return
        if args.action == "supervise":
            supervisor.run()
            return
        if args.action == "status":
            print(status_summary(supervisor))
            return
        if args.action == "results":
            print(
                results_summary(
                    supervisor.state, supervisor.root, supervisor.output, args.campaign
                )
            )
            return
        if args.action == "backfill-labels":
            from knee_abnormality.autoresearch.diagnostics import backfill

            if supervisor.state is None:
                raise ValueError("Campaign has not been started")
            print(json.dumps(backfill(supervisor.state, supervisor.root), indent=2))
            return
        if args.action == "usage":
            print(
                json.dumps(
                    campaign_usage(supervisor.state)
                    if supervisor.state
                    else {"status": "not_started"},
                    indent=2,
                )
            )
            return
        with lock(supervisor.output / "control.lock"):
            current = status(supervisor)
            if args.action == "start":
                if args.provider is None:
                    raise ValueError(
                        "start requires --provider codex or --provider claude"
                    )
                preflight(args.provider, supervisor.root)
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        args.gpu,
                        "--query-gpu=uuid",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=True,
                )
                gpu = result.stdout.strip()
                if not gpu.startswith("GPU-") or "\n" in gpu:
                    raise ValueError("Select exactly one local GPU")
                supervisor.initialize(args.provider, gpu)
                launch(supervisor)
            elif args.action == "import-ideas":
                if args.ideas_file is None:
                    raise ValueError("import-ideas requires --ideas-file")
                with lock(supervisor.output / "supervisor.lock"):
                    supervisor = Supervisor(args.root.resolve(), args.campaign)
                    if (
                        supervisor.state is None
                        or supervisor.state["status"] != "stopped"
                    ):
                        raise ValueError(
                            "Stop the campaign and wait for STOPPED before importing ideas"
                        )
                    if any(
                        not job["processed"]
                        for job in supervisor.state["jobs"].values()
                    ):
                        raise ValueError("Cannot import while jobs remain unprocessed")
                    supervisor.import_ideas(args.ideas_file)
            elif args.action == "resume":
                if supervisor.state is None:
                    raise ValueError("Campaign has not been started")
                if (
                    args.provider is not None
                    and args.provider != supervisor.state["provider"]
                ):
                    raise ValueError("Resume uses the campaign's recorded provider")
                preflight(supervisor.state["provider"], supervisor.root)
                write_json(supervisor.output / "resume.json", {})
                if not current["supervisor_alive"]:
                    launch(supervisor)
            else:
                if supervisor.state is None:
                    raise ValueError("Campaign has not been started")
                write_json(supervisor.output / "stop.json", {"abort": args.abort})
                if not current["supervisor_alive"]:
                    launch(supervisor)
        print(json.dumps({"requested": args.action, **status(supervisor)}, indent=2))
    except (
        OSError,
        ValueError,
        KeyError,
        SettingsError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Autoresearch error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
