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
    metric_line,
    progress_lines,
    results_summary,
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


def status_summary(supervisor: Supervisor):
    current = status(supervisor)
    lines = [
        f"{current['campaign']} | {current['provider'] or 'no provider'} | {current['status'].upper()}"
    ]
    lines.append(
        f"Supervisor: {'alive' if current['supervisor_alive'] else 'not running'}"
    )
    if supervisor.state is None:
        return "\n".join(lines)
    state = supervisor.state
    if current["reason"]:
        lines.append(f"Reason: {current['reason']}")
    retry_after = state.get("retry_after", 0)
    if current["status"] == "running" and retry_after > time.time():
        remaining = int(retry_after - time.time())
        lines.append(
            f"Provider wait: no agent jobs until {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(retry_after))} "
            f"({remaining // 60}m {remaining % 60:02d}s left); training and comparison continue."
        )
    if (supervisor.output / "stop.json").exists() and current["status"] != "stopped":
        lines.append("Stop requested; waiting for the supervisor to finish stopping.")
    if current["status"] == "calibrating":
        path = supervisor.output / "baseline/triplet_accounting.json"
        ledger = read_json(path) if path.exists() else {"train": [], "validation": []}
        lines.append(
            f"Calibration: {len(ledger['train'])}/{state['limits']['epochs']} training passes; "
            f"{len(ledger['validation'])}/2 validation passes completed (CPU only)"
        )
    active = [job for job in state["jobs"].values() if not job["processed"]]
    for job in active:
        directory = Path(job["path"])
        elapsed = max(0, int(time.time() - job["created_at"]))
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
        lines.append(
            f"Job: {job['kind']} — {job_status}, elapsed {elapsed // 60}m {elapsed % 60:02d}s"
        )
    if not active:
        lines.append("Jobs: none active")
    queue = state["queue"]
    lines.append(
        f"Backlog: {len(queue)} ideas (refill below {state['refill_below']}, target {state['refill_to']})"
    )
    for index, idea_id in enumerate(queue[:3], 1):
        idea = state["ideas"][idea_id]
        lines.append(f"  {index}. {idea['title']} [{idea['status']}]")
    claimed = state["experiments"].get(state["claimed"])
    lines.append(
        f"Next candidate: {claimed['idea']} — {claimed['stage']}"
        if claimed
        else "Next candidate: none"
    )
    running = state["experiments"].get(state["running"])
    if running:
        lines.append(f"Experiment: {running['idea']} — {running['stage']}")
        fold = Path(running["directory"]) / "training/fold_0"
        path = fold / "triplet_accounting.json"
        if path.exists():
            ledger = read_json(path)
            lines.append(
                f"Training epochs completed: {len(ledger['train'])}/{state['limits']['epochs']}"
            )
        metrics = sorted((fold / "epochs").glob("metrics_epoch_*.json"))
        if metrics:
            try:
                score = read_json(metrics[-1])["macro_auc"]
                lines.append(f"Latest validation AUC: {score:.6f}")
            except (OSError, ValueError, KeyError):
                pass
    completed = [exp for exp in state["experiments"].values() if exp.get("decision")]
    accepted = sum(exp["decision"] == "provisional_accept" for exp in completed)
    lines.append(
        f"Results: {len(completed)} decided; {accepted} provisionally accepted"
    )
    if completed:
        latest = completed[-1]
        lines.append(f"Latest decision: {latest['idea']} — {decision_text(latest)}")
        lines.append(
            f"  Reference: {latest.get('reference', {}).get('id', 'unavailable')}"
        )
        lines.append(f"  {metric_line(latest)}")
    lines.extend(progress_lines(state, supervisor.root, supervisor.output))
    usage = campaign_usage(state)
    completion_reporting = usage["completion_reporting"]
    if completion_reporting:
        lines.append(
            "Completion reporting: "
            f"{completion_reporting.get('reported', 0)} reported; "
            f"{completion_reporting.get('escalated', 0)} escalated; "
            f"{completion_reporting.get('orchestrator', 0)} direct to orchestrator; "
            f"{completion_reporting.get('assigned', 0)} pending"
        )
    totals = usage["total"]
    if totals.get("jobs"):
        lines.append(
            f"Agent usage: {totals.get('jobs', 0)} jobs; "
            f"{totals.get('input', 0) + totals.get('cache_write', 0):,} new input, "
            f"{totals.get('cache_read', 0):,} cached input, {totals.get('output', 0):,} output tokens"
        )
        lines.append(
            f"Usage coverage: {totals.get('complete_jobs', 0)} complete, "
            f"{totals.get('partial_jobs', 0)} partial, {totals.get('unavailable_jobs', 0)} unavailable; "
            f"{totals.get('failed_jobs', 0)} failed jobs included"
        )
        if totals.get("cost_known_jobs"):
            lines.append(
                f"Recorded API cost equivalent: ${totals['recorded_cost_usd']:.2f} (not subscription quota)"
            )
        per_valid = usage["recorded_cost_per_valid_experiment_usd"]
        if per_valid is not None:
            coverage = (
                "complete coverage"
                if usage["cost_coverage_complete"]
                else "incomplete coverage"
            )
            lines.append(
                f"Recorded cost per valid experiment: ${per_valid:.2f} ({coverage})"
            )
        for name, values in usage["by_role_event"].items():
            cost = (
                f"${values['recorded_cost_usd']:.2f}"
                if values.get("cost_known_jobs")
                else "cost unavailable"
            )
            lines.append(
                f"  {name}: {values['jobs']} jobs, {cost}, {values.get('output', 0):,} output tokens"
            )
    seconds = state["limits"]["max_run_seconds"]
    lines.append(
        f"Budget: {seconds // 60}m {seconds % 60:02d}s per experiment; {state['limits']['epochs']} epochs; fold 0"
    )
    if (
        current["status"] == "running"
        and not active
        and not queue
        and not state["events"]
    ):
        lines.append(
            "Idle: backlog exhausted; waiting for a new research direction or resume."
        )
    lines.append(f"Artifacts: {supervisor.output}")
    return "\n".join(lines)


def validate_campaign(supervisor: Supervisor):
    recipe = load_config(supervisor.root / supervisor.campaign["recipe"], folds=[0])
    config = recipe.model_dump(mode="json")
    reference = supervisor.root / supervisor.campaign["reference_run"]
    limits = initial_limits(
        reference, config, supervisor.campaign["reference_equivalences"]
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
