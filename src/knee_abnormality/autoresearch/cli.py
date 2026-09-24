import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import CliImplicitFlag, CliPositionalArg, SettingsError

from knee_abnormality.autoresearch.campaign.gates import initial_limits, validate_config
from knee_abnormality.autoresearch.campaign.reports import ResearchReport
from knee_abnormality.autoresearch.reporting.progress import (
    results_summary,
)
from knee_abnormality.autoresearch.reporting.status import status, status_summary
from knee_abnormality.autoresearch.runtime.isolation import environment, sandbox
from knee_abnormality.autoresearch.runtime.providers import auth_environment
from knee_abnormality.autoresearch.runtime.storage import (
    lock,
    process_identity,
    write_json,
)
from knee_abnormality.autoresearch.runtime.usage import campaign_usage
from knee_abnormality.autoresearch.supervisor import Supervisor, seed_files
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
    seed_order = supervisor.campaign["backlog"]["seed_order"]
    if seed_order == "forced" and any(idea.dependencies for idea in ideas):
        raise ValueError(
            "Forced seeds cannot declare dependencies; their file order sets the sequence"
        )
    print(
        json.dumps(
            {
                "campaign": supervisor.campaign["id"],
                "valid": True,
                "seed_ideas": len(ideas),
                "seed_order": seed_order,
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
            from knee_abnormality.autoresearch.runtime.jobs import run_job

            if args.job_dir is None:
                raise ValueError("job requires --job-dir")
            run_job(args.job_dir)
            return
        if args.action == "calibrate":
            from knee_abnormality.autoresearch.scoring.accounting import calibrate

            if args.config is None or args.destination is None:
                raise ValueError("calibrate requires --config and --destination")
            calibrate(args.config, args.destination)
            return
        if args.action == "compare":
            from knee_abnormality.autoresearch.scoring.diagnostics import compare

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
            from knee_abnormality.autoresearch.scoring.diagnostics import backfill

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
