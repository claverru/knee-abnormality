import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator
from pydantic_settings import CliImplicitFlag, CliPositionalArg, SettingsError

from knee_abnormality.autoresearch.campaign.reports import ResearchReport
from knee_abnormality.autoresearch.reporting.progress import (
    results_summary,
)
from knee_abnormality.autoresearch.reporting.status import status, status_summary
from knee_abnormality.autoresearch.runtime.gpu import gpu_uuid
from knee_abnormality.autoresearch.runtime.jobs import run_job
from knee_abnormality.autoresearch.runtime.providers import preflight
from knee_abnormality.autoresearch.runtime.storage import (
    lock,
    write_json,
)
from knee_abnormality.autoresearch.runtime.usage import campaign_usage
from knee_abnormality.autoresearch.scoring.diagnostics import backfill, compare
from knee_abnormality.autoresearch.supervisor import Supervisor, seed_files
from knee_abnormality.cli import CLISettings

ACTION_ARGUMENTS = {
    "start": ("provider",),
    "job": ("job_dir",),
    "calibrate": ("config", "destination"),
    "compare": ("config",),
    "import-ideas": ("ideas_file",),
}


RUNTIME_ACTIONS = {"job", "calibrate", "compare"}


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
    campaign: CliPositionalArg[str | None] = Field(
        default=None, pattern=r"^[a-z][a-z0-9-]*$"
    )
    provider: Literal["codex", "claude"] | None = None
    gpu: str = "0"
    abort: CliImplicitFlag[bool] = False
    root: Path = Path(__file__).resolve().parents[3]
    job_dir: Path | None = None
    config: Path | None = None
    destination: Path | None = None
    ideas_file: Path | None = None

    @model_validator(mode="after")
    def action_arguments(self):
        missing = [
            f"--{name.replace('_', '-')}"
            for name in ACTION_ARGUMENTS.get(self.action, ())
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"{self.action} requires {' and '.join(missing)}")
        if (self.campaign is None) != (self.action in RUNTIME_ACTIONS):
            raise ValueError(
                f"{self.action} "
                + ("takes no campaign" if self.campaign else "requires a campaign ID")
            )
        return self


def validate_campaign(supervisor: Supervisor):
    _, _, limits = supervisor.recipe_limits()
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
            run_job(args.job_dir)
            return
        if args.action == "calibrate":
            from knee_abnormality.autoresearch.scoring.accounting import calibrate

            calibrate(args.config, args.destination)
            return
        if args.action == "compare":
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
                preflight(args.provider, supervisor.root)
                supervisor.initialize(args.provider, gpu_uuid(args.gpu))
                supervisor.launch()
            elif args.action == "import-ideas":
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
                    supervisor.launch()
            else:
                if supervisor.state is None:
                    raise ValueError("Campaign has not been started")
                write_json(supervisor.output / "stop.json", {"abort": args.abort})
                if not current["supervisor_alive"]:
                    supervisor.launch()
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
