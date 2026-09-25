import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, StringConstraints
from pydantic_settings import CliApp, CliPositionalArg, CliSubCommand

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
from knee_abnormality.cli import CLISettings, command, describe

ROOT = Path(__file__).resolve().parents[3]
CampaignId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]*$")]


class RootCommand(BaseModel):
    root: Path = Field(default=ROOT, description="Repository root")


class CampaignCommand(RootCommand):
    campaign: CliPositionalArg[CampaignId] = Field(
        description="Campaign ID under autoresearch/campaigns"
    )

    def supervisor(self) -> Supervisor:
        return Supervisor(self.root.resolve(), self.campaign)


def control(name: str, command: CampaignCommand, action) -> None:
    supervisor = command.supervisor()
    with lock(supervisor.output / "control.lock"):
        supervisor = action(supervisor, status(supervisor)) or supervisor
    print(json.dumps({"requested": name, **status(supervisor)}, indent=2))


def started(supervisor: Supervisor) -> Supervisor:
    if supervisor.state is None:
        raise ValueError("Campaign has not been started")
    return supervisor


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


class Start(CampaignCommand):
    model_config = describe("Seal the campaign policy and launch its supervisor")

    provider: Literal["codex", "claude"] = Field(description="Agent subscription")
    gpu: str = Field(default="0", description="GPU index or UUID for training")

    def cli_cmd(self) -> None:
        def action(supervisor, current):
            preflight(self.provider, supervisor.root)
            supervisor.initialize(self.provider, gpu_uuid(self.gpu))
            supervisor.launch()

        control("start", self, action)


class Validate(CampaignCommand):
    model_config = describe("Check the campaign files and seed ideas without starting")

    def cli_cmd(self) -> None:
        validate_campaign(self.supervisor())


class Status(CampaignCommand):
    model_config = describe("Summarize the campaign state")

    def cli_cmd(self) -> None:
        print(status_summary(self.supervisor()))


class Results(CampaignCommand):
    model_config = describe("Summarize macro and per-label experiment results")

    def cli_cmd(self) -> None:
        supervisor = self.supervisor()
        print(
            results_summary(
                supervisor.state, supervisor.root, supervisor.output, self.campaign
            )
        )


class BackfillLabels(CampaignCommand):
    model_config = describe("Backfill per-label diagnostics for finished experiments")

    def cli_cmd(self) -> None:
        supervisor = started(self.supervisor())
        print(json.dumps(backfill(supervisor.state, supervisor.root), indent=2))


class Usage(CampaignCommand):
    model_config = describe("Account for the campaign's subscription usage")

    def cli_cmd(self) -> None:
        supervisor = self.supervisor()
        print(
            json.dumps(
                campaign_usage(supervisor.state)
                if supervisor.state
                else {"status": "not_started"},
                indent=2,
            )
        )


class Stop(CampaignCommand):
    model_config = describe("Ask the supervisor to stop after the current job")

    abort: bool = Field(default=False, description="Cancel running jobs instead")

    def cli_cmd(self) -> None:
        def action(supervisor, current):
            started(supervisor)
            write_json(supervisor.output / "stop.json", {"abort": self.abort})
            if not current["supervisor_alive"]:
                supervisor.launch()

        control("stop", self, action)


class Resume(CampaignCommand):
    model_config = describe("Resume a stopped campaign with its recorded provider")

    def cli_cmd(self) -> None:
        def action(supervisor, current):
            started(supervisor)
            preflight(supervisor.state["provider"], supervisor.root)
            write_json(supervisor.output / "resume.json", {})
            if not current["supervisor_alive"]:
                supervisor.launch()

        control("resume", self, action)


class ImportIdeas(CampaignCommand):
    model_config = describe("Add ideas to the backlog of a stopped campaign")

    ideas_file: Path = Field(description="Research report YAML with the new ideas")

    def cli_cmd(self) -> None:
        def action(supervisor, current):
            with lock(supervisor.output / "supervisor.lock"):
                supervisor = self.supervisor()
                if supervisor.state is None or supervisor.state["status"] != "stopped":
                    raise ValueError(
                        "Stop the campaign and wait for STOPPED before importing ideas"
                    )
                if any(
                    not job["processed"] for job in supervisor.state["jobs"].values()
                ):
                    raise ValueError("Cannot import while jobs remain unprocessed")
                supervisor.import_ideas(self.ideas_file)
            return supervisor

        control("import-ideas", self, action)


class Supervise(CampaignCommand):
    model_config = describe("Run the supervisor loop in the foreground")

    def cli_cmd(self) -> None:
        self.supervisor().run()


class Job(BaseModel):
    model_config = describe("Run one queued supervisor job")

    job_dir: Path = Field(description="Job directory written by the supervisor")

    def cli_cmd(self) -> None:
        run_job(self.job_dir)


class Calibrate(BaseModel):
    model_config = describe("Measure the reference recipe's runtime limits")

    config: Path = Field(description="Experiment YAML to calibrate")
    destination: Path = Field(description="Output directory for the calibration run")

    def cli_cmd(self) -> None:
        from knee_abnormality.autoresearch.scoring.accounting import calibrate

        calibrate(self.config, self.destination)


class Compare(RootCommand):
    model_config = describe("Compare an experiment with its reference run")

    config: Path = Field(
        description="Comparison command JSON written by the supervisor"
    )

    def cli_cmd(self) -> None:
        compare(self.config, self.root.resolve())


class AutoresearchCLI(CLISettings):
    model_config = describe(
        "Validate, run, stop and inspect local autoresearch campaigns"
    )

    start: CliSubCommand[Start] = command(Start)
    validate_campaign: CliSubCommand[Validate] = command(Validate, alias="validate")
    status: CliSubCommand[Status] = command(Status)
    results: CliSubCommand[Results] = command(Results)
    backfill_labels: CliSubCommand[BackfillLabels] = command(BackfillLabels)
    usage: CliSubCommand[Usage] = command(Usage)
    stop: CliSubCommand[Stop] = command(Stop)
    resume: CliSubCommand[Resume] = command(Resume)
    import_ideas: CliSubCommand[ImportIdeas] = command(ImportIdeas)
    supervise: CliSubCommand[Supervise] = command(Supervise)
    job: CliSubCommand[Job] = command(Job)
    calibrate: CliSubCommand[Calibrate] = command(Calibrate)
    compare: CliSubCommand[Compare] = command(Compare)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main():
    CliApp.run(AutoresearchCLI)


if __name__ == "__main__":
    main()
