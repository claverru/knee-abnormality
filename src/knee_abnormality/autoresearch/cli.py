import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, StringConstraints, model_validator
from pydantic_settings import CliApp, CliPositionalArg, CliSubCommand

from knee_abnormality.autoresearch.campaign.reports import IdeaReport
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
from knee_abnormality.autoresearch.supervisor import Supervisor
from knee_abnormality.autoresearch.topics.loop import TopicLoop
from knee_abnormality.autoresearch.topics.status import loop_status, loop_summary
from knee_abnormality.cli import CLISettings, command, describe

ROOT = Path(__file__).resolve().parents[3]
CampaignId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]*$")]
Provider = Literal["codex", "claude"]


class RootCommand(BaseModel):
    root: Path = Field(default=ROOT, description="Repository root")


class GpuChoice(BaseModel):
    gpu: str = Field(default="0", description="GPU index or UUID for training")


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


class Start(CampaignCommand, GpuChoice):
    model_config = describe("Seal the campaign policy and launch its supervisor")

    provider: Provider = Field(description="Agent subscription")

    def cli_cmd(self) -> None:
        def action(supervisor, current):
            preflight(self.provider, supervisor.root)
            supervisor.initialize(self.provider, gpu_uuid(self.gpu))
            supervisor.launch()

        control("start", self, action)


class Validate(CampaignCommand):
    model_config = describe("Check the campaign files and seed ideas without starting")

    def cli_cmd(self) -> None:
        print(json.dumps(self.supervisor().validate(), indent=2))


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


def deliver(name: str, command: CampaignCommand, request) -> None:
    def action(supervisor, current):
        request(started(supervisor))
        if not current["supervisor_alive"]:
            with lock(supervisor.output / "supervisor.lock"):
                supervisor = command.supervisor()
                supervisor.drain_inbox()
        return supervisor

    control(name, command, action)


class ImportIdeas(CampaignCommand):
    model_config = describe(
        "Add ideas to a campaign; a running supervisor applies them at its next safe point"
    )

    ideas_file: Path = Field(description="Research report YAML with the new ideas")
    forced: bool = Field(
        default=False,
        description="Queue the ideas in file order ahead of every other idea except the reserved head, without orchestrator admission",
    )

    def cli_cmd(self) -> None:
        report = IdeaReport.model_validate(yaml.safe_load(self.ideas_file.read_text()))
        deliver(
            "import-ideas",
            self,
            lambda supervisor: supervisor.request_ideas(
                report, self.forced, self.ideas_file
            ),
        )


class Adopt(CampaignCommand):
    model_config = describe(
        "Make a completed experiment the incumbent by user decision, applied when no training runs"
    )

    experiment: str = Field(description="Experiment ID to adopt")
    rationale: str = Field(description="The user's reason for the adoption")

    def cli_cmd(self) -> None:
        deliver(
            "adopt",
            self,
            lambda supervisor: supervisor.request_adoption(
                self.experiment, self.rationale
            ),
        )


class Supervise(CampaignCommand):
    model_config = describe("Run the supervisor loop in the foreground")

    def cli_cmd(self) -> None:
        self.supervisor().run()


class LoopCommand(RootCommand):
    def loop(self) -> TopicLoop:
        return TopicLoop(self.root.resolve())


def loop_control(name: str, command: LoopCommand, action) -> None:
    loop = command.loop()
    loop.output.mkdir(parents=True, exist_ok=True)
    with lock(loop.output / "control.lock"):
        extra = action(loop, loop.alive()) or {}
    print(
        json.dumps(
            {"requested": name, **extra, **loop_status(command.loop())}, indent=2
        )
    )


def loop_started(loop: TopicLoop) -> TopicLoop:
    if loop.state is None:
        raise ValueError("The topic loop has not been started")
    return loop


class LoopStart(LoopCommand, GpuChoice):
    model_config = describe(
        "Start the topic loop: it runs the first topic and rotates topics until stopped"
    )

    provider: Provider = Field(description="Agent subscription for every topic")
    campaign: CampaignId | None = Field(
        default=None,
        description="Existing campaign to run as the first topic, started or resumed if needed",
    )
    reference_run: Path | None = Field(
        default=None,
        description="Starting reference run on the main recipe; the first topic comes from the queue or a researcher draft",
    )

    @model_validator(mode="after")
    def one_start(self):
        if (self.campaign is None) == (self.reference_run is None):
            raise ValueError("Give exactly one of --campaign and --reference-run")
        return self

    def cli_cmd(self) -> None:
        def action(loop, running):
            preflight(self.provider, loop.root)
            loop.initialize(
                self.provider, gpu_uuid(self.gpu), self.campaign, self.reference_run
            )
            loop.launch()

        loop_control("loop-start", self, action)


class LoopStatus(LoopCommand):
    model_config = describe(
        "Summarize the topic loop: current topic, streak, rotation, topics and queue"
    )

    def cli_cmd(self) -> None:
        print(loop_summary(self.loop()))


class LoopStop(LoopCommand):
    model_config = describe(
        "Stop the topic loop and its topics after their current jobs"
    )

    abort: bool = Field(default=False, description="Cancel running training too")

    def cli_cmd(self) -> None:
        def action(loop, running):
            loop_started(loop)
            write_json(loop.output / "stop.json", {"abort": self.abort})
            if not running:
                loop.launch()

        loop_control("loop-stop", self, action)


class LoopResume(LoopCommand):
    model_config = describe(
        "Resume a stopped topic loop, the topics it stopped and the watchdog that the stop ended"
    )

    def cli_cmd(self) -> None:
        from knee_abnormality.autoresearch.watchdog import resume_loop_watchdog

        def action(loop, running):
            preflight(loop_started(loop).state["provider"], loop.root)
            write_json(loop.output / "resume.json", {})
            if not running:
                loop.launch()
            try:
                return {"watchdog": resume_loop_watchdog(loop.root)}
            except (OSError, ValueError, RuntimeError) as error:
                return {"watchdog": f"not restarted: {error}"}

        loop_control("loop-resume", self, action)


class LoopSupervise(LoopCommand):
    model_config = describe("Run the topic loop in the foreground")

    def cli_cmd(self) -> None:
        loop_started(self.loop()).run()


class Job(BaseModel):
    model_config = describe("Run one queued supervisor job")

    job_dir: Path = Field(description="Job directory written by the supervisor")

    def cli_cmd(self) -> None:
        run_job(self.job_dir)


class Calibrate(BaseModel):
    model_config = describe(
        "Replay a recipe's data pipeline on CPU to measure its triplet and visit counts"
    )

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
        "Validate, run, stop and inspect local autoresearch campaigns and the topic loop"
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
    adopt: CliSubCommand[Adopt] = command(Adopt)
    supervise: CliSubCommand[Supervise] = command(Supervise)
    loop_start: CliSubCommand[LoopStart] = command(LoopStart)
    loop_status: CliSubCommand[LoopStatus] = command(LoopStatus)
    loop_stop: CliSubCommand[LoopStop] = command(LoopStop)
    loop_resume: CliSubCommand[LoopResume] = command(LoopResume)
    loop_supervise: CliSubCommand[LoopSupervise] = command(LoopSupervise)
    job: CliSubCommand[Job] = command(Job)
    calibrate: CliSubCommand[Calibrate] = command(Calibrate)
    compare: CliSubCommand[Compare] = command(Compare)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main():
    CliApp.run(AutoresearchCLI)


if __name__ == "__main__":
    main()
