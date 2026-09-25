import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliSubCommand

from knee_abnormality.cli import CLISettings, command, describe
from knee_abnormality.inference.bundle import verify_bundle
from knee_abnormality.kaggle import SubmissionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _unit(record: Path) -> str:
    identity = hashlib.sha256(str(record).encode()).hexdigest()[:16]
    return f"knee-submit-{identity}.service"


def _log(record: Path) -> Path:
    return record.with_name(f"{record.stem}-runner.log")


def start(request: SubmissionRequest) -> dict:
    record = request.record.expanduser().resolve()
    bundle = request.bundle.expanduser().resolve()
    manifest = verify_bundle(bundle)
    if not manifest.get("ready"):
        raise ValueError(f"Bundle is not ready: {manifest.get('unmet_prerequisites')}")
    log = _log(record)
    for path in (record, log):
        if path.exists():
            raise FileExistsError(f"Choose a new submission record: {path} exists")
    record.parent.mkdir(parents=True, exist_ok=True)
    unit = _unit(record)
    subprocess.run(
        [
            "systemd-run",
            "--user",
            f"--unit={unit}",
            "--collect",
            f"--working-directory={PROJECT_ROOT}",
            f"--property=StandardOutput=append:{log}",
            f"--property=StandardError=append:{log}",
            sys.executable,
            "-u",
            "-m",
            "scripts.pipeline.submit",
            *CliApp.serialize(
                request.model_copy(update={"bundle": bundle, "record": record})
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"state": "launched", "unit": unit, "record": str(record), "log": str(log)}


def status(record: Path) -> dict:
    if record.is_file():
        value = json.loads(record.read_text())
    elif _log(record).is_file():
        value = {"state": "starting", "detail": "Read runner log if startup fails"}
    else:
        raise FileNotFoundError(f"No submission record or runner log for {record}")
    submission = value.get("submission") or {}
    history = value.get("history", [])
    terminal = value.get("state") in {
        "submission_complete",
        "submission_pending",
        "failed",
        "superseded_before_notebook",
    }
    end = (
        datetime.fromisoformat(value["updated_at"])
        if terminal and value.get("updated_at")
        else datetime.now(UTC)
    )
    stages = []
    for index, event in enumerate(history):
        next_at = (
            datetime.fromisoformat(history[index + 1]["at"])
            if index + 1 < len(history)
            else end
        )
        stages.append(
            {
                **event,
                "seconds": round(
                    (next_at - datetime.fromisoformat(event["at"])).total_seconds(), 1
                ),
            }
        )
    return {
        "record": str(record),
        "log": str(_log(record)),
        "state": value.get("state"),
        "updated_at": value.get("updated_at"),
        "notebook": value.get("notebook"),
        "notebook_version": value.get("notebook_version"),
        "description": value.get("description"),
        "public_score": submission.get("publicScore", submission.get("public_score")),
        "submission": submission,
        "stages": stages,
        "error": value.get("error"),
        "detail": value.get("detail"),
    }


class JobCommand(BaseModel):
    record: Path = Field(
        description="Submission record JSON; the runner log is written beside it"
    )


class Start(SubmissionRequest):
    model_config = describe(
        "Launch scripts.pipeline.submit as a detached systemd user service whose log "
        "is written beside the record"
    )

    def cli_cmd(self) -> None:
        print(json.dumps(start(self), indent=2))


class Status(JobCommand):
    model_config = describe("Show the submission state and stage durations")

    def cli_cmd(self) -> None:
        print(json.dumps(status(self.record.expanduser().resolve()), indent=2))


class Logs(JobCommand):
    model_config = describe("Print the end of the runner log")

    lines: int = Field(default=60, gt=0, description="Log lines to print")
    follow: bool = Field(default=False, description="Keep printing new lines")

    def cli_cmd(self) -> None:
        tail = ["tail", "-n", str(self.lines), *(["-F"] if self.follow else [])]
        log = _log(self.record.expanduser().resolve())
        subprocess.run([*tail, "--", str(log)], check=True)


class SubmissionJobCLI(CLISettings):
    model_config = describe(
        "Launch a detached submission service or inspect its record and runner log"
    )

    start: CliSubCommand[Start] = command(Start)
    status: CliSubCommand[Status] = command(Status)
    logs: CliSubCommand[Logs] = command(Logs)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(SubmissionJobCLI)


if __name__ == "__main__":
    main()
