import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import CliImplicitFlag, CliPositionalArg, SettingsError

from knee_abnormality.bundle import verify_bundle
from knee_abnormality.cli import NonExitingCLISettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SubmissionJobCLI(NonExitingCLISettings):
    command: CliPositionalArg[Literal["start", "status", "logs"]]
    record: Path
    bundle: Path | None = None
    dataset: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
    notebook: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
    message: str = "Fold 0 inference bundle"
    lines: int = Field(default=60, gt=0)
    follow: CliImplicitFlag[bool] = False


def _unit(record: Path) -> str:
    identity = hashlib.sha256(str(record).encode()).hexdigest()[:16]
    return f"knee-submit-{identity}.service"


def _log(record: Path) -> Path:
    return record.with_name(f"{record.stem}-runner.log")


def start(args: SubmissionJobCLI, record: Path) -> dict:
    if args.bundle is None or args.dataset is None or args.notebook is None:
        raise ValueError("start requires --bundle, --dataset and --notebook")
    bundle = args.bundle.expanduser().resolve()
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
            "--bundle",
            str(bundle),
            "--dataset",
            args.dataset,
            "--notebook",
            args.notebook,
            "--record",
            str(record),
            "--message",
            args.message,
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
        "public_score": submission.get("publicScore", submission.get("public_score")),
        "submission": submission,
        "stages": stages,
        "error": value.get("error"),
        "detail": value.get("detail"),
    }


def main() -> None:
    try:
        args = SubmissionJobCLI()
        record = args.record.expanduser().resolve()
        if args.command == "start":
            print(json.dumps(start(args, record), indent=2))
        elif args.command == "status":
            print(json.dumps(status(record), indent=2))
        else:
            command = ["tail", "-n", str(args.lines)]
            if args.follow:
                command.append("-F")
            subprocess.run([*command, "--", str(_log(record))], check=True)
    except subprocess.CalledProcessError as error:
        print(f"Submission job error: {error.stderr or error}", file=sys.stderr)
        raise SystemExit(2) from None
    except (OSError, ValueError, TypeError, KeyError, SettingsError) as error:
        print(f"Submission job error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
