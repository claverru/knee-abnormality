import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import CliPositionalArg

from knee_abnormality.autoresearch.campaign.binding import digest
from knee_abnormality.autoresearch.campaign.reports import WatchdogReport
from knee_abnormality.autoresearch.runtime.isolation import (
    changed_files,
    environment,
    manifest,
    scrubbed_environment,
)
from knee_abnormality.autoresearch.runtime.layout import (
    PREDICTIONS,
    campaign_fold,
    fold_dir,
)
from knee_abnormality.autoresearch.runtime.providers import codex_command, preflight
from knee_abnormality.autoresearch.runtime.storage import (
    alive,
    atomic_write,
    lock,
    optional_json,
    process_identity,
    read_json,
    sha256,
    spawn,
    write_json,
)
from knee_abnormality.autoresearch.runtime.usage import capture_usage
from knee_abnormality.autoresearch.supervisor import Supervisor
from knee_abnormality.cli import CLISettings


class WatchCLI(CLISettings):
    action: CliPositionalArg[Literal["start", "status", "stop", "watch"]]
    campaign: CliPositionalArg[str] = Field(pattern=r"^[a-z][a-z0-9-]*$")
    root: Path = Path(__file__).resolve().parents[3]


PROTECTED = (
    "autoresearch/roles/watchdog.md",
    *(
        f"src/knee_abnormality/autoresearch/{name}.py"
        for name in (
            "campaign/gates",
            "campaign/rules",
            "runtime/layout",
            "scoring/diagnostics",
            "campaign/binding",
            "campaign/completion",
            "scoring/accounting",
            "campaign/definition",
            "supervisor/results",
            "supervisor/experiments",
            "runtime/isolation",
        )
    ),
)
REPAIRABLE_FILES = (
    "autoresearch/runtime.md",
    "autoresearch/lifecycle.md",
    "autoresearch/watchdog.md",
    "autoresearch/status.md",
    "autoresearch/scripts.md",
    "scripts/tools/autoresearch.py",
    "scripts/tools/autoresearch_watch.py",
)
PROJECTIONS = ("backlog.yaml", "ideas.yaml", "findings.md")


def repairable(path):
    return path not in PROTECTED and (
        path in REPAIRABLE_FILES
        or path.startswith(
            ("src/knee_abnormality/autoresearch/", "autoresearch/roles/")
        )
    )


class Watchdog:
    def __init__(self, root, campaign):
        self.root = root.resolve()
        self.campaign = campaign
        supervisor = Supervisor(self.root, campaign)
        if supervisor.state is None:
            raise ValueError("Start the campaign before enabling its watchdog")
        self.output = supervisor.output
        self.directory = self.output / "watchdog"
        self.record = optional_json(self.directory / "state.json") or {}
        self.record.setdefault("incidents", [])
        self.record.setdefault("resumes", [])

    def save(self, **fields):
        self.record.update(fields, updated_at=time.time())
        write_json(self.directory / "state.json", self.record)

    def supervisor_alive(self):
        return alive(optional_json(self.output / "supervisor.json"))

    def alive(self):
        return alive(self.record.get("owner")) or alive(
            optional_json(self.directory / "launcher.json")
        )

    def permitted(self, token):
        return (
            not (self.directory / "stop.json").exists()
            and optional_json(self.output / "stop.json")
            == {"abort": False, "watchdog": token}
            and not (self.output / "resume.json").exists()
        )

    def workspace_manifest(self):
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=self.root,
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout.decode()
        return {
            path: sha256(self.root / path) if (self.root / path).is_file() else None
            for path in listed.split("\0")
            if path
            and not (
                path.startswith("autoresearch/campaigns/")
                and path.rsplit("/", 1)[-1] in PROJECTIONS
            )
        }

    def recovery_guard(self):
        state = read_json(self.output / "state.json")
        sources = {state["base"], state["incumbent"]["source"]}
        for key in ("claimed", "running"):
            exp = state["experiments"].get(state[key])
            if exp:
                sources.update(exp[name] for name in ("source", "workspace"))
        files = [
            self.output / "limits.json",
            Path(state["headers"]),
            fold_dir(state["incumbent"]["run"], campaign_fold(state)) / PREDICTIONS,
            self.root / "autoresearch/campaigns" / self.campaign / "campaign.yaml",
            self.root / "autoresearch/campaigns" / self.campaign / "contract.md",
            self.output / "baseline/input_identity.json",
            self.output / "baseline/triplet_accounting.json",
        ]
        return digest(
            {
                "state": {
                    key: value
                    for key, value in state.items()
                    if key
                    not in {"status", "reason", "pause_kind", "revision", "updated_at"}
                },
                "files": {
                    str(path): sha256(path) if path.exists() else None for path in files
                },
                "sources": {
                    source: manifest(Path(source)) if Path(source).exists() else None
                    for source in sorted(sources)
                },
            }
        )

    def quiesce(self, state, token):
        with lock(self.output / "control.lock"):
            current = read_json(self.output / "state.json")
            if (
                current["revision"] != state["revision"]
                or (self.output / "stop.json").exists()
                or (self.output / "resume.json").exists()
                or (self.directory / "stop.json").exists()
            ):
                return False
            write_json(self.output / "stop.json", {"abort": False, "watchdog": token})
        deadline = time.monotonic() + 30
        while self.supervisor_alive():
            if not self.permitted(token) or time.monotonic() >= deadline:
                raise ValueError("Supervisor did not stop cleanly for repair")
            time.sleep(1)
        return self.permitted(token)

    def repair(self, state, token):
        attempt = self.directory / token
        attempt.mkdir()
        recent = sorted(
            state["jobs"].values(), key=lambda job: job["created_at"], reverse=True
        )[:3]
        write_json(
            attempt / "incident.json",
            {
                **{
                    key: state[key]
                    for key in (
                        "campaign",
                        "status",
                        "reason",
                        "revision",
                        "incumbent",
                        "claimed",
                        "running",
                    )
                },
                "state_path": str(self.output / "state.json"),
                "supervisor_log": str(self.output / "supervisor.log"),
                "recent_jobs": [
                    {key: job[key] for key in ("id", "kind", "path", "processed")}
                    for job in recent
                ],
            },
        )
        write_json(attempt / "schema.json", WatchdogReport.model_json_schema())
        prompt = (
            (self.root / "autoresearch/roles/watchdog.md").read_text()
            + f"""
# Incident

- Campaign: {self.campaign}
- Incident record (pre-stop state): {attempt / "incident.json"}
- Control files: {self.output / "stop.json"} must still contain watchdog={token}, and
  {self.directory / "stop.json"} must not exist.
- You may change only files under src/knee_abnormality/autoresearch/ and
  autoresearch/roles/, plus {", ".join(REPAIRABLE_FILES)}.
- You must never change {", ".join(PROTECTED)}.
- The watchdog verifies the actual changed files against this scope; any other change
  blocks resume.
"""
        )
        atomic_write(attempt / "stdin.txt", prompt)
        profile = yaml.safe_load((self.root / "autoresearch/agents.yaml").read_text())[
            "codex"
        ]["watchdog"]
        write_json(
            attempt / "assignment.json",
            {
                "profile": profile,
                "context": {
                    "role": "watchdog",
                    "assignment": {"event": {"kind": "operational_failure"}},
                },
            },
        )
        command = codex_command(
            profile,
            attempt / "schema.json",
            attempt / "report.json",
            "workspace-write",
            "--cd",
            str(self.root),
        )
        before = self.workspace_manifest()
        with (
            (attempt / "stdin.txt").open() as source,
            (attempt / "stdout.log").open("w") as stdout,
            (attempt / "stderr.log").open("w") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=scrubbed_environment(),
                stdin=source,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            write_json(attempt / "owner.json", process_identity(process.pid))
            deadline = time.monotonic() + 900
            try:
                while process.poll() is None:
                    if not self.permitted(token) or time.monotonic() >= deadline:
                        raise ValueError(
                            "Repair cancelled or exceeded its 15-minute limit"
                        )
                    time.sleep(2)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                write_json(attempt / "receipt.json", {"exit_code": process.returncode})
                capture_usage("codex", attempt)
        changed = changed_files(before, self.workspace_manifest())
        outside = [path for path in changed if not repairable(path)]
        write_json(
            attempt / "changes.json", {"actual": changed, "outside_scope": outside}
        )
        if outside:
            raise ValueError(
                f"Repair changed files outside its scope: {', '.join(outside)}"
            )
        if process.returncode:
            raise ValueError(f"Repair agent failed; inspect {attempt}")
        result = WatchdogReport.model_validate(read_json(attempt / "report.json"))
        if result.status != "ready":
            raise ValueError(result.summary)
        return {**result.model_dump(mode="json"), "actual_changed_files": changed}

    def resume(self, token, guard):
        with lock(self.output / "control.lock"):
            if not self.permitted(token) or self.supervisor_alive():
                raise ValueError(
                    "Recovery ownership changed; automatic resume cancelled"
                )
            if self.recovery_guard() != guard:
                raise ValueError(
                    "Protected campaign state or artifacts changed during repair"
                )
            supervisor = Supervisor(self.root, self.campaign)
            preflight(supervisor.state["provider"], self.root)
            write_json(self.output / "resume.json", {})
            supervisor.launch()
        self.await_running()

    def plain_resume(self, state):
        with lock(self.output / "control.lock"):
            current = read_json(self.output / "state.json")
            if (
                current["status"] != "paused"
                or current.get("reason") != state.get("reason")
                or (self.output / "stop.json").exists()
                or (self.output / "resume.json").exists()
                or (self.directory / "stop.json").exists()
            ):
                return False
            supervisor = Supervisor(self.root, self.campaign)
            preflight(supervisor.state["provider"], self.root)
            write_json(self.output / "resume.json", {})
            if not self.supervisor_alive():
                supervisor.launch()
        self.await_running()
        return True

    def await_running(self):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = read_json(self.output / "state.json")
            if (
                state["status"] in {"running", "calibrating"}
                and self.supervisor_alive()
            ):
                return
            if (self.directory / "stop.json").exists():
                break
            time.sleep(1)
        raise ValueError("Resume did not reach a live running or calibrating state")

    def watch(self):
        with lock(self.directory / "watch.lock"):
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(
                    signum, lambda *_: write_json(self.directory / "stop.json", {})
                )
            self.save(owner=process_identity(os.getpid()), status="watching")
            while not (self.directory / "stop.json").exists():
                state = read_json(self.output / "state.json")
                if (
                    state["status"] in {"stopping", "stopped"}
                    or (self.output / "stop.json").exists()
                ):
                    self.save(status="stopped", reason="Campaign stop requested")
                    return
                live = self.supervisor_alive()
                self.save(
                    campaign_status=state["status"],
                    supervisor_alive=live,
                    retry_after=state.get("retry_after"),
                )
                if state["status"] == "paused" or not live:
                    outstanding = [
                        job
                        for job in state["jobs"].values()
                        if not job["processed"] and not job.get("application_error")
                    ]
                    busy = (
                        bool(outstanding)
                        if live
                        else any(
                            alive(optional_json(Path(job["path"]) / name))
                            for job in outstanding
                            for name in ("owner.json", "launcher.json")
                        )
                    )
                    reason = state.get("reason") or "Supervisor exited unexpectedly"
                    fingerprint = re.sub(
                        r"\d{8}T\d{6}Z-[0-9a-f]{8}", "<attempt>", reason
                    )
                    if (
                        state["status"] == "paused"
                        and state["pause_kind"] == "rejection"
                        and fingerprint not in self.record["resumes"]
                        and len(self.record["resumes"]) < 3
                    ):
                        self.save(status="resuming", reason=reason)
                        if self.plain_resume(state):
                            self.record["resumes"].append(fingerprint)
                        self.save(status="watching", reason=None)
                    elif busy:
                        self.save(status="waiting_for_jobs")
                    else:
                        if (
                            fingerprint in self.record["incidents"]
                            or len(self.record["incidents"]) >= 3
                        ):
                            self.save(
                                status="blocked",
                                reason="Repeated failure or three-repair limit",
                                failure=reason,
                            )
                            return
                        token = uuid.uuid4().hex
                        guard = self.recovery_guard()
                        if not self.quiesce(state, token):
                            self.save(
                                status="stopped", reason="Recovery was superseded"
                            )
                            return
                        self.record["incidents"].append(fingerprint)
                        self.save(status="repairing", attempt=token, reason=reason)
                        result = self.repair(state, token)
                        self.resume(token, guard)
                        self.save(status="watching", reason=None, last_repair=result)
                for _ in range(60):
                    if (self.directory / "stop.json").exists():
                        break
                    time.sleep(1)
            self.save(status="stopped", reason="Watchdog stop requested")


def main():
    args = WatchCLI()
    watcher = Watchdog(args.root, args.campaign)
    if args.action == "watch":
        try:
            watcher.watch()
        except BlockingIOError:
            print("Watchdog already locked by another process", file=sys.stderr)
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as error:
            watcher.save(status="blocked", reason=str(error))
        return
    with lock(watcher.directory / "control.lock"):
        if args.action == "start":
            if not watcher.alive():
                attempt = watcher.record.get("attempt")
                if attempt and alive(
                    optional_json(watcher.directory / attempt / "owner.json")
                ):
                    raise ValueError("A previous repair agent is still alive")
                state = read_json(watcher.output / "state.json")
                if (
                    state["status"] in {"stopping", "stopped"}
                    or (watcher.output / "stop.json").exists()
                ):
                    raise ValueError("Cannot watch a deliberately stopped campaign")
                preflight("codex", watcher.root)
                (watcher.directory / "stop.json").unlink(missing_ok=True)
                watcher.save(
                    status="starting",
                    reason=None,
                    incidents=watcher.record.get("incidents", []),
                )
                spawn(
                    [
                        "knee_abnormality.autoresearch.watchdog",
                        "watch",
                        args.campaign,
                        "--root",
                        str(watcher.root),
                    ],
                    watcher.root,
                    watcher.directory / "watch.log",
                    watcher.directory / "launcher.json",
                    environment(watcher.root),
                )
        elif args.action == "stop":
            write_json(watcher.directory / "stop.json", {})
        watcher.record = optional_json(watcher.directory / "state.json") or {}
        print(
            json.dumps(
                {
                    **watcher.record,
                    "alive": watcher.alive(),
                    "artifacts": str(watcher.directory),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
