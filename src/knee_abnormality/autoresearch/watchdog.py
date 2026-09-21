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

from knee_abnormality.autoresearch.cli import launch, preflight
from knee_abnormality.autoresearch.integration import digest
from knee_abnormality.autoresearch.isolation import manifest
from knee_abnormality.autoresearch.providers import auth_environment
from knee_abnormality.autoresearch.storage import (
    alive,
    atomic_write,
    lock,
    process_identity,
    read_json,
    sha256,
    write_json,
)
from knee_abnormality.autoresearch.supervisor import Supervisor
from knee_abnormality.autoresearch.usage import capture_usage
from knee_abnormality.cli import CLISettings


class WatchCLI(CLISettings):
    action: CliPositionalArg[Literal["start", "status", "stop", "watch"]]
    campaign: CliPositionalArg[str] = Field(pattern=r"^[a-z][a-z0-9-]*$")
    root: Path = Path(__file__).resolve().parents[3]


def optional_json(path):
    return read_json(path) if path.exists() else None


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
            Path(state["incumbent"]["run"]) / "fold_0/validation_predictions.csv",
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
                    if key not in {"status", "reason", "revision", "updated_at"}
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
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["ready", "blocked"]},
                "summary": {"type": "string"},
                "changed_files": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status", "summary", "changed_files"],
        }
        write_json(attempt / "schema.json", schema)
        prompt = (
            f"Repair one operational failure in autoresearch campaign {self.campaign}. "
            f"The user explicitly authorized failure-triggered repairs and resume. "
            f"Read {attempt / 'incident.json'} for the pre-stop state and only the relevant failed job logs. "
            f"The watchdog has quiesced the supervisor and owns recovery. "
            "Do not start, resume, stop or launch any campaign, model, GPU job or other agent yourself. "
            "Return ready only after a concrete root cause is fixed and relevant static checks pass. "
            "For a dead supervisor with intact artifacts, verified ordinary resume readiness can suffice. "
            "Quota/authentication failures, lost training, scientific decisions, source/data mismatches, "
            "and fixes requiring changes to sealed inputs must return blocked with evidence. "
            "Never loosen gates, limits, scope, sandboxing, acceptance criteria or dependencies. "
            "Never rewrite state, canonical decisions, predictions, identities, hashes, frozen sources, "
            "trained artifacts, policy seals or the campaign's reference. Do not change campaign proposals. "
            "You may fix autoresearch runtime code, role briefs and operational documentation only. "
            "Follow AGENTS.md and applicable skills. No tests. All shell commands use rtk. "
            "Other work is present: preserve unrelated edits, never reset, checkout or commit. "
            "Use the existing environment and no dependency installation. Keep the investigation narrow. "
            f"Before editing, confirm {self.output / 'stop.json'} still contains watchdog={token}; "
            f"if changed or {self.directory / 'stop.json'} exists, stop and return blocked. "
            "Do not remove or alter those control files. The watchdog will recheck them before resume. "
            "Exit with a concise structured result; no polling or repeated attempts."
        )
        atomic_write(attempt / "stdin.txt", prompt)
        profile = yaml.safe_load((self.root / "autoresearch/agents.yaml").read_text())[
            "codex"
        ]["orchestrator"]
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
        command = [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--sandbox",
            "workspace-write",
            "-c",
            'approval_policy="never"',
            "-c",
            "features.multi_agent=false",
            "-m",
            profile["model"],
            "-c",
            f'model_reasoning_effort="{profile["effort"]}"',
            "--cd",
            str(self.root),
            "--json",
            "--output-schema",
            str(attempt / "schema.json"),
            "--output-last-message",
            str(attempt / "report.json"),
            "-",
        ]
        with (
            (attempt / "stdin.txt").open() as source,
            (attempt / "stdout.log").open("w") as stdout,
            (attempt / "stderr.log").open("w") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=auth_environment(),
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
        if process.returncode:
            raise ValueError(f"Repair agent failed; inspect {attempt}")
        result = read_json(attempt / "report.json")
        if result["status"] != "ready":
            raise ValueError(result["summary"])
        return result

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
            launch(supervisor)
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
                self.save(campaign_status=state["status"], supervisor_alive=live)
                if state["status"] == "paused" or not live:
                    outstanding = [
                        job for job in state["jobs"].values() if not job["processed"]
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
                    if busy:
                        self.save(status="waiting_for_jobs")
                    else:
                        reason = state.get("reason") or "Supervisor exited unexpectedly"
                        fingerprint = re.sub(
                            r"\d{8}T\d{6}Z-[0-9a-f]{8}", "<attempt>", reason
                        )
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
                with (watcher.directory / "watch.log").open("ab") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "scripts.tools.autoresearch_watch",
                            "watch",
                            args.campaign,
                            "--root",
                            str(watcher.root),
                        ],
                        cwd=watcher.root,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                write_json(
                    watcher.directory / "launcher.json", process_identity(process.pid)
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
