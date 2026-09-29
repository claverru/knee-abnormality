import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import CliApp, CliSubCommand

from knee_abnormality.autoresearch.campaign.binding import digest
from knee_abnormality.autoresearch.campaign.reports import WatchdogReport
from knee_abnormality.autoresearch.cli import CampaignId, RootCommand
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
    tail,
    write_json,
)
from knee_abnormality.autoresearch.runtime.usage import capture_usage
from knee_abnormality.autoresearch.supervisor import Supervisor
from knee_abnormality.autoresearch.topics.history import OUTPUTS, campaign_state
from knee_abnormality.autoresearch.topics.loop import LOOP, TopicLoop
from knee_abnormality.cli import CLISettings, command, describe

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
            "topics/checks",
            "topics/history",
            "topics/packet",
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
LIMIT = 3
STUCK_SECONDS = 1800
LOOP_STOPPED = "Topic loop stop requested"
WATCHDOG_STOPPED = "Watchdog stop requested"
LOOP_VOLATILE = {"status", "reason", "error", "retry_after", "revision", "updated_at"}
ERRORS = (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError)


def repairable(path):
    return path not in PROTECTED and (
        path in REPAIRABLE_FILES
        or path.startswith(
            ("src/knee_abnormality/autoresearch/", "autoresearch/roles/")
        )
    )


def fingerprint(reason):
    return re.sub(r"\d{8}T\d{6}Z-[0-9a-f]{8}", "<attempt>", reason)


def loop_targets(root, state):
    rotation = state["rotation"] or {}
    successor = rotation.get("successor")
    targets = {}
    if state["current"] is not None:
        targets[state["current"]] = False
    if successor is not None:
        targets[successor["campaign"]] = rotation["phase"] == "successor"
    return {
        campaign: deferred
        for campaign, deferred in targets.items()
        if campaign_state(root, campaign) is not None
    }


def looped(root):
    state = optional_json(root / LOOP / "loop.json")
    if state is None or state["status"] == "stopped":
        return set()
    return set(loop_targets(root, state))


def loop_failure(log):
    lines = [line.strip() for line in tail(log).splitlines() if line.strip()]
    return lines[-1] if lines else "Topic loop exited unexpectedly"


def workspace_manifest(root):
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout.decode()
    return {
        path: sha256(root / path) if (root / path).is_file() else None
        for path in listed.split("\0")
        if path
        and not (
            path.startswith("autoresearch/campaigns/")
            and path.rsplit("/", 1)[-1] in PROJECTIONS
        )
    }


def run_repair(root, attempt, incident, permitted):
    write_json(attempt / "schema.json", WatchdogReport.model_json_schema())
    prompt = (
        (root / "autoresearch/roles/watchdog.md").read_text()
        + "\n# Incident\n\n"
        + "".join(f"- {line}\n" for line in incident)
        + f"""- You may change only files under src/knee_abnormality/autoresearch/ and
  autoresearch/roles/, plus {", ".join(REPAIRABLE_FILES)}.
- You must never change {", ".join(PROTECTED)}.
- The watchdog verifies the actual changed files against this scope; any other change
  blocks resume.
"""
    )
    atomic_write(attempt / "stdin.txt", prompt)
    profile = yaml.safe_load((root / "autoresearch/agents.yaml").read_text())["codex"][
        "watchdog"
    ]
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
        str(root),
    )
    before = workspace_manifest(root)
    with (
        (attempt / "stdin.txt").open() as source,
        (attempt / "stdout.log").open("w") as stdout,
        (attempt / "stderr.log").open("w") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=root,
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
                if not permitted() or time.monotonic() >= deadline:
                    raise ValueError("Repair cancelled or exceeded its 15-minute limit")
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
    changed = changed_files(before, workspace_manifest(root))
    outside = [path for path in changed if not repairable(path)]
    write_json(attempt / "changes.json", {"actual": changed, "outside_scope": outside})
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


def repair_alive(directory, record):
    attempt = record.get("attempt")
    return bool(attempt) and alive(optional_json(directory / attempt / "owner.json"))


class CampaignRecovery:
    def __init__(self, root, campaign, stops, member=None):
        self.root = root
        self.campaign = campaign
        self.stops = stops
        self.member = member or (lambda: True)
        supervisor = Supervisor(root, campaign)
        if supervisor.state is None:
            raise ValueError(f"Start {campaign} before enabling its watchdog")
        self.output = supervisor.output
        self.directory = self.output / "watchdog"
        self.record = optional_json(self.directory / "recovery.json") or {}
        self.record.setdefault("incidents", [])
        self.record.setdefault("resumes", [])

    def save(self, **fields):
        self.record.update(fields, updated_at=time.time())
        write_json(self.directory / "recovery.json", self.record)

    def interrupted(self):
        return any(path.exists() for path in self.stops)

    def supervisor_alive(self):
        return alive(optional_json(self.output / "supervisor.json"))

    def permitted(self, token):
        return (
            not self.interrupted()
            and self.member()
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
                or self.interrupted()
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
        attempt.mkdir(parents=True)
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
        return run_repair(
            self.root,
            attempt,
            [
                f"Campaign: {self.campaign}",
                f"Incident record (pre-stop state): {attempt / 'incident.json'}",
                (
                    f"Control files: {self.output / 'stop.json'} must still contain "
                    f"watchdog={token}, and {', '.join(map(str, self.stops))} must not exist."
                ),
            ],
            lambda: self.permitted(token),
        )

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
                or self.interrupted()
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
            if self.interrupted():
                break
            time.sleep(1)
        raise ValueError("Resume did not reach a live running or calibrating state")

    def check(self, deferred, blocked):
        state = read_json(self.output / "state.json")
        if "watchdog" in (optional_json(self.output / "stop.json") or {}):
            if self.record.get("status") != "blocked":
                self.save(
                    status="blocked",
                    reason="An interrupted repair left the campaign stopped for inspection",
                )
            return "blocked"
        if (
            state["status"] in {"stopping", "stopped"}
            or (self.output / "stop.json").exists()
        ):
            self.save(
                status="stopped",
                reason="Campaign stop requested",
                campaign_status=state["status"],
            )
            return "ended"
        live = self.supervisor_alive()
        self.save(
            campaign_status=state["status"],
            supervisor_alive=live,
            retry_after=state.get("retry_after"),
        )
        if state["status"] != "paused" and live:
            if self.record.get("status") != "watching":
                self.save(status="watching", reason=None)
            return "watching"
        if blocked:
            return "blocked"
        if deferred and state["status"] == "paused":
            self.save(
                status="deferred",
                reason="The loop cancels a successor that pauses before its first approval",
            )
            return "watching"
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
        failure = fingerprint(reason)
        if (
            state["status"] == "paused"
            and state["pause_kind"] == "rejection"
            and failure not in self.record["resumes"]
            and len(self.record["resumes"]) < LIMIT
        ):
            self.save(status="resuming", reason=reason)
            if self.plain_resume(state):
                self.record["resumes"].append(failure)
            self.save(status="watching", reason=None)
        elif busy:
            self.save(status="waiting_for_jobs")
        else:
            if (
                failure in self.record["incidents"]
                or len(self.record["incidents"]) >= LIMIT
            ):
                self.save(
                    status="blocked",
                    reason="Repeated failure or three-repair limit",
                    failure=reason,
                )
                return "blocked"
            token = uuid.uuid4().hex
            guard = self.recovery_guard()
            if not self.quiesce(state, token):
                self.save(status="stopped", reason="Recovery was superseded")
                return "ended"
            self.record["incidents"].append(failure)
            self.save(status="repairing", attempt=token, reason=reason)
            result = self.repair(state, token)
            self.resume(token, guard)
            self.save(status="watching", reason=None, last_repair=result)
        return "watching"


class Watchdog:
    def __init__(self, root, campaign=None):
        self.root = root.resolve()
        self.campaign = campaign
        self.loop = None
        self.recoveries = {}
        self.blocked = set()
        self.loop_blocked = False
        if campaign is None:
            self.loop = TopicLoop(self.root)
            if self.loop.state is None:
                raise ValueError("Start the topic loop before enabling its watchdog")
            self.directory = self.loop.output / "watchdog"
        else:
            recovery = CampaignRecovery(self.root, campaign, [])
            self.directory = recovery.directory
            recovery.stops = self.stops()
            self.recoveries[campaign] = recovery
        self.record = optional_json(self.directory / "state.json") or {}
        if self.loop is not None:
            self.record.setdefault("loop_restarts", [])
            self.record.setdefault("loop_incidents", [])

    def save(self, **fields):
        self.record.update(fields, updated_at=time.time())
        write_json(self.directory / "state.json", self.record)

    def stops(self):
        return [self.directory / "stop.json"] + (
            [self.loop.output / "stop.json"] if self.loop is not None else []
        )

    def alive(self):
        return alive(self.record.get("owner")) or alive(
            optional_json(self.directory / "launcher.json")
        )

    def targets(self):
        if self.loop is None:
            return {self.campaign: False}
        return loop_targets(self.root, read_json(self.loop.state_path))

    def recovery(self, campaign):
        if campaign not in self.recoveries:
            self.recoveries[campaign] = CampaignRecovery(
                self.root,
                campaign,
                self.stops(),
                lambda: campaign in self.targets(),
            )
        return self.recoveries[campaign]

    def loop_permitted(self):
        output = self.loop.output
        return (
            not any(path.exists() for path in self.stops())
            and not (output / "resume.json").exists()
            and not self.loop.alive()
        )

    def loop_guard(self):
        state = read_json(self.loop.state_path)
        return digest(
            {key: value for key, value in state.items() if key not in LOOP_VOLATILE}
        )

    def relaunch_loop(self, guard=None):
        with lock(self.loop.output / "control.lock"):
            state = read_json(self.loop.state_path)
            if not self.loop_permitted() or state["status"] != "running":
                return False
            if guard is not None and self.loop_guard() != guard:
                raise ValueError("Topic loop state changed during repair")
            preflight(state["provider"], self.root)
            (self.loop.output / "restart.json").unlink(missing_ok=True)
            self.loop.launch()
        return True

    def recoverable(self, failure):
        return (
            failure not in self.record["loop_restarts"]
            and len(self.record["loop_restarts"]) < LIMIT
        ) or (
            failure not in self.record["loop_incidents"]
            and len(self.record["loop_incidents"]) < LIMIT
        )

    def watch_error(self, error):
        if error is None:
            if self.record.get("loop_error_since") or self.record.get("loop_recovery"):
                self.save(loop_error_since=None, loop_recovery=None)
            return
        failure = fingerprint(error["message"])
        since = self.record.get("loop_error_since")
        if since is None or since["failure"] != failure:
            self.save(
                loop_error_since={
                    "failure": failure,
                    "message": error["message"],
                    "at": time.time(),
                }
            )
            return
        if time.time() - since["at"] < STUCK_SECONDS:
            return
        if not self.recoverable(failure):
            if self.record.get("loop_recovery") != "stuck":
                self.save(
                    loop_recovery="stuck",
                    loop_failure=error["message"],
                    loop_reason="Persistent loop error after the restart and repair limits",
                )
            return
        output = self.loop.output
        with lock(output / "control.lock"):
            if (
                any(path.exists() for path in self.stops())
                or (output / "resume.json").exists()
            ):
                return
            write_json(output / "restart.json", {"watchdog": True})
        self.save(
            loop_stuck={
                "failure": failure,
                "message": error["message"],
                "requested_at": time.time(),
            },
            loop_error_since=None,
            loop_recovery="restarting",
            loop_failure=error["message"],
        )

    def repair_loop(self, state, reason, token):
        attempt = self.directory / token
        attempt.mkdir(parents=True)
        rotation = state["rotation"] or {}
        write_json(
            attempt / "incident.json",
            {
                "subject": "topic loop",
                "failure": reason,
                "status": state["status"],
                "current": state["current"],
                "rotation": rotation.get("id"),
                "phase": rotation.get("phase"),
                "error": state.get("error"),
                "state_path": str(self.loop.state_path),
                "loop_log": str(self.loop.output / "loop.log"),
            },
        )
        return run_repair(
            self.root,
            attempt,
            [
                (
                    "Subject: the topic loop process, which exited while running or was "
                    "restarted between passes after the same error persisted for 30 "
                    "minutes; its campaigns keep their own supervisors"
                ),
                f"Incident record: {attempt / 'incident.json'}",
                (
                    f"Control files: the loop must stay stopped, and "
                    f"{', '.join(map(str, self.stops()))} and "
                    f"{self.loop.output / 'resume.json'} must not exist."
                ),
            ],
            self.loop_permitted,
        )

    def check_loop(self):
        state = read_json(self.loop.state_path)
        if (
            state["status"] in {"stopping", "stopped"}
            or (self.loop.output / "stop.json").exists()
        ):
            return False
        live = self.loop.alive()
        self.save(
            loop_status=state["status"], loop_alive=live, loop_error=state.get("error")
        )
        if live:
            if self.record.get("loop_stuck"):
                return True
            if self.loop_blocked or self.record.get("loop_recovery") not in {
                None,
                "stuck",
            }:
                self.loop_blocked = False
                self.save(loop_recovery=None)
            self.watch_error(state.get("error"))
            return True
        if self.loop_blocked:
            return True
        stuck = self.record.get("loop_stuck")
        reason = (
            stuck["message"] if stuck else loop_failure(self.loop.output / "loop.log")
        )
        failure = fingerprint(reason)
        self.save(loop_stuck=None)
        if (
            failure not in self.record["loop_restarts"]
            and len(self.record["loop_restarts"]) < LIMIT
        ):
            self.save(loop_recovery="relaunching", loop_failure=reason)
            if self.relaunch_loop():
                self.record["loop_restarts"].append(failure)
            self.save(loop_recovery=None)
            return True
        if (
            failure in self.record["loop_incidents"]
            or len(self.record["loop_incidents"]) >= LIMIT
        ):
            self.loop_blocked = True
            self.save(
                loop_recovery="blocked",
                loop_failure=reason,
                loop_reason="Repeated loop failure or three-repair limit",
            )
            return True
        token = uuid.uuid4().hex
        guard = self.loop_guard()
        self.record["loop_incidents"].append(failure)
        self.save(loop_recovery="repairing", attempt=token, loop_failure=reason)
        result = self.repair_loop(state, reason, token)
        if not self.relaunch_loop(guard):
            raise ValueError(
                "Loop recovery was superseded; automatic relaunch cancelled"
            )
        self.save(loop_recovery=None, last_loop_repair=result)
        return True

    def cycle(self):
        if self.loop is not None:
            try:
                if not self.check_loop():
                    return "stopped", LOOP_STOPPED
            except ERRORS as error:
                self.loop_blocked = True
                self.save(loop_recovery="blocked", loop_reason=str(error))
        elif self.campaign in looped(self.root):
            return (
                "stopped",
                "The topic loop runs this campaign; watch the loop instead",
            )
        targets = self.targets()
        for campaign, deferred in targets.items():
            try:
                recovery = self.recovery(campaign)
                outcome = recovery.check(deferred, campaign in self.blocked)
                reason = recovery.record.get("reason")
            except ERRORS as error:
                outcome, reason = "blocked", str(error)
                if campaign in self.recoveries:
                    self.recoveries[campaign].save(status="blocked", reason=reason)
            if outcome == "blocked":
                self.blocked.add(campaign)
            else:
                self.blocked.discard(campaign)
            if self.loop is None and outcome != "watching":
                return ("blocked" if outcome == "blocked" else "stopped"), reason
        self.save(campaigns=sorted(targets), blocked=sorted(self.blocked))
        return None

    def watch(self):
        with lock(self.directory / "watch.lock"):
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(
                    signum, lambda *_: write_json(self.directory / "stop.json", {})
                )
            self.save(owner=process_identity(os.getpid()), status="watching")
            while not (self.directory / "stop.json").exists():
                ended = self.cycle()
                if ended is not None:
                    status, reason = ended
                    self.save(status=status, reason=reason)
                    return
                for _ in range(60):
                    if (self.directory / "stop.json").exists():
                        break
                    time.sleep(1)
            self.save(status="stopped", reason=WATCHDOG_STOPPED)

    def check_start(self):
        targets = self.targets()
        for campaign in targets:
            directory = self.root / OUTPUTS / campaign / "watchdog"
            if repair_alive(
                directory, optional_json(directory / "recovery.json") or {}
            ):
                raise ValueError(
                    f"A previous repair agent for {campaign} is still alive"
                )
        if self.loop is None:
            state = read_json(self.recoveries[self.campaign].output / "state.json")
            if (
                state["status"] in {"stopping", "stopped"}
                or (self.recoveries[self.campaign].output / "stop.json").exists()
            ):
                raise ValueError("Cannot watch a deliberately stopped campaign")
            if self.campaign in looped(self.root):
                raise ValueError(
                    "The topic loop runs this campaign; start the watchdog without --campaign"
                )
            return
        state = read_json(self.loop.state_path)
        if (
            state["status"] in {"stopping", "stopped"}
            or (self.loop.output / "stop.json").exists()
        ):
            raise ValueError("Cannot watch a deliberately stopped topic loop")
        if repair_alive(self.directory, self.record):
            raise ValueError("A previous loop repair agent is still alive")
        for campaign in targets:
            directory = self.root / OUTPUTS / campaign / "watchdog"
            if alive(
                (optional_json(directory / "state.json") or {}).get("owner")
            ) or alive(optional_json(directory / "launcher.json")):
                raise ValueError(
                    f"A campaign watchdog runs on {campaign}; stop it first"
                )


def report(watcher: Watchdog) -> None:
    watcher.record = optional_json(watcher.directory / "state.json") or {}
    print(
        json.dumps(
            {
                **watcher.record,
                "alive": watcher.alive(),
                "recovery": {
                    campaign: optional_json(
                        watcher.root / OUTPUTS / campaign / "watchdog/recovery.json"
                    )
                    for campaign in watcher.targets()
                },
                "artifacts": str(watcher.directory),
            },
            indent=2,
        )
    )


class WatchTarget(RootCommand):
    campaign: CampaignId | None = Field(
        default=None,
        description="Campaign run outside the topic loop; omit to watch the topic loop and every campaign it runs",
    )

    def watchdog(self) -> Watchdog:
        return Watchdog(self.root, self.campaign)


def launch(watcher: Watchdog) -> None:
    if watcher.alive():
        return
    watcher.check_start()
    preflight("codex", watcher.root)
    (watcher.directory / "stop.json").unlink(missing_ok=True)
    loop = (
        {}
        if watcher.loop is None
        else {"loop_error_since": None}
        if (watcher.loop.output / "restart.json").exists()
        else {"loop_error_since": None, "loop_stuck": None}
    )
    watcher.save(status="starting", reason=None, **loop)
    spawn(
        [
            "knee_abnormality.autoresearch.watchdog",
            "watch",
            *(["--campaign", watcher.campaign] if watcher.campaign else []),
            "--root",
            str(watcher.root),
        ],
        watcher.root,
        watcher.directory / "watch.log",
        watcher.directory / "launcher.json",
        environment(watcher.root),
    )


def resume_loop_watchdog(root: Path) -> str:
    watcher = Watchdog(root)
    if watcher.record.get("reason") == WATCHDOG_STOPPED or not watcher.record:
        return (
            "not started: it was never started for the loop or was stopped by the user"
        )
    deadline = time.monotonic() + 60
    while read_json(watcher.loop.state_path)["status"] != "running":
        if time.monotonic() >= deadline:
            return "not restarted: the loop did not reach running within 60 s"
        time.sleep(1)
    with lock(watcher.directory / "control.lock"):
        watcher.record = optional_json(watcher.directory / "state.json") or {}
        if watcher.alive():
            return "already running"
        if watcher.record.get("reason") != LOOP_STOPPED:
            return f"not restarted: it ended with {watcher.record.get('reason')!r}"
        launch(watcher)
    return "restarted"


class Start(WatchTarget):
    model_config = describe(
        "Launch a detached watchdog for the running topic loop or a running campaign"
    )

    def cli_cmd(self) -> None:
        watcher = self.watchdog()
        with lock(watcher.directory / "control.lock"):
            launch(watcher)
            report(watcher)


class Status(WatchTarget):
    model_config = describe("Show the watchdog state")

    def cli_cmd(self) -> None:
        watcher = self.watchdog()
        with lock(watcher.directory / "control.lock"):
            report(watcher)


class Stop(WatchTarget):
    model_config = describe("Ask the watchdog to stop")

    def cli_cmd(self) -> None:
        watcher = self.watchdog()
        with lock(watcher.directory / "control.lock"):
            write_json(watcher.directory / "stop.json", {})
            report(watcher)


class Watch(WatchTarget):
    model_config = describe("Run the watchdog loop in the foreground")

    def cli_cmd(self) -> None:
        watcher = self.watchdog()
        try:
            watcher.watch()
        except BlockingIOError:
            print("Watchdog already locked by another process", file=sys.stderr)
        except ERRORS as error:
            watcher.save(status="blocked", reason=str(error))


class WatchCLI(CLISettings):
    model_config = describe(
        "Monitor the topic loop and its campaigns, or one campaign, without model polling "
        "and dispatch bounded Codex repairs for operational failures"
    )

    start: CliSubCommand[Start] = command(Start)
    status: CliSubCommand[Status] = command(Status)
    stop: CliSubCommand[Stop] = command(Stop)
    watch: CliSubCommand[Watch] = command(Watch)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main():
    CliApp.run(WatchCLI)


if __name__ == "__main__":
    main()
