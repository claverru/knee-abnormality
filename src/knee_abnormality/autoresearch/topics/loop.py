import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml
from pydantic import ValidationError

from knee_abnormality.autoresearch.campaign.definition import RECIPE, chain_recipe
from knee_abnormality.autoresearch.campaign.gates import initial_limits
from knee_abnormality.autoresearch.campaign.reports import (
    TOPIC_REPORTS,
    TopicDraftReport,
)
from knee_abnormality.autoresearch.runtime import providers
from knee_abnormality.autoresearch.runtime.isolation import environment
from knee_abnormality.autoresearch.runtime.storage import (
    alive,
    atomic_write,
    lock,
    optional_json,
    process_identity,
    read_json,
    spawn,
    terminate,
    write_json,
)
from knee_abnormality.autoresearch.supervisor import Supervisor
from knee_abnormality.autoresearch.supervisor.common import (
    REJECTION_LIMIT,
    identifier,
    seed_files,
)
from knee_abnormality.autoresearch.supervisor.jobs import JobMixin
from knee_abnormality.autoresearch.supervisor.prompts import role_prompt, topic_rules
from knee_abnormality.autoresearch.topics.checks import check_draft, stage
from knee_abnormality.autoresearch.topics.files import (
    CAMPAIGNS,
    REJECTED,
    free_id,
    load_pool,
    queued_topics,
    register_campaign,
    write_draft,
)
from knee_abnormality.autoresearch.topics.history import (
    OUTPUTS,
    QUIET_TOPICS,
    campaign_records,
    campaign_state,
    decided_runs,
    outcome,
    researcher_streak,
    seeded_ideas,
)
from knee_abnormality.autoresearch.topics.packet import build_packet

LOOP = OUTPUTS / ".loop"
SEALED = {"mode": "sealed"}
POLL_SECONDS = 30
FIRST_TOPIC_RETRY_SECONDS = 3600
ERRORS = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
    ValidationError,
    subprocess.SubprocessError,
)


def approved(state):
    return any(
        exp.get("implementation_review", {}).get("decision") == "approve"
        for exp in state["experiments"].values()
    )


class TopicLoop(JobMixin):
    def __init__(self, root: Path):
        self.root = root
        self.output = (root / LOOP).resolve()
        self.state_path = self.output / "loop.json"
        self.state = optional_json(self.state_path)
        self.wakeup = threading.Event()

    def save(self):
        self.state["revision"] += 1
        self.state["updated_at"] = time.time()
        write_json(self.state_path, self.state)

    def relative(self, path):
        path = Path(path).resolve()
        return (
            path.relative_to(self.root).as_posix()
            if path.is_relative_to(self.root)
            else str(path)
        )

    def launch(self):
        spawn(
            [
                "knee_abnormality.autoresearch.cli",
                "loop-supervise",
                "--root",
                str(self.root),
            ],
            self.root,
            self.output / "loop.log",
            self.output / "process.json",
            environment(self.root),
        )

    def alive(self):
        return alive(optional_json(self.output / "process.json"))

    def initialize(self, provider, gpu, campaign, reference_run):
        if self.state is not None:
            raise ValueError("The topic loop already exists; use loop-resume")
        load_pool(self.root)
        if reference_run is not None:
            initial_limits(
                (self.root / reference_run).resolve(),
                read_json(
                    (self.root / reference_run).resolve() / "effective_config.json"
                ),
                {},
                [],
                [],
                SEALED,
            )
        self.output.mkdir(parents=True, exist_ok=True)
        self.state = {
            "revision": 0,
            "provider": provider,
            "gpu": gpu,
            "status": "running",
            "reason": None,
            "error": None,
            "profiles": yaml.safe_load(
                (self.root / "autoresearch/agents.yaml").read_text()
            ),
            "reference": self.relative(self.root / reference_run)
            if reference_run is not None
            else None,
            "current": campaign,
            "topics": [],
            "rotation": None,
            "rotations": [],
            "jobs": {},
            "pool_taken": {},
            "retry_after_decided": {},
            "first_attempt_after": 0,
            "to_resume": [],
            "attention": None,
        }
        if campaign is not None:
            supervisor = Supervisor(self.root, campaign)
            self.state["topics"].append(
                self.topic_record(campaign, "user", None, campaign, None, None)
            )
            if supervisor.state is None:
                supervisor.initialize(provider, gpu)
                supervisor.launch()
            elif supervisor.state["status"] == "stopped":
                self.resume_campaign(campaign)
        self.save()

    def topic_record(self, campaign, origin, source, title, predecessor, chain):
        return {
            "campaign": campaign,
            "origin": origin,
            "source": source,
            "title": title,
            "predecessor": predecessor,
            "chain_incumbent": chain["incumbent"] if chain else None,
            "status": "active",
            "started_at": time.time(),
            "retired_at": None,
            "streak_at_rotation": None,
            "outcome": None,
            "pool_seeds": [],
        }

    def topic(self, campaign):
        return next(
            item for item in self.state["topics"] if item["campaign"] == campaign
        )

    def supervisor_alive(self, campaign):
        return alive(optional_json(self.root / OUTPUTS / campaign / "supervisor.json"))

    def request_stop(self, campaign, abort):
        output = self.root / OUTPUTS / campaign
        state = campaign_state(self.root, campaign)
        if state is None or state["status"] == "stopped":
            return
        if not (output / "stop.json").exists():
            write_json(output / "stop.json", {"abort": abort})
        if not self.supervisor_alive(campaign):
            Supervisor(self.root, campaign).launch()

    def repairing(self, campaign):
        return "watchdog" in (
            optional_json(self.root / OUTPUTS / campaign / "stop.json") or {}
        )

    def resume_campaign(self, campaign):
        providers.preflight(self.state["provider"], self.root)
        write_json(self.root / OUTPUTS / campaign / "resume.json", {})
        if not self.supervisor_alive(campaign):
            Supervisor(self.root, campaign).launch()

    def mark(self, campaign, phase):
        path = self.root / OUTPUTS / campaign / "rotation.json"
        if phase is None:
            path.unlink(missing_ok=True)
        else:
            write_json(path, {"phase": phase, "rotation": self.state["rotation"]["id"]})

    def run(self):
        with lock(self.output / "loop.lock"):
            write_json(self.output / "process.json", process_identity(os.getpid()))

            def stop(signum, frame):
                write_json(self.output / "stop.json", {"abort": False})
                self.wakeup.set()

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            while True:
                try:
                    self.state["error"] = None
                    if self.control():
                        return
                except ERRORS as error:
                    self.state["error"] = {
                        "message": f"{type(error).__name__}: {error}",
                        "at": time.time(),
                    }
                    self.save()
                self.wakeup.wait(POLL_SECONDS)
                self.wakeup.clear()

    def control(self):
        resume = self.output / "resume.json"
        restart = self.output / "restart.json"
        if resume.exists():
            (self.output / "stop.json").unlink(missing_ok=True)
            restart.unlink(missing_ok=True)
            self.state.update(status="running", reason=None)
            self.state.pop("retry_after", None)
            self.save()
            resume.unlink()
        request = self.output / "stop.json"
        if not request.exists() and restart.exists():
            restart.unlink()
            return True
        if not request.exists():
            self.tick()
            return False
        abort = read_json(request)["abort"]
        if self.state["status"] != "stopping":
            self.state["status"] = "stopping"
            self.save()
        rotation = self.state["rotation"]
        campaigns = [
            campaign
            for campaign in (
                self.state["current"],
                rotation and rotation["predecessor"],
                rotation and (rotation.get("successor") or {}).get("campaign"),
            )
            if campaign
        ]
        for campaign in dict.fromkeys(campaigns):
            state = campaign_state(self.root, campaign)
            if state is None or state["status"] == "stopped":
                continue
            self.request_stop(campaign, abort)
            if campaign not in self.state["to_resume"]:
                self.state["to_resume"].append(campaign)
        for job in self.outstanding():
            owner = Path(job["path"]) / "owner.json"
            if owner.exists():
                terminate(read_json(owner))
        self.harvest()
        stopped = all(
            (campaign_state(self.root, campaign) or {"status": "stopped"})["status"]
            == "stopped"
            for campaign in campaigns
        )
        if stopped and not self.outstanding():
            self.state["status"] = "stopped"
            self.save()
            return True
        self.save()
        return False

    def tick(self):
        if "retry_after" in self.state and not self.waiting():
            self.state.pop("retry_after")
            self.state["reason"] = None
        self.harvest()
        for campaign in list(self.state["to_resume"]):
            self.resume_campaign(campaign)
            self.state["to_resume"].remove(campaign)
            self.save()
        current = self.state["current"]
        if current is not None:
            self.refresh(current)
        rotation = self.state["rotation"]
        if rotation is not None:
            self.advance(rotation)
        elif (current is None and time.time() >= self.state["first_attempt_after"]) or (
            current is not None and self.due(current)
        ):
            self.begin(current)
        self.save()

    def refresh(self, campaign):
        state = campaign_state(self.root, campaign)
        if state is not None:
            self.topic(campaign)["outcome"] = outcome(
                state, seeded_ideas(self.root / OUTPUTS / campaign)
            )

    def due(self, campaign):
        state = campaign_state(self.root, campaign)
        if state is None or state["status"] != "running":
            return False
        decided = len(decided_runs(state))
        return (
            decided > self.state["retry_after_decided"].get(campaign, -1)
            and researcher_streak(state, seeded_ideas(self.root / OUTPUTS / campaign))
            >= state["rotate_after"]
        )

    def begin(self, predecessor):
        rotation_id = identifier("rotation")
        directory = self.output / "rotations" / rotation_id
        if predecessor is None:
            reference = self.state["reference"]
            start = {
                "recipe": RECIPE,
                "reference_run": reference,
                "chain": None,
                "budget": SEALED,
            }
            incumbent = {
                "campaign": None,
                "id": Path(reference).name,
                "run": str((self.root / reference).resolve()),
                "source": str(self.root),
            }
            streak = None
        else:
            state = campaign_state(self.root, predecessor)
            current = state["incumbent"]
            chain = {
                "predecessor": predecessor,
                "incumbent": current["id"],
                "source": self.relative(current["source"]),
                "run": self.relative(current["run"]),
                "predictions_sha256": current["predictions_sha256"],
            }
            start = {
                "recipe": chain_recipe(chain["run"]),
                "reference_run": chain["run"],
                "chain": chain,
                "budget": state["budget"],
            }
            incumbent = {"campaign": predecessor, **current}
            streak = researcher_streak(
                state, seeded_ideas(self.root / OUTPUTS / predecessor)
            )
        self.state["rotation"] = {
            "id": rotation_id,
            "directory": str(directory),
            "started_at": time.time(),
            "phase": "sourcing",
            "predecessor": predecessor,
            "incumbent": incumbent,
            "start": start,
            "streak": streak,
            "rejections": 0,
            "previous_rejection": None,
            "job": None,
            "attempts": [],
            "queue_rejections": [],
            "valid": [],
            "successor": None,
        }
        directory.mkdir(parents=True)
        if predecessor is not None:
            self.mark(predecessor, "retiring")
        self.save()
        self.source(self.state["rotation"])

    def source(self, rotation):
        for topic in queued_topics(self.root):
            try:
                campaign = free_id(self.root, topic.name)
                definition, _ = self.stage(rotation, topic, campaign, "queue")
            except ERRORS as error:
                self.reject_queue(rotation, topic, error)
                continue
            self.materialize(rotation, topic, campaign, "queue", definition.title)
            return
        packet = Path(rotation["directory"]) / "packet.json"
        if not packet.exists():
            pool = load_pool(self.root)
            write_json(
                packet,
                build_packet(
                    self.root,
                    rotation["incumbent"],
                    campaign_records(self.root),
                    {item["campaign"]: item["origin"] for item in self.state["topics"]},
                    [
                        seed.model_dump(mode="json")
                        for seed in pool.seeds
                        if seed.id not in self.state["pool_taken"]
                    ],
                ),
            )
        rotation["phase"] = "drafting"
        self.draft(rotation)

    def stage(self, rotation, topic, campaign, origin):
        target = Path(rotation["directory"]) / "staged" / campaign
        if target.exists():
            shutil.rmtree(target)
        return stage(self.root, topic, target, campaign, rotation["start"], origin)

    def reject_queue(self, rotation, topic, error):
        target = (
            self.root
            / REJECTED
            / f"{topic.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(topic, target)
        atomic_write(target / "rejection.txt", f"{type(error).__name__}: {error}\n")
        rotation["queue_rejections"].append(
            {
                "topic": topic.name,
                "moved_to": self.relative(target),
                "reason": str(error),
            }
        )
        self.save()

    def agent_source(self, rotation):
        return Path(rotation["incumbent"]["source"])

    def agent(self, rotation, role, context):
        source = self.agent_source(rotation)
        job_id, directory = self.create_job(
            role, source, [], payload={"rotation": rotation["id"]}
        )
        profile = providers.resolved(
            self.state["provider"],
            self.state["profiles"][self.state["provider"]][role],
        )
        command = providers.invocation(
            self.state["provider"], TOPIC_REPORTS[role], profile, source, directory
        )
        context = {
            **context,
            "python_executable": sys.executable,
            "previous_rejection": rotation["previous_rejection"],
        }
        prompt = role_prompt(
            (self.root / "AGENTS.md").read_text(),
            (self.root / "autoresearch/roles" / f"{role}.md").read_text(),
            topic_rules(rotation["streak"]),
            context,
        )
        atomic_write(directory / "stdin.txt", prompt)
        spec = read_json(directory / "spec.json")
        spec.update(command=command, provider=self.state["provider"])
        write_json(directory / "spec.json", spec)
        write_json(
            directory / "assignment.json", {"profile": profile, "context": context}
        )
        rotation["job"] = job_id
        self.save()
        self.launch_worker(self.state["jobs"][job_id])

    def packet(self, rotation):
        return read_json(Path(rotation["directory"]) / "packet.json")

    def draft(self, rotation):
        packet = self.packet(rotation)
        self.agent(
            rotation,
            "researcher",
            {
                "assignment": "topic_draft",
                "drafts": "two or three",
                "budget": rotation["start"]["budget"],
                "packet": str(Path(rotation["directory"]) / "packet.json"),
                "incumbent": packet["incumbent"],
                "seed_pool": packet["seed_pool"],
                "closed_families": packet["closed_families"],
                "near_misses": packet["near_misses"],
                "past_topics": packet["topics"],
                "codebase_map": packet["codebase_map"],
                "incumbent_source": rotation["incumbent"]["source"],
            },
        )

    def pick(self, rotation):
        packet = self.packet(rotation)
        self.agent(
            rotation,
            "orchestrator",
            {
                "assignment": "topic_pick",
                "packet": str(Path(rotation["directory"]) / "packet.json"),
                "incumbent": {
                    key: packet["incumbent"][key]
                    for key in ("id", "campaign", "macro_auc", "per_label")
                },
                "past_topics": packet["topics"],
                "drafts": rotation["valid"],
            },
        )

    def harvest(self):
        for job in list(self.state["jobs"].values()):
            if job["processed"]:
                continue
            directory = Path(job["path"])
            receipt = directory / "receipt.json"
            if receipt.exists():
                job["processed"] = True
                self.result(job, read_json(receipt))
                self.save()
                continue
            if alive(optional_json(directory / "owner.json")) or alive(
                optional_json(directory / "launcher.json")
            ):
                continue
            if time.time() - job["created_at"] < 10:
                continue
            if (directory / "started.json").exists() or self.state[
                "status"
            ] == "stopping":
                write_json(
                    receipt,
                    {
                        "exit_code": -1,
                        "cancelled": self.state["status"] == "stopping",
                        "error": "Worker lost or stopped; attempt will not be replayed",
                        "seconds": None,
                    },
                )
            else:
                self.launch_worker(job)

    def result(self, job, receipt):
        rotation = self.state["rotation"]
        if rotation is None or rotation["job"] != job["id"]:
            return
        rotation["job"] = None
        if receipt.get("cancelled") or self.state["status"] != "running":
            return
        directory = Path(job["path"])
        problem = providers.failure(self.state["provider"], directory, receipt)
        if problem and self.provider_wait(job, problem):
            return
        if receipt["exit_code"] != 0 or receipt.get("error") or problem:
            self.rejected(
                rotation,
                f"{job['kind']} failed: {problem or receipt}; inspect {directory}",
            )
            return
        self.state.get("provider_retries", {}).pop(f"{job['kind']}:", None)
        try:
            report = providers.result(
                self.state["provider"], TOPIC_REPORTS[job["kind"]], directory
            )
        except (ValidationError, ValueError, KeyError, IndexError, OSError) as error:
            self.rejected(rotation, f"{job['kind']} report rejected: {error}")
            return
        if job["kind"] == "researcher":
            self.apply_drafts(rotation, job, TopicDraftReport.model_validate(report))
        else:
            self.apply_pick(rotation, job, report)

    def rejected(self, rotation, reason):
        rotation["rejections"] += 1
        rotation["previous_rejection"] = reason
        if rotation["rejections"] >= REJECTION_LIMIT:
            self.cancel(
                rotation,
                f"{rotation['rejections']} consecutive failed topic attempts; last: {reason}",
            )

    def apply_drafts(self, rotation, job, report):
        packet = self.packet(rotation)
        attempt = Path(rotation["directory"]) / "drafts" / job["id"]
        verdicts, valid = {}, []
        ids = [draft.id for draft in report.drafts]
        for draft in report.drafts:
            reasons = check_draft(draft, packet, self.root)
            if ids.count(draft.id) > 1:
                reasons.append("Draft IDs must be unique within a report")
            if not reasons:
                directory = attempt / draft.id
                write_draft(draft, directory)
                try:
                    self.stage(rotation, directory, draft.id, "researcher")
                except ERRORS as error:
                    reasons.append(f"Generated campaign fails validation: {error}")
            verdicts[draft.id] = reasons
            if not reasons:
                valid.append(
                    {
                        "id": draft.id,
                        "title": draft.title,
                        "directory": str(attempt / draft.id),
                        "draft": str(attempt / draft.id / "draft.yaml"),
                        "weaknesses": [item.model_dump() for item in draft.weaknesses],
                        "mechanism_families": draft.mechanism_families,
                        "scope": draft.scope.model_dump(
                            exclude={
                                "developer_instructions",
                                "researcher_instructions",
                            }
                        ),
                        "seeds": [
                            {
                                "id": seed.id,
                                "title": seed.title,
                                "complexity": seed.complexity,
                                "structural": seed.structural,
                            }
                            for seed in draft.seeds
                        ],
                    }
                )
        rotation["attempts"].append(
            {
                "job": job["id"],
                "report": str(Path(job["path"]) / "report.json"),
                "checks": verdicts,
            }
        )
        write_json(attempt / "checks.json", verdicts)
        if not valid:
            self.rejected(
                rotation,
                "Every draft failed the topic checks: "
                + "; ".join(
                    f"{key}: {' | '.join(value)}" for key, value in verdicts.items()
                ),
            )
            return
        rotation.update(rejections=0, previous_rejection=None, valid=valid)
        if len(valid) == 1:
            self.accept(rotation, valid[0], None)
            return
        rotation["phase"] = "picking"
        self.pick(rotation)

    def apply_pick(self, rotation, job, report):
        choice = next(
            (item for item in rotation["valid"] if item["id"] == report["draft_id"]),
            None,
        )
        if choice is None:
            self.rejected(
                rotation,
                f"Pick {report['draft_id']!r} is not one of the valid drafts "
                f"{[item['id'] for item in rotation['valid']]}",
            )
            return
        rotation.update(rejections=0, previous_rejection=None)
        self.accept(rotation, choice, str(Path(job["path"]) / "report.json"))

    def accept(self, rotation, choice, pick):
        rotation["pick"] = {"draft": choice["id"], "report": pick}
        self.materialize(
            rotation,
            Path(choice["directory"]),
            choice["id"],
            "researcher",
            choice["title"],
        )

    def materialize(self, rotation, topic, campaign, origin, title):
        target = self.root / CAMPAIGNS / campaign
        rotation["successor"] = {
            "campaign": campaign,
            "origin": origin,
            "source": self.relative(topic) if origin == "queue" else str(topic),
            "title": title,
            "pool_seeds": [],
        }
        rotation["phase"] = "successor"
        self.save()
        try:
            shutil.copytree(Path(rotation["directory"]) / "staged" / campaign, target)
            seeds = {
                idea["id"]
                for path in seed_files(target)
                for idea in yaml.safe_load(path.read_text())["ideas"]
            }
            rotation["successor"]["pool_seeds"] = [
                seed.id
                for seed in load_pool(self.root).seeds
                if seed.id in seeds and seed.id not in self.state["pool_taken"]
            ]
            register_campaign(self.root, campaign, title, origin, rotation["start"])
            supervisor = Supervisor(self.root, campaign)
            supervisor.validate()
            if rotation["predecessor"] is not None:
                self.mark(campaign, "successor")
            supervisor.initialize(self.state["provider"], self.state["gpu"])
            supervisor.launch()
        except ERRORS as error:
            if origin == "queue":
                self.reject_queue(rotation, topic, error)
            self.cancel(rotation, f"{campaign} failed to start: {error}")
            return
        self.save()
        if rotation["predecessor"] is None:
            self.commit(rotation)

    def adopted(self, rotation):
        state = campaign_state(self.root, rotation["predecessor"])
        current = state["incumbent"]
        return (current["id"], current["predictions_sha256"]) != (
            rotation["incumbent"]["id"],
            rotation["incumbent"]["predictions_sha256"],
        )

    def advance(self, rotation):
        predecessor = rotation["predecessor"]
        if predecessor is not None and self.adopted(rotation):
            self.cancel(
                rotation, f"{predecessor} adopted a candidate; its streak reset"
            )
            return
        phase = rotation["phase"]
        if phase in {"sourcing", "drafting", "picking"}:
            if rotation["job"] is None and not self.waiting():
                if phase == "sourcing":
                    self.source(rotation)
                elif phase == "drafting":
                    self.draft(rotation)
                else:
                    self.pick(rotation)
            return
        successor = rotation["successor"]["campaign"]
        if phase == "successor":
            state = campaign_state(self.root, successor)
            if state is not None and state["status"] == "paused":
                self.cancel(
                    rotation,
                    f"{successor} paused before its first approval: {state['reason']}",
                )
            elif state is not None and approved(state):
                rotation["phase"] = "handover"
                self.mark(predecessor, "handover")
            return
        state = campaign_state(self.root, predecessor)
        if state["status"] == "stopped" and not self.repairing(predecessor):
            self.commit(rotation)
        elif state["running"] is None and not any(
            event["kind"] == "completion" for event in state["events"]
        ):
            self.request_stop(predecessor, False)

    def cancel(self, rotation, reason):
        successor = (rotation.get("successor") or {}).get("campaign")
        if successor is not None:
            self.request_stop(successor, True)
            self.mark(successor, None)
            record = self.topic_record(
                successor,
                rotation["successor"]["origin"],
                rotation["successor"]["source"],
                rotation["successor"]["title"],
                rotation["predecessor"],
                rotation["start"]["chain"],
            )
            record.update(status="superseded", retired_at=time.time(), reason=reason)
            self.state["topics"].append(record)
            register_campaign(
                self.root,
                successor,
                rotation["successor"]["title"],
                rotation["successor"]["origin"],
                rotation["start"],
                superseded=True,
            )
        predecessor = rotation["predecessor"]
        if predecessor is not None:
            self.mark(predecessor, None)
            state = campaign_state(self.root, predecessor)
            if (
                state["status"] in {"stopping", "stopped"}
                and predecessor not in self.state["to_resume"]
                and not self.repairing(predecessor)
            ):
                self.state["to_resume"].append(predecessor)
            self.state["retry_after_decided"][predecessor] = len(decided_runs(state))
        else:
            self.state["first_attempt_after"] = time.time() + FIRST_TOPIC_RETRY_SECONDS
        self.close(rotation, "cancelled", reason)

    def close(self, rotation, result, reason):
        job = self.state["jobs"].get(rotation["job"])
        if job and not job["processed"]:
            owner = Path(job["path"]) / "owner.json"
            if owner.exists():
                terminate(read_json(owner))
        rotation.update(result=result, reason=reason, ended_at=time.time())
        write_json(Path(rotation["directory"]) / "rotation.json", rotation)
        self.state["rotations"].append(
            {
                key: rotation.get(key)
                for key in (
                    "id",
                    "directory",
                    "started_at",
                    "ended_at",
                    "result",
                    "reason",
                )
            }
        )
        self.state["rotation"] = None
        self.save()

    def commit(self, rotation):
        predecessor = rotation["predecessor"]
        successor = rotation["successor"]
        if predecessor is not None:
            self.refresh(predecessor)
            self.topic(predecessor).update(
                status="retired",
                retired_at=time.time(),
                streak_at_rotation=rotation["streak"],
            )
        self.mark(successor["campaign"], None)
        if predecessor is not None:
            self.mark(predecessor, None)
        record = self.topic_record(
            successor["campaign"],
            successor["origin"],
            successor["source"],
            successor["title"],
            predecessor,
            rotation["start"]["chain"],
        )
        record["pool_seeds"] = successor["pool_seeds"]
        self.state["topics"].append(record)
        for seed in successor["pool_seeds"]:
            self.state["pool_taken"][seed] = successor["campaign"]
        if successor["origin"] == "queue":
            queued = self.root / successor["source"]
            if queued.is_dir():
                shutil.rmtree(queued)
        self.state["current"] = successor["campaign"]
        retired = [item for item in self.state["topics"] if item["status"] == "retired"]
        quiet = retired[-QUIET_TOPICS:]
        self.state["attention"] = (
            f"{QUIET_TOPICS} consecutive topics without an adoption: "
            + ", ".join(item["campaign"] for item in quiet)
            if len(quiet) == QUIET_TOPICS
            and all((item["outcome"] or {}).get("adoptions", 0) == 0 for item in quiet)
            else None
        )
        self.close(rotation, "committed", None)
