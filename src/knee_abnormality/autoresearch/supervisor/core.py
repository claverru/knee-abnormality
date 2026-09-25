import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import yaml

from knee_abnormality.autoresearch.campaign.binding import (
    reference_seed,
)
from knee_abnormality.autoresearch.campaign.definition import (
    CampaignPolicy,
    load_campaign,
)
from knee_abnormality.autoresearch.campaign.gates import (
    initial_limits,
    validate_config,
)
from knee_abnormality.autoresearch.campaign.reports import ResearchReport
from knee_abnormality.autoresearch.campaign.rules import SEED_TERMINAL
from knee_abnormality.autoresearch.reporting.progress import findings_summary
from knee_abnormality.autoresearch.runtime.isolation import (
    environment,
    manifest,
    snapshot,
)
from knee_abnormality.autoresearch.runtime.storage import (
    atomic_write,
    lock,
    optional_json,
    process_identity,
    read_json,
    sha256,
    spawn,
    terminate,
    write_json,
)
from knee_abnormality.autoresearch.supervisor.common import (
    identifier,
    seed_files,
)
from knee_abnormality.autoresearch.supervisor.experiments import ExperimentMixin
from knee_abnormality.autoresearch.supervisor.jobs import JobMixin
from knee_abnormality.autoresearch.supervisor.prompts import IMPORT_INSTRUCTIONS
from knee_abnormality.autoresearch.supervisor.results import ResultMixin
from knee_abnormality.autoresearch.supervisor.scheduling import SchedulingMixin
from knee_abnormality.config import STUDY_HEADERS, load_config


class Supervisor(JobMixin, ExperimentMixin, ResultMixin, SchedulingMixin):
    def __init__(self, root: Path, campaign: str):
        self.root = root
        self.docs = root / "autoresearch/campaigns" / campaign
        definition = load_campaign(self.docs / "campaign.yaml")
        self.campaign = definition.model_dump(mode="json")
        self.policy = definition.policy
        if self.campaign["id"] != campaign:
            raise ValueError(f"Mismatched campaign: {campaign}")
        self.output = (root / self.campaign["artifacts_root"]).resolve()
        self.state_path = self.output / "state.json"
        self.state = optional_json(self.state_path)
        if self.state is not None:
            if self.state["campaign"] != campaign:
                raise ValueError("Artifact directory belongs to a different campaign")
            sealed = CampaignPolicy.model_validate(self.state["policy"])
            if sealed != self.policy:
                raise ValueError(
                    "Campaign policy differs from its seal; create a new campaign"
                )
            self.policy = sealed
        self.wakeup = threading.Event()

    def save(self):
        self.state["revision"] += 1
        self.state["updated_at"] = time.time()
        write_json(self.state_path, self.state)
        self.project()

    def project(self):
        revision = self.state["revision"]
        atomic_write(
            self.docs / "backlog.yaml",
            yaml.safe_dump(
                {
                    "revision": revision,
                    "queue": self.state["queue"],
                },
                sort_keys=False,
            ),
        )
        atomic_write(
            self.docs / "ideas.yaml",
            yaml.safe_dump(
                {
                    "revision": revision,
                    "ideas": list(self.state["ideas"].values()),
                },
                sort_keys=False,
            ),
        )
        findings = "\n\n".join(
            f"## {entry['event']}\n\n{entry['finding']}\n\n{entry['rationale']}"
            + (
                "\n\nUnresolved concerns:\n"
                + "\n".join(f"- {value}" for value in entry["concerns"])
                if entry.get("concerns")
                else ""
            )
            + (
                "\n\nEvidence:\n"
                + "\n".join(f"- {value}" for value in entry["evidence_refs"])
                if entry.get("evidence_refs")
                else ""
            )
            for entry in self.state["findings"]
        )
        atomic_write(
            self.output / "findings.md", f"# Campaign findings\n\n{findings}\n"
        )
        atomic_write(
            self.docs / "findings.md", findings_summary(self.state, self.campaign)
        )

    def launch(self):
        spawn(
            [
                "knee_abnormality.autoresearch.cli",
                "supervise",
                self.state["campaign"],
                "--root",
                str(self.root),
            ],
            self.root,
            self.output / "supervisor.log",
            self.output / "supervisor.json",
            environment(self.root),
        )

    def recipe_limits(self):
        config = load_config(self.root / self.campaign["recipe"]).model_dump(
            mode="json"
        )
        reference = (self.root / self.campaign["reference_run"]).resolve()
        limits = initial_limits(
            reference,
            config,
            self.campaign["reference_equivalences"],
            self.campaign["reference_config_exceptions"],
            self.campaign["reference_identity_exceptions"],
        )
        validate_config(config, config, self.policy)
        return config, reference, limits

    def initialize(self, provider: str, gpu: str):
        if self.state is not None:
            raise ValueError("Campaign already exists; use resume")
        if (
            yaml.safe_load((self.docs / "ideas.yaml").read_text())["ideas"]
            or yaml.safe_load((self.docs / "backlog.yaml").read_text())["queue"]
        ):
            raise ValueError(
                "Research records exist without runtime state; restore state instead of resetting history"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        base = self.output / "baseline/source"
        if base.exists():
            raise ValueError(
                "Incomplete initialization exists; inspect baseline/ before starting again"
            )
        config, reference, limits = self.recipe_limits()
        headers = self.root / STUDY_HEADERS
        if not headers.is_file():
            raise ValueError(f"Missing bootstrap acquisition metadata: {headers}")
        snapshot(self.root, base)
        config["seed"] = reference_seed({"run": str(reference)})
        atomic_write(
            base / "configs/training.yaml", yaml.safe_dump(config, sort_keys=False)
        )
        write_json(self.output / "baseline/manifest.json", manifest(base))
        write_json(self.output / "baseline/config.json", config)
        write_json(self.output / "limits.json", limits)
        self.state = {
            "revision": 0,
            "campaign": self.campaign["id"],
            "policy": self.policy.model_dump(mode="json"),
            "provider": provider,
            "gpu": gpu,
            "status": "calibrating",
            "reason": None,
            "pause_kind": None,
            "profiles": yaml.safe_load(
                (self.root / "autoresearch/agents.yaml").read_text()
            ),
            "base": str(base),
            "reference": str(reference),
            "headers": str(headers),
            "headers_sha256": sha256(headers),
            "limits": limits,
            "config": config,
            "incumbent": {
                "id": "historical-control",
                "source": str(base),
                "run": str(reference),
                "predictions_sha256": limits["reference_predictions_sha256"],
            },
            "queue": [],
            "ideas": {},
            "experiments": {},
            "jobs": {},
            "events": [],
            "findings": [],
            "claimed": None,
            "running": None,
            "evidence_revision": 0,
            "last_refill_revision": -1,
            "refill_below": self.campaign["backlog"]["refill_below"],
            "refill_to": self.campaign["backlog"]["refill_to"],
            "seed_order": self.campaign["backlog"]["seed_order"],
            "seed_ideas": [],
            "control_retrain": None,
            "control_retrains": [],
        }
        seeds = seed_files(self.docs)
        for path in seeds:
            if self.state["seed_order"] == "forced":
                self.queue_seeds(path)
            else:
                self.import_ideas(path)
        if not seeds:
            self.save()

    def queue_seeds(self, path: Path):
        report = ResearchReport.model_validate(yaml.safe_load(path.read_text()))
        ids = [idea.id for idea in report.ideas]
        if not ids or len(ids) != len(set(ids)) or set(ids) & set(self.state["ideas"]):
            raise ValueError("Forced seeds need nonempty, unique, unused idea IDs")
        if any(idea.dependencies for idea in report.ideas):
            raise ValueError(
                "Forced seeds cannot declare dependencies; their file order sets the sequence"
            )
        artifact = self.output / "imports" / identifier("seeds") / "report.json"
        write_json(artifact, report.model_dump(mode="json"))
        for idea in report.ideas:
            self.state["ideas"][idea.id] = {
                **idea.model_dump(mode="json"),
                "status": "queued",
                "created_revision": self.state["revision"],
                "seeded": str(path.resolve()),
            }
        self.state["queue"].extend(ids)
        self.state["seed_ideas"].extend(ids)
        self.save()

    def seeds_pending(self):
        return self.state["seed_order"] == "forced" and any(
            self.state["ideas"][idea_id]["status"] not in SEED_TERMINAL
            for idea_id in self.state["seed_ideas"]
        )

    def event(self, kind, payload):
        self.state["events"].append({"id": identifier(kind), "kind": kind, **payload})

    def request_review(self, exp, gate_error=None, refresh_reason=None):
        exp["stage"] = "awaiting_review"
        payload = {"experiment": exp["id"], "gate_error": gate_error}
        if refresh_reason:
            payload["refresh_reason"] = refresh_reason
        self.event("review", payload)

    def route_completion(self, event, record):
        event["completion_reporting"] = record
        self.state["experiments"][event["experiment"]]["completion_reporting"] = dict(
            record
        )

    def pause(self, reason, kind=None):
        self.state.update(status="paused", reason=reason, pause_kind=kind)

    def import_ideas(self, path: Path):
        report = ResearchReport.model_validate(yaml.safe_load(path.read_text()))
        ids = [idea.id for idea in report.ideas]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Import requires nonempty ideas with unique IDs")
        known = set(self.state["ideas"])
        for event in self.state["events"]:
            known.update(
                idea["id"] for idea in event.get("report", {}).get("ideas", [])
            )
        if known.intersection(ids):
            raise ValueError(
                "Imported IDs already exist in the registry or pending proposals"
            )
        artifact = self.output / "imports" / identifier("ideas") / "report.json"
        value = report.model_dump(mode="json")
        write_json(artifact, value)
        self.event(
            "proposals",
            {
                "report": value,
                "artifact": str(artifact),
                "source": str(path.resolve()),
                "instructions": IMPORT_INSTRUCTIONS,
            },
        )
        self.save()

    def run(self):
        with lock(self.output / "supervisor.lock"):
            write_json(self.output / "supervisor.json", process_identity(os.getpid()))
            self.project()

            def stop(signum, frame):
                write_json(self.output / "stop.json", {"abort": False})
                self.wakeup.set()

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            while True:
                try:
                    resume = self.output / "resume.json"
                    if resume.exists():
                        (self.output / "stop.json").unlink(missing_ok=True)
                        self.state.update(
                            status="running"
                            if self.state.get("sealed_sha256")
                            else "calibrating",
                            reason=None,
                            pause_kind=None,
                        )
                        self.state.pop("retry_after", None)
                        control = self.state["control_retrain"]
                        if control and control["stage"] == "failed":
                            control["stage"] = "pending"
                        for job in self.state["jobs"].values():
                            job.pop("application_error", None)
                        self.save()
                        resume.unlink()
                    request = self.output / "stop.json"
                    if request.exists() and self.state["status"] != "stopping":
                        self.state["status"] = "stopping"
                        self.save()
                    self.harvest()
                    if request.exists():
                        abort = read_json(request)["abort"]
                        if self.state["status"] != "stopping":
                            self.state["status"] = "stopping"
                            self.save()
                        running = self.state["experiments"].get(self.state["running"])
                        if (
                            not abort
                            and running
                            and running["stage"] == "needs_comparison"
                        ):
                            self.comparison(running)
                        for job in self.outstanding():
                            if abort or job["kind"] not in {
                                "training",
                                "comparison",
                                "control",
                            }:
                                owner = Path(job["path"]) / "owner.json"
                                if owner.exists():
                                    terminate(read_json(owner))
                        if not self.outstanding():
                            self.state["status"] = "stopped"
                            self.save()
                            return
                    else:
                        self.schedule()
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    subprocess.SubprocessError,
                ) as error:
                    self.pause(f"{type(error).__name__}: {error}")
                    self.save()
                self.wakeup.wait(5)
                self.wakeup.clear()
