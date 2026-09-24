import sys
from pathlib import Path

import yaml

from knee_abnormality.autoresearch.campaign.binding import parent_changed
from knee_abnormality.autoresearch.campaign.completion import completion_blockers
from knee_abnormality.autoresearch.campaign.gates import validate_config
from knee_abnormality.autoresearch.runtime.isolation import sandbox
from knee_abnormality.autoresearch.runtime.storage import atomic_write
from knee_abnormality.autoresearch.supervisor.common import gpu_available
from knee_abnormality.autoresearch.supervisor.prompts import (
    FORCED_SEED_INSTRUCTIONS,
    ORCHESTRATOR_INSTRUCTIONS,
    researcher_instructions,
)


class SchedulingMixin:
    def schedule(self):
        if self.state["status"] not in {"running", "calibrating"}:
            return
        if self.state["status"] == "calibrating":
            self.schedule_calibration()
            return
        if "retry_after" in self.state and not self.waiting():
            self.state.pop("retry_after")
            self.state["reason"] = None
            self.save()
        running = self.state["experiments"].get(self.state["running"])
        if running and running["stage"] == "needs_comparison":
            self.comparison(running)
        if self.active("orchestrator") or self.active("completion_reporter"):
            return
        if (
            self.state["running"]
            and self.state["experiments"][self.state["running"]]["stage"] == "launching"
        ):
            return
        if self.state["events"]:
            self.schedule_event()
            return
        if self.state.get("control_retrain"):
            self.schedule_control()
            return
        if self.advance_claimed():
            return
        self.claim_next()
        self.start_developer()
        self.refill()

    def schedule_calibration(self):
        if self.active("calibration"):
            return
        base = Path(self.state["base"])
        destination = self.output / "baseline"
        self.submit(
            "calibration",
            base,
            sandbox(
                [
                    sys.executable,
                    "-m",
                    "knee_abnormality.autoresearch.cli",
                    "calibrate",
                    "--config",
                    str(base / "configs/training.yaml"),
                    "--destination",
                    str(destination),
                ],
                base,
                [destination],
                readonly=[base],
            ),
            timeout=14400,
        )

    def schedule_event(self):
        if self.waiting():
            return
        self.state["events"].sort(
            key=lambda event: {"completion": 0, "review": 1, "proposals": 2}[
                event["kind"]
            ]
        )
        event = self.state["events"][0]
        if event["kind"] == "review" and not event.get("gate_error"):
            exp = self.state["experiments"][event["experiment"]]
            if parent_changed(self.state, exp) or not exp.get("integration"):
                self.reconcile(exp)
                self.state["events"].pop(0)
                self.save()
                return
            event["review_binding"] = self.review_binding(exp)
        if event["kind"] == "completion" and not event.get("completion_reporting"):
            blockers = completion_blockers(self.state, event)
            if not blockers and not event.get("previous_rejection"):
                self.role(
                    "completion_reporter",
                    Path(self.state["base"]),
                    {"event": event},
                    payload={"event": event},
                )
                return
            event["completion_reporting"] = {
                "status": "orchestrator",
                "reasons": blockers or [event["previous_rejection"]],
            }
            self.state["experiments"][event["experiment"]]["completion_reporting"] = (
                dict(event["completion_reporting"])
            )
        context = {
            "event": event,
            "instructions": ORCHESTRATOR_INSTRUCTIONS
            + (FORCED_SEED_INSTRUCTIONS if self.seeds_pending() else ""),
        }
        if event.get("experiment"):
            context["experiment"] = self.state["experiments"][event["experiment"]]
        self.role(
            "orchestrator",
            Path(self.state["base"]),
            context,
            payload={"event": event},
        )

    def advance_claimed(self):
        running = self.state["experiments"].get(self.state["running"])
        claimed = self.state["experiments"].get(self.state["claimed"])
        if (
            claimed
            and claimed["stage"]
            in {"needs_prepare", "needs_config", "repair_requested", "prepared"}
            and (
                parent_changed(self.state, claimed)
                or (claimed["stage"] == "prepared" and not claimed.get("integration"))
            )
        ):
            self.reconcile(claimed)
            return True
        if claimed and claimed["stage"] == "needs_prepare":
            self.prepare(claimed, claimed["developer_report"])
            return True
        if claimed and claimed["stage"] == "needs_config":
            self.config_gate(claimed)
            return True
        if claimed and claimed["stage"] == "repair_requested":
            self.claim(repair=claimed)
            claimed = self.state["experiments"][self.state["claimed"]]
        if claimed and claimed["stage"] == "prepared":
            binding = self.review_binding(claimed)
            approval = claimed.get("implementation_review", {})
            if (
                approval.get("decision") != "approve"
                or approval.get("binding") != binding
            ):
                claimed["stage"] = "awaiting_review"
                self.event(
                    "review",
                    {
                        "experiment": claimed["id"],
                        "gate_error": None,
                        "refresh_reason": "Approval is absent or its parent, candidate or external evidence changed",
                    },
                )
                self.save()
                return True
        if (
            claimed
            and claimed["stage"] == "prepared"
            and running is None
            and gpu_available(self.state["gpu"])
        ):
            self.training(claimed)
        return False

    def claim_next(self):
        if (
            self.state["claimed"] is None
            and self.state.get("crash_repair")
            and (
                len(self.state["queue"]) < self.state["refill_to"]
                or self.state.get("seed_order") == "forced"
            )
        ):
            crashed = self.state["experiments"][self.state.pop("crash_repair")]
            self.state["queue"].insert(0, crashed["idea"])
            self.claim(repair=crashed)
        if self.state["claimed"] is None and self.state["queue"]:
            self.claim()

    def start_developer(self):
        claimed = self.state["experiments"].get(self.state["claimed"])
        if (
            not claimed
            or claimed["stage"] != "needs_developer"
            or self.active("developer")
            or self.waiting()
        ):
            return
        claimed["stage"] = "developing"
        self.role(
            "developer",
            Path(claimed["workspace"]),
            {
                "idea": self.state["ideas"][claimed["idea"]],
                "experiment": claimed,
                "repair_instruction": self.state["experiments"][
                    claimed["repair_of"]
                ].get("repair_instruction")
                if claimed["repair_of"]
                else None,
                "previous_rejection": claimed.get("previous_rejection"),
                "completion": self.policy.developer_instructions,
            },
            payload={"experiment": claimed["id"]},
        )

    def refill(self):
        if (
            len(self.state["queue"]) >= self.state["refill_below"]
            or self.active("researcher")
            or self.waiting()
            or self.state["last_refill_revision"] == self.state["evidence_revision"]
            or self.seeds_pending()
        ):
            return
        self.state["last_refill_revision"] = self.state["evidence_revision"]
        self.role(
            "researcher",
            Path(self.state["incumbent"]["source"]),
            {
                "needed": self.state["refill_to"] - len(self.state["queue"]),
                "instructions": researcher_instructions(
                    self.policy.researcher_instructions
                ),
            },
        )

    def schedule_control(self):
        control = self.state["control_retrain"]
        if control["stage"] != "pending" or self.active("control"):
            return
        if self.state["running"] is not None or not gpu_available(self.state["gpu"]):
            return
        adopted = self.state["experiments"][control["adopted"]]
        control["attempts"] += 1
        directory = self.output / "controls" / f"{control['id']}-{control['attempts']}"
        directory.mkdir(parents=True)
        config = dict(adopted["config"])
        config["seed"] = control["seed"]
        config["name"] = control["id"]
        config["paths"] = {**config["paths"], "output_root": str(directory)}
        validate_config(self.state["config"], config, self.policy)
        config_path = directory / "config.yaml"
        atomic_write(config_path, yaml.safe_dump(config, sort_keys=False))
        source = Path(adopted["source"])
        writable = [directory]
        cache = Path.home() / ".cache"
        if cache.exists():
            writable.append(cache)
        control.update(stage="training", directory=str(directory))
        self.submit(
            "control",
            source,
            sandbox(
                [
                    sys.executable,
                    "-m",
                    "scripts.pipeline.train",
                    str(config_path),
                    "--folds",
                    "0",
                    "--output-root",
                    str(directory),
                    "--run-id",
                    "training",
                ],
                source,
                writable,
                gpu=True,
                readonly=[source, config_path],
            ),
            payload={"control": control["id"]},
            timeout=self.state["limits"]["max_run_seconds"],
            gpu=self.state["gpu"],
            extra_env={"KNEE_AUTORESEARCH_LIMITS": str(self.output / "limits.json")},
        )
