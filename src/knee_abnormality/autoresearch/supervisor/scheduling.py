import sys
from pathlib import Path

from knee_abnormality.autoresearch.campaign.binding import parent_changed
from knee_abnormality.autoresearch.campaign.completion import completion_blockers
from knee_abnormality.autoresearch.runtime.gpu import gpu_available
from knee_abnormality.autoresearch.runtime.isolation import sandbox
from knee_abnormality.autoresearch.supervisor.prompts import (
    FORCED_SEED_INSTRUCTIONS,
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
        if self.next_event():
            self.schedule_event()
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

    def next_event(self):
        order = {"completion": 0, "review": 1, "proposals": 2}
        self.state["events"].sort(key=lambda event: order[event["kind"]])
        return self.state["events"][0] if self.state["events"] else None

    def schedule_event(self):
        if self.waiting():
            return
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
            self.route_completion(
                event,
                {
                    "status": "orchestrator",
                    "reasons": blockers or [event["previous_rejection"]],
                },
            )
        context = {"event": event}
        if self.seeds_pending():
            context["instructions"] = FORCED_SEED_INSTRUCTIONS
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
                self.request_review(
                    claimed,
                    refresh_reason="Approval is absent or its parent, candidate or external evidence changed",
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
                or self.state["seed_ideas"]
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
                "campaign_instructions": self.policy.developer_instructions,
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
        research = self.state["research"]
        needed = self.state["refill_to"] - len(self.state["queue"])
        new_results = [
            key
            for key, value in self.state["experiments"].items()
            if value.get("decision") and key not in research["reviewed"]
        ]
        self.role(
            "researcher",
            Path(self.state["incumbent"]["source"]),
            {
                "needed": needed,
                "new_results": new_results,
                "previous_rejection": research["previous_rejection"],
                "campaign_instructions": self.policy.researcher_instructions,
            },
            payload={"new_results": new_results, "needed": needed},
        )
