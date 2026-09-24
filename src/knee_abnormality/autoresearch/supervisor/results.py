import secrets
from pathlib import Path

from knee_abnormality.autoresearch.campaign.binding import (
    parent_changed,
    reference_seed,
)
from knee_abnormality.autoresearch.campaign.completion import completion_blockers
from knee_abnormality.autoresearch.campaign.gates import (
    active_incumbent_ideas,
    dependency_blockers,
    evaluate,
    validate_config,
    verdict,
)
from knee_abnormality.autoresearch.campaign.reports import (
    REPORTS,
    OrchestratorReport,
)
from knee_abnormality.autoresearch.runtime import providers
from knee_abnormality.autoresearch.runtime.storage import (
    read_json,
    sha256,
    tail,
    write_json,
)
from knee_abnormality.autoresearch.scoring.diagnostics import (
    diagnostics_path,
    read_diagnostics,
)
from knee_abnormality.autoresearch.supervisor.common import (
    identifier,
    trained_epochs,
    work_item,
)
from knee_abnormality.autoresearch.supervisor.prompts import COMPLETION_RATIONALE


class ResultMixin:
    def process_result(self, job, receipt):
        kind = job["kind"]
        directory = Path(job["path"])
        exp = self.state["experiments"].get(job["payload"].get("experiment"))
        failed = receipt["exit_code"] != 0 or receipt.get("error")
        if (
            kind == "comparison"
            and receipt.get("cancelled")
            and not (directory / "started.json").exists()
        ):
            exp["stage"] = "needs_comparison"
            return
        if kind == "control":
            self.control_result(receipt)
            return
        if receipt.get("cancelled") and kind not in {"training", "comparison"}:
            self.cancelled_result(job, exp)
            return
        if kind == "training":
            self.training_result(exp, receipt, directory)
            return
        if kind == "comparison":
            self.comparison_result(exp, receipt, directory, failed)
            return
        problem = (
            providers.failure(self.state["provider"], directory, receipt)
            if kind in REPORTS
            else None
        )
        if failed or problem:
            self.failed_result(job, exp, directory, problem)
            return
        if kind == "calibration":
            self.seal()
        elif kind == "config":
            self.config_result(exp, receipt, directory)
        else:
            self.agent_result(job, exp, directory)

    def control_result(self, receipt):
        control = self.state["control_retrain"]
        if receipt.get("cancelled") or receipt["exit_code"] == 75:
            control["stage"] = "pending"
            return
        directory = Path(control["directory"])
        reasons = evaluate(directory, self.state["limits"], receipt)
        try:
            seed = read_json(directory / "training/effective_config.json")["seed"]
            if seed != control["seed"]:
                reasons.append("Control training seed differs from the drawn seed")
        except (OSError, KeyError, ValueError) as error:
            reasons.append(f"Missing or malformed control seed evidence: {error}")
        if reasons:
            control.update(stage="failed", reasons=reasons)
            self.state.update(
                status="paused",
                reason=f"Control retrain of {control['adopted']} failed; inspect {directory}: "
                + "; ".join(reasons)
                + ". Fix the cause and resume to retrain.",
            )
            return
        run = directory / "training"
        previous = self.state["incumbent"]
        self.state["incumbent"] = {
            **previous,
            "run": str(run),
            "predictions_sha256": sha256(run / "fold_0/validation_predictions.csv"),
        }
        self.state.setdefault("control_retrains", []).append(
            {
                "id": control["id"],
                "adopted": control["adopted"],
                "seed": control["seed"],
                "adopted_run": previous["run"],
                "control_run": str(run),
            }
        )
        self.state["control_retrain"] = None
        self.state["evidence_revision"] += 1

    def cancelled_result(self, job, exp):
        kind = job["kind"]
        if kind == "completion_reporter":
            self.escalate_completion(
                job,
                "Reporter cancelled; orchestrator handles completion after resume",
            )
        elif kind == "developer":
            exp["stage"] = "needs_developer"
        elif kind == "config":
            exp["stage"] = "needs_config"
        elif kind == "researcher":
            self.state["last_refill_revision"] = -1

    def training_result(self, exp, receipt, directory):
        if receipt["exit_code"] == 75:
            exp["stage"] = "prepared"
            self.state["running"] = None
            return
        self.dequeue_launched(exp)
        exp["training_receipt"] = receipt
        if (
            exp["launched"]
            and exp["repair_count"] < 1
            and receipt["exit_code"] != 0
            and not receipt.get("error")
            and not receipt.get("cancelled")
            and trained_epochs(Path(exp["directory"])) == 0
        ):
            self.state["running"] = None
            self.state["crash_repair"] = exp["id"]
            exp.update(
                stage="repair_requested",
                repair_instruction=(
                    f"Training exited with code {receipt['exit_code']} before completing an epoch; "
                    "no result was recorded. Fix the cause within the same idea; the repaired candidate "
                    f"returns to review. Full log: {directory / 'stderr.log'}. Stderr tail:\n"
                    f"{tail(directory / 'stderr.log')}"
                ),
            )
            return
        measured = {
            **receipt,
            "seconds": receipt["seconds"] + exp["preparation_seconds"]
            if receipt.get("seconds") is not None
            else None,
        }
        reasons = evaluate(Path(exp["directory"]), self.state["limits"], measured)
        if "training_seed" in exp.get("launch_binding", {}):
            try:
                actual_seed = read_json(
                    Path(exp["directory"]) / "training/effective_config.json"
                )["seed"]
                if (
                    actual_seed != exp["launch_binding"]["training_seed"]
                    or actual_seed != exp["seed"]
                    or actual_seed != reference_seed(exp["reference"])
                ):
                    reasons.append("Training seed differs from the recorded incumbent")
            except (OSError, KeyError, ValueError) as error:
                reasons.append(f"Missing or malformed training seed evidence: {error}")
        if not exp["launched"]:
            self.state["running"] = None
            exp["stage"] = "awaiting_review"
            self.event(
                "review",
                {
                    "experiment": exp["id"],
                    "gate_error": f"Launch failed: {reasons}",
                },
            )
        elif reasons:
            self.completion(exp, "invalid", {"reasons": reasons})
        else:
            exp["stage"] = "needs_comparison"

    def comparison_result(self, exp, receipt, directory, failed):
        total_seconds = (
            exp["preparation_seconds"]
            + exp["training_receipt"]["seconds"]
            + (receipt.get("seconds") or 0)
        )
        exp["measured_compute_seconds"] = total_seconds
        failed = failed or total_seconds > self.state["limits"]["max_run_seconds"]
        if failed:
            self.completion(
                exp,
                "invalid",
                {"reasons": [f"Comparison failed; inspect {directory}"]},
            )
        else:
            report = read_json(Path(exp["directory"]) / "comparison/comparison.json")
            diagnostics = read_diagnostics(
                Path(exp["directory"]) / "comparison/comparison.json"
            )
            macro_bootstrap = (diagnostics or {}).get("macro_bootstrap")
            self.completion(
                exp,
                verdict(report, macro_bootstrap),
                {
                    "comparison": str(
                        Path(exp["directory"]) / "comparison/comparison.json"
                    ),
                    "macro_auc": report["family_means"]["all"],
                    "adoption_bootstrap": macro_bootstrap,
                    "per_label": str(
                        diagnostics_path(
                            Path(exp["directory"]) / "comparison/comparison.json"
                        )
                    ),
                    "bootstrap": report["paired_report_group_bootstrap"]["intervals"][
                        "all"
                    ],
                    "soft_macro_auc": (diagnostics or {}).get("soft_macro"),
                },
            )

    def failed_result(self, job, exp, directory, problem):
        kind = job["kind"]
        if kind == "completion_reporter":
            self.escalate_completion(job, f"Reporter failed; inspect {directory}")
        if kind == "config":
            exp["stage"] = "awaiting_review"
            self.event(
                "review",
                {
                    "experiment": exp["id"],
                    "gate_error": f"Configuration failed; inspect {directory}",
                },
            )
            return
        if kind == "developer":
            exp["stage"] = "needs_developer"
        if problem and self.provider_wait(job, problem):
            if kind == "researcher":
                self.state["last_refill_revision"] = -1
            return
        detail = f" ({problem['kind']}: {problem['message'][:500]})" if problem else ""
        self.state.update(
            status="paused",
            reason=f"{kind} failed; inspect {directory}{detail}. Fix the cause and use resume.",
        )
        return

    def config_result(self, exp, receipt, directory):
        exp["preparation_seconds"] = receipt["seconds"]
        try:
            config = read_json(directory / "stdout.log")
            config.pop("config_file")
            config.pop("split_dir")
            validate_config(self.state["config"], config, self.policy)
            exp["config"] = config
            if not parent_changed(self.state, exp):
                self.review_binding(exp)
            exp["stage"] = "awaiting_review"
            self.event("review", {"experiment": exp["id"], "gate_error": None})
        except (ValueError, KeyError) as error:
            exp["stage"] = "awaiting_review"
            self.event("review", {"experiment": exp["id"], "gate_error": str(error)})

    def agent_result(self, job, exp, directory):
        kind = job["kind"]
        self.state.get("provider_retries", {}).pop(work_item(job), None)
        report = providers.result(self.state["provider"], kind, directory)
        if kind == "researcher":
            self.state["last_refill_revision"] = self.state["evidence_revision"]
            if report["ideas"]:
                self.event(
                    "proposals",
                    {"report": report, "artifact": str(directory / "report.json")},
                )
        elif kind == "developer":
            exp.pop("rejections", None)
            exp.pop("previous_rejection", None)
            exp["developer_report"] = report
            exp["integration"] = report["integration"]
            if report["status"] == "blocked":
                exp["stage"] = "awaiting_review"
                self.event(
                    "review",
                    {
                        "experiment": exp["id"],
                        "gate_error": report["blocker"] or "Developer blocked",
                    },
                )
            elif self.state["status"] != "running":
                exp["stage"] = "needs_prepare"
            else:
                try:
                    self.prepare(exp, report)
                except (OSError, ValueError, SyntaxError) as error:
                    exp["stage"] = "awaiting_review"
                    self.event(
                        "review",
                        {"experiment": exp["id"], "gate_error": str(error)},
                    )
        elif kind == "completion_reporter":
            self.apply_completion_report(job, report)
        else:
            self.apply_orchestrator(job["payload"]["event"], report)

    def completion_event(self, job):
        event_id = job["payload"]["event"]["id"]
        if not self.state["events"] or self.state["events"][0]["id"] != event_id:
            raise ValueError("Completion reporter event is stale")
        event = self.state["events"][0]
        if event["kind"] != "completion":
            raise ValueError("Completion reporter cannot handle another event type")
        routing = event.get("completion_reporting", {})
        if routing.get("job") != job["id"] or routing.get("status") != "assigned":
            raise ValueError("Completion reporter no longer owns the assigned event")
        return event

    def escalate_completion(self, job, reason):
        event = self.completion_event(job)
        record = {
            "status": "escalated",
            "job": job["id"],
            "reason": reason,
            "report": str(Path(job["path"]) / "report.json"),
        }
        event["completion_reporting"] = record
        self.state["experiments"][event["experiment"]]["completion_reporting"] = dict(
            record
        )

    def apply_completion_report(self, job, report):
        event = self.completion_event(job)
        reasons = completion_blockers(self.state, event)
        if report["status"] == "escalate" or report["concerns"] or reasons:
            reason = "; ".join([*reasons, *report["concerns"]]) or report["summary"]
            self.escalate_completion(job, reason)
            return
        exp = self.state["experiments"][event["experiment"]]
        self.apply_orchestrator(
            event,
            {
                "accepted_proposal_ids": [],
                "ranked_ids": None,
                "park_ids": [],
                "review": "none",
                "integration_review": None,
                "rationale": COMPLETION_RATIONALE,
                "finding": report["summary"],
                "concerns": [],
                "evidence_refs": [
                    str(Path(job["path"]) / "report.json"),
                    *report["evidence_refs"],
                ],
            },
        )
        exp["completion_reporting"] = {
            "status": "reported",
            "job": job["id"],
            "report": str(Path(job["path"]) / "report.json"),
        }

    def apply_orchestrator(self, event, report):
        OrchestratorReport.model_validate(report)
        if not self.state["events"] or self.state["events"][0]["id"] != event["id"]:
            raise ValueError("Orchestrator event is stale")
        proposed = {
            idea["id"]: idea for idea in event.get("report", {}).get("ideas", [])
        }
        if len(proposed) != len(event.get("report", {}).get("ideas", [])):
            raise ValueError("Researcher supplied duplicate idea IDs")
        if proposed.keys() & self.state["ideas"].keys():
            raise ValueError("New research reports cannot overwrite existing ideas")
        available = {
            key: value
            for key, value in self.state["ideas"].items()
            if value["status"] == "proposed"
        } | proposed
        accepted = report["accepted_proposal_ids"]
        parked = report["park_ids"]
        if len(accepted) != len(set(accepted)) or not set(accepted) <= available.keys():
            raise ValueError("Invalid accepted proposal IDs")
        if not set(parked) <= set(self.state["queue"]):
            raise ValueError("Can only park queued ideas")
        expected = (set(self.state["queue"]) | set(accepted)) - set(parked)
        ranking = (
            self.state["queue"]
            if report["ranked_ids"] is None
            else report["ranked_ids"]
        )
        seeds = (
            set(self.state["seed_ideas"])
            if self.state.get("seed_order") == "forced"
            else set()
        )
        if (
            set(ranking) != expected
            or len(ranking) != len(expected)
            or len([idea_id for idea_id in ranking if idea_id not in seeds])
            > self.state["refill_to"]
        ):
            raise ValueError(
                "Ranking must contain each eligible live queue entry once, within the target"
            )
        if seeds:
            kept = [
                idea_id
                for idea_id in self.state["queue"]
                if idea_id in seeds and idea_id not in parked
            ]
            if ranking[: len(kept)] != kept:
                raise ValueError(
                    "User-ordered seeds must keep their order ahead of every other idea"
                )
            reviewed = (
                self.state["experiments"][event["experiment"]]["idea"]
                if event["kind"] == "review"
                else None
            )
            if any(
                idea_id in seeds
                and (idea_id != reviewed or report["review"] != "block")
                for idea_id in parked
            ):
                raise ValueError(
                    "A user-ordered seed can only be parked by blocking it at its own review"
                )
        claimed = self.state["experiments"].get(self.state["claimed"])
        if (
            claimed
            and claimed["idea"] in parked
            and not (
                event["kind"] == "review" and event.get("experiment") == claimed["id"]
            )
        ):
            raise ValueError("An unrelated event cannot park work in progress")
        if (
            claimed
            and claimed["idea"] not in parked
            and (not ranking or ranking[0] != claimed["idea"])
        ):
            raise ValueError("Reserved queue head cannot be displaced")
        for idea_id in accepted:
            idea = available[idea_id]
            if not idea["novelty"].strip() or not idea["precedents"]:
                raise ValueError(
                    "A proposal needs precedent references and a material difference"
                )
            blockers = dependency_blockers(
                idea, self.state["ideas"], active_incumbent_ideas(self.state)
            )
            if blockers:
                raise ValueError(
                    f"Cannot queue {idea_id}: dependencies must be accepted and active in the incumbent: "
                    f"{'; '.join(blockers)}. Omit this ID from accepted_proposal_ids and ranked_ids; "
                    "it will remain proposed. Queue position does not resolve dependencies."
                )
        exp = self.state["experiments"].get(event.get("experiment"))
        binding = None
        if event["kind"] == "review":
            if report["review"] not in {"approve", "repair", "block"}:
                raise ValueError("Candidate review needs approve, repair, or block")
            if report["review"] == "approve" and (
                event.get("gate_error") or not exp.get("config")
            ):
                raise ValueError("Orchestrator cannot override a failed gate")
            if report["review"] == "repair" and exp["repair_count"] >= 1:
                raise ValueError("Only one bounded implementation repair is allowed")
            if report["review"] in {"repair", "approve"} and exp["idea"] in parked:
                raise ValueError("A repaired or approved head must remain reserved")
            if report["review"] == "block" and exp["idea"] not in parked:
                raise ValueError("Blocked review must explicitly park its idea")
            if report["review"] == "approve" or event.get("review_binding"):
                binding = self.review_binding(exp)
                if event.get("review_binding") != binding:
                    current_event = self.state["events"][0]
                    current_event.pop("review_binding", None)
                    current_event["refresh_reason"] = (
                        "Review inputs changed; inspect the updated evidence"
                    )
                    self.save()
                    return
        if event["kind"] == "completion" and (
            self.state["running"] != exp["id"]
            or exp["reference"] != self.state["incumbent"]
        ):
            raise ValueError(
                "Completion reference differs from the current incumbent; explicit recovery is required"
            )
        if event["kind"] != "review" and report["review"] != "none":
            raise ValueError(
                "Only candidate review events can approve, repair, or block"
            )
        for idea_id, idea in proposed.items():
            if idea_id not in self.state["ideas"]:
                self.state["ideas"][idea_id] = {
                    **idea,
                    "status": "proposed",
                    "created_revision": self.state["revision"],
                }
        for idea_id in accepted:
            self.state["ideas"][idea_id]["status"] = "queued"
        for idea_id in parked:
            self.state["ideas"][idea_id].update(
                status="blocked", blocked_reason=report["rationale"]
            )
            if claimed and claimed["idea"] == idea_id:
                claimed["stage"] = "blocked"
                self.state["claimed"] = None
            self.state["evidence_revision"] += 1
        self.state["queue"] = list(ranking)
        if report["ranked_ids"] is not None:
            for idea_id in ranking:
                self.state["ideas"][idea_id]["ranking_rationale"] = report["rationale"]
        if exp:
            exp.update(
                finding=report["finding"],
                concerns=report["concerns"],
                evidence_refs=report["evidence_refs"],
            )
        if event["kind"] == "review":
            exp["implementation_review"] = {
                "event": event["id"],
                "decision": report["review"],
                "concerns": report["concerns"],
                "integration_review": report["integration_review"],
                "binding": binding,
            }
            write_json(
                Path(exp["directory"]) / f"review-{event['id']}.json",
                exp["implementation_review"],
            )
            if report["review"] == "approve":
                exp["stage"] = "prepared"
                exp["active_idea_ids"] = binding["active_idea_ids"]
                self.state["ideas"][exp["idea"]]["status"] = "prepared"
            elif report["review"] == "repair":
                exp["stage"] = "repair_requested"
                exp["repair_instruction"] = report["rationale"]
        elif event["kind"] == "completion":
            exp.update(
                stage="completed",
                decision=event["eligible_decision"],
                rationale=report["rationale"],
            )
            self.state["ideas"][exp["idea"]].update(
                status="completed", decision=event["eligible_decision"]
            )
            self.state["running"] = None
            self.state["evidence_revision"] += 1
            if event["eligible_decision"] == "provisional_accept":
                run = Path(exp["directory"]) / "training"
                self.state["incumbent"] = {
                    "id": exp["id"],
                    "source": exp["source"],
                    "run": str(run),
                    "predictions_sha256": sha256(
                        run / "fold_0/validation_predictions.csv"
                    ),
                }
                current_seed = reference_seed(self.state["incumbent"])
                seed = secrets.randbits(32)
                while seed == current_seed:
                    seed = secrets.randbits(32)
                self.state["control_retrain"] = {
                    "id": identifier("control"),
                    "adopted": exp["id"],
                    "seed": seed,
                    "stage": "pending",
                    "attempts": 0,
                }
        self.state["findings"].append(
            {
                "event": event["id"],
                "finding": report["finding"],
                "rationale": report["rationale"],
                "concerns": report["concerns"],
                "evidence_refs": report["evidence_refs"],
            }
        )
        self.state["events"].pop(0)
