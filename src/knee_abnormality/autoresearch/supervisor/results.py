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
from knee_abnormality.autoresearch.runtime.gpu import GPU_BUSY
from knee_abnormality.autoresearch.runtime.layout import (
    PREDICTIONS,
    campaign_fold,
    comparison_path,
    diagnostics_path,
    fold_dir,
    read_diagnostics,
)
from knee_abnormality.autoresearch.runtime.storage import (
    read_json,
    sha256,
    tail,
    write_json,
)
from knee_abnormality.autoresearch.supervisor.common import (
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
        if receipt["exit_code"] == GPU_BUSY:
            exp["stage"] = "prepared"
            self.state["running"] = None
            return
        self.dequeue_launched(exp)
        exp["training_receipt"] = receipt
        if (
            exp["launched"]
            and exp["repair_count"] < 2
            and receipt["exit_code"] != 0
            and not receipt.get("error")
            and not receipt.get("cancelled")
            and trained_epochs(Path(exp["directory"]), campaign_fold(self.state)) == 0
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
        reasons = evaluate(
            Path(exp["directory"]),
            campaign_fold(self.state),
            self.state["limits"],
            measured,
        )
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
            self.request_review(exp, f"Launch failed: {reasons}")
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
            comparison = comparison_path(exp)
            report = read_json(comparison)
            diagnostics = read_diagnostics(comparison)
            macro_bootstrap = (diagnostics or {}).get("macro_bootstrap")
            self.completion(
                exp,
                verdict(report, macro_bootstrap),
                {
                    "comparison": str(comparison),
                    "macro_auc": report["family_means"]["all"],
                    "adoption_bootstrap": macro_bootstrap,
                    "per_label": str(diagnostics_path(comparison)),
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
            self.request_review(exp, f"Configuration failed; inspect {directory}")
            return
        if kind == "developer":
            exp["stage"] = "needs_developer"
        if problem and self.provider_wait(job, problem):
            if kind == "researcher":
                self.state["last_refill_revision"] = -1
            return
        detail = f" ({problem['kind']}: {problem['message'][:500]})" if problem else ""
        self.pause(
            f"{kind} failed; inspect {directory}{detail}. Fix the cause and use resume."
        )
        return

    def config_result(self, exp, receipt, directory):
        exp["preparation_seconds"] = receipt["seconds"]
        try:
            config = read_json(directory / "stdout.log")
            config.pop("config_file")
            config.pop("overrides")
            config.pop("split_dir")
            validate_config(self.state["config"], config, self.policy)
            exp["config"] = config
            if not parent_changed(self.state, exp):
                self.review_binding(exp)
            self.request_review(exp)
        except (ValueError, KeyError) as error:
            self.request_review(exp, str(error))

    def agent_result(self, job, exp, directory):
        kind = job["kind"]
        self.state.get("provider_retries", {}).pop(work_item(job), None)
        report = providers.result(self.state["provider"], REPORTS[kind], directory)
        if kind == "researcher":
            self.apply_research(job, report)
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
                self.request_review(exp, report["blocker"] or "Developer blocked")
            elif self.state["status"] != "running":
                exp["stage"] = "needs_prepare"
            else:
                try:
                    self.prepare(exp, report)
                except (OSError, ValueError, SyntaxError) as error:
                    self.request_review(exp, str(error))
        elif kind == "completion_reporter":
            self.apply_completion_report(job, report)
        else:
            self.apply_orchestrator(job["payload"]["event"], report)

    def apply_research(self, job, report):
        research = self.state["research"]
        ideas = self.state["ideas"]
        reused = sorted({idea["id"] for idea in report["ideas"]} & ideas.keys())
        if reused:
            raise ValueError(
                "Idea IDs already exist in the registry; give new proposals new IDs and "
                f"leave existing proposals to the orchestrator: {reused}"
            )
        verdicts = {item["mechanism_id"]: item for item in report["family_verdicts"]}
        if len(verdicts) != len(report["family_verdicts"]):
            raise ValueError("Give each family at most one verdict")
        unknown = verdicts.keys() - {idea["mechanism_id"] for idea in ideas.values()}
        if unknown:
            raise ValueError(f"Verdicts name unknown families: {sorted(unknown)}")
        results = {}
        for key in job["payload"]["new_results"]:
            exp = self.state["experiments"][key]
            results.setdefault(ideas[exp["idea"]]["mechanism_id"], []).append(
                exp["decision"]
            )
        missing = results.keys() - verdicts.keys()
        if missing:
            raise ValueError(
                f"Families with new results need a verdict: {sorted(missing)}"
            )
        for family, item in verdicts.items():
            previous = research["verdicts"].get(family)
            decisions = results.get(family, [])
            if previous and previous["verdict"] == "close" and not decisions:
                raise ValueError(f"{family} is closed; only a new result reopens it")
            misses = (
                0
                if previous is None or "provisional_accept" in decisions
                else previous["misses"] + decisions.count("reject")
            )
            if misses >= 2 and item["verdict"] != "close":
                raise ValueError(
                    f"{family}: two consecutive follow-ups were not adopted; close it"
                )
            research["verdicts"][family] = {
                **item,
                "misses": misses,
                "revision": self.state["evidence_revision"],
            }
        proposed = {idea["mechanism_id"] for idea in report["ideas"]}
        closed = {
            family
            for family, item in research["verdicts"].items()
            if item["verdict"] == "close"
        }
        if proposed & closed:
            raise ValueError(
                f"Closed families cannot receive proposals: {sorted(proposed & closed)}"
            )
        live = {
            idea["mechanism_id"]
            for idea in ideas.values()
            if idea["status"]
            in {"proposed", "queued", "developing", "prepared", "launched"}
        }
        uncovered = {
            family
            for family, item in research["verdicts"].items()
            if item["verdict"] != "close" and family not in live
        }
        if len(uncovered & proposed) < min(len(uncovered), job["payload"]["needed"]):
            raise ValueError(
                "Open extend/retry lines take refill slots first; propose their next steps "
                f"or close them: {sorted(uncovered - proposed)}"
            )
        research["reviewed"].extend(job["payload"]["new_results"])
        research.update(rejections=0, previous_rejection=None)

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
        self.route_completion(
            event,
            {
                "status": "escalated",
                "job": job["id"],
                "reason": reason,
                "report": str(Path(job["path"]) / "report.json"),
            },
        )

    def apply_completion_report(self, job, report):
        event = self.completion_event(job)
        reasons = completion_blockers(self.state, event)
        if report["status"] == "escalate" or reasons:
            reason = "; ".join([*reasons, *report["concerns"]]) or report["summary"]
            self.escalate_completion(job, reason)
            return
        exp = self.state["experiments"][event["experiment"]]
        concerns = [*exp["implementation_review"]["concerns"], *report["concerns"]]
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
                "concerns": concerns,
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
        seeds = set(self.state["seed_ideas"])
        claimed = self.state["experiments"].get(self.state["claimed"])
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
            head = (
                [claimed["idea"]]
                if claimed
                and claimed["idea"] not in seeds
                and claimed["idea"] not in parked
                else []
            )
            if ranking[: len(head) + len(kept)] != head + kept:
                raise ValueError(
                    "User-ordered seeds must keep their order ahead of every other idea except the reserved head"
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
                        fold_dir(run, campaign_fold(self.state)) / PREDICTIONS
                    ),
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
