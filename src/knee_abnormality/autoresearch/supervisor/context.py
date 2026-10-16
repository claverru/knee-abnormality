import json
from pathlib import Path

from knee_abnormality.autoresearch.campaign.gates import (
    active_incumbent_ideas,
    config_differences,
    dependency_blockers,
    flatten,
)
from knee_abnormality.autoresearch.campaign.reports import REPORTS
from knee_abnormality.autoresearch.reporting.curves import (
    campaign_curves,
    compact_curves,
)
from knee_abnormality.autoresearch.runtime.layout import (
    campaign_fold,
    comparison_path,
    diagnostics_path,
    experiment_diagnostics,
    job_experiment,
)
from knee_abnormality.autoresearch.runtime.storage import read_json, sha256, write_json
from knee_abnormality.autoresearch.scoring.diagnostics import INTERPRETATION
from knee_abnormality.autoresearch.supervisor.monitoring import run_failure


def select(value, keys):
    return {key: value[key] for key in keys if key in value}


RANGE_FREE_LIMITS = {
    "epochs",
    "optimizer_steps",
    "training_visits",
    "training_triplets",
}


def agent_limits(state):
    if state["budget"]["mode"] != "range":
        return state["limits"]
    return {
        key: value
        for key, value in state["limits"].items()
        if key not in RANGE_FREE_LIMITS
    }


def artifact(path):
    path = Path(path)
    value = {"path": str(path), "available": path.is_file()}
    if value["available"]:
        value.update(bytes=path.stat().st_size, sha256=sha256(path))
    return value


def idea_card(idea):
    return select(
        idea,
        (
            "id",
            "title",
            "mechanism_id",
            "parent_idea_id",
            "status",
            "decision",
            "dependencies",
            "experiments",
        ),
    )


def experiment_card(exp):
    return select(
        exp,
        (
            "id",
            "idea",
            "stage",
            "implementation_parent",
            "implementation_reference",
            "integration",
            "activation",
            "run_checks",
            "active_idea_ids",
            "repair_of",
            "repair_count",
            "decision",
            "eligible_decision",
            "finding",
            "concerns",
            "directory",
            "source",
            "parent_source",
            "workspace",
            "reconciliation",
        ),
    )


def family_ledger(state):
    families = {}
    for idea in state["ideas"].values():
        families.setdefault(
            idea["mechanism_id"], {"family": idea["mechanism_id"], "attempts": []}
        )["attempts"].append(select(idea, ("id", "parent_idea_id", "status")))
    for exp in state["experiments"].values():
        if not exp.get("decision"):
            continue
        idea = state["ideas"][exp["idea"]]
        scores = exp.get("evidence", {}).get("macro_auc")
        validation = exp.get("evidence", {}).get("validation_macro_auc")
        bounds = (exp.get("evidence", {}).get("adoption_bootstrap") or {}).get(
            "delta_percentile_80"
        )
        families[idea["mechanism_id"]].setdefault("results", []).append(
            {
                "experiment": exp["id"],
                "idea": idea["id"],
                "decision": exp["decision"],
                "macro_delta": scores["candidate"] - scores["reference"]
                if scores
                else None,
                "validation_macro_delta": validation["candidate"]
                - validation["reference"]
                if validation
                else None,
                "adoption_interval_80": bounds,
            }
        )
    for family, verdict in state.get("research", {}).get("verdicts", {}).items():
        families[family]["verdict"] = select(
            verdict, ("verdict", "reason", "next_step", "misses")
        )
    return list(families.values())


def experiment_evidence(state, exp):
    directory = Path(exp["directory"])
    fold = campaign_fold(state)
    packet = {
        "experiment": experiment_card(exp),
        "limits": agent_limits(state),
        "artifacts": {
            name: artifact(directory / name)
            for name in (
                "changes.diff",
                "manifest.json",
                "config.yaml",
                "run-limits.json",
                "evaluation.json",
                "comparison/comparison.json",
                "comparison/per_label.json",
                "training/effective_config.json",
                f"training/fold_{fold}/triplet_accounting.json",
                f"training/fold_{fold}/timing.json",
                f"training/fold_{fold}/input_identity.json",
            )
        },
        "accounting_available": False,
        "config_available": "config" in exp,
        **select(
            exp,
            ("reference", "seed", "preparation_seconds", "measured_compute_seconds"),
        ),
    }
    if "config" in exp:
        packet["config_differences"] = config_differences(
            state["config"], exp["config"]
        )
        candidate = flatten(exp["config"])
        packet["candidate_config_changes"] = {
            key: candidate.get(key) for key in packet["config_differences"]
        }
    ledger_path = directory / f"training/fold_{fold}/triplet_accounting.json"
    if ledger_path.is_file():
        ledger = read_json(ledger_path)
        packet["accounting_available"] = True
        packet["accounting"] = {
            **select(
                ledger,
                (
                    "complete",
                    "training_triplets",
                    "training_visits",
                    "changed_training_studies",
                    "unit",
                ),
            ),
            **{
                phase: [
                    select(
                        row,
                        (
                            "epoch",
                            "triplets",
                            "visits",
                            "cohort_sha256",
                            "selection_sha256",
                        ),
                    )
                    for row in ledger[phase]
                ]
                for phase in ("train", "validation")
            },
        }
    timing_path = directory / f"training/fold_{fold}/timing.json"
    if timing_path.is_file():
        packet["timing"] = read_json(timing_path)
    evaluation_path = directory / "evaluation.json"
    if evaluation_path.is_file():
        packet["evaluation"] = read_json(evaluation_path)
    if diagnostics := experiment_diagnostics(exp):
        packet["per_label"] = diagnostics["per_target"]
        packet["per_label_interpretation"] = INTERPRETATION
    packet["jobs"] = [
        {
            "id": job["id"],
            "kind": job["kind"],
            "receipt": artifact(Path(job["path"]) / "receipt.json"),
            "report": artifact(Path(job["path"]) / "report.json"),
            "stdout": str(Path(job["path"]) / "stdout.log"),
            "stderr": str(Path(job["path"]) / "stderr.log"),
        }
        for job in state["jobs"].values()
        if job_experiment(job) == exp["id"]
    ]
    return packet


def campaign_defects(state, exclude):
    logs = {
        job["payload"]["experiment"]: str(Path(job["path"]) / "stderr.log")
        for job in state["jobs"].values()
        if job["kind"] == "training"
    }
    rows = []
    for value in state["experiments"].values():
        if value["id"] == exclude:
            continue
        if value.get("repair_instruction"):
            text = value["repair_instruction"]
        elif value.get("decision") == "invalid" and (
            run_failure(value) or not value.get("training_receipt", {}).get("cancelled")
        ):
            text = "; ".join(value.get("evidence", {}).get("reasons", [])) + (
                f" Log: {logs[value['id']]}" if value["id"] in logs else ""
            )
        else:
            continue
        rows.append(
            {"experiment": value["id"], "idea": value["idea"], "defect": text[:800]}
        )
    return rows[-12:]


def build_context(state, role, assignment, directory, docs, state_path):
    if role == "completion_reporter":
        event = assignment["event"]
        exp = state["experiments"][event["experiment"]]
        evidence = experiment_evidence(state, exp)
        evidence_path = directory / "evidence.json"
        write_json(evidence_path, evidence)
        comparison = read_json(Path(event["evidence"]["comparison"]))
        scores = comparison["family_means"]["all"]
        return {
            "role": role,
            "revision": state["revision"],
            "assignment": {
                "event": select(
                    event, ("id", "kind", "experiment", "eligible_decision")
                ),
                "experiment": experiment_card(exp),
                "idea": idea_card(state["ideas"][exp["idea"]]),
                "evidence_packet": str(evidence_path),
                "facts": {
                    **event["evidence"],
                    "macro_auc_delta": scores["candidate"] - scores["reference"],
                    "per_target": evidence["per_label"],
                    "per_label_interpretation": INTERPRETATION,
                    "timing": evidence["timing"],
                    "measured_compute_seconds": exp["measured_compute_seconds"],
                    "max_run_seconds": state["limits"]["max_run_seconds"],
                    "implementation_review": exp["implementation_review"],
                },
            },
        }
    directory = Path(directory)
    event = assignment.get("event", {})
    exp_id = event.get("experiment") or assignment.get("experiment", {}).get("id")
    exp = state["experiments"].get(exp_id)
    ideas = state["ideas"]
    proposals = {
        key: value for key, value in ideas.items() if value["status"] == "proposed"
    } | {idea["id"]: idea for idea in event.get("report", {}).get("ideas", [])}
    relevant = set()
    if role == "orchestrator":
        relevant.update(
            dependency
            for proposal in proposals.values()
            for dependency in proposal["dependencies"]
        )
    if exp:
        idea = ideas[exp["idea"]]
        relevant.update([idea["id"], *idea["dependencies"]])
        if idea["parent_idea_id"]:
            relevant.add(idea["parent_idea_id"])
        relevant.update(
            key
            for key, value in ideas.items()
            if value["mechanism_id"] == idea["mechanism_id"]
        )
    incumbent = state["experiments"].get(state["incumbent"]["id"])
    if incumbent:
        relevant.add(incumbent["idea"])
    if role == "researcher" or event.get("kind") in {"proposals", "completion"}:
        relevant.update(state["queue"])
    history = [
        {
            "id": job["id"],
            "role": job["kind"],
            "event": job["payload"].get("event", {}).get("kind"),
            "experiment": job_experiment(job),
            "report": str(Path(job["path"]) / "report.json"),
            "assignment": str(Path(job["path"]) / "assignment.json"),
        }
        for job in state["jobs"].values()
        if job["kind"] in REPORTS and job["processed"]
    ]
    index_path = directory / "history-index.json"
    write_json(
        index_path,
        {
            "ideas": [idea_card(idea) for idea in ideas.values()],
            "experiments": [
                {
                    **experiment_card(value),
                    "per_label_path": str(diagnostics_path(comparison_path(value))),
                }
                for value in state["experiments"].values()
            ],
            "reports": history,
            "full_registry": str(docs / "ideas.yaml"),
            "full_findings": str(Path(state_path).parent / "findings.md"),
            "state": str(state_path),
        },
    )
    results = [
        value
        for value in state["experiments"].values()
        if value.get("decision")
        and (
            value["id"] in assignment["new_results"]
            if role == "researcher"
            else value["idea"] in relevant
        )
    ]
    context = {
        "role": role,
        "budget": state["budget"],
        "limits": agent_limits(state),
        "incumbent": state["incumbent"],
        "active_incumbent_ideas": active_incumbent_ideas(state),
        "history_index": str(index_path),
        "prior_work": str(docs / "prior-work.md"),
        "relevant_ideas": [
            idea_card(ideas[key]) for key in sorted(relevant) if key in ideas
        ],
        "recent_results": [
            {
                **experiment_card(value),
                "evidence": value.get("evidence", {}),
                "per_label": diagnostics["per_target"] if diagnostics else None,
            }
            for value in (results if role == "researcher" else results[-6:])
            for diagnostics in [experiment_diagnostics(value)]
        ],
        "per_label_interpretation": INTERPRETATION,
        "earlier_results": len(state["research"]["reviewed"])
        if role == "researcher"
        else max(0, len(results) - 6),
        "research_reports": [item for item in history if item["role"] == "researcher"][
            -1:
        ]
        if role == "researcher"
        else [],
        "assignment": {
            key: value
            for key, value in assignment.items()
            if key not in {"experiment", "event", "idea"}
        },
        "revision": state["revision"],
    }
    if role != "developer":
        context["queue"] = [idea_card(ideas[key]) for key in state["queue"]]
        context["reserved_head"] = (
            state["experiments"][state["claimed"]]["idea"] if state["claimed"] else None
        )
        context["active_experiments"] = [
            experiment_card(state["experiments"][key])
            for key in (state["claimed"], state["running"])
            if key is not None
        ]
    if role == "developer" or event.get("kind") == "review":
        context["campaign_defects"] = campaign_defects(state, exp_id)
    if role == "researcher" or event.get("kind") == "proposals":
        context["family_ledger"] = family_ledger(state)
    if role == "researcher":
        curves = campaign_curves(state)
        curves_path = directory / "curves.json"
        write_json(curves_path, curves)
        context["run_curves"] = compact_curves(
            curves, set(assignment["new_results"]), curves_path
        )
    if role == "orchestrator":
        context["proposal_admission"] = [
            {
                "id": idea["id"],
                "dependencies_satisfied": not blockers,
                "unresolved_dependencies": blockers,
            }
            for idea in proposals.values()
            for blockers in [
                dependency_blockers(idea, ideas, active_incumbent_ideas(state))
            ]
        ]
    if "idea" in assignment:
        context["assignment"]["idea"] = select(
            assignment["idea"],
            (
                "id",
                "title",
                "mechanism_id",
                "parent_idea_id",
                "opportunity",
                "mechanism",
                "precedents",
                "novelty",
                "coverage",
                "exploitation",
                "budget",
                "implementation",
                "open_choices",
                "dependencies",
                "source_change",
                "targeted_curve",
            ),
        )
    if event:
        context["assignment"]["event"] = {
            key: value
            for key, value in event.items()
            if key not in {"report", "evidence"}
        }
        if "report" in event:
            context["assignment"]["proposals"] = event["report"]
        if event.get("kind") == "proposals":
            context["available_proposals"] = [
                idea_card(idea)
                for idea in ideas.values()
                if idea["status"] == "proposed"
            ]
    if exp:
        evidence = experiment_evidence(state, exp)
        evidence_path = directory / "evidence.json"
        write_json(evidence_path, evidence)
        context["assignment"]["experiment"] = experiment_card(exp)
        context["assignment"]["evidence_packet"] = str(evidence_path)
        if role == "orchestrator":
            context["assignment"]["facts"] = {
                key: value
                for key, value in evidence.items()
                if key not in {"experiment", "limits", "artifacts", "jobs"}
            }
        related_ids = {
            value["id"]
            for value in state["experiments"].values()
            if value["idea"] in relevant
        }
        context["related_reports"] = [
            item for item in history if item["experiment"] in related_ids
        ][-6:]
    return context


def prompt_sizes(context, prompt):
    return {
        "prompt_bytes": len(prompt.encode()),
        "context_bytes": len(json.dumps(context, ensure_ascii=False).encode()),
        "sections_bytes": {
            key: len(json.dumps(value, ensure_ascii=False).encode())
            for key, value in context.items()
        },
    }
