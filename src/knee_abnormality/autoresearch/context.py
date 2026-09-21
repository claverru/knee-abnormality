import json
from pathlib import Path

from knee_abnormality.autoresearch.diagnostics import (
    INTERPRETATION,
    experiment_diagnostics,
)
from knee_abnormality.autoresearch.gates import (
    active_incumbent_ideas,
    config_differences,
    dependency_blockers,
    flatten,
)
from knee_abnormality.autoresearch.models import REPORTS
from knee_abnormality.autoresearch.storage import read_json, sha256, write_json


def select(value, keys):
    return {key: value[key] for key in keys if key in value}


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


def experiment_evidence(state, exp):
    directory = Path(exp["directory"])
    packet = {
        "experiment": experiment_card(exp),
        "limits": state["limits"],
        "artifacts": {
            name: artifact(directory / name)
            for name in (
                "changes.diff",
                "manifest.json",
                "config.yaml",
                "evaluation.json",
                "comparison/comparison.json",
                "comparison/per_label.json",
                "training/effective_config.json",
                "training/fold_0/triplet_accounting.json",
                "training/fold_0/timing.json",
                "training/fold_0/input_identity.json",
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
    ledger_path = directory / "training/fold_0/triplet_accounting.json"
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
    timing_path = directory / "training/fold_0/timing.json"
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
        if job["payload"].get(
            "experiment", job["payload"].get("event", {}).get("experiment")
        )
        == exp["id"]
    ]
    return packet


def build_context(state, role, assignment, directory, docs, state_path):
    if role == "completion_reporter":
        event = assignment["event"]
        exp = state["experiments"][event["experiment"]]
        evidence = experiment_evidence(state, exp)
        evidence_path = directory / "evidence.json"
        write_json(evidence_path, evidence)
        comparison = read_json(event["evidence"]["comparison"])
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
            "experiment": job["payload"].get(
                "experiment", job["payload"].get("event", {}).get("experiment")
            ),
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
                    "per_label_path": str(
                        Path(value["directory"]) / "comparison/per_label.json"
                    ),
                }
                for value in state["experiments"].values()
            ],
            "reports": history,
            "full_registry": str(docs / "ideas.yaml"),
            "full_findings": str(docs / "findings.md"),
            "state": str(state_path),
        },
    )
    results = [
        value
        for value in state["experiments"].values()
        if value.get("decision") and (role == "researcher" or value["idea"] in relevant)
    ]
    context = {
        "role": role,
        "limits": state["limits"],
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
            for value in results[-6:]
            for diagnostics in [experiment_diagnostics(value)]
        ],
        "per_label_interpretation": INTERPRETATION,
        "earlier_results": max(0, len(results) - 6),
        "research_reports": [item for item in history if item["role"] == "researcher"][
            -3:
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
    if role == "researcher":
        context["mechanism_registry"] = [idea_card(idea) for idea in ideas.values()]
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
                if key
                not in {
                    "experiment",
                    "limits",
                    "artifacts",
                    "jobs",
                    "candidate_sampling_config",
                }
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
