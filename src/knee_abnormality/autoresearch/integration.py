import hashlib
import json
from pathlib import Path

from knee_abnormality.autoresearch.gates import (
    ACCEPTANCE_RULE,
    active_incumbent_ideas,
    dependency_blockers,
)
from knee_abnormality.autoresearch.isolation import manifest
from knee_abnormality.autoresearch.models import DeveloperReport
from knee_abnormality.autoresearch.storage import read_json, sha256


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def reference_seed(reference):
    seed = read_json(Path(reference["run"]) / "effective_config.json")["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Incumbent must have a recorded unsigned 32-bit training seed")
    return seed


def comparison_identity_keys(state, exp, policy):
    allowed = list(policy.input_identity_keys)
    transition = state.get("incumbent_transition")
    if (
        not transition
        or exp["reference"]["id"] != transition["incumbent_id"]
        or "preprocessing_identity" in allowed
    ):
        return allowed
    reference_path = Path(exp["reference"]["run"]) / "fold_0/input_identity.json"
    if sha256(reference_path) != transition["reference_input_identity_sha256"]:
        raise ValueError("Adopted reference input identity changed")
    candidate = read_json(
        Path(exp["directory"]) / "training/fold_0/input_identity.json"
    )
    reference = read_json(reference_path)
    if candidate["preprocessing_identity"] != reference["preprocessing_identity"]:
        if candidate["preprocessing_identity"] != transition["preprocessing_identity"]:
            raise ValueError("Candidate preprocessing differs from the adopted cleanup")
        allowed.append("preprocessing_identity")
    return allowed


def reference_snapshot(reference):
    source = Path(reference["source"])
    run = Path(reference["run"])
    if not source.is_dir():
        raise ValueError(f"Missing incumbent source: {source}")
    predictions = sha256(run / "fold_0/validation_predictions.csv")
    if predictions != reference["predictions_sha256"]:
        raise ValueError("Incumbent predictions changed")
    return {
        **reference,
        "source_sha256": digest(manifest(source)),
        "config_sha256": sha256(run / "effective_config.json"),
        "input_identity_sha256": sha256(run / "fold_0/input_identity.json"),
    }


def external_evidence(root, campaign):
    root = Path(root)
    runs = {}
    for name in sorted(set(campaign["additional_evidence_runs"])):
        run = root / name
        runs[name] = {
            filename: sha256(path) if path.is_file() else None
            for filename in (
                "effective_config.json",
                "source_snapshot.zip",
                "fold_0/input_identity.json",
                "fold_0/selected_checkpoint.json",
                "fold_0/validation_predictions.csv",
                "fold_0/timing.json",
            )
            for path in [run / filename]
        }
    campaigns = {}
    for path in sorted((root / "outputs/autoresearch").glob("*/state.json")):
        state = read_json(path)
        if state["campaign"] == campaign["id"]:
            continue
        campaigns[state["campaign"]] = {
            "state_path": str(path),
            "incumbent": state["incumbent"],
            "decisions": {
                exp["id"]: exp["decision"]
                for exp in state["experiments"].values()
                if exp.get("decision")
            },
        }
    return {"runs": runs, "campaigns": campaigns}


def parent_changed(state, exp):
    current = reference_snapshot(state["incumbent"])
    recorded = exp.get("implementation_reference")
    if recorded and recorded["id"] == current["id"] and recorded != current:
        raise ValueError(
            "Implementation parent artifacts changed without a new incumbent identity"
        )
    return recorded != current


def candidate_binding(state, exp, evidence):
    reference = reference_snapshot(state["incumbent"])
    seed = reference_seed(reference)
    if exp["config"]["seed"] != seed or exp.get("seed") != seed:
        raise ValueError("Candidate training seed differs from the incumbent")
    if exp.get("implementation_reference") != reference:
        raise ValueError("Candidate implementation parent differs from the incumbent")
    report = DeveloperReport.model_validate(exp["developer_report"])
    if report.status != "prepared":
        raise ValueError("Candidate has no prepared integration plan")
    integration = report.integration.model_dump(mode="json")
    if exp.get("integration") != integration:
        raise ValueError("Recorded integration differs from the developer report")
    active = set(active_incumbent_ideas(state))
    removed = set(integration["removed_idea_ids"])
    if not removed <= active:
        raise ValueError(
            "Integration removes idea IDs that are not active in the incumbent"
        )
    retained = active - removed
    for idea_id in sorted(retained):
        blockers = dependency_blockers(
            state["ideas"][idea_id], state["ideas"], retained
        )
        if blockers:
            raise ValueError(
                f"Retained idea {idea_id} loses required dependencies: "
                + "; ".join(blockers)
            )
    idea = state["ideas"][exp["idea"]]
    blockers = dependency_blockers(idea, state["ideas"], retained)
    if blockers:
        raise ValueError(
            "Candidate dependencies are not retained: " + "; ".join(blockers)
        )
    actual = manifest(Path(exp["source"]))
    if actual != read_json(Path(exp["directory"]) / "manifest.json"):
        raise ValueError("Reviewed candidate source changed")
    if sha256(Path(state["headers"])) != state["headers_sha256"]:
        raise ValueError("Acquisition headers changed")
    return {
        "reference": reference,
        "training_seed": seed,
        "acceptance_rule": dict(ACCEPTANCE_RULE),
        "incumbent_transition": state.get("incumbent_transition"),
        "candidate_source_sha256": digest(actual),
        "candidate_config_sha256": digest(exp["config"]),
        "integration": integration,
        "active_idea_ids": sorted(retained | {exp["idea"]}),
        "external_evidence": evidence,
        "headers_sha256": state["headers_sha256"],
        "limits_sha256": state["sealed_sha256"],
    }


def launch_blockers(state, exp):
    blockers = []
    if state["status"] != "running":
        blockers.append("Campaign is not running")
    if state["running"] is not None:
        blockers.append("Previous training or its decision is still outstanding")
    if state["events"]:
        blockers.append("Campaign events are still pending")
    if any(
        not job["processed"]
        and job["kind"]
        in {"training", "comparison", "orchestrator", "completion_reporter"}
        for job in state["jobs"].values()
    ):
        blockers.append("Training, comparison or decision work is still active")
    if state["claimed"] != exp["id"] or exp["stage"] != "prepared":
        blockers.append("Candidate is not the prepared reserved attempt")
    if not state["queue"] or state["queue"][0] != exp["idea"]:
        blockers.append("Candidate is not the reserved queue head")
    return blockers
