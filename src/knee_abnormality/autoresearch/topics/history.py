from pathlib import Path

from pydantic import ValidationError

from knee_abnormality.autoresearch.campaign.definition import load_campaign
from knee_abnormality.autoresearch.runtime.storage import optional_json, read_json

QUIET_TOPICS = 3
CAMPAIGNS = Path("autoresearch/campaigns")
OUTPUTS = Path("outputs/autoresearch")


def seeded_ideas(output: Path):
    return {
        idea["id"]
        for path in sorted(output.glob("imports/*/report.json"))
        for idea in read_json(path)["ideas"]
    }


def decided_runs(state):
    launched = {}
    for job in state["jobs"].values():
        if job["kind"] == "training":
            experiment = job["payload"]["experiment"]
            launched[experiment] = max(launched.get(experiment, 0), job["created_at"])
    return sorted(
        (exp for exp in state["experiments"].values() if exp.get("decision")),
        key=lambda exp: launched.get(exp["id"], 0),
    )


def researcher_run(state, exp, seeded):
    return exp["idea"] not in seeded and "seeded" not in state["ideas"][exp["idea"]]


def failed_run_check(exp):
    return any(not outcome["passed"] for outcome in exp.get("run_checks", {}).values())


def researcher_streak(state, seeded):
    streak = 0
    for exp in decided_runs(state):
        if exp["decision"] == "provisional_accept":
            streak = 0
        elif (
            exp["decision"] != "invalid" or failed_run_check(exp)
        ) and researcher_run(state, exp, seeded):
            streak += 1
    return streak


def outcome(state, seeded):
    runs = decided_runs(state)
    valid = [exp for exp in runs if exp["decision"] != "invalid"]
    return {
        "decided": len(runs),
        "valid": len(valid),
        "researcher_runs": sum(researcher_run(state, exp, seeded) for exp in valid),
        "adoptions": sum(exp["decision"] == "provisional_accept" for exp in runs),
        "invalid": len(runs) - len(valid),
        "streak": researcher_streak(state, seeded),
    }


def campaign_state(root: Path, campaign: str):
    return optional_json(root / OUTPUTS / campaign / "state.json")


def campaign_records(root: Path):
    records = []
    for docs in sorted((root / CAMPAIGNS).iterdir()):
        if not (docs / "campaign.yaml").is_file():
            continue
        try:
            definition = load_campaign(docs / "campaign.yaml").model_dump(mode="json")
        except (ValidationError, ValueError) as error:
            definition = {"id": docs.name, "error": str(error)}
        output = root / OUTPUTS / docs.name
        state = optional_json(output / "state.json")
        records.append(
            {
                "id": docs.name,
                "docs": docs,
                "output": output,
                "definition": definition,
                "state": state,
                "seeded": seeded_ideas(output) if state else set(),
            }
        )
    return records
