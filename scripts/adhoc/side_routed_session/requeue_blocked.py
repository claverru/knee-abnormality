import json
import sys
from pathlib import Path

import yaml

FIELDS = (
    "id", "mechanism_id", "parent_idea_id", "title", "opportunity", "mechanism", "precedents", "novelty",
    "coverage", "exploitation", "budget", "implementation", "open_choices", "dependencies", "priority_rationale",
)
state = json.loads(Path("outputs/autoresearch/routing-and-shifts/state.json").read_text())
root = Path("autoresearch/campaigns/routing-and-shifts")
out = []
for name in sys.argv[2:]:
    source = state["ideas"][name]
    assert source["status"] == "blocked", (name, source["status"])
    idea = {field: source.get(field) for field in FIELDS}
    idea["id"] = name + "-requeue"
    idea["dependencies"] = []
    idea["priority_rationale"] = (
        "Keeper's requeue of 2026-10-07: parked without training because the developer returned blocked without "
        "running the guardrail check and tried to write the frozen source itself; the contract now states where "
        "and how the check runs. " + str(source.get("priority_rationale") or "")
    )
    out.append(idea)
(root / sys.argv[1]).write_text(yaml.safe_dump({"ideas": out, "exhausted_directions": ""}, sort_keys=False, allow_unicode=True, width=96))
print(sys.argv[1], [idea["id"] for idea in out])
