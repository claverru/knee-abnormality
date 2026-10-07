import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

CAMPAIGN = "routing-and-shifts"
MARK = "Environment, read this first."
ENVIRONMENT = (
    MARK + " (1) You have no GPU and must not use one. Run `rtk proxy uv run --locked python -m "
    "scripts.tools.check_side_routing --config configs/training.yaml` from your candidate workspace as "
    "it is: the script selects the CPU by itself when no GPU is visible, takes a few minutes there, and "
    "that CPU run is the required check. Never return blocked for lack of a GPU. (2) Make every change "
    "inside the candidate workspace you were assigned and nowhere else. Do not stage the candidate under "
    "/tmp, and do not write source/ or changes.diff in the experiment directory: it is read-only to you "
    "by design, and the supervisor freezes the source and the diff from your workspace after you return. "
    "(3) When the change is complete and the check passes, return status prepared with the integration "
    "and activation declarations and the guardrail JSON in the report. An unwritable experiment "
    "directory is never a reason to return blocked. "
)
FIELDS = (
    "id", "mechanism_id", "parent_idea_id", "title", "opportunity", "mechanism", "precedents", "novelty",
    "coverage", "exploitation", "budget", "implementation", "open_choices", "dependencies", "priority_rationale",
)
root = Path("autoresearch/campaigns") / CAMPAIGN
state = json.loads(Path(f"outputs/autoresearch/{CAMPAIGN}/state.json").read_text())
ideas = state["ideas"]
protected_from = {name[: -len("-k")] for name in ideas if name.endswith("-k")}
chosen = []
for name, idea in ideas.items():
    if MARK in str(idea.get("implementation")) or name in protected_from:
        continue
    if idea.get("status") == "queued" or (idea.get("status") == "blocked" and name in sys.argv[1:]):
        chosen.append(name)
if not chosen:
    print("nothing to protect")
    raise SystemExit
out = []
for name in chosen:
    idea = {field: ideas[name].get(field) for field in FIELDS}
    idea["id"] = name + "-k"
    idea["dependencies"] = []
    idea["implementation"] = ENVIRONMENT + " ".join(str(idea["implementation"]).split())
    idea["priority_rationale"] = "Keeper's copy with the developer-environment paragraph. " + str(idea["priority_rationale"] or "")
    out.append(idea)
stamp = time.strftime("%Y%m%dT%H%M%S")
for start in range(0, len(out), 5):
    file = root / f"keeper-copies-{stamp}-{start // 5}.yaml"
    file.write_text(yaml.safe_dump({"ideas": out[start : start + 5], "exhausted_directions": ""}, sort_keys=False, allow_unicode=True, width=96))
    run = subprocess.run(
        ["uv", "run", "--locked", "python", "-m", "scripts.tools.autoresearch", "import-ideas", CAMPAIGN, "--ideas-file", str(file), "--forced"],
        capture_output=True, text=True,
    )
    print("imported", file.name, [idea["id"] for idea in out[start : start + 5]], "ok" if run.returncode == 0 else run.stderr[-300:])
time.sleep(40)
for name in chosen:
    if ideas[name].get("status") != "queued":
        continue
    run = subprocess.run(
        ["uv", "run", "--locked", "python", "-m", "scripts.tools.autoresearch", "remove-idea", CAMPAIGN, "--idea", name, "--rationale",
         "Keeper, 2026-10-07: replaced by its copy with the developer-environment paragraph; without it developers return blocked without running the guardrail check"],
        capture_output=True, text=True,
    )
    print("removed", name, "ok" if run.returncode == 0 else run.stderr.strip().splitlines()[-1][:200])
