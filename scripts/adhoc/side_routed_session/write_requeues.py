import sys
from pathlib import Path

import yaml

root = Path("autoresearch/campaigns/routing-and-shifts")
CPU = (
    "Environment, read this first. (1) You have no GPU and must not use one. Run `rtk proxy uv run "
    "--locked python -m scripts.tools.check_side_routing --config configs/training.yaml` from your "
    "candidate workspace as it is: the script selects the CPU by itself when no GPU is visible, takes "
    "a few minutes there, and that CPU run is the required check. Never return blocked for lack of a "
    "GPU. (2) Make every change inside the candidate workspace you were assigned and nowhere else. Do "
    "not stage the candidate under /tmp, and do not write source/ or changes.diff in the experiment "
    "directory: it is read-only to you by design, and the supervisor freezes the source and the diff "
    "from your workspace after you return. (3) When the change is complete and the check passes, "
    "return status prepared with the integration and activation declarations and the guardrail JSON "
    "in the report. An unwritable experiment directory is never a reason to return blocked. "
)
sources = {}
for name in ("seed-ideas-2.yaml", "seed-ideas-3.yaml", "light-port-shift-ideas.yaml"):
    for idea in yaml.safe_load((root / name).read_text())["ideas"]:
        sources[idea["id"]] = idea
routing = sys.argv[1].split(",")
out = []
for old in routing:
    idea = dict(sources[old])
    idea["id"] = old + "-2"
    idea["implementation"] = CPU + " ".join(idea["implementation"].split())
    idea["priority_rationale"] = (
        "Keeper's requeue of 2026-10-07: the first candidate was parked because the developer did not run the "
        "guardrail check, reading it as needing a GPU. " + idea["priority_rationale"]
    )
    out.append(idea)
(root / sys.argv[2]).write_text(yaml.safe_dump({"ideas": out, "exhausted_directions": ""}, sort_keys=False, allow_unicode=True, width=96))
print(sys.argv[2], [idea["id"] for idea in out])
if len(sys.argv) > 3:
    light = []
    names = [idea["id"] for idea in yaml.safe_load((root / "light-port-shift-ideas.yaml").read_text())["ideas"]]
    rename = {name: name + "-c" for name in names}
    for old in names:
        idea = dict(sources[old])
        idea["id"] = rename[old]
        for field in ("mechanism", "implementation", "exploitation", "budget", "opportunity", "title", "novelty", "coverage"):
            text = " ".join(idea[field].split())
            for before, after in rename.items():
                text = text.replace(before + "-c", before).replace(before, after)
            idea[field] = text
        idea["implementation"] = CPU + idea["implementation"]
        light.append(idea)
    (root / sys.argv[3]).write_text(yaml.safe_dump({"ideas": light, "exhausted_directions": ""}, sort_keys=False, allow_unicode=True, width=96))
    print(sys.argv[3], [idea["id"] for idea in light])
