from pathlib import Path

import yaml

directory = Path("autoresearch/campaigns/side-routing")
seeds = yaml.safe_load((directory / "seed-ideas.yaml").read_text())
idea = dict(seeds["ideas"][0])
idea["id"] = "routed-report-attribute-heads-requeue"
idea["mechanism"] = (
    idea["mechanism"].strip()
    + " Route names follow the incumbent's ROUTES at the time of the run: if the OA sides are"
    " routed as medial_rest and lateral_rest and PF OA as front_up, the cartilage questions use"
    " those three routes in place of medial_back, lateral_back and front."
)
idea["exploitation"] = (
    idea["exploitation"].strip()
    + " Activation: log, per epoch, the number of observed, positively weighted routed question"
    " pools whose attention differs from the same head's unrouted attention by a total variation"
    " above 1e-5; the run is void if that count is zero in the first epoch."
)
idea["implementation"] = (
    idea["implementation"].strip()
    + " Keeper note, 2026-10-05: the first candidate of this idea"
    " (routed-report-attribute-heads) implemented the mechanism and was blocked only because"
    " the guardrail check needed a GPU the developer's environment does not have."
    " scripts.tools.check_side_routing now runs on the CPU when no GPU is visible (about ten"
    " seconds, about 7 GB); run it as the developer instructions say and report its JSON."
)
idea["priority_rationale"] = (
    "Requeue of the first forced seed after a harness failure (no GPU in the developer's"
    " environment for the mandatory guardrail check), not a result."
)
text = yaml.safe_dump(
    {"ideas": [idea], "exhausted_directions": ""},
    sort_keys=False,
    allow_unicode=True,
    width=98,
)
(directory / "routed-report-attribute-heads-requeue-ideas.yaml").write_text(text)
index = directory / "INDEX.md"
content = index.read_text()
row = "| Reading the design, every run's tables, the drawings and the measurements |"
assert content.count(row) == 1
index.write_text(
    content.replace(
        row,
        "| Reading the keeper requeue of the routed report-attribute heads, after the first candidate was blocked because the guardrail check needed a GPU the developer's environment lacks | [Routed report-attribute heads requeue](routed-report-attribute-heads-requeue-ideas.yaml) |\n"
        + row,
    )
)
print(text[:400])
