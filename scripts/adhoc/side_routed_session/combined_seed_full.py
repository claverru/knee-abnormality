import json
from pathlib import Path

import yaml

directory = Path("autoresearch/campaigns/side-routing")
state = json.loads(Path("outputs/autoresearch/side-routing/state.json").read_text())
ideas = state["ideas"]
items = list(ideas.values()) if isinstance(ideas, dict) else ideas
priors = next(i for i in items if i["id"] == "spatial-priors-for-routed-report-heads")
term = yaml.safe_load((directory / "knee-level-term-ideas.yaml").read_text())["ideas"][0]
idea = {
    "id": "localising-design-all-three-parts",
    "mechanism_id": "knee-level-disease-term",
    "parent_idea_id": None,
    "title": "Combine the three parts that improved compartment localisation or cost nothing:"
    " PF OA from front-and-up, the knee-level term, and the routed report heads with priors",
    "opportunity": (
        "Evidence-backed on each part; that they hold together is a guess. Ten runs of the"
        " campaign are within 0.0004 of the incumbent on the x4 macro, except one at -0.0009,"
        " so the macro of a single run does not rank them. Two effects repeated and are larger"
        " than their run-to-run spread. (1) Routing PF OA from the front-and-up region: PF OA"
        " localisation +0.0153 and +0.0129 against the incumbent (tolerance 0.0057), in"
        " front-and-up-region-for-pf-oa and knee-level-term-with-front-and-up-routing. (2) The"
        " knee-level term, learned as a contrast (strengths -0.14 for meniscus and -0.02 to"
        " -0.04 for OA every time): Medial and Lateral Meniscus localisation +0.0018/+0.0042,"
        " +0.0041/+0.0059, +0.0048/+0.0040 and +0.0036/+0.0020 in its four runs. (3) The"
        " shared, routed report heads with position priors (spatial-priors-for-routed-report-heads)"
        " cost nothing measurable (macro +0.0003, routed sum -0.0001, no own-channel flag) and"
        " remove the last part of the model that asks the backbone for the compartment. The"
        " design that carries all three has not been run; it is the campaign's candidate for"
        " the best compartment localisation, which is what the public score followed across"
        " the line's three submissions."
    ),
    "mechanism": (
        "Parts (A) and (B) exactly as knee-level-term-with-front-and-up-routing, whose source"
        " is under outputs/autoresearch/side-routing/experiments/knee-level-term-with-front-and-up-routing-*/source: "
        + term["mechanism"].strip()
        + " Part (C) exactly as spatial-priors-for-routed-report-heads, whose source is under"
        " outputs/autoresearch/side-routing/experiments/spatial-priors-for-routed-report-heads-*/source,"
        " with its cartilage routes renamed to this design's ROUTES (medial_back ->"
        " medial_rest, lateral_back -> lateral_rest, front -> front_up): "
        + (priors.get("mechanism") or "").strip()
    ),
    "precedents": [
        "autoresearch/campaigns/side-routing/run-notes.md, decisions 1, 4, 5, 7 and 9 and the summary after nine runs.",
    ],
    "novelty": "The three parts have been run alone and in pairs, never together.",
    "coverage": "Every study: the five routed targets at training and inference; the report heads at training only.",
    "exploitation": (
        "Read the run report against the incumbent: the five localisation values against the"
        " lineage's best (expected: PF OA and both meniscus targets above, the OA sides within"
        " tolerance), the routed sum, the macro, the own-targets flag, the two knee-level"
        " strengths and routing_width_up_mm. Shown wrong by: PF OA or meniscus localisation not"
        " above the incumbent's, which would mean the parts interfere."
    ),
    "budget": "Same cost as the incumbent.",
    "implementation": (
        "Apply the changes as the two named experiments did: modeling/model.py for ROUTES,"
        " TARGET_ROUTES, PRIOR_VARIANTS, ROUTING_WIDTHS with \"up\", routing_weights, the"
        " knee_level_scale parameter and the knee-level term in _attention_pool;"
        " training/report_attributes.py and training/lightning.py for the shared, routed report"
        " heads with their priors, using this design's route names; modeling/spatial_tokens.py"
        " for the identity entries; training/lightning.py logs the widths, the routing means"
        " and knee_level_scale_<channel>. No config field and no option."
        " scripts.tools.check_side_routing runs on the CPU when no GPU is visible; all nine"
        " guardrails must pass and its JSON goes in the report."
    ),
    "open_choices": "None.",
    "dependencies": [],
    "priority_rationale": "Keeper seed: the one design that carries every repeated localisation gain of the campaign, for the user to judge in the morning.",
}
(directory / "localising-design-ideas.yaml").write_text(
    yaml.safe_dump(
        {"ideas": [idea], "exhausted_directions": ""},
        sort_keys=False,
        allow_unicode=True,
        width=98,
    )
)
index = directory / "INDEX.md"
content = index.read_text()
row = "| Reading the keeper's repairs, adoption decisions and added seeds |"
assert content.count(row) == 1
index.write_text(
    content.replace(
        row,
        "| Reading the keeper seed that carries all three parts with a repeated localisation gain or no cost (front-and-up PF OA, the knee-level term, routed report heads with priors) | [Localising design](localising-design-ideas.yaml) |\n"
        + row,
    )
)
print("ok")
