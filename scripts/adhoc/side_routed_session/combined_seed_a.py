import json
from pathlib import Path

import yaml

directory = Path("autoresearch/campaigns/side-routing")
state = json.loads(Path("outputs/autoresearch/side-routing/state.json").read_text())
ideas = state["ideas"]
items = list(ideas.values()) if isinstance(ideas, dict) else ideas
priors = next(i for i in items if i["id"] == "spatial-priors-for-routed-report-heads")
term = yaml.safe_load(
    (directory / "knee-level-term-on-incumbent-routing-ideas.yaml").read_text()
)["ideas"][0]
idea = {
    "id": "knee-level-term-with-routed-report-heads-and-priors",
    "mechanism_id": "knee-level-disease-term",
    "parent_idea_id": None,
    "title": "Combine the knee-level term of the routed targets with the shared, routed report"
    " heads and their position priors",
    "opportunity": (
        "Evidence-backed on each part, a guess that they add up. The two runs of the campaign"
        " above the incumbent on the x4 macro are knee-level-term-on-incumbent-routing"
        " (0.924834, +0.000358) and spatial-priors-for-routed-report-heads (0.924807,"
        " +0.000331). Each was kept out of the lineage by one flag the other does not raise."
        " The knee-level run lifted Medial and Lateral Meniscus localisation by +0.0041 and"
        " +0.0059, above the fourth run's mark, and was flagged only on ACL (-0.0012, 0.0002"
        " under the lowest sibling run). The report-heads run raised no own-channel flag (ACL"
        " -0.0004, Fracture +0.0023, Baker's no longer below every sibling) and was flagged"
        " only on Medial Meniscus localisation (0.0036 under the mark, tolerance 0.0025), the"
        " one number the knee-level term moves. The two changes touch different parts: the"
        " logits of the five routed targets, and the training-only report heads."
    ),
    "mechanism": (
        "Both mechanisms exactly as in their own runs, whose sources are under"
        " outputs/autoresearch/side-routing/experiments/knee-level-term-on-incumbent-routing-*/source"
        " and outputs/autoresearch/side-routing/experiments/spatial-priors-for-routed-report-heads-*/source."
        " (A) " + term["mechanism"].strip() + " (B) " + (priors.get("mechanism") or "").strip()
    ),
    "precedents": [
        "autoresearch/campaigns/side-routing/run-notes.md, decisions 5 and 7: the two runs this design combines.",
    ],
    "novelty": "The two changes have only been run apart.",
    "coverage": "Every study: the five routed targets at training and inference through the knee-level term; the report heads at training only.",
    "exploitation": (
        "Read the run report against the incumbent: the macro (expected at or above both"
        " parts), Medial Meniscus localisation against the lineage's best (expected above it,"
        " as in the knee-level run), ACL and the own-targets flag, the routed sum, the two"
        " knee-level strengths (expected negative again) and the report questions'"
        " divergences. Shown wrong by: Medial Meniscus localisation flagged with negative"
        " knee-level strengths, or a macro below the incumbent."
    ),
    "budget": "Same cost as the incumbent.",
    "implementation": (
        "Apply both changes as their own experiments did (see the two sources named in the"
        " mechanism): modeling/model.py for the knee_level_scale parameter and the knee-level"
        " term in _attention_pool; training/report_attributes.py and training/lightning.py for"
        " the shared, routed report heads with their priors; modeling/spatial_tokens.py for the"
        " identity entries; training/lightning.py also logs knee_level_scale_<channel>. No"
        " config field and no option. scripts.tools.check_side_routing runs on the CPU when no"
        " GPU is visible; all nine guardrails must pass and its JSON goes in the report."
    ),
    "open_choices": "None.",
    "dependencies": [],
    "priority_rationale": "Keeper seed: the two runs above the incumbent, each blocked by a flag the other clears, as one design.",
}
(directory / "knee-level-term-with-report-heads-ideas.yaml").write_text(
    yaml.safe_dump(
        {"ideas": [idea], "exhausted_directions": ""},
        sort_keys=False,
        allow_unicode=True,
        width=98,
    )
)
print(len(idea["mechanism"]))
