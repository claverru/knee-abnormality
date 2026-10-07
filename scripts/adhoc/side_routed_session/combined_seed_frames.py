import json
from pathlib import Path

import yaml

directory = Path("autoresearch/campaigns/side-routing")
state = json.loads(Path("outputs/autoresearch/side-routing/state.json").read_text())
ideas = state["ideas"]
items = list(ideas.values()) if isinstance(ideas, dict) else ideas
frames = next(i for i in items if i["id"] == "separate-oa-frame-for-octant-routing")
term = yaml.safe_load(
    (directory / "knee-level-term-on-incumbent-routing-ideas.yaml").read_text()
)["ideas"][0]
idea = {
    "id": "separate-oa-frame-with-knee-level-term",
    "mechanism_id": "knee-level-disease-term",
    "parent_idea_id": None,
    "title": "Add the knee-level term to the design with a separate OA frame",
    "opportunity": (
        "Evidence-backed on each part; that they add up is a guess."
        " separate-oa-frame-for-octant-routing (2026-10-06) has the highest x4 macro of the"
        " campaign, 0.924876 (+0.000400), PF OA localisation +0.0137 and an own-channel sum of"
        " +0.0067. Its two frames turned apart about the vertical axis, +6.2 degrees for the"
        " meniscus frame and -6.1 degrees for the OA frame, where the single shared frame had"
        " stayed within 1.6 degrees in every run: the two diseases ask for different"
        " orientations of the midline. It was kept out by Medial Meniscus localisation, 0.0045"
        " under the fourth run's mark (tolerance 0.0025), and by Medial OA localisation, 0.0031"
        " under the best (tolerance 0.0030). The knee-level term has raised Medial Meniscus"
        " localisation in each of its four runs (+0.0018, +0.0041, +0.0048, +0.0036 against the"
        " incumbent) and cleared that mark in three of them, with strengths learned negative"
        " every time (meniscus -0.14, OA -0.02 to -0.04)."
    ),
    "mechanism": (
        "Two parts. (A) The design of separate-oa-frame-for-octant-routing unchanged, whose"
        " source is under"
        " outputs/autoresearch/side-routing/experiments/separate-oa-frame-for-octant-routing-*/source: "
        + (frames.get("mechanism") or "").strip()
        + " (B) The knee-level term of knee-level-term-on-incumbent-routing, whose source is"
        " under outputs/autoresearch/side-routing/experiments/knee-level-term-on-incumbent-routing-*/source,"
        " on top of that routing: "
        + term["mechanism"].strip()
    ),
    "precedents": [
        "autoresearch/campaigns/side-routing/run-notes.md, decisions 5 and 11 and the knee-level term's four runs (4, 5, 9, 10).",
    ],
    "novelty": "The separate OA frame has been run once, without the knee-level term.",
    "coverage": "Every study with a fitted centre, training and inference, for the five routed targets.",
    "exploitation": (
        "Read the run report against the incumbent: Medial Meniscus and Medial OA localisation"
        " against the lineage's best (the two flags of the separate-frame run), PF OA"
        " localisation (expected to stay near +0.013), the macro, the two frames' angles"
        " (expected to turn apart again) and the two knee-level strengths (expected negative)."
        " Shown wrong by: frames that stay together, or Medial Meniscus localisation not above"
        " the separate-frame run's."
    ),
    "budget": "Same cost as the incumbent.",
    "implementation": (
        "Start from the separate-frame experiment's changes (modeling/model.py,"
        " modeling/spatial_tokens.py, training/lightning.py as in its source) and add the"
        " knee_level_scale parameter and the knee-level term in _attention_pool, logging"
        " knee_level_scale_<channel> per epoch. No config field and no option."
        " scripts.tools.check_side_routing runs on the CPU when no GPU is visible; all nine"
        " guardrails must pass and its JSON goes in the report."
    ),
    "open_choices": "None.",
    "dependencies": [],
    "priority_rationale": "Keeper seed: the highest macro of the campaign with the one mechanism that has cleared its main flag every time.",
}
(directory / "separate-oa-frame-with-knee-level-term-ideas.yaml").write_text(
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
        "| Reading the keeper seed that adds the knee-level term to the researcher's separate OA frame | [Separate OA frame with knee-level term](separate-oa-frame-with-knee-level-term-ideas.yaml) |\n"
        + row,
    )
)
notes = directory / "run-notes.md"
text = notes.read_text()
old = "| `localising-design-all-three-parts` | 2026-10-06 03:42 |"
i = text.index(old)
j = text.index("\n", i) + 1
notes.write_text(
    text[:j]
    + "| `separate-oa-frame-with-knee-level-term` | 2026-10-06 04:05 | The highest macro of the campaign (separate OA frame, +0.0004) with the mechanism that cleared its main flag, the Medial Meniscus mark, in three of four runs |\n"
    + text[j:]
)
print("ok")
