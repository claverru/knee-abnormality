import sys
from pathlib import Path

import yaml

directory = Path("autoresearch/campaigns/side-routing")
codes = yaml.safe_load((directory / "image-frame-position-codes-ideas.yaml").read_text())["ideas"][0]
term = yaml.safe_load(
    (directory / "knee-level-term-on-incumbent-routing-ideas.yaml").read_text()
)["ideas"][0]
evidence = sys.argv[1]
idea = {
    "id": "knee-level-term-with-image-frame-position-codes",
    "mechanism_id": "knee-level-disease-term",
    "parent_idea_id": None,
    "title": "Combine the knee-level term of the routed targets with the image-frame position"
    " codes of the own-channel classifiers",
    "opportunity": (
        "Evidence-backed on each part, a guess that they add up. (1)"
        " knee-level-term-on-incumbent-routing (2026-10-06): x4 0.924834, +0.000358 over the"
        " incumbent and the highest of the line; Medial and Lateral Meniscus localisation"
        " +0.0041 and +0.0059, above the fourth run's mark; the two learned strengths settled"
        " at -0.138 (meniscus) and -0.021 (OA), so each compartment's evidence is contrasted"
        " with the knee-level evidence; its one flag was ACL, 0.0012 below the incumbent's and"
        " 0.0002 under the lowest sibling run. (2) image-frame-position-codes-for-own-channels"
        " (2026-10-06): " + evidence + " The two changes touch different parts of the model:"
        " the first only the logits of the five routed targets, the second only the tokens the"
        " seven own-channel classifiers read."
    ),
    "mechanism": (
        "Both mechanisms exactly as in their own runs, whose sources are under"
        " outputs/autoresearch/side-routing/experiments/knee-level-term-on-incumbent-routing-*/source"
        " and outputs/autoresearch/side-routing/experiments/image-frame-position-codes-for-own-channels-*/source."
        " (A) " + term["mechanism"].strip() + " (B) " + codes["mechanism"].strip()
    ),
    "precedents": [
        "autoresearch/campaigns/side-routing/run-notes.md, decisions 5 and 6: the two runs this design combines.",
    ],
    "novelty": "The two changes have only been run apart.",
    "coverage": "Every study, training and inference: the five routed targets through the knee-level term, the seven own-channel targets through the token codes.",
    "exploitation": (
        "Read the run report against the incumbent: the macro (expected above both parts),"
        " ACL and the own-targets flag, Fracture and Baker's, the meniscus localisation values"
        " against the lineage's best, the routed sum, and the two knee-level strengths"
        " (expected negative again). Shown wrong by: a macro below the better of the two parts,"
        " or the own-targets flag on ACL."
    ),
    "budget": "Same cost as the incumbent.",
    "implementation": (
        "modeling/model.py, modeling/spatial_tokens.py and training/lightning.py: apply both"
        " changes as their own experiments did (see the two sources named in the mechanism);"
        " they edit different functions (_attention_pool and the knee_level_scale parameter;"
        " encode_batch, study_tokens and the token codes). No config field and no option."
        " scripts.tools.check_side_routing runs on the CPU when no GPU is visible; all nine"
        " guardrails must pass and its JSON goes in the report."
    ),
    "open_choices": "None.",
    "dependencies": [],
    "priority_rationale": "Keeper seed: the two parts that each moved their own targets, as one design.",
}
(directory / "knee-level-term-with-image-frame-codes-ideas.yaml").write_text(
    yaml.safe_dump(
        {"ideas": [idea], "exhausted_directions": ""},
        sort_keys=False,
        allow_unicode=True,
        width=98,
    )
)
print("written")
