from pathlib import Path

import yaml

directory = Path("autoresearch/campaigns/side-routing")
source = yaml.safe_load((directory / "knee-level-term-ideas.yaml").read_text())["ideas"][0]
idea = dict(source)
idea["id"] = "knee-level-term-on-incumbent-routing"
idea["title"] = (
    "Add to every routed target the shared classifier's knee-level evidence, on the"
    " incumbent's routing"
)
idea["opportunity"] = (
    source["opportunity"].strip()
    + " knee-level-term-with-front-and-up-routing changes the PF OA region and adds the"
    " knee-level term in one design. This seed is the same term on the incumbent's regions"
    " (PF OA in front, the OA sides behind), so the two runs together show what the term is"
    " worth with each region, and the campaign has a design ready that returns the other"
    " compartments' evidence without moving any cut."
)
idea["mechanism"] = (
    "Part (B) of knee-level-term-with-front-and-up-routing only, on the incumbent's routing,"
    " which does not change. In KneeAbnormalityModel._attention_pool, for each shared channel"
    " c compute an unrouted attention over the study's valid tokens, softmax(score_c + view"
    " bias) with the same scorer output and view label bias the routed targets use, with no"
    " routing term and no position prior, and pool the channel's evidence with it: g_c = sum"
    " over tokens of that attention times evidence_c (the same evidence, from the shared"
    " classifiers' position-free tokens). Every target k of channel c then has logit_k = its"
    " routed logit + scale_c * g_c, where scale is a new learned parameter with one scalar per"
    " shared channel (meniscus, oa), started at zero under torch.random.fork_rng. The seven"
    " own-channel targets are unchanged. Nothing else changes."
)
idea["novelty"] = "The knee-level term without any change of region."
idea["exploitation"] = (
    "Read the run report against the incumbent: Medial OA and Lateral OA (expected above the"
    " incumbent's, towards the fourth run's +0.0017 and +0.0019 over it), the five localisation"
    " deltas (expected unchanged: the term is the same for the targets of a channel), the"
    " routed sum against the lineage's best, and the two logged scales. Shown wrong by: the OA"
    " scale near zero, or the OA sides not above the incumbent's with a scale that moved."
)
idea["implementation"] = (
    "modeling/model.py: the new parameter knee_level_scale and the knee-level term in"
    " _attention_pool; SPATIAL_TOKEN_IDENTITY aggregation in modeling/spatial_tokens.py."
    " training/lightning.py: log knee_level_scale_<channel> per epoch. The temperature needs"
    " no change: the unrouted attention uses the scorer and the view bias, which"
    " apply_attention_temperature already scales. No config field and no option."
    " scripts.tools.check_side_routing runs on the CPU when no GPU is visible; all nine"
    " guardrails must pass and its JSON goes in the report."
)
idea["priority_rationale"] = (
    "Keeper seed: the knee-level term on its own regions, beside the combined design, from the"
    " first run's finding that exclusive regions cost the OA sides the other compartments'"
    " evidence."
)
(directory / "knee-level-term-on-incumbent-routing-ideas.yaml").write_text(
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
        "| Reading the keeper seed that adds the knee-level term on the incumbent's regions, beside the combined design | [Knee-level term on the incumbent's routing](knee-level-term-on-incumbent-routing-ideas.yaml) |\n"
        + row,
    )
)
