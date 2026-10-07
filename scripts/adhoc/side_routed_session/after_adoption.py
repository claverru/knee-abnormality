import json
from pathlib import Path

import yaml

directory = Path("autoresearch/campaigns/side-routing")
run = "outputs/autoresearch/side-routing/experiments/separate-oa-frame-with-knee-level-term-20261006T020231Z-52418f08/training"
lineage_file = Path("outputs/analysis/side-routing-lineage.json")
lineage = json.loads(lineage_file.read_text())
if run not in lineage:
    lineage.append(run)
    lineage_file.write_text(json.dumps(lineage, indent=2))

state = json.loads(Path("outputs/autoresearch/side-routing/state.json").read_text())
ideas = state["ideas"]
items = list(ideas.values()) if isinstance(ideas, dict) else ideas
source = next(i for i in items if i["id"] == "calibrated-shared-report-outputs")
fields = (
    "mechanism_id",
    "parent_idea_id",
    "title",
    "opportunity",
    "mechanism",
    "precedents",
    "novelty",
    "coverage",
    "exploitation",
    "budget",
    "implementation",
    "open_choices",
    "dependencies",
    "priority_rationale",
)
idea = {"id": "calibrated-shared-report-outputs-requeue"}
for field in fields:
    idea[field] = source.get(field)
idea["parent_idea_id"] = None
idea["dependencies"] = []
idea["implementation"] = (
    (idea["implementation"] or "").strip()
    + " Keeper note, 2026-10-06: requeue of the researcher's calibrated-shared-report-outputs."
    " Its first candidate was built on the previous incumbent and its training was cancelled"
    " seconds after launch, when separate-oa-frame-with-knee-level-term was adopted; build"
    " this one on the current incumbent, using its route names."
)
idea["priority_rationale"] = (
    "Requeue after a cancelled launch on a superseded incumbent, not a result."
)
(directory / "calibrated-shared-report-outputs-requeue-ideas.yaml").write_text(
    yaml.safe_dump(
        {"ideas": [idea], "exhausted_directions": ""},
        sort_keys=False,
        allow_unicode=True,
        width=98,
    )
)

prior = directory / "prior-work.md"
text = prior.read_text()
old = "`outputs/training/20261005-205251-side-routed-run5`); the keeper appends every adopted run's\ntraining directory."
assert text.count(old) == 1
prior.write_text(
    text.replace(
        old,
        "`outputs/training/20261005-205251-side-routed-run5`); the keeper appends every adopted run's\ntraining directory. Adopted so far: `" + run + "` (2026-10-06).",
    )
)

index = directory / "INDEX.md"
content = index.read_text()
row = "| Reading the keeper's repairs, adoption decisions and added seeds |"
assert content.count(row) == 1
index.write_text(
    content.replace(
        row,
        "| Reading the keeper requeue of the researcher's calibrated report outputs, after its first launch was cancelled at the first adoption | [Calibrated report outputs requeue](calibrated-shared-report-outputs-requeue-ideas.yaml) |\n"
        + row,
    ).replace(
        "**Running since 2026-10-05 23:30 CEST, Codex, three forced seeds, open-ended.**",
        "**Running since 2026-10-05 23:30 CEST, Codex, open-ended; first adoption 2026-10-06 04:50 (`separate-oa-frame-with-knee-level-term`).**",
    )
)

notes = directory / "run-notes.md"
text = notes.read_text()
old = "### A standing obstacle: the fourth run's Medial Meniscus mark\n"
assert text.count(old) == 1
new = '''### 13. `separate-oa-frame-with-knee-level-term` — **adopted** (2026-10-06 04:50)

×4 0.924485 against the incumbent's 0.924476: +0.000010. **No flag**: the first run of the
campaign whose report says `joins_lineage: true`. It is the new incumbent.

| | Value | Condition |
|---|---:|---|
| Macro | +0.00001 | not below the incumbent |
| Routed sum against the best | −0.0088 | tolerance 0.0090 |
| Medial Meniscus localisation | +0.0062 (0.0028 above the fourth run's mark) | |
| Lateral Meniscus localisation | +0.0039 | |
| PF OA localisation | +0.0114 | |
| Medial OA, Lateral OA localisation | −0.0019, −0.0061 | tolerances 0.0030, 0.0126 |
| Own-channel targets | sum +0.0048; none below every sibling (ACL +0.0014, Baker's +0.0008) | |
| Side head | 99.8% | |

- **The design:** the researcher's separate frame for OA with PF OA routed from front-and-up,
  plus the keeper's knee-level term. The frames turned apart again (meniscus +6.3°, OA −5.9°
  about the vertical axis; nods −2.5° and +3.7°) and the knee-level strengths were negative
  again: both learned behaviours repeated.
- **What it is and is not.** On the plain macro it equals the fifth run; it was adopted because
  it holds that macro while telling compartments apart better than any earlier base on PF OA
  and both meniscus targets. It is not evidence of a better score.
- **Mechanics.** The supervisor stopped with this run awaiting its decision, and `adopt` needs
  a decided run. I resumed, let the decision be recorded, aborted the training that launched
  in the same pass (`calibrated-shared-report-outputs`, built on the old incumbent, seconds
  in), adopted, and resumed. That idea is requeued for the new incumbent. GPU idle: about 12
  minutes, until the first candidate on the new incumbent is built.

'''
text = text.replace(old, new + old)
old = "| `separate-oa-frame-with-knee-level-term` | 2026-10-06 04:05 |"
i = text.index(old)
j = text.index("\n", i) + 1
text = (
    text[:j]
    + "| `calibrated-shared-report-outputs-requeue` | 2026-10-06 04:52 | Requeue of the researcher's idea on the new incumbent, after its launch on the old one was cancelled at the adoption |\n"
    + text[j:]
)
notes.write_text(text)
print(lineage)
