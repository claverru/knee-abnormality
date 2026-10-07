from pathlib import Path

import yaml

root = Path("autoresearch/campaigns/routing-and-shifts")
source = yaml.safe_load((root / "light-port-shift-ideas-c.yaml").read_text())["ideas"][0]
idea = dict(source)
idea["id"] = "shift-per-series-type-unit-parameter"
idea["title"] = "Learn one shift per view of each series type with the parameter in units of the margin, so the optimiser can move it"
idea["opportunity"] = (
    "Evidence-backed defect, a guess for the gain. shift-per-series-type-light-port-c "
    "(20261007T003634Z-50765893) ran to the end within budget (1,455 s) and normal memory, and its "
    "shifts did not move: the largest of the eight coronal and sagittal values at the last epoch is "
    "0.07 mm (coronal fluid, rows), the others between -0.03 and +0.02 mm. The parameter is held in "
    "millimetres, shift = limit * tanh(view_shift / limit), and trained at the head learning rate of "
    "0.0006, so one optimiser step moves it by about 0.0006 mm: 4,420 steps cannot take it past 2.7 mm "
    "even if every step agreed in sign. The run therefore measured the port with a zero shift (x4 "
    "0.924827 against 0.925389, x1 0.916431 against 0.917411: a rerun of the incumbent through the "
    "cutter), not a learned shift. In the learnable-views campaign the same parameter was held in "
    "units of its limit and moved by millimetres."
)
idea["mechanism"] = (
    "Exactly shift-per-series-type-light-port-c (reuse its candidate's code from "
    "outputs/autoresearch/routing-and-shifts/experiments/shift-per-series-type-light-port-c-20261007T003634Z-50765893: "
    "changes.diff and source/), with one change in step (5): the parameter view_shift[type, view, 2] "
    "is dimensionless, zero at the start, and the shift in millimetres is limit * tanh(view_shift) "
    "with limit = M * pixel spacing (15 mm for the coronal and sagittal views). One optimiser step at "
    "the head learning rate then moves a shift by about 0.009 mm near zero, enough to cross the "
    "margin within the run if the gradient agrees in sign. Steps (1) to (4), the uint8 canvases "
    "normalised with the incumbent's bounds, the cut, the translated patient maps and valid "
    "rectangles, the logging and the checks are unchanged. Nothing else changes."
)
idea["exploitation"] = (
    "Read the logged shifts per epoch first: whether they leave zero by more than 1 mm, whether they "
    "settle, and their sign per view against the hand offsets (sagittal +7 mm forward, +10 mm "
    "vertical). Then both multipliers and the front targets. Shown wrong by: shifts still within 1 mm "
    "of zero at the last epoch, which would mean the gradient has no consistent sign, the "
    "learnable-views finding."
)
idea["priority_rationale"] = "Keeper's seed after a defect the first complete shift run exposed: the shift could not move at the optimiser's step size."
idea["novelty"] = "First run in which a learned shift per series type can actually move on the routed model."
(root / "unit-parameter-shift-ideas.yaml").write_text(
    yaml.safe_dump({"ideas": [idea], "exhausted_directions": ""}, sort_keys=False, allow_unicode=True, width=96)
)
print(idea["id"], len(idea["mechanism"]), len(idea["opportunity"]))
