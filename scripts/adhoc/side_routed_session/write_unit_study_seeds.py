from pathlib import Path

import yaml

root = Path("autoresearch/campaigns/routing-and-shifts")
source = {idea["id"]: idea for idea in yaml.safe_load((root / "light-port-shift-ideas-c.yaml").read_text())["ideas"]}
RENAME = {
    "shift-per-study-one-point-light-port-c": "shift-per-study-one-point-unit-output",
    "shift-per-study-per-view-light-port-c": "shift-per-study-per-view-unit-output",
    "shift-per-study-from-the-spots-light-port-c": "shift-per-study-from-the-spots-unit-output",
    "crops-and-frame-share-one-centre-light-port-c": "crops-and-frame-share-one-centre-unit-output",
}
EVIDENCE = (
    " Keeper's seed of 2026-10-07 after a defect the first per-study run exposed. "
    "shift-per-study-one-point-light-port-c (20261007T005816Z-9a106989) trained within budget and its "
    "point barely moved: at the last epoch the mean over validation knees is -0.62, +0.59 and +0.20 mm "
    "(anterior, left-right, superior) and the spread across knees 0.07, 0.09 and 0.06 mm, so every knee "
    "received the same sub-millimetre offset (x4 0.924344 and x1 0.915927 against 0.925389 and "
    "0.917411). The locator's last layer gives millimetres directly and learns at the head rate of "
    "0.0006, as the per-series parameter did when it stayed within 0.07 mm of zero."
)
CHANGE = (
    "One change against {old}, whose code is reused (changes.diff and source/ of the latest candidate "
    "of shift-per-study-one-point-light-port-c under outputs/autoresearch/routing-and-shifts/experiments/ "
    "for the port and the locator): the locator's last layer is dimensionless. Wherever {old} bounds a "
    "millimetre output x as limit * tanh(x / limit), this seed uses limit * tanh(x) with the same "
    "limit, and the last layer still starts at zero. A step of the optimiser then moves a shift about "
    "fifteen times further. Everything else is as follows. "
)
out = []
for old, new in RENAME.items():
    idea = dict(source[old])
    idea["id"] = new
    for field in ("title", "opportunity", "mechanism", "exploitation", "budget", "implementation", "novelty", "coverage"):
        text = " ".join(idea[field].split())
        for before, after in RENAME.items():
            text = text.replace(before, after)
        idea[field] = text
    idea["opportunity"] = idea["opportunity"] + EVIDENCE
    idea["mechanism"] = CHANGE.format(old=old) + idea["mechanism"]
    idea["priority_rationale"] = "Keeper's seed: the per-study shift could not move at the optimiser's step size. " + idea["priority_rationale"]
    out.append(idea)
(root / "unit-output-study-shift-ideas.yaml").write_text(
    yaml.safe_dump({"ideas": out, "exhausted_directions": ""}, sort_keys=False, allow_unicode=True, width=96)
)
print([idea["id"] for idea in out])
