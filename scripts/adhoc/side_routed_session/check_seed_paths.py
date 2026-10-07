import zipfile
from pathlib import Path

import yaml

base = Path("outputs/autoresearch/learnable-views/experiments")
for name in (
    "hand-sizes-learned-shifts-20261003T183532Z-8d11d7a5",
    "knee-anchored-learned-offsets-20261002T194645Z-d4e331e1",
    "hand-views-routed-shift-learning-20261003T212020Z-0863da22",
):
    folder = base / name
    print(name, "| source dir:", (folder / "source").is_dir(), "| changes.diff:", (folder / "changes.diff").is_file())
for run in ("20261006-184303-side-routed-midway-cuts", "20261006-200553-side-routed-mirrored-held-division"):
    archive = Path("outputs/training") / run / "source_snapshot.zip"
    names = zipfile.ZipFile(archive).namelist()
    print(run, "| model.py in snapshot:", any(name.endswith("modeling/model.py") for name in names))
print("division json:", Path("outputs/analysis/side-routing-cut-forces-20261006/symmetric-division-0.15.json").is_file())
print("guardrail allows the locator loss:", "shift_locator_loss" in Path("scripts/tools/check_side_routing.py").read_text())
root = Path("autoresearch/campaigns/routing-and-shifts")
order = []
for file in ("seed-ideas.yaml", "seed-ideas-2.yaml", "seed-ideas-3.yaml"):
    for idea in yaml.safe_load((root / file).read_text())["ideas"]:
        order.append(idea["id"])
        long = {key: len(value) for key, value in idea.items() if isinstance(value, str) and len(value) > 2600}
        if long:
            print("  long fields in", idea["id"], long)
print("order:", order)
