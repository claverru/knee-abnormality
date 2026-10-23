import json
import os
from pathlib import Path

BASE = Path("/workspace/data/labels/prepared")
OLD = BASE / "anatomy-mrannotator-sample550-fastmri1230-joints"
EXTRA = BASE / "anatomy-mrannotator-fastmri-extra-joints"
FIRST = BASE / "anatomy-mrannotator-fastmri1230-joints"
OUT = BASE / "anatomy-mrannotator-sample550-fastmri1966-joints"
FEMUR, TIBIA = 80.0, 45.0


def study_of(entry):
    return entry["study"] if isinstance(entry, dict) else entry


def reason(knee):
    if knee.get("problem"):
        return f"problem: {knee['problem']}"
    if (knee.get("femur_ml") or 0) < FEMUR or (knee.get("tibia_ml") or 0) < TIBIA:
        return f"femur {knee.get('femur_ml') or 0:.0f} ml, tibia {knee.get('tibia_ml') or 0:.0f} ml"
    return None


old_knees = json.loads((OLD / "knees.json").read_text())
extra_knees = json.loads((EXTRA / "knees.json").read_text())
dropped = {knee["study"]: reason(knee) for knee in extra_knees if reason(knee)}
kept_extra = [knee for knee in extra_knees if knee["study"] not in dropped]
overlap = {study_of(entry) for entry in old_knees} & {knee["study"] for knee in kept_extra}
assert not overlap, overlap
OUT.mkdir(exist_ok=True)
(OUT / "volumes").mkdir(exist_ok=True)
for path in (OLD / "volumes").iterdir():
    link = OUT / "volumes" / path.name
    if not link.exists():
        link.symlink_to(os.path.relpath(path.resolve(), OUT / "volumes"))
for knee in kept_extra:
    source = EXTRA / "volumes" / f"{knee['study']}.npz"
    link = OUT / "volumes" / source.name
    if not link.exists():
        link.symlink_to(os.path.relpath(source.resolve(), OUT / "volumes"))
(OUT / "knees.json").write_text(json.dumps(old_knees + kept_extra, indent=1))
(OUT / "classes.json").write_text((OLD / "classes.json").read_text())
old_measurements = json.loads((OLD / "measurements.json").read_text())
extra_measurements = json.loads((EXTRA / "measurements.json").read_text())
if isinstance(old_measurements, list):
    merged = old_measurements + [entry for entry in extra_measurements if study_of(entry) not in dropped]
else:
    merged = {**old_measurements, **{key: value for key, value in extra_measurements.items() if key not in dropped}}
(OUT / "measurements.json").write_text(json.dumps(merged, indent=1))
sources = json.loads((OLD / "sources.json").read_text())
sources["fastmri_extra"] = {
    "folder": EXTRA.name,
    "segmented": len(extra_knees),
    "kept": len(kept_extra),
    "rule": sources["fastmri"]["rule"],
    "selection": "20% of the fastMRI training studies in total (1,966), the first 1,230 kept; extra knees fill each scanner group to its square-root quota, seed 20261014",
    "dropped": dropped,
}
(OUT / "sources.json").write_text(json.dumps(sources, indent=1))
volumes = len(list((OUT / "volumes").iterdir()))
print("kaggle", sources["kaggle"]["knees"], "fastmri first kept", sources["fastmri"]["kept"], "extra kept", len(kept_extra), "of", len(extra_knees), "volumes", volumes, "knees.json", len(old_knees) + len(kept_extra))
print("dropped", dropped)
print("measurements type", type(old_measurements).__name__, len(merged))
