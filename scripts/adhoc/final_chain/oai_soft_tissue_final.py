import json
import os
from pathlib import Path

PREPARED = Path("/workspace/data/labels/prepared")
OAI = PREPARED / "soft-tissue-nnunet-knee-oai1887"
KAGGLE = PREPARED / "soft-tissue-nnunet-knee-dess265"
COMBINED = PREPARED / "soft-tissue-nnunet-knee-dess265-oai"
CASES = Path("/root/.cache/knee-segmenters/oai-dess/cases.json")

cases = json.loads(CASES.read_text())
mismatched = sorted(
    case["study"]
    for case in cases
    if "size" in case and not (case["frame_matches"] and all(case["frame_matches"].values()))
)
removed = []
for study in mismatched:
    path = OAI / "volumes" / f"{study}.npz"
    if path.exists():
        path.unlink()
        removed.append(study)
knees = [knee for knee in json.loads((OAI / "knees.json").read_text()) if knee["study"] not in set(mismatched)]
(OAI / "knees.json").write_text(json.dumps(knees, indent=1) + "\n")
(OAI / "excluded_frames.json").write_text(json.dumps(mismatched, indent=1) + "\n")
oai_volumes = sorted(path.stem for path in (OAI / "volumes").glob("*.npz"))
print("mismatched", len(mismatched), "removed volumes", len(removed), "oai volumes", len(oai_volumes), "oai knees", len(knees))

kaggle_classes = json.loads((KAGGLE / "classes.json").read_text())["classes"]
oai_classes = json.loads((OAI / "classes.json").read_text())["classes"]
if kaggle_classes != oai_classes:
    raise ValueError("class lists differ")
(COMBINED / "volumes").mkdir(parents=True, exist_ok=True)
for source in (KAGGLE, OAI):
    for path in (source / "volumes").glob("*.npz"):
        link = COMBINED / "volumes" / path.name
        if not link.exists():
            os.symlink(Path("../..") / source.name / "volumes" / path.name, link)
combined_knees = json.loads((KAGGLE / "knees.json").read_text()) + knees
(COMBINED / "knees.json").write_text(json.dumps(combined_knees, indent=1) + "\n")
(COMBINED / "classes.json").write_text((KAGGLE / "classes.json").read_text())
links = list((COMBINED / "volumes").glob("*.npz"))
broken = [link for link in links if not link.resolve().exists()]
print("combined volumes", len(links), "broken", len(broken), "knees", len(combined_knees))
