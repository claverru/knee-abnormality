import json
from pathlib import Path

outputs = Path("outputs").resolve()
ledger = json.loads((outputs / "submission" / "kaggle-ledger.json").read_text())
items = ledger if isinstance(ledger, list) else list(ledger.get("submissions", ledger).values()) if isinstance(ledger.get("submissions", ledger), dict) else ledger.get("submissions", ledger)
print("ledger entries", len(items), "| keys", sorted(items[-1])[:14])
names = set()
for item in items:
    for key in ("bundle", "bundle_name", "export", "name", "bundle_dir"):
        if key in item and item[key]:
            names.add(Path(str(item[key])).name)
exports = outputs / "submission" / "exports"
found = {}
for bundle in sorted(exports.iterdir()):
    if not bundle.is_dir() or "side-routed" not in bundle.name and "side-routing" not in bundle.name and "octant" not in bundle.name:
        continue
    text = ""
    for file in list(bundle.glob("*.json")) + list(exports.glob(bundle.name + "*.json")):
        text += file.read_text()
    runs = sorted({part for part in text.replace('"', " ").replace(",", " ").split() if "/training/" in part or "/experiments/" in part})
    found[bundle.name] = (bundle.name in names, [r[-150:] for r in runs[:4]])
for name, (in_ledger, runs) in found.items():
    print(name, "| in ledger:", in_ledger)
    for run in runs:
        print("     ", run)
print(sorted(names)[-10:])
