import json
from pathlib import Path

root = Path("outputs/autoresearch/side-routing")
manifest = json.loads((root / "baseline/manifest.json").read_text())
print("manifest keys", list(manifest.keys())[:15])
for key, value in manifest.items():
    if isinstance(value, dict):
        hits = [k for k in value if "check_side" in k]
        print(key, len(value), hits[:2])
    elif isinstance(value, list):
        print(key, len(value), [v for v in value if "check_side" in str(v)][:2])
    else:
        print(key, str(value)[:100])
state = json.loads((root / "state.json").read_text())
print("status", state["status"], "reason", state["reason"])
print("experiments", {k: {kk: str(vv)[:80] for kk, vv in v.items() if kk in ("status", "idea_id", "repairs", "parent_source", "workspace", "stage")} for k, v in state["experiments"].items()})
print("jobs", {k: (v.get("role"), v.get("status")) for k, v in state["jobs"].items()})
print("profiles", json.dumps(state["profiles"])[:600])
