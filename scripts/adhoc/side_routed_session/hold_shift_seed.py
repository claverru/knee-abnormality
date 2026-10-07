import json
import shutil
from pathlib import Path

path = Path("outputs/autoresearch/routing-and-shifts/state.json")
shutil.copy(path, path.with_name("state.before-holding-shift-seeds-20261007.json"))
s = json.loads(path.read_text())
assert s["status"] == "stopped"
idea = "learned-shift-per-study-one-point"
experiment = s["claimed"]
assert s["experiments"][experiment]["idea"] == idea and s["experiments"][experiment]["stage"] == "prepared"
s["experiments"][experiment]["stage"] = "superseded_by_reconciliation"
s["claimed"] = None
s["ideas"][idea]["status"] = "queued"
queue = [item for item in s["queue"] if item != idea]
queue.insert(queue.index("synovitis-shares-effusion-attention") + 1, idea)
s["queue"] = queue
path.write_text(json.dumps(s, indent=2))
print("queue", s["queue"], "| claimed", s["claimed"], "| running", s["running"])
