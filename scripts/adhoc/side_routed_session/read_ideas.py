import json
import sys
from pathlib import Path

state = json.loads(Path("outputs/autoresearch/side-routing/state.json").read_text())
queue = state["queue"]
print("queue", queue)
ideas = state["ideas"]
items = ideas.values() if isinstance(ideas, dict) else ideas
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 700
for idea in items:
    if idea["id"] in queue and idea.get("origin", "") != "seed":
        print("\n==", idea["id"], "| origin", idea.get("origin"), "| status", idea.get("status"))
        print("TITLE:", idea.get("title"))
        print("OPPORTUNITY:", (idea.get("opportunity") or "")[:limit])
        print("MECHANISM:", (idea.get("mechanism") or "")[:limit])
