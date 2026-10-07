import json
import subprocess
import time
from pathlib import Path

MARK = "Environment, read this first."
state_path = Path("outputs/autoresearch/routing-and-shifts/state.json")
deadline = time.time() + 3 * 3600
while time.time() < deadline:
    try:
        ideas = json.loads(state_path.read_text())["ideas"]
    except (OSError, ValueError):
        time.sleep(10)
        continue
    copied = {name[:-2] for name in ideas if name.endswith("-k")}
    waiting = [
        name
        for name, idea in ideas.items()
        if idea.get("status") == "queued" and MARK not in str(idea.get("implementation")) and name not in copied
    ]
    if waiting:
        print("new queued proposals:", waiting, flush=True)
        run = subprocess.run(
            ["uv", "run", "--locked", "python", "/tmp/claude-1000/-home-claudio-src-knee-abnormality/d2f4bec7-d221-4005-8675-56c8c41acf6d/scratchpad/protect_ideas.py"],
            capture_output=True, text=True,
        )
        print(run.stdout[-1200:], run.stderr[-300:], flush=True)
        break
    time.sleep(20)
else:
    print("no new proposals in three hours")
