import json
import sys

path = sys.argv[1]
turns = []
for line in open(path):
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    message = entry.get("message") or {}
    role = message.get("role")
    content = message.get("content")
    if role not in ("user", "assistant"):
        continue
    if isinstance(content, str):
        text = content
    else:
        text = "\n".join(
            part.get("text", "")
            for part in content or []
            if isinstance(part, dict) and part.get("type") == "text"
        )
    text = text.strip()
    if text and not text.startswith("<system-reminder>") and "tool_use_id" not in text[:40]:
        turns.append((role, text))
for role, text in turns[-14:]:
    print(f"--- {role}\n{text[:1800]}\n")
