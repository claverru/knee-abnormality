import json

s = json.load(open("outputs/autoresearch/routing-and-shifts/state.json"))


def flat(value):
    return {k: (v if not isinstance(v, (dict, list)) else f"<{type(v).__name__} {len(v)}>") for k, v in value.items() if k not in ("directory", "workspace", "source")}


for key in (
    "learned-shift-per-series-type-20261006T220210Z-47b5085a",
    "learned-shift-per-series-type-20261006T221425Z-19a2bc48",
    "learned-shift-per-study-one-point-20261006T223612Z-af66b09d",
    "menisci-read-their-whole-side-20261006T220713Z-3e5f6af8",
):
    print(key[-30:], flat(s["experiments"][key]))
print("jobs", type(s["jobs"]).__name__, len(s["jobs"]))
jobs = s["jobs"] if isinstance(s["jobs"], list) else list(s["jobs"].values())
for job in jobs[-6:]:
    print("  ", flat(job))
print("findings", json.dumps(s["findings"])[:600])
print("events tail", json.dumps(s["events"][-3:])[:900] if isinstance(s["events"], list) else type(s["events"]))
print("seed_ideas", json.dumps(s["seed_ideas"])[:300], "| research", json.dumps(s["research"])[:200])
