from pathlib import Path

import yaml

root = Path("autoresearch/campaigns/routing-and-shifts")
path = root / "light-port-shift-ideas.yaml"
data = yaml.safe_load(path.read_text())
for idea in data["ideas"]:
    for field in ("mechanism", "implementation"):
        text = " ".join(idea[field].split())
        text = text.replace("the normalisation after the cut, ", "")
        text = text.replace("the zero-shift equality of the port (true at initialisation, since the last layer is zero)", "checks (a) and (b) of shift-per-series-type-light-port (true at initialisation, since the last layer is zero)")
        idea[field] = text
    assert "normalisation after the cut," not in idea["mechanism"], idea["id"]
path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=96))
for idea in data["ideas"][1:3]:
    print(idea["id"], "\n ", idea["mechanism"][:520], "\n")
index = root / "INDEX.md"
s = index.read_text()
anchor = "| Reading the keeper's repairs, adoption decisions and each run's per-target details |"
assert s.count(anchor) == 1
index.write_text(
    s.replace(
        anchor,
        "| Reading the keeper's requeue of the five shift seeds on a lighter port (uint8 canvases normalised in the loader with the incumbent's bounds, no normalisation after the cut), forced on 2026-10-07 after the first port was invalid on time and memory; they replace the four shift seeds still queued | [Light-port shift seeds](light-port-shift-ideas.yaml) |\n"
        + anchor,
    )
)
