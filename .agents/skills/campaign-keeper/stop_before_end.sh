#!/usr/bin/env bash
campaign="$1"
root="$2"
cd "$root" || exit 1
status() { rtk proxy uv run --locked python -m scripts.tools.autoresearch status "$campaign" 2>&1; }
until status | grep -qE "^ +training .*epoch (8|9|10)/10"; do sleep 30; done
rtk proxy uv run --locked python -m scripts.tools.autoresearch stop "$campaign" >/dev/null 2>&1
until status | head -1 | grep -q "STOPPED"; do sleep 15; done
latest=$(ls -t "outputs/autoresearch/$campaign/experiments"/*/evaluation.json | head -1)
python3 - "$latest" <<'PY'
import json
import sys

path = sys.argv[1]
evaluation = json.load(open(path))
macro = evaluation["macro_auc"]
print(
    path.split("/")[-2],
    evaluation["eligible_decision"],
    f"reference {macro['reference']:.6f}",
    f"candidate {macro['candidate']:.6f}",
    f"delta {macro['candidate'] - macro['reference']:+.6f}",
    f"x1 delta {evaluation['validation_macro_auc']['candidate'] - evaluation['validation_macro_auc']['reference']:+.6f}",
    "80%",
    [round(value, 6) for value in evaluation["adoption_bootstrap"]["delta_percentile_80"]],
)
PY
