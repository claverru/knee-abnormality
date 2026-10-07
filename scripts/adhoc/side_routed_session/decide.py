import json
import subprocess
import sys
from pathlib import Path

campaign = Path("outputs/autoresearch/side-routing")
lineage_file = Path("outputs/analysis/side-routing-lineage.json")
if not lineage_file.exists():
    lineage_file.write_text(
        json.dumps(
            [
                "outputs/training/20261005-202001-side-routed-run4",
                "outputs/training/20261005-205251-side-routed-run5",
            ],
            indent=2,
        )
    )
lineage = json.loads(lineage_file.read_text())
state = json.loads((campaign / "state.json").read_text())
incumbent = state["incumbent"]
base = Path(incumbent["run"])
if not (base / "fold_1").exists() and (base / "training" / "fold_1").exists():
    base = base / "training"
experiment = sorted((campaign / "experiments").glob(f"{sys.argv[1]}*"))[-1]
run = experiment / "training"
old = Path("outputs/autoresearch/fixes-and-removals/experiments")
reference = old / "slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc/training"
siblings = [
    str(path.parents[1])
    for path in sorted(old.glob("*/training/fold_1/validation_predictions.csv"))
    if path.parents[1] != reference
]
out = experiment / "side_routing_report.json"
subprocess.run(
    [
        sys.executable,
        "-m",
        "scripts.tools.report_side_routing",
        "--run",
        str(run),
        "--base",
        str(base),
        "--lineage",
        json.dumps(lineage),
        "--siblings",
        json.dumps(siblings),
        "--output",
        str(out),
    ],
    check=True,
    stdout=subprocess.DEVNULL,
)
report = json.loads(out.read_text())
print("experiment", experiment.name, "| incumbent", incumbent["id"])
print("macro", round(report["macro_auc"], 6), "delta", round(report["macro_delta"], 6))
print("routed sum vs base", round(report["routed_sum"], 4), "| vs best", round(report["routed_sum_against_best"], 4), "| tolerance", round(2 * report["routed_sum_sibling_sd"], 4))
print("own sum vs base", round(report["own_sum"], 4))
for name, value in report["per_target"].items():
    print(f"  {name:18s} {value['delta']:+.4f}{'  BELOW ALL SIBLINGS' if value['below_all_siblings'] else ''}")
print("localisation (vs base, vs best, tolerance):")
for name, value in report["localisation"].items():
    print(f"  {name:18s} {value['delta']:+.4f} {value['against_best']:+.4f} {2 * value['sibling_sd']:.4f}")
print("side head", report["side_head"])
print("geometry", {k: round(v, 2) for k, v in report["learned_geometry"].items()})
print("flags", report["flags"])
print("new own flags", report["new_own_targets_below_all_siblings"])
print("JOINS_LINEAGE", report["joins_lineage"])
