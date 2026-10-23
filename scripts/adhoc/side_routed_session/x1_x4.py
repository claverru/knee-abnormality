import json
from pathlib import Path

PUBLIC = {
    "knee-frame": "0.952 (S112)",
    "run4": "0.953 (S113)",
    "run5": "0.952 (S114)",
    "soft-gradient-own-routes": "0.952 (S116)",
    "side-routed-study-frames": "0.952 (S117)",
    "spot-frame-ring-sectors": "0.953 (S118)",
}
runs = [
    Path(
        "outputs/autoresearch/fixes-and-removals/experiments/slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc/training"
    )
] + sorted(
    p
    for p in Path("outputs/training").glob("2026100[56]-*side-routed*")
    if "rn50" not in p.name
)
print(f"{'run':50}{'x1':>9}{'x4':>9}{'x4-x1':>8}  public")
for run in runs:
    file = run / "fold_1" / "selected_checkpoint.json"
    if not file.exists():
        continue
    data = json.loads(file.read_text())
    if "prediction_macro_auc" not in data:
        continue
    name = "old recipe" if "autoresearch" in str(run) else run.name[16:]
    public = (
        "0.950 (S108/S111)"
        if name == "old recipe"
        else next((v for k, v in PUBLIC.items() if name.endswith(k)), "")
    )
    print(
        f"{name:50}{data['score']:9.4f}{data['prediction_macro_auc']:9.4f}{data['prediction_macro_auc'] - data['score']:8.4f}  {public}"
    )
