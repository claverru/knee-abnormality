from pathlib import Path

RUNS = {
    "old recipe (no octants)": "outputs/autoresearch/fixes-and-removals/experiments/slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc/training",
    "run 5 (octants)": "outputs/training/20261005-205251-side-routed-run5",
    "run 7": "outputs/training/20261006-081303-side-routed-soft-gradient-own-routes",
    "run 11 (per-knee frame)": "outputs/training/20261006-103317-side-routed-study-frames-landmarks",
    "run 12 (regions, routed reports)": "outputs/training/20261006-112944-side-routed-eighteen-regions",
    "run 18": "outputs/training/20261006-135237-side-routed-regions-soft-gradient-knee-level",
    "run 22": "outputs/training/20261006-153716-side-routed-regions-kneecap-above-lower-cut",
}
for name, path in RUNS.items():
    run = Path(path)
    metrics = next(run.glob("fold_*/logs/version_0/metrics.csv"))
    start = (run / "effective_config.json").stat().st_mtime
    done = (run / "fold_1" / "selected_checkpoint.json").stat().st_mtime
    print(
        f"{name:34} ten epochs {(metrics.stat().st_mtime - start) / 60:5.1f} min   "
        f"with the final prediction {(done - start) / 60:5.1f} min"
    )
