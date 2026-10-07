from pathlib import Path

base = Path("/home/claudio/src/knee-abnormality/outputs/autoresearch/fixes-and-removals/experiments/slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc")
target = Path("/tmp/claude-1000/-home-claudio-src-knee-abnormality/d2f4bec7-d221-4005-8675-56c8c41acf6d/scratchpad/old-experiment")
target.mkdir(parents=True, exist_ok=True)
for name in ("source", "training"):
    link = target / name
    if not link.exists():
        link.symlink_to(base / name)
print(sorted(p.name for p in target.iterdir()))
