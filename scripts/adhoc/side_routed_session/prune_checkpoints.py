import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

DELETE = len(sys.argv) > 1 and sys.argv[1] == "delete"
REFERENCE = "20261006-163650-side-routed-spot-frame-ring-sectors"
outputs = Path("outputs").resolve()
live = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
campaign = outputs / "autoresearch" / "side-routing"
rows = []
for path in sorted(campaign.rglob("*.ckpt")):
    rows.append(("side-routing campaign", path.relative_to(campaign).parts[1], path))
for run in sorted((outputs / "training").glob("2026100[56]-*side-routed*")):
    for path in sorted(run.rglob("*.ckpt")):
        rows.append(("manual runs", run.name, path))
plan, kept = [], []
for group, run, path in rows:
    size = path.stat().st_size
    name = run[16:] if group == "manual runs" else run
    if run == REFERENCE:
        kept.append((group, run, path, size, "the reference"))
    elif f"--name {name}" in live or str(path.parent) in live:
        kept.append((group, run, path, size, "a live training has this run's name"))
    else:
        plan.append((group, run, path, size))
gib = 1024**3
for group in ("side-routing campaign", "manual runs"):
    chosen = [row for row in plan if row[0] == group]
    print(f"{group}: delete {len(chosen)} checkpoints in {len({row[1] for row in chosen})} runs, {sum(row[3] for row in chosen) / gib:.2f} GiB")
print("kept:")
for reason in sorted({row[4] for row in kept}):
    chosen = [row for row in kept if row[4] == reason]
    print(f"  {reason}: {len(chosen)} checkpoints, {sum(row[3] for row in chosen) / gib:.2f} GiB, runs {sorted({row[1] for row in chosen})}")
print("submitted bundles of this line and their weights:")
for bundle in sorted((outputs / "submission" / "exports").glob("side-rout*")):
    if bundle.is_dir():
        weights = [p for p in bundle.rglob("*.ckpt")]
        print(f"  {bundle.name}: {len(weights)} weight file(s), {sum(p.stat().st_size for p in weights) / gib:.2f} GiB")
if DELETE:
    failed = []
    for _, _, path, _ in plan:
        try:
            path.unlink()
        except OSError as error:
            failed.append((str(path), str(error)))
    folder = outputs / "archive" / "maintenance" / f"disk-cleanup-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "request": "user, 2026-10-06: delete every checkpoint of the side-routing campaign and of the manual side-routed runs except the reference",
                "deleted": [{"group": g, "run": r, "path": str(p), "bytes": s} for g, r, p, s in plan],
                "kept": [{"group": g, "run": r, "path": str(p), "bytes": s, "reason": why} for g, r, p, s, why in kept],
                "failed": failed,
            },
            indent=1,
        )
    )
    print("deleted", len(plan) - len(failed), "failed", len(failed), "manifest", folder / "manifest.json")
