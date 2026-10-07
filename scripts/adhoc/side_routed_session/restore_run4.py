import shutil
import sys
import zipfile
from pathlib import Path

snapshot = Path("outputs/training/20261005-202001-side-routed-run4/source_snapshot.zip")
keep = Path(
    "/tmp/claude-1000/-home-claudio-src-knee-abnormality/d2f4bec7-d221-4005-8675-56c8c41acf6d/scratchpad/no-front-scorer"
)
apply = len(sys.argv) > 1 and sys.argv[1] == "apply"
with zipfile.ZipFile(snapshot) as archive:
    names = [name for name in archive.namelist() if not name.endswith("/")]
    print("snapshot files", len(names), "top-level", sorted({name.split("/")[0] for name in names}))
    different = []
    for name in names:
        current = Path(name)
        data = archive.read(name)
        if not current.exists() or current.read_bytes() != data:
            different.append(name)
    print("differ from the snapshot:", different)
    if apply:
        for name in different:
            current = Path(name)
            if current.exists():
                target = keep / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current, target)
            current.write_bytes(archive.read(name))
        print("restored", len(different))
