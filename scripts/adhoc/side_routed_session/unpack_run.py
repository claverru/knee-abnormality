import sys
import zipfile
from pathlib import Path

run = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
archive = zipfile.ZipFile(run / "source_snapshot.zip")
archive.extractall(target / "source")
link = target / "training"
if not link.exists():
    link.symlink_to(run)
print(sorted(p.name for p in (target / "source").iterdir()), len(archive.namelist()))
print(sorted(p.name for p in (link / "fold_1").glob("best-*")))
