import filecmp
import shutil
import sys
from pathlib import Path

experiment = Path(sys.argv[1])
target = Path(sys.argv[2])
frozen = experiment / "source"
changed = []
for path in sorted((frozen / "src" / "knee_abnormality").rglob("*.py")):
    relative = path.relative_to(frozen)
    destination = target / relative
    if not destination.exists() or not filecmp.cmp(path, destination, shallow=False):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        changed.append(str(relative))
print("\n".join(changed))
