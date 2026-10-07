import filecmp
import sys
from pathlib import Path

E = Path("outputs/autoresearch/routing-and-shifts/experiments")
TREES = {
    "spot": E / "spot-shift-locator-keeps-spatial-maps-k-20261007T043843Z-632ceac9" / "source",
    "incumbent": E / "height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402" / "source",
    "worktree": Path("."),
}
a, b = TREES[sys.argv[1]], TREES[sys.argv[2]]
for top in ("src", "scripts", "configs"):
    for path in sorted((a / top).rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        other = b / path.relative_to(a)
        if not other.exists():
            print("only in", sys.argv[1], path.relative_to(a))
        elif not filecmp.cmp(path, other, shallow=False):
            print("differs", path.relative_to(a))
    for path in sorted((b / top).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and not (a / path.relative_to(b)).exists():
            print("only in", sys.argv[2], path.relative_to(b))
