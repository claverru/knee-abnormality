import shutil
from pathlib import Path

keep = Path(
    "/tmp/claude-1000/-home-claudio-src-knee-abnormality/d2f4bec7-d221-4005-8675-56c8c41acf6d/scratchpad/no-front-scorer"
)
for source in sorted(path for path in keep.rglob("*") if path.is_file()):
    target = Path(source.relative_to(keep))
    shutil.copy2(source, target)
    print("reapplied", target)
