import shutil
import zipfile
from pathlib import Path

FILES = (
    "src/knee_abnormality/modeling/model.py",
    "src/knee_abnormality/modeling/spatial_tokens.py",
    "src/knee_abnormality/training/lightning.py",
    "src/knee_abnormality/training/report_attributes.py",
    "configs/training.yaml",
    "configs/refined-resnet50.yaml",
    "scripts/tools/check_side_routing.py",
    "scripts/tools/diagnose_encoding_usage.py",
    "scripts/adhoc/render_routing_views.py",
)
keep = Path("/tmp/claude-1000/-home-claudio-src-knee-abnormality/d2f4bec7-d221-4005-8675-56c8c41acf6d/scratchpad/run10-build")
archive = zipfile.ZipFile("outputs/training/20261006-084210-side-routed-study-frames/source_snapshot.zip")
names = set(archive.namelist())
for name in FILES:
    target = keep / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(name, target)
    assert name in names, name
    Path(name).write_bytes(archive.read(name))
    print("restored", name)
