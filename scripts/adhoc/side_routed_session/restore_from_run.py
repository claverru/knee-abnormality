import shutil
import sys
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
)
run = Path(sys.argv[1])
keep = Path(sys.argv[2])
archive = zipfile.ZipFile(run / "source_snapshot.zip")
names = set(archive.namelist())
for name in FILES:
    target = keep / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(name, target)
    assert name in names, name
    Path(name).write_bytes(archive.read(name))
    print("restored", name)
