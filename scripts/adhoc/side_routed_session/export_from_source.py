import runpy
import sys
from pathlib import Path

experiment = Path(sys.argv[1]).resolve()
frozen = experiment / "source"
checkpoint = next((experiment / "training").glob("fold_*/best-*.ckpt"))
sys.path[:0] = [str(frozen), str(frozen / "src")]
sys.argv = ["export.py", "--checkpoint", str(checkpoint), *sys.argv[2:]]
runpy.run_module("scripts.pipeline.export", run_name="__main__")
