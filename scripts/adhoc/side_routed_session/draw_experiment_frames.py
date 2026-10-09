import sys
from pathlib import Path

experiment = Path(sys.argv[1]).resolve()
rows = sys.argv[2]
label = sys.argv[3]
sys.path[:0] = [str(experiment / "source"), str(experiment / "source" / "src")]
script = Path("/tmp/claude-1000/-home-claudio-src-knee-abnormality/d2f4bec7-d221-4005-8675-56c8c41acf6d/scratchpad/draw_regions.py").read_text()
old = 'OUT = Path("outputs/analysis/s/side-routing-loop-20261006") / RUN.name[16:]'
assert script.count(old) == 1
script = script.replace(old, f'OUT = Path("outputs/analysis/r/routing-and-shifts-frames-20261007") / "{label}"')
sys.argv = ["draw_regions.py", str(experiment / "training"), rows]
exec(compile(script, "draw_regions.py", "exec"))
