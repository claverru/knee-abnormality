import json
import os
import random
import time
from pathlib import Path


def competition_root() -> Path:
    override = os.environ.get("KNEE_PROBE_DATA")
    if override:
        return Path(override)
    root = Path("/kaggle/input")
    candidates = sorted(
        {
            path.parent
            for pattern in (
                "*/train_series.csv",
                "*/*/train_series.csv",
                "*/*/*/train_series.csv",
            )
            for path in root.glob(pattern)
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one competition root, found {candidates}")
    return candidates[0]


COMPETITION = competition_root()
WORKING = Path(os.environ.get("KNEE_PROBE_OUT", "/kaggle/working"))
AUDIT_ONLY = bool(os.environ.get("KNEE_PROBE_AUDIT_ONLY"))
SEED = 20260917


def environment() -> dict:
    cpuinfo = Path("/proc/cpuinfo")
    model = None
    if cpuinfo.exists():
        model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        )
    limit = Path("/sys/fs/cgroup/cpu.max")
    return {
        "competition_root": str(COMPETITION),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "cpu_model": model,
        "cgroup_cpu_max": limit.read_text().strip() if limit.exists() else None,
    }


def series_directories(split: str = "train") -> list[Path]:
    root = COMPETITION / f"{split}_series"
    studies = sorted(entry.path for entry in os.scandir(root) if entry.is_dir())
    random.Random(SEED).shuffle(studies)
    return [Path(study) for study in studies]


def series_paths(study_dir: Path) -> list[Path]:
    return [
        Path(entry.path)
        for series in os.scandir(study_dir)
        if series.is_dir()
        for entry in os.scandir(series.path)
        if entry.name.endswith(".dcm")
    ]


def file_blocks(
    cells: int, files_per_cell: int, split: str = "train"
) -> list[list[Path]]:
    blocks: list[list[Path]] = []
    current: list[Path] = []
    studies = iter(series_directories(split))
    while len(blocks) < cells:
        current.extend(series_paths(next(studies)))
        if len(current) >= files_per_cell:
            blocks.append(sorted(current))
            current = []
    return blocks


class Timer:
    def __enter__(self):
        self.wall = time.perf_counter()
        self.cpu = time.process_time()
        return self

    def __exit__(self, *_):
        self.wall = time.perf_counter() - self.wall
        self.cpu = time.process_time() - self.cpu


def report(name: str, result: dict) -> None:
    result["probe"] = name
    result["environment"] = environment()
    text = json.dumps(result, indent=1, default=str)
    (WORKING / f"{name}.json").write_text(text + "\n")
    print(text, flush=True)
