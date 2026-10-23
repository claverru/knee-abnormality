import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STUDIES_PER_CELL = 60
CELLS = [
    ("double_glob", 1),
    ("single_scandir", 1),
    ("single_scandir", 8),
    ("single_scandir", 16),
    ("single_scandir", 64),
]


def series_dirs(study: Path) -> list[Path]:
    return [Path(entry.path) for entry in os.scandir(study) if entry.is_dir()]


def double_glob(study: Path) -> tuple[int, int]:
    series = series_dirs(study)
    sum(sum(1 for _ in directory.glob("*.dcm")) for directory in series)
    listed = sum(len(sorted(directory.glob("*.dcm"))) for directory in series)
    return len(series), listed


def single_scandir(study: Path) -> tuple[int, int]:
    series = series_dirs(study)
    total = 0
    for directory in series:
        names = sorted(
            entry.name for entry in os.scandir(directory) if entry.name.endswith(".dcm")
        )
        total += len(names)
    return len(series), total


def run_cell(name: str, threads: int, studies: list[Path]) -> dict:
    function = double_glob if name == "double_glob" else single_scandir
    with Timer() as timer:
        if threads == 1:
            results = [function(study) for study in studies]
        else:
            with ThreadPoolExecutor(threads) as pool:
                results = list(pool.map(function, studies))
    series = sum(count for count, _ in results)
    files = sum(count for _, count in results)
    return {
        "implementation": name,
        "threads": threads,
        "studies": len(studies),
        "series": series,
        "files": files,
        "wall_seconds": round(timer.wall, 3),
        "cpu_seconds": round(timer.cpu, 3),
        "milliseconds_per_study": round(1000 * timer.wall / len(studies), 2),
        "series_per_second": round(series / timer.wall, 1),
    }


def audit(studies: list[Path]) -> dict:
    mismatches = 0
    for study in studies:
        expected = sorted(
            path.name
            for directory in series_dirs(study)
            for path in directory.glob("*.dcm")
        )
        actual = sorted(
            entry.name
            for directory in series_dirs(study)
            for entry in os.scandir(directory)
            if entry.name.endswith(".dcm")
        )
        if expected != actual:
            mismatches += 1
    return {"compared": len(studies), "mismatched_studies": mismatches}


def main() -> None:
    studies = series_directories()
    needed = STUDIES_PER_CELL * (len(CELLS) + 1)
    blocks = [
        studies[index * STUDIES_PER_CELL : (index + 1) * STUDIES_PER_CELL]
        for index in range(len(CELLS) + 1)
    ]
    result = {"available_studies": len(studies), "needed": needed}
    result["audit"] = audit(blocks[0][:20])
    result["cells"] = (
        []
        if AUDIT_ONLY
        else [
            run_cell(name, threads, block)
            for (name, threads), block in zip(CELLS, blocks[1:])
        ]
    )
    report("indexing", result)


main()
