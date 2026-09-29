import csv
import math
from pathlib import Path


def metrics_path(fold: Path):
    versions = [
        path
        for path in Path(fold).glob("logs/version_*/metrics.csv")
        if path.parent.name.removeprefix("version_").isdigit()
    ]
    return max(
        versions,
        key=lambda path: int(path.parent.name.removeprefix("version_")),
        default=None,
    )


def read_metrics(fold: Path):
    path = metrics_path(fold)
    if path is None:
        return []
    try:
        lines = path.read_text().splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines.pop()
        table = list(csv.reader(lines))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    if not table:
        return []
    header, *body = table
    rows = []
    for cells in body:
        if len(cells) != len(header):
            continue
        row = {}
        for name, cell in zip(header, cells, strict=True):
            try:
                value = float(cell)
            except ValueError:
                continue
            if math.isfinite(value):
                row[name] = value
        if "epoch" in row:
            rows.append(row)
    return rows
