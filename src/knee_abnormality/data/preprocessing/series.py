import math
from dataclasses import dataclass
from pathlib import Path

from knee_abnormality.data.preprocessing.dicom import (
    PREPARATION_ERRORS,
    NativePixelDescriptor,
    read_header,
)
from knee_abnormality.data.preprocessing.geometry import (
    NativePose,
    SeriesGeometry,
    summarize_series,
)
from knee_abnormality.data.sources import KNOWN_BAD_INSTANCES


@dataclass(frozen=True, slots=True)
class SliceRecord:
    path: Path
    physical_position: float
    original_index: int
    relative_position_mm: float | None = None
    descriptor: NativePixelDescriptor | None = None
    position_source: str = "filename"
    pose: NativePose | None = None


@dataclass(frozen=True, slots=True)
class IndexedSeries:
    records: tuple[SliceRecord, ...]
    geometry: SeriesGeometry


def index_series(series_dir: Path) -> IndexedSeries:
    candidates = sorted(series_dir.glob("*.dcm"))
    records = []
    layouts = {}
    for fallback, path in enumerate(candidates):
        if path.stem in KNOWN_BAD_INSTANCES:
            continue
        try:
            header = read_header(path)
            physical = header.physical_position
            position = (
                physical
                if physical is not None
                else header.instance_number
                if header.instance_number is not None
                else float(fallback)
            )
            if math.isfinite(position):
                pose = header.pose
                if pose is not None:
                    pose = NativePose(
                        pose.position, layouts.setdefault(pose.layout, pose.layout)
                    )
                records.append(
                    (
                        path,
                        position,
                        physical,
                        header.descriptor,
                        header.position_source,
                        pose,
                    )
                )
        except PREPARATION_ERRORS:
            continue
    records.sort(key=lambda item: (item[1], item[0].name))
    physical_positions = [item[2] for item in records if item[2] is not None]
    origin = (
        (min(physical_positions) + max(physical_positions)) / 2
        if physical_positions
        else 0
    )
    slices = tuple(
        SliceRecord(
            path,
            position,
            index,
            physical - origin if physical is not None else None,
            descriptor,
            source,
            pose,
        )
        for index, (path, position, physical, descriptor, source, pose) in enumerate(
            records
        )
    )
    return IndexedSeries(slices, summarize_series(slices))
