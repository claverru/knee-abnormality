import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from knee_abnormality.data.preprocessing.dicom import (
    PREPARATION_ERRORS,
    NativePixelDescriptor,
    SeriesTags,
    io_pool,
    parse_prefix,
    read_prefix,
)
from knee_abnormality.data.preprocessing.exclusions import KNOWN_BAD_INSTANCES
from knee_abnormality.data.preprocessing.geometry import (
    RELIABILITY,
    NativePose,
    PoseLayout,
    SeriesGeometry,
    summarize_series,
)
from knee_abnormality.data.sources import list_series

REFORMAT_IMAGE_TYPES = ("DERIVED", "SECONDARY", "PROJECTION")
REFORMAT_DESCRIPTION = re.compile(r"\bmpr\b|reformat")


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
class SeriesSummary:
    series_number: float | None = None
    signature: tuple | None = None
    reformat: bool = False
    acquired_resolution_mm: float | None = None
    receive_coil: str | None = None
    removed_instances: int = 0


@dataclass(frozen=True, slots=True)
class IndexedSeries:
    records: tuple[SliceRecord, ...]
    geometry: SeriesGeometry
    summary: SeriesSummary


def modal_layout(layouts: Counter) -> tuple[PoseLayout | None, set[PoseLayout]]:
    distinct = list(layouts)
    if not distinct:
        return None, set()
    tolerance = RELIABILITY["orientation_tolerance"]
    shapes = np.asarray([(layout.rows, layout.columns) for layout in distinct])
    values = np.asarray(
        [(*layout.spacing, *layout.orientation) for layout in distinct],
        dtype=np.float64,
    )
    compatible = (shapes[:, None] == shapes[None]).all(-1) & (
        np.abs(values[:, None] - values[None]) <= tolerance
    ).all(-1)
    support = compatible @ np.asarray([layouts[layout] for layout in distinct])
    best = max(
        range(len(distinct)),
        key=lambda index: (support[index], layouts[distinct[index]]),
    )
    if not support[best]:
        return None, set(distinct)
    return distinct[best], {
        layout for layout, keep in zip(distinct, compatible[best]) if keep
    }


def _acquired_resolution(
    matrix: tuple[float, ...], layout: PoseLayout | None
) -> float | None:
    if layout is None or len(matrix) != 4:
        return None
    horizontal = [value for value in (matrix[0], matrix[2]) if value > 0]
    vertical = [value for value in (matrix[1], matrix[3]) if value > 0]
    if len(horizontal) != 1 or len(vertical) != 1:
        return None
    value = max(
        layout.columns * layout.spacing[1] / horizontal[0],
        layout.rows * layout.spacing[0] / vertical[0],
    )
    return value if math.isfinite(value) and value > 0 else None


def summarize_tags(
    tags: SeriesTags | None, layout: PoseLayout | None, removed: int
) -> SeriesSummary:
    if tags is None:
        return SeriesSummary(removed_instances=removed)
    signature = (
        (
            round(tags.repetition_time),
            round(tags.echo_time),
            tags.echo_train_length,
            tags.flip_angle,
            tags.acquisition_type,
            tags.sequence_variant,
            tags.scan_options,
            layout.rows if layout is not None else None,
            layout.columns if layout is not None else None,
        )
        if tags.repetition_time is not None and tags.echo_time is not None
        else None
    )
    reformat = any(
        word in value for value in tags.image_type for word in REFORMAT_IMAGE_TYPES
    ) or bool(REFORMAT_DESCRIPTION.search((tags.description or "").lower()))
    return SeriesSummary(
        tags.series_number,
        signature,
        reformat,
        _acquired_resolution(tags.acquisition_matrix, layout),
        tags.receive_coil,
        removed,
    )


def _safe_prefix(path: Path) -> tuple[bytes, os.stat_result] | None:
    try:
        return read_prefix(path)
    except PREPARATION_ERRORS:
        return None


def index_series(
    series_dir: Path,
    threads: int = 1,
    candidates: list[Path] | None = None,
) -> IndexedSeries:
    if candidates is None:
        candidates = list_series(series_dir)
    prefixes: dict[Path, tuple[bytes, os.stat_result]] = {}
    if threads > 1 and len(candidates) > 1:
        for path, prefix in zip(
            candidates, io_pool(threads).map(_safe_prefix, candidates), strict=True
        ):
            if prefix is not None:
                prefixes[path] = prefix
    records = []
    layouts = {}
    for fallback, path in enumerate(candidates):
        if path.stem in KNOWN_BAD_INSTANCES:
            continue
        try:
            header = parse_prefix(path, prefixes.get(path))
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
                        header.tags,
                    )
                )
        except PREPARATION_ERRORS:
            continue
    reference, compatible = modal_layout(
        Counter(item[5].layout for item in records if item[5] is not None)
    )
    retained = [
        item for item in records if item[5] is None or item[5].layout in compatible
    ]
    removed = len(records) - len(retained)
    retained.sort(key=lambda item: (item[1], item[0].name))
    tags = Counter(item[6] for item in retained).most_common(1)
    physical_positions = [item[2] for item in retained if item[2] is not None]
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
        for index, (path, position, physical, descriptor, source, pose, _) in enumerate(
            retained
        )
    )
    return IndexedSeries(
        slices,
        summarize_series(slices),
        summarize_tags(tags[0][0] if tags else None, reference, removed),
    )
