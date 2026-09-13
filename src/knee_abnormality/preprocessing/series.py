import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from knee_abnormality.preprocessing.dicom import (
    PREPARATION_ERRORS,
    NativePixelDescriptor,
    read_header,
)
from knee_abnormality.preprocessing.geometry import (
    NativePose,
    SeriesGeometry,
    summarize_series,
)

KNOWN_BAD_INSTANCES = {
    "1.2.826.0.1.3680043.8.498.66460523372602171398246641420761543626",
    "1.2.826.0.1.3680043.8.498.41947538458330624797291612781685746320",
}


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


def index_series(
    series_dir: Path, position_reference: Literal["origin", "center"] = "origin"
) -> IndexedSeries:
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
    origin = next((item[2] for item in records if item[2] is not None), 0)
    if position_reference == "center":
        physical_positions = [item[2] for item in records if item[2] is not None]
        if physical_positions:
            origin = (min(physical_positions) + max(physical_positions)) / 2
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


def _center_weights(
    records: tuple[SliceRecord, ...],
    group_size: int,
    center_weight: float,
) -> np.ndarray:
    if all(record.relative_position_mm is not None for record in records):
        positions = np.asarray(
            [record.relative_position_mm for record in records], dtype=np.float64
        )
    else:
        positions = np.asarray(
            [record.original_index for record in records], dtype=np.float64
        )
    middle = (positions[0] + positions[-1]) / 2
    half_span = (positions[-1] - positions[0]) / 2
    centers = positions[group_size // 2 : len(records) - group_size // 2]
    distance = (
        np.abs(centers - middle) / half_span
        if half_span > 0
        else np.zeros_like(centers)
    )
    return 1 + (center_weight - 1) * (1 - distance.clip(0, 1))


def _weighted_random_starts(
    weights: np.ndarray, groups: int, group_size: int, generator: torch.Generator
) -> list[int]:
    count = len(weights) + group_size - 1
    log_weights = np.log(weights)
    partitions = np.full((groups + 1, count + 1), -np.inf)
    partitions[0] = 0
    for remaining in range(1, groups + 1):
        for start in range(count - remaining * group_size, -1, -1):
            partitions[remaining, start] = np.logaddexp(
                partitions[remaining, start + 1],
                log_weights[start] + partitions[remaining - 1, start + group_size],
            )
    draws = torch.rand(len(weights), generator=generator).numpy()
    starts = []
    start = 0
    remaining = groups
    while remaining:
        probability = np.exp(
            log_weights[start]
            + partitions[remaining - 1, start + group_size]
            - partitions[remaining, start]
        )
        if draws[start] < probability:
            starts.append(start)
            remaining -= 1
            start += group_size
        else:
            start += 1
    return starts


def _weighted_validation_starts(
    weights: np.ndarray, groups: int, group_size: int
) -> list[int]:
    coordinates = np.arange(len(weights))
    cumulative = (weights.cumsum() - 0.5 * weights) / weights.sum()
    desired = np.interp((np.arange(groups) + 0.5) / groups, cumulative, coordinates)
    costs = (coordinates[None] - desired[:, None]) ** 2
    previous = np.full(costs.shape, -1, dtype=np.int64)
    for group in range(1, groups):
        best_cost = np.inf
        best_start = -1
        for start in range(len(weights)):
            before = start - group_size
            if before >= 0 and costs[group - 1, before] < best_cost:
                best_cost = costs[group - 1, before]
                best_start = before
            costs[group, start] += best_cost
            previous[group, start] = best_start
    start = int(costs[-1].argmin())
    starts = [start]
    for group in range(groups - 1, 0, -1):
        start = int(previous[group, start])
        starts.append(start)
    return starts[::-1]


def select_slices(
    records: tuple[SliceRecord, ...],
    limit: int,
    training: bool,
    generator: torch.Generator,
    group_size: Literal[1, 3] = 1,
    sampling: Literal["uniform", "center_weighted"] = "uniform",
    center_weight: float = 3.0,
) -> tuple[SliceRecord, ...]:
    count = len(records)
    if group_size not in {1, 3} or limit < 1 or limit % group_size:
        raise ValueError(
            "Slice limits must be positive multiples of the input group size"
        )
    if sampling == "center_weighted":
        if not math.isfinite(center_weight) or center_weight < 1:
            raise ValueError(
                "The center sampling weight must be finite and at least one"
            )
        groups = min(count, limit) // group_size
        if groups == 0:
            return ()
        if count == groups * group_size:
            return records
        weights = _center_weights(records, group_size, center_weight)
        starts = (
            _weighted_random_starts(weights, groups, group_size, generator)
            if training
            else _weighted_validation_starts(weights, groups, group_size)
        )
        return tuple(
            records[start + offset] for start in starts for offset in range(group_size)
        )
    if sampling != "uniform":
        raise ValueError(f"Unsupported slice sampling: {sampling}")
    if group_size == 3:
        groups = min(count, limit) // 3
        if groups == 0:
            return ()
        compressed_count = count - 2 * groups
        if compressed_count == groups:
            selected = torch.arange(groups)
        elif training:
            selected = (
                torch.randperm(compressed_count, generator=generator)[:groups]
                .sort()
                .values
            )
        elif groups == 1:
            selected = torch.tensor([compressed_count // 2])
        else:
            selected = (
                torch.linspace(0, compressed_count - 1, groups).round().to(torch.int64)
            )
        starts = selected + 2 * torch.arange(groups)
        return tuple(
            records[start + offset] for start in starts.tolist() for offset in range(3)
        )
    if count <= limit:
        return records
    if training:
        selected = (
            torch.randperm(count, generator=generator)[:limit].sort().values.tolist()
        )
    elif limit == 1:
        selected = [count // 2]
    else:
        selected = torch.linspace(0, count - 1, limit).round().to(torch.int64).tolist()
    return tuple(records[index] for index in selected)
