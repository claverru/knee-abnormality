import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pydicom
import torch
from scipy.sparse import csr_matrix

KNOWN_BAD_INSTANCES = {
    "1.2.826.0.1.3680043.8.498.66460523372602171398246641420761543626",
    "1.2.826.0.1.3680043.8.498.41947538458330624797291612781685746320",
}
PREPARATION_ERRORS = (
    AttributeError,
    EOFError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class SliceRecord:
    path: Path
    physical_position: float
    original_index: int
    relative_position_mm: float | None = None


@dataclass(frozen=True)
class PreparedSeries:
    images: torch.Tensor
    positions: torch.Tensor
    failures: tuple[str, ...]
    constant: bool
    uid: str = ""

    @property
    def slice_count(self) -> int:
        return len(self.images) * (self.images.shape[1] if self.images.ndim == 4 else 1)


@dataclass(frozen=True)
class _ArraySlice:
    image: np.ndarray
    rect: tuple[int, int, int, int]


def _physical_position(dataset: pydicom.Dataset) -> float | None:
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    location = getattr(dataset, "ImagePositionPatient", None)
    if (
        orientation is not None
        and location is not None
        and len(orientation) == 6
        and len(location) == 3
    ):
        row = np.asarray(orientation[:3], dtype=np.float64)
        column = np.asarray(orientation[3:], dtype=np.float64)
        normal = np.cross(row, column)
        position = float(np.dot(np.asarray(location, dtype=np.float64), normal))
        if math.isfinite(position) and np.linalg.norm(normal) > 0:
            return position
    if hasattr(dataset, "SliceLocation"):
        position = float(dataset.SliceLocation)
        if math.isfinite(position):
            return position
    return None


def index_series(
    series_dir: Path, position_reference: Literal["origin", "center"] = "origin"
) -> tuple[SliceRecord, ...]:
    candidates = sorted(series_dir.glob("*.dcm"))
    records = []
    for fallback, path in enumerate(candidates):
        if path.stem in KNOWN_BAD_INSTANCES:
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
            physical = _physical_position(dataset)
            position = (
                physical
                if physical is not None
                else float(getattr(dataset, "InstanceNumber", fallback))
            )
            if math.isfinite(position):
                records.append((path, position, physical))
        except (
            AttributeError,
            EOFError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            continue
    records.sort(key=lambda item: (item[1], item[0].name))
    origin = next((physical for _, _, physical in records if physical is not None), 0)
    if position_reference == "center":
        physical_positions = [
            physical for _, _, physical in records if physical is not None
        ]
        if physical_positions:
            origin = (min(physical_positions) + max(physical_positions)) / 2
    return tuple(
        SliceRecord(
            path,
            position,
            index,
            physical - origin if physical is not None else None,
        )
        for index, (path, position, physical) in enumerate(records)
    )


def _center_weights(
    records: tuple[SliceRecord, ...], group_size: int, center_weight: float
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


def _crop_geometry(
    target_height: int, target_width: int, image_size: int
) -> tuple[int, int, int, int, int, int]:
    source_top = max(0, (target_height - image_size) // 2)
    source_left = max(0, (target_width - image_size) // 2)
    output_top = max(0, (image_size - target_height) // 2)
    output_left = max(0, (image_size - target_width) // 2)
    rows = min(target_height, image_size)
    columns = min(target_width, image_size)
    return source_top, source_left, output_top, output_left, rows, columns


@lru_cache(maxsize=128)
def _axis_filter(source: int, target: int, first: int, count: int) -> csr_matrix:
    scale = source / target
    support = max(1.0, scale)
    rows = []
    columns = []
    values = []
    for output_index in range(first, first + count):
        center = (output_index + 0.5) * scale - 0.5
        start = math.floor(center - support)
        end = math.ceil(center + support)
        weights = {}
        for index in range(start, end + 1):
            weight = max(0.0, 1.0 - abs(center - index) / support)
            if weight == 0 or index < 0 or index >= source:
                continue
            weights[index] = weight
        total = sum(weights.values())
        for index, weight in weights.items():
            rows.append(output_index - first)
            columns.append(index)
            values.append(weight / total)
    return csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (rows, columns),
        ),
        shape=(count, source),
        dtype=np.float32,
    )


def _resample_crop(path: Path, spacing_mm: float, image_size: int) -> _ArraySlice:
    dataset = pydicom.dcmread(path)
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"Expected one two-dimensional frame, got {pixels.shape}")
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    source_spacing = getattr(dataset, "PixelSpacing", None)
    if source_spacing is None or len(source_spacing) != 2:
        raise ValueError("Missing two-dimensional PixelSpacing")
    target_height = max(
        1, round(pixels.shape[0] * float(source_spacing[0]) / spacing_mm)
    )
    target_width = max(
        1, round(pixels.shape[1] * float(source_spacing[1]) / spacing_mm)
    )
    source_top, source_left, output_top, output_left, rows, columns = _crop_geometry(
        target_height, target_width, image_size
    )
    vertical = _axis_filter(pixels.shape[0], target_height, source_top, rows)
    horizontal = _axis_filter(pixels.shape[1], target_width, source_left, columns)
    middle = vertical.dot(pixels)
    region = horizontal.dot(middle.T).T
    if slope != 1.0 or intercept != 0.0:
        region *= np.float32(slope)
        region += np.float32(intercept)
    image = np.zeros((image_size, image_size), dtype=np.float32)
    image[
        output_top : output_top + rows,
        output_left : output_left + columns,
    ] = region
    return _ArraySlice(image, (output_top, output_left, rows, columns))


def _linear_quantiles(
    values: np.ndarray, lower_quantile: float, upper_quantile: float
) -> tuple[float, float]:
    quantiles = np.asarray([lower_quantile, upper_quantile], dtype=np.float32)
    ranks = quantiles * np.float32(values.size - 1)
    lower = np.floor(ranks).astype(np.int64)
    upper = np.ceil(ranks).astype(np.int64)
    values.partition(np.unique(np.concatenate((lower, upper))))
    lower_values = values[lower]
    upper_values = values[upper]
    bounds = lower_values + (upper_values - lower_values) * (
        ranks - lower
    ).astype(np.float32)
    return float(bounds[0]), float(bounds[1])


def _normalize_pixels(
    pixels: np.ndarray, rects: np.ndarray, low: float, high: float
) -> np.ndarray:
    low_value = np.float32(low)
    high_value = np.float32(high)
    if (
        not np.isfinite(low_value)
        or not np.isfinite(high_value)
        or high_value <= low_value
    ):
        return np.zeros(pixels.shape, dtype=np.uint8)
    gain = (np.float32(1.0) / (high_value - low_value)) * np.float32(255.0)
    np.clip(pixels, low_value, high_value, out=pixels)
    np.subtract(pixels, low_value, out=pixels)
    np.multiply(pixels, gain, out=pixels)
    np.rint(pixels, out=pixels)
    np.clip(pixels, np.float32(0.0), np.float32(255.0), out=pixels)
    output = pixels.astype(np.uint8)
    for index, (top, left, rows, columns) in enumerate(rects):
        output[index, :top] = 0
        output[index, top + rows :] = 0
        output[index, top : top + rows, :left] = 0
        output[index, top : top + rows, left + columns :] = 0
    return output


DecodedSlice = _ArraySlice | str
SeriesDecoder = Callable[
    [tuple[SliceRecord, ...], float, int], tuple[DecodedSlice, ...]
]


def decode_slice(path: Path, spacing_mm: float, image_size: int) -> DecodedSlice:
    try:
        value = _resample_crop(path, spacing_mm, image_size)
    except PREPARATION_ERRORS as error:
        return f"{path}: {type(error).__name__}: {error}"
    top, left, rows, columns = value.rect
    region = value.image[top : top + rows, left : left + columns]
    if not np.isfinite(region).all():
        return f"{path}: non-finite pixels"
    return value


def decode_series(
    records: tuple[SliceRecord, ...], spacing_mm: float, image_size: int
) -> tuple[DecodedSlice, ...]:
    return tuple(
        decode_slice(record.path, spacing_mm, image_size) for record in records
    )


def prepare_series(
    records: tuple[SliceRecord, ...],
    spacing_mm: float,
    image_size: int,
    lower_quantile: float,
    upper_quantile: float,
    position_encoding: Literal["index", "physical_mm"] = "index",
    group_size: Literal[1, 3] = 1,
    decoder: SeriesDecoder = decode_series,
) -> PreparedSeries:
    if group_size not in {1, 3} or len(records) % group_size:
        raise ValueError("Preparation requires complete input groups")
    if group_size == 3:
        if len({record.path for record in records}) != len(records):
            raise ValueError("Triplets cannot reuse slices")
        if any(
            records[start + offset].original_index
            != records[start].original_index + offset
            for start in range(0, len(records), 3)
            for offset in (1, 2)
        ):
            raise ValueError("Triplet slices must be consecutive in the indexed series")
    if position_encoding not in {"index", "physical_mm"}:
        raise ValueError(f"Unsupported position encoding: {position_encoding}")
    if position_encoding == "physical_mm" and any(
        record.relative_position_mm is None for record in records
    ):
        raise ValueError(
            "Physical position encoding requires DICOM geometry or SliceLocation"
        )
    decoded = decoder(records, spacing_mm, image_size)
    if len(decoded) != len(records):
        raise ValueError("Series decoder must return one result per selected slice")
    images = []
    rectangles = []
    positions = []
    failures = []
    for start in range(0, len(records), group_size):
        group_records = records[start : start + group_size]
        group = decoded[start : start + group_size]
        group_images = []
        group_masks = []
        for result in group:
            if isinstance(result, str):
                failures.append(result)
                continue
            group_images.append(result.image)
            group_masks.append(result.rect)
        if len(group_images) != group_size:
            continue
        images.extend(group_images)
        rectangles.extend(group_masks)
        center = group_records[group_size // 2]
        positions.append(
            center.relative_position_mm
            if position_encoding == "physical_mm"
            else center.original_index
        )
    if not images:
        if group_size == 3:
            raise ValueError("No complete selected triplet could be decoded")
        raise ValueError("No selected image could be decoded")
    pixels = np.empty((len(images), image_size, image_size), dtype=np.float32)
    rects = np.asarray(rectangles, dtype=np.int32)
    scratch = np.empty(
        sum(rows * columns for _, _, rows, columns in rectangles),
        dtype=np.float32,
    )
    offset = 0
    for index, (image, (top, left, rows, columns)) in enumerate(
        zip(images, rectangles, strict=True)
    ):
        pixels[index] = image
        size = rows * columns
        scratch[offset : offset + size] = image[
            top : top + rows, left : left + columns
        ].reshape(-1)
        offset += size
    low, high = _linear_quantiles(scratch, lower_quantile, upper_quantile)
    constant = not math.isfinite(low) or not math.isfinite(high) or high <= low
    output = torch.from_numpy(_normalize_pixels(pixels, rects, low, high))
    if group_size == 3:
        output = output.reshape(-1, 3, image_size, image_size)
    return PreparedSeries(
        output, torch.tensor(positions, dtype=torch.float32), tuple(failures), constant
    )
