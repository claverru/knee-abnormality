import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pydicom
import torch
from torch.nn import functional as F

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


def _resample_crop(
    path: Path, spacing_mm: float, image_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = pydicom.dcmread(path)
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"Expected one two-dimensional frame, got {pixels.shape}")
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * np.float32(slope) + np.float32(intercept)
    source_spacing = getattr(dataset, "PixelSpacing", None)
    if source_spacing is None or len(source_spacing) != 2:
        raise ValueError("Missing two-dimensional PixelSpacing")
    target_height = max(
        1, round(pixels.shape[0] * float(source_spacing[0]) / spacing_mm)
    )
    target_width = max(
        1, round(pixels.shape[1] * float(source_spacing[1]) / spacing_mm)
    )
    image = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)
    image = F.interpolate(
        image,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0, 0]
    valid = torch.ones_like(image, dtype=torch.bool)
    pad_height = max(0, image_size - target_height)
    pad_width = max(0, image_size - target_width)
    if pad_height or pad_width:
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        image = F.pad(image, (left, right, top, bottom))
        valid = F.pad(valid, (left, right, top, bottom))
    top = max(0, (image.shape[0] - image_size) // 2)
    left = max(0, (image.shape[1] - image_size) // 2)
    return image[top : top + image_size, left : left + image_size], valid[
        top : top + image_size, left : left + image_size
    ]


def linear_quantiles(values: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    ranks = quantiles * (values.numel() - 1)
    lower = ranks.floor().to(torch.int64)
    upper = ranks.ceil().to(torch.int64)
    partitioned = values.numpy()
    partitioned.partition(torch.cat((lower, upper)).unique().numpy())
    return torch.lerp(values[lower], values[upper], ranks - lower)


DecodedSlice = tuple[torch.Tensor, torch.Tensor] | str
SeriesDecoder = Callable[
    [tuple[SliceRecord, ...], float, int], tuple[DecodedSlice, ...]
]


def decode_slice(path: Path, spacing_mm: float, image_size: int) -> DecodedSlice:
    try:
        image, valid = _resample_crop(path, spacing_mm, image_size)
    except PREPARATION_ERRORS as error:
        return f"{path}: {type(error).__name__}: {error}"
    if not torch.isfinite(image).all():
        return f"{path}: non-finite pixels"
    return image, valid


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
    valid_masks = []
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
            image, valid = result
            group_images.append(image)
            group_masks.append(valid)
        if len(group_images) != group_size:
            continue
        images.extend(group_images)
        valid_masks.extend(group_masks)
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
    stacked = torch.stack(images).to(torch.float32)
    valid = torch.stack(valid_masks)
    values = stacked[valid]
    bounds = linear_quantiles(values, torch.tensor([lower_quantile, upper_quantile]))
    low, high = bounds.unbind()
    constant = (
        not math.isfinite(float(low))
        or not math.isfinite(float(high))
        or bool(high <= low)
    )
    if constant:
        output = torch.zeros_like(stacked, dtype=torch.uint8)
    else:
        output = (
            ((stacked.clamp(low, high) - low) * (255.0 / (high - low)))
            .round()
            .to(torch.uint8)
        )
        output.masked_fill_(~valid, 0)
    if group_size == 3:
        output = output.reshape(-1, 3, image_size, image_size)
    return PreparedSeries(
        output, torch.tensor(positions, dtype=torch.float32), tuple(failures), constant
    )
