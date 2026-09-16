import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from knee_abnormality.data.preprocessing.dicom import PREPARATION_ERRORS, read_frame
from knee_abnormality.data.preprocessing.geometry import RELIABILITY
from knee_abnormality.data.preprocessing.intensity import (
    _linear_quantiles,
    _normalize_pixels,
)
from knee_abnormality.data.preprocessing.series import SliceRecord
from knee_abnormality.data.preprocessing.spatial import ResampledSlice, resample_crop


@dataclass(frozen=True)
class PreparedSeries:
    images: torch.Tensor
    positions: torch.Tensor
    failures: tuple[str, ...]
    constant: bool
    patient_maps: torch.Tensor
    valid_rects: torch.Tensor
    geometry_valid: torch.Tensor
    selected_paths: tuple[tuple[str, ...], ...]
    uid: str = ""
    failed_paths: tuple[Path, ...] = ()
    unique_slice_count: int = 0
    intensity_pixel_count: int = 0
    intensity_bounds: tuple[float, float] = (0.0, 0.0)
    sampling_path: str = "physical"
    reflection_eligible: bool = False
    reflection_reason: str = "unavailable"

    @property
    def slice_count(self) -> int:
        return self.images.shape[0] * self.images.shape[1]


DecodedSlice = ResampledSlice | str
SeriesDecoder = Callable[
    [tuple[SliceRecord, ...], float, int], tuple[DecodedSlice, ...]
]


def decode_slice(
    record: SliceRecord, spacing_mm: float, image_size: int
) -> DecodedSlice:
    path = record.path
    try:
        value = resample_crop(
            read_frame(path, record.descriptor), spacing_mm, image_size
        )
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
    return tuple(decode_slice(record, spacing_mm, image_size) for record in records)


def prepare_series(
    records: tuple[SliceRecord, ...],
    spacing_mm: float,
    image_size: int,
    lower_quantile: float,
    upper_quantile: float,
    decoder: SeriesDecoder = decode_series,
    cache: dict[Path, DecodedSlice] | None = None,
) -> PreparedSeries:
    group_size = 3
    if len(records) % 3:
        raise ValueError("Preparation requires complete triplets")
    if (
        len({records[i].original_index for i in range(1, len(records), 3)})
        != len(records) // 3
    ):
        raise ValueError("Selected triplet middles must be distinct")
    if any(
        len({r.path for r in records[i : i + 3]}) != 3
        for i in range(0, len(records), 3)
    ):
        raise ValueError("Each triplet requires three distinct slices")
    if any(
        records[start + offset].original_index != records[start].original_index + offset
        for start in range(0, len(records), 3)
        for offset in (1, 2)
    ):
        raise ValueError("Triplet slices must be consecutive in the indexed series")
    if any(record.relative_position_mm is None for record in records):
        raise ValueError(
            "Physical position encoding requires DICOM geometry or SliceLocation"
        )
    cache = {} if cache is None else cache
    unique = tuple(
        sorted(
            {r.path: r for r in records}.values(),
            key=lambda r: (r.original_index, str(r.path)),
        )
    )
    missing = tuple(r for r in unique if r.path not in cache)
    decoded = decoder(missing, spacing_mm, image_size)
    if len(decoded) != len(missing):
        raise ValueError("Series decoder must return one result per unique slice")
    cache.update((r.path, value) for r, value in zip(missing, decoded, strict=True))
    failed_paths = tuple(r.path for r in unique if isinstance(cache[r.path], str))
    failures = tuple(cache[path] for path in failed_paths)
    groups = tuple(
        records[i : i + group_size]
        for i in range(0, len(records), group_size)
        if all(not isinstance(cache[r.path], str) for r in records[i : i + group_size])
    )
    retained = tuple(
        sorted(
            {r.path: r for group in groups for r in group}.values(),
            key=lambda r: (r.original_index, str(r.path)),
        )
    )
    if not retained:
        return PreparedSeries(
            torch.empty((0, group_size, image_size, image_size), dtype=torch.uint8),
            torch.empty(0),
            failures,
            False,
            torch.empty((0, group_size, 3, 3)),
            torch.empty((0, group_size, 4)),
            torch.empty(0, dtype=torch.bool),
            (),
            failed_paths=failed_paths,
            reflection_reason="empty_prepared_series",
        )
    lookup = {r.path: i for i, r in enumerate(retained)}
    successful_records = [r for group in groups for r in group]
    gather = np.asarray([lookup[r.path] for r in successful_records])
    values = [cache[r.path] for r in retained]
    rectangles = [v.rect for v in values]
    pixels = np.stack([v.image for v in values]).astype(np.float32)
    rects = np.asarray(rectangles, dtype=np.int32)
    scratch = np.concatenate(
        [
            v.image[top : top + rows, left : left + columns].reshape(-1)
            for v, (top, left, rows, columns) in zip(values, rectangles, strict=True)
        ]
    )
    low, high = _linear_quantiles(scratch, lower_quantile, upper_quantile)
    constant = not math.isfinite(low) or not math.isfinite(high) or high <= low
    normalized = _normalize_pixels(pixels, rects, low, high)
    output = torch.from_numpy(normalized[gather])
    output = output.reshape(-1, 3, image_size, image_size)
    rects = rects[gather]
    transforms = [cache[r.path].output_to_native for r in successful_records]
    selected_paths = tuple(tuple(str(r.path) for r in group) for group in groups)
    positions = [group[1].relative_position_mm for group in groups]
    pose_values = np.asarray(
        [
            (
                *record.pose.position,
                *record.pose.layout.orientation,
                *record.pose.layout.spacing,
                record.pose.layout.rows,
                record.pose.layout.columns,
            )
            if record.pose is not None
            else (0.0,) * 13
            for record in successful_records
        ],
        dtype=np.float64,
    )
    axes = pose_values[:, 3:9].reshape(-1, 2, 3)
    tolerance = RELIABILITY["orientation_tolerance"]
    valid = (
        np.isfinite(pose_values).all(1)
        & (pose_values[:, 9:11] > 0).all(1)
        & (pose_values[:, 11:] >= 1).all(1)
        & (np.abs(np.linalg.norm(axes, axis=-1) - 1) <= tolerance).all(1)
        & (np.abs((axes[:, 0] * axes[:, 1]).sum(-1)) <= tolerance)
    )
    pose_values[~valid] = 0
    native_maps = np.stack(
        (
            axes[:, 0] * pose_values[:, 10:11],
            axes[:, 1] * pose_values[:, 9:10],
            pose_values[:, :3],
        ),
        axis=-1,
    )
    patient_maps = (native_maps @ np.asarray(transforms)).reshape(-1, group_size, 3, 3)
    geometry_valid = valid.reshape(-1, group_size).all(1)
    return PreparedSeries(
        output,
        torch.tensor(positions, dtype=torch.float32),
        tuple(failures),
        constant,
        torch.tensor(np.asarray(patient_maps), dtype=torch.float32),
        torch.tensor(rects.reshape(-1, group_size, 4), dtype=torch.float32),
        torch.tensor(geometry_valid, dtype=torch.bool),
        tuple(selected_paths),
        failed_paths=failed_paths,
        unique_slice_count=len(retained),
        intensity_pixel_count=len(scratch),
        intensity_bounds=(float(low), float(high)),
    )
