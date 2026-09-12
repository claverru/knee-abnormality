import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from knee_abnormality.preprocessing.dicom import PREPARATION_ERRORS, read_frame
from knee_abnormality.preprocessing.intensity import (
    _linear_quantiles,
    _normalize_pixels,
)
from knee_abnormality.preprocessing.series import SliceRecord
from knee_abnormality.preprocessing.spatial import ResampledSlice, resample_crop


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
