import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix

from knee_abnormality.data.preprocessing.dicom import DecodedFrame


@dataclass(frozen=True)
class ResampledSlice:
    image: np.ndarray
    rect: tuple[int, int, int, int]
    output_to_native: np.ndarray


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


def resample_crop(
    frame: DecodedFrame, spacing_mm: float, image_size: int
) -> ResampledSlice:
    pixels = frame.pixels
    source_spacing = (frame.row_spacing, frame.column_spacing)
    slope, intercept = frame.slope, frame.intercept
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
    scale_x = pixels.shape[1] / target_width
    scale_y = pixels.shape[0] / target_height
    output_to_native = np.array(
        [
            [scale_x, 0, (source_left - output_left + 0.5) * scale_x - 0.5],
            [0, scale_y, (source_top - output_top + 0.5) * scale_y - 0.5],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return ResampledSlice(
        image, (output_top, output_left, rows, columns), output_to_native
    )
