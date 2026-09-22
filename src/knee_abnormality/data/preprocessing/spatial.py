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


@dataclass(frozen=True)
class ResamplePlan:
    key: tuple[int, int, int, int, int]
    row_start: int
    row_stop: int
    vertical_index: np.ndarray
    vertical_weight: np.ndarray
    vertical_matrix: csr_matrix
    column_start: int
    column_stop: int
    horizontal_index: np.ndarray
    horizontal_weight: np.ndarray
    horizontal_matrix: csr_matrix
    rect: tuple[int, int, int, int]
    output_to_native: np.ndarray


def resample_targets(
    rows: int,
    columns: int,
    row_spacing: float,
    column_spacing: float,
    spacing_mm: float,
) -> tuple[int, int]:
    return (
        max(1, round(rows * float(row_spacing) / spacing_mm)),
        max(1, round(columns * float(column_spacing) / spacing_mm)),
    )


def _output_to_native(
    source_rows: int,
    source_columns: int,
    target_height: int,
    target_width: int,
    image_size: int,
) -> np.ndarray:
    source_top, source_left, output_top, output_left, _, _ = _crop_geometry(
        target_height, target_width, image_size
    )
    scale_x = source_columns / target_width
    scale_y = source_rows / target_height
    return np.array(
        [
            [scale_x, 0, (source_left - output_left + 0.5) * scale_x - 0.5],
            [0, scale_y, (source_top - output_top + 0.5) * scale_y - 0.5],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def _taps(
    matrix: csr_matrix,
) -> tuple[int, int, np.ndarray, np.ndarray, csr_matrix]:
    start = int(matrix.indices.min())
    stop = int(matrix.indices.max()) + 1
    counts = np.diff(matrix.indptr)
    rows = np.repeat(np.arange(matrix.shape[0]), counts)
    positions = np.arange(matrix.nnz) - np.repeat(matrix.indptr[:-1], counts)
    index = np.zeros((matrix.shape[0], int(counts.max())), dtype=np.int64)
    weight = np.zeros(index.shape, dtype=np.float32)
    index[rows, positions] = matrix.indices - start
    weight[rows, positions] = matrix.data
    return start, stop, index, weight, matrix[:, start:stop]


@lru_cache(maxsize=256)
def resample_plan(
    source_rows: int,
    source_columns: int,
    target_height: int,
    target_width: int,
    image_size: int,
) -> ResamplePlan | None:
    source_top, source_left, output_top, output_left, rows, columns = _crop_geometry(
        target_height, target_width, image_size
    )
    vertical = _axis_filter(source_rows, target_height, source_top, rows)
    horizontal = _axis_filter(source_columns, target_width, source_left, columns)
    if vertical.nnz == 0 or horizontal.nnz == 0:
        return None
    return ResamplePlan(
        (source_rows, source_columns, target_height, target_width, image_size),
        *_taps(vertical),
        *_taps(horizontal),
        (output_top, output_left, rows, columns),
        _output_to_native(
            source_rows, source_columns, target_height, target_width, image_size
        ),
    )


def resample_block(block: np.ndarray, plan: ResamplePlan) -> np.ndarray:
    middle = plan.vertical_matrix.dot(block.astype(np.float32))
    return plan.horizontal_matrix.dot(middle.T).T


def resample_crop(
    frame: DecodedFrame, spacing_mm: float, image_size: int
) -> ResampledSlice:
    pixels = frame.pixels
    slope, intercept = frame.slope, frame.intercept
    target_height, target_width = resample_targets(
        *pixels.shape, frame.row_spacing, frame.column_spacing, spacing_mm
    )
    plan = resample_plan(*pixels.shape, target_height, target_width, image_size)
    source_top, source_left, output_top, output_left, rows, columns = _crop_geometry(
        target_height, target_width, image_size
    )
    if plan is None:
        middle = _axis_filter(pixels.shape[0], target_height, source_top, rows).dot(
            pixels.astype(np.float32)
        )
        region = (
            _axis_filter(pixels.shape[1], target_width, source_left, columns)
            .dot(middle.T)
            .T
        )
    else:
        region = resample_block(
            pixels[
                plan.row_start : plan.row_stop, plan.column_start : plan.column_stop
            ],
            plan,
        )
    if slope != 1.0 or intercept != 0.0:
        region *= np.float32(slope)
        region += np.float32(intercept)
    image = np.zeros((image_size, image_size), dtype=np.float32)
    image[
        output_top : output_top + rows,
        output_left : output_left + columns,
    ] = region
    return ResampledSlice(
        image,
        (output_top, output_left, rows, columns),
        _output_to_native(
            pixels.shape[0], pixels.shape[1], target_height, target_width, image_size
        ),
    )
