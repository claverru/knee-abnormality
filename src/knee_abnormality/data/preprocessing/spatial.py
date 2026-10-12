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
    target_height: int,
    target_width: int,
    image_size: int,
    offset_y: int = 0,
    offset_x: int = 0,
    margin: int = 0,
) -> tuple[int, int, int, int, int, int]:
    origins = (
        min(max(0, (extent - image_size) // 2 + offset), max(0, extent - image_size))
        - max(0, (image_size - extent) // 2)
        - margin
        for extent, offset in ((target_height, offset_y), (target_width, offset_x))
    )
    origin_y, origin_x = origins
    size = image_size + 2 * margin
    source_top, source_left = max(0, origin_y), max(0, origin_x)
    output_top, output_left = max(0, -origin_y), max(0, -origin_x)
    rows = min(target_height - source_top, size - output_top)
    columns = min(target_width - source_left, size - output_left)
    return source_top, source_left, output_top, output_left, rows, columns


@lru_cache(maxsize=1024)
def _full_axis_filter(source: int, target: int) -> csr_matrix:
    scale = source / target
    support = max(1.0, scale)
    rows = []
    columns = []
    values = []
    for output_index in range(target):
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
            rows.append(output_index)
            columns.append(index)
            values.append(weight / total)
    return csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (rows, columns),
        ),
        shape=(target, source),
        dtype=np.float32,
    )


@lru_cache(maxsize=128)
def _axis_filter(source: int, target: int, first: int, count: int) -> csr_matrix:
    return _full_axis_filter(source, target)[first : first + count]


@dataclass(frozen=True)
class ResamplePlan:
    key: tuple[int, ...]
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
    offset_y: int = 0,
    offset_x: int = 0,
    margin: int = 0,
) -> np.ndarray:
    source_top, source_left, output_top, output_left, _, _ = _crop_geometry(
        target_height, target_width, image_size, offset_y, offset_x, margin
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
    offset_y: int = 0,
    offset_x: int = 0,
    margin: int = 0,
) -> ResamplePlan | None:
    source_top, source_left, output_top, output_left, rows, columns = _crop_geometry(
        target_height, target_width, image_size, offset_y, offset_x, margin
    )
    vertical = _axis_filter(source_rows, target_height, source_top, rows)
    horizontal = _axis_filter(source_columns, target_width, source_left, columns)
    if vertical.nnz == 0 or horizontal.nnz == 0:
        return None
    return ResamplePlan(
        (
            source_rows,
            source_columns,
            target_height,
            target_width,
            image_size,
            offset_y,
            offset_x,
            margin,
        ),
        *_taps(vertical),
        *_taps(horizontal),
        (output_top, output_left, rows, columns),
        _output_to_native(
            source_rows,
            source_columns,
            target_height,
            target_width,
            image_size,
            offset_y,
            offset_x,
            margin,
        ),
    )


def resample_block(block: np.ndarray, plan: ResamplePlan) -> np.ndarray:
    middle = plan.vertical_matrix.dot(block.astype(np.float32))
    return plan.horizontal_matrix.dot(middle.T).T


@lru_cache(maxsize=512)
def _window_filters(
    key: tuple[int, ...], rows: tuple[int, int], columns: tuple[int, int]
) -> tuple[csr_matrix, csr_matrix, int, int, int, int]:
    plan = resample_plan(*key)
    vertical = plan.vertical_matrix[rows[0] : rows[1]]
    horizontal = plan.horizontal_matrix[columns[0] : columns[1]]
    first_row = int(vertical.indices.min())
    last_row = int(vertical.indices.max()) + 1
    first_column = int(horizontal.indices.min())
    last_column = int(horizontal.indices.max()) + 1
    return (
        vertical[:, first_row:last_row],
        horizontal[:, first_column:last_column],
        first_row,
        last_row,
        first_column,
        last_column,
    )


def resample_window(
    block: np.ndarray,
    plan: ResamplePlan,
    rows: tuple[int, int],
    columns: tuple[int, int],
) -> np.ndarray:
    vertical, horizontal, first_row, last_row, first_column, last_column = (
        _window_filters(plan.key, rows, columns)
    )
    middle = vertical.dot(
        block[first_row:last_row, first_column:last_column].astype(np.float32)
    )
    return horizontal.dot(middle.T).T


def resample_crop(
    frame: DecodedFrame,
    spacing_mm: float,
    image_size: int,
    offset_y: int = 0,
    offset_x: int = 0,
    margin: int = 0,
) -> ResampledSlice:
    pixels = frame.pixels
    slope, intercept = frame.slope, frame.intercept
    target_height, target_width = resample_targets(
        *pixels.shape, frame.row_spacing, frame.column_spacing, spacing_mm
    )
    plan = resample_plan(
        *pixels.shape,
        target_height,
        target_width,
        image_size,
        offset_y,
        offset_x,
        margin,
    )
    source_top, source_left, output_top, output_left, rows, columns = _crop_geometry(
        target_height, target_width, image_size, offset_y, offset_x, margin
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
    size = image_size + 2 * margin
    image = np.zeros((size, size), dtype=np.float32)
    image[
        output_top : output_top + rows,
        output_left : output_left + columns,
    ] = region
    return ResampledSlice(
        image,
        (output_top, output_left, rows, columns),
        _output_to_native(
            pixels.shape[0],
            pixels.shape[1],
            target_height,
            target_width,
            image_size,
            offset_y,
            offset_x,
            margin,
        ),
    )


PLANE_AXES = {"Axial": (0, 1), "Coronal": (0, 2), "Sagittal": (1, 2)}


def record_layout(record):
    if record.descriptor is not None:
        value = record.descriptor
        return value.rows, value.columns, value.row_spacing, value.column_spacing
    if record.pose is not None:
        value = record.pose.layout
        return value.rows, value.columns, *value.spacing
    return None


def record_transform(record, spacing, size, offset_y=0, offset_x=0):
    rows, columns, row_spacing, column_spacing = record_layout(record)
    return _output_to_native(
        rows,
        columns,
        *resample_targets(rows, columns, row_spacing, column_spacing, spacing),
        size,
        offset_y,
        offset_x,
    )


def image_rule(view):
    return tuple(
        axis.offset_mm if axis.anchor == "image" else 0.0
        for axis in (view.horizontal, view.vertical)
    )


def view_position(records, plane, view, side, knee, trusted, frame_spacing, size):
    settings = (view.horizontal, view.vertical)
    spacing = view.field_mm / size
    available = [record for record in records if record_layout(record) is not None]
    if not available:
        offset_x, offset_y = (round(offset / spacing) for offset in image_rule(view))
        return offset_y, offset_x, None, 0.0
    record = available[len(available) // 2]
    rows, columns, row_spacing, column_spacing = record_layout(record)
    detail = record_transform(record, spacing, size)
    output_center = np.array([(size - 1) / 2, (size - 1) / 2, 1])
    native_center = detail @ output_center
    middle = ((columns - 1) / 2, (rows - 1) / 2)
    pitch = (column_spacing, row_spacing)
    directions = (
        -1.0 if plane == "Sagittal" else side,
        1.0 if plane == "Axial" else -1.0,
    )
    located = trusted and record.pose is not None
    anchored = [axis for axis in (0, 1) if settings[axis].anchor == "knee"]
    native = [native_center[0], native_center[1]]
    rule = [0.0, 0.0]
    requested = [0, 0]
    for axis in (
        *(axis for axis in (0, 1) if axis not in anchored),
        *anchored * len(anchored),
    ):
        setting = settings[axis]
        if setting.anchor == "image":
            rule[axis] = setting.offset_mm
        elif located:
            pose = record.pose
            patient = PLANE_AXES[plane][axis]
            other = 1 - axis
            along = pose.layout.orientation[3 * axis + patient] * pitch[axis]
            across = pose.layout.orientation[3 * other + patient] * pitch[other]
            reached = (
                pose.position[patient] + along * middle[axis] + across * native[other]
            )
            target = knee[patient] + directions[axis] * setting.offset_mm
            if abs(along) > 1e-6:
                rule[axis] = (target - reached) * pitch[axis] / along
        requested[axis] = round(
            (middle[axis] + rule[axis] / pitch[axis] - native_center[axis])
            / detail[axis, axis]
        )
        native[axis] = native_center[axis] + requested[axis] * detail[axis, axis]
    rule = np.array(rule, dtype=np.float64)
    requested = np.array(requested, dtype=np.int64)
    lower = np.full(2, -np.inf)
    upper = np.full(2, np.inf)
    for item in available:
        wide = record_transform(item, frame_spacing, size)
        detail = record_transform(item, spacing, size)
        rows, columns, _, _ = record_layout(item)
        for axis, extent in enumerate((columns, rows)):
            wide_scale, detail_scale = wide[axis, axis], detail[axis, axis]
            wide_first = wide[axis, 2] - wide_scale / 2
            detail_first = detail[axis, 2] - detail_scale / 2
            first = max(-0.5, wide_first)
            last = min(extent - 0.5, wide_first + size * wide_scale)
            lower[axis] = max(
                lower[axis], math.ceil((first - detail_first) / detail_scale - 1e-5)
            )
            upper[axis] = min(
                upper[axis],
                math.floor(
                    (last - detail_first - size * detail_scale) / detail_scale + 1e-5
                ),
            )
    applied = np.where(lower <= upper, np.clip(requested, lower, upper), 0).astype(
        np.int64
    )
    offset_x, offset_y = map(int, applied)
    transform = record_transform(record, spacing, size, offset_y, offset_x)
    rows, columns, row_spacing, column_spacing = record_layout(record)
    native_center = transform @ output_center
    center = np.array(
        [
            (native_center[0] - (columns - 1) / 2) * column_spacing,
            (native_center[1] - (rows - 1) / 2) * row_spacing,
        ]
    )
    return (
        offset_y,
        offset_x,
        tuple(map(float, center)),
        float(np.linalg.norm(center - rule)),
    )


def fallback_window_center(value, view, size):
    if not value.triplet_count:
        return (0.0, 0.0), 0.0
    frame = value.frames[int(value.gather[1])]
    rows, columns, row_spacing, column_spacing = frame.native_layout
    native = frame.output_to_native @ np.array(
        [(size - 1) / 2 + frame.margin, (size - 1) / 2 + frame.margin, 1]
    )
    center = np.array(
        [
            (native[0] - (columns - 1) / 2) * column_spacing,
            (native[1] - (rows - 1) / 2) * row_spacing,
        ]
    )
    return tuple(map(float, center)), float(np.linalg.norm(center - image_rule(view)))
