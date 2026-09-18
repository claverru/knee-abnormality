import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydicom
from scipy.sparse import csr_matrix

SERIES_COUNT = 24
REPEATS = 3
SPACING_MM = 0.625
IMAGE_SIZE = 224

PREPARATION_ERRORS = (AttributeError, EOFError, OSError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    pixels: np.ndarray
    row_spacing: float
    column_spacing: float
    slope: float
    intercept: float


@dataclass(frozen=True)
class ResampledSlice:
    image: np.ndarray
    rect: tuple
    output_to_native: np.ndarray


def _crop_geometry(target_height: int, target_width: int, image_size: int):
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
    rows, columns, values = [], [], []
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
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(count, source),
        dtype=np.float32,
    )


def _target_shape(frame: DecodedFrame):
    return (
        max(1, round(frame.pixels.shape[0] * float(frame.row_spacing) / SPACING_MM)),
        max(1, round(frame.pixels.shape[1] * float(frame.column_spacing) / SPACING_MM)),
    )


def resample_crop(frame: DecodedFrame) -> ResampledSlice:
    pixels = frame.pixels
    slope, intercept = frame.slope, frame.intercept
    target_height, target_width = _target_shape(frame)
    source_top, source_left, output_top, output_left, rows, columns = _crop_geometry(
        target_height, target_width, IMAGE_SIZE
    )
    vertical = _axis_filter(pixels.shape[0], target_height, source_top, rows)
    horizontal = _axis_filter(pixels.shape[1], target_width, source_left, columns)
    middle = vertical.dot(pixels)
    region = horizontal.dot(middle.T).T
    if slope != 1.0 or intercept != 0.0:
        region *= np.float32(slope)
        region += np.float32(intercept)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    image[output_top : output_top + rows, output_left : output_left + columns] = region
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
    return ResampledSlice(image, (output_top, output_left, rows, columns), output_to_native)


def baseline(frames: list) -> list:
    return [resample_crop(frame) for frame in frames]


def stacked(frames: list) -> list:
    groups: dict = {}
    for index, frame in enumerate(frames):
        key = (
            frame.pixels.shape,
            frame.row_spacing,
            frame.column_spacing,
            frame.slope,
            frame.intercept,
        )
        groups.setdefault(key, []).append(index)
    output: list = [None] * len(frames)
    for key, indexes in groups.items():
        frame = frames[indexes[0]]
        height, width = frame.pixels.shape
        target_height, target_width = _target_shape(frame)
        source_top, source_left, output_top, output_left, rows, columns = _crop_geometry(
            target_height, target_width, IMAGE_SIZE
        )
        vertical = _axis_filter(height, target_height, source_top, rows)
        horizontal = _axis_filter(width, target_width, source_left, columns)
        count = len(indexes)
        stack = np.empty((height, count * width), dtype=np.float32)
        for position, index in enumerate(indexes):
            stack[:, position * width : (position + 1) * width] = frames[index].pixels
        middle = vertical.dot(stack).reshape(rows, count, width)
        middle = np.ascontiguousarray(middle.transpose(2, 1, 0).reshape(width, count * rows))
        region = horizontal.dot(middle).reshape(columns, count, rows)
        scale_x = width / target_width
        scale_y = height / target_height
        matrix = np.array(
            [
                [scale_x, 0, (source_left - output_left + 0.5) * scale_x - 0.5],
                [0, scale_y, (source_top - output_top + 0.5) * scale_y - 0.5],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        for position, index in enumerate(indexes):
            piece = region[:, position, :].T
            if frame.slope != 1.0 or frame.intercept != 0.0:
                piece = piece * np.float32(frame.slope) + np.float32(frame.intercept)
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            image[output_top : output_top + rows, output_left : output_left + columns] = piece
            output[index] = ResampledSlice(
                image, (output_top, output_left, rows, columns), matrix
            )
    return output


def batched(frames: list) -> list:
    groups: dict = {}
    for index, frame in enumerate(frames):
        key = (
            frame.pixels.shape,
            frame.row_spacing,
            frame.column_spacing,
            frame.slope,
            frame.intercept,
        )
        groups.setdefault(key, []).append(index)
    output: list = [None] * len(frames)
    for key, indexes in groups.items():
        frame = frames[indexes[0]]
        height, width = frame.pixels.shape
        target_height, target_width = _target_shape(frame)
        source_top, source_left, output_top, output_left, rows, columns = _crop_geometry(
            target_height, target_width, IMAGE_SIZE
        )
        vertical = _axis_filter(height, target_height, source_top, rows)
        horizontal = _axis_filter(width, target_width, source_left, columns).toarray().T
        count = len(indexes)
        stack = np.empty((height, count * width), dtype=np.float32)
        for position, index in enumerate(indexes):
            stack[:, position * width : (position + 1) * width] = frames[index].pixels
        middle = vertical.dot(stack).reshape(rows, count, width).transpose(1, 0, 2)
        region = np.matmul(middle, horizontal)
        scale_x = width / target_width
        scale_y = height / target_height
        matrix = np.array(
            [
                [scale_x, 0, (source_left - output_left + 0.5) * scale_x - 0.5],
                [0, scale_y, (source_top - output_top + 0.5) * scale_y - 0.5],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        for position, index in enumerate(indexes):
            piece = region[position]
            if frame.slope != 1.0 or frame.intercept != 0.0:
                piece = piece * np.float32(frame.slope) + np.float32(frame.intercept)
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            image[output_top : output_top + rows, output_left : output_left + columns] = piece
            output[index] = ResampledSlice(
                image, (output_top, output_left, rows, columns), matrix
            )
    return output


def load_frames(study_dirs: list) -> list:
    frames = []
    for study in study_dirs:
        for series in sorted(os.scandir(study), key=lambda entry: entry.name):
            if not series.is_dir():
                continue
            paths = sorted(Path(series.path).glob("*.dcm"))
            for path in paths:
                try:
                    dataset = pydicom.dcmread(path)
                    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
                    spacing = dataset.PixelSpacing
                    if pixels.ndim != 2 or len(spacing) != 2:
                        continue
                    frames.append(
                        DecodedFrame(
                            pixels,
                            float(spacing[0]),
                            float(spacing[1]),
                            float(getattr(dataset, "RescaleSlope", 1.0)),
                            float(getattr(dataset, "RescaleIntercept", 0.0)),
                        )
                    )
                except PREPARATION_ERRORS:
                    continue
            if frames:
                break
        if len(frames) >= 24 * len(study_dirs):
            break
    return frames


def measure(name: str, function, frames: list) -> dict:
    function(frames[:4])
    best = None
    for _ in range(REPEATS):
        with Timer() as timer:
            function(frames)
        best = timer.wall if best is None else min(best, timer.wall)
    return {
        "implementation": name,
        "slices": len(frames),
        "wall_seconds": round(best, 4),
        "milliseconds_per_slice": round(1000 * best / len(frames), 4),
    }


VARIANTS = {"stacked": stacked, "batched": batched}


def audit(frames: list) -> dict:
    expected = baseline(frames)
    results = {}
    for name, function in VARIANTS.items():
        exact = 0
        worst = 0.0
        for left, right in zip(expected, function(frames)):
            if left.rect != right.rect or not np.array_equal(
                left.output_to_native, right.output_to_native
            ):
                worst = float("inf")
                continue
            if np.array_equal(left.image, right.image):
                exact += 1
            worst = max(worst, float(np.max(np.abs(left.image - right.image))))
        results[name] = {"bit_exact": exact, "max_absolute_difference": worst}
    results["compared"] = len(frames)
    return results


def main() -> None:
    studies = series_directories()[:SERIES_COUNT]
    frames = load_frames(studies)
    result = {"frames_loaded": len(frames), "audit": audit(frames)}
    result["cells"] = (
        []
        if AUDIT_ONLY
        else [measure("baseline", baseline, frames)]
        + [measure(name, function, frames) for name, function in VARIANTS.items()]
    )
    report("resample", result)


main()
