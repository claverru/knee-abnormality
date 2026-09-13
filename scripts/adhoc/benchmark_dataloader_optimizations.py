import ctypes
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import psutil
import pydicom
import torch
from pydantic import Field
from pydantic_settings import CliPositionalArg
from pydicom.dataelem import RawDataElement
from pydicom.filereader import read_partial
from pydicom.tag import Tag
from scipy.sparse import csr_matrix
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import knee_abnormality.dataset as production_dataset
import knee_abnormality.preprocessing.dicom as production_dicom
import knee_abnormality.preprocessing.preparation as production_preprocessing
from knee_abnormality.cli import CLISettings
from knee_abnormality.config import DataConfig, ExperimentConfig, load_config
from knee_abnormality.dataset import (
    KNOWN_BAD_SERIES,
    SERIES,
    STUDY,
    SeriesCandidate,
    StudySeries,
    allocate_acquisition_inputs,
    resolve_raw_dir,
)
from knee_abnormality.inference import load_checkpoint_model
from knee_abnormality.labels import load_fold_labels
from knee_abnormality.preprocessing.dicom import PREPARATION_ERRORS
from knee_abnormality.preprocessing.preparation import (
    prepare_series as production_prepare_series,
)
from knee_abnormality.preprocessing.series import (
    KNOWN_BAD_INSTANCES,
    SliceRecord,
)
from knee_abnormality.runtime import move_batch
from scripts.adhoc import dataloader_reference as REFERENCE_DATASET

PreparedSeries = REFERENCE_DATASET.PreparedSeries
REFERENCE_PREPROCESSING = REFERENCE_DATASET


CANDIDATES = (
    "baseline",
    "numpy-native",
    "affine-late",
    "direct-region",
    "combined-numpy",
    "combined",
    "production",
    "production-reference",
    "dicom-deferred",
    "dicom-native",
)
DICOM_CANDIDATES = ("dicom-deferred", "dicom-native")
WORKERS = 8
THREADS = 1
STAGE_TOTALS = {}
PROFILE_STAGES = False
AUDIT_SERIES = []
CAPTURE_AUDIT = False
NATIVE_LIBRARY = None
FILTER_CACHE = {}
CORE_FILES = (
    "configs/baseline.yaml",
    "src/knee_abnormality/config.py",
    "src/knee_abnormality/dataset.py",
    "src/knee_abnormality/inference.py",
    "src/knee_abnormality/model.py",
    *tuple(
        str(path)
        for path in sorted(Path("src/knee_abnormality/preprocessing").rglob("*.py"))
    ),
    "src/knee_abnormality/runtime.py",
    "src/knee_abnormality/training.py",
    "scripts/pipeline/export.py",
    "scripts/adhoc/dataloader_reference/__init__.py",
)
NATIVE_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <stddef.h>
int normalize_rects(const float *images, uint8_t *output, const int32_t *rects,
                    int32_t count, int32_t height, int32_t width,
                    float low, float high) {
    if (!isfinite(low) || !isfinite(high) || high <= low) return 1;
    float gain = (1.0f / (high - low)) * 255.0f;
    size_t plane = (size_t)height * (size_t)width;
    for (int32_t n = 0; n < count; ++n) {
        int32_t top = rects[4 * n];
        int32_t left = rects[4 * n + 1];
        int32_t rows = rects[4 * n + 2];
        int32_t cols = rects[4 * n + 3];
        const float *source = images + (size_t)n * plane;
        uint8_t *target = output + (size_t)n * plane;
        for (int32_t row = top; row < top + rows; ++row) {
            size_t offset = (size_t)row * (size_t)width;
            for (int32_t col = left; col < left + cols; ++col) {
                float value = source[offset + (size_t)col];
                if (!isfinite(value)) return 2;
                if (value < low) value = low;
                if (value > high) value = high;
                float mapped = nearbyintf((value - low) * gain);
                if (mapped < 0.0f) mapped = 0.0f;
                if (mapped > 255.0f) mapped = 255.0f;
                target[offset + (size_t)col] = (uint8_t)mapped;
            }
        }
    }
    return 0;
}
"""


class BenchmarkCLI(CLISettings):
    config: CliPositionalArg[Path]
    output_dir: Path
    mode: Literal[
        "suite",
        "run",
        "audit",
        "profile",
        "model",
        "reference-audit",
        "dicom-audit",
        "dicom-suite",
    ] = "suite"
    candidate: str = "baseline"
    cohort: Path | None = None
    samples: int = Field(default=128, ge=8, le=1024)
    repeats: int = Field(default=3, ge=1, le=10)
    fold: int = Field(default=0, ge=0, le=4)
    native_library: Path | None = None
    checkpoint: Path | None = None
    audit_samples: int = Field(default=8, ge=1, le=24)


@dataclass(frozen=True)
class ArraySlice:
    image: np.ndarray
    rect: tuple[int, int, int, int]
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    slope: float
    intercept: float


@dataclass(frozen=True, slots=True)
class DicomDescriptor:
    offset: int
    length: int
    file_size: int
    modified_ns: int
    rows: int
    columns: int
    bits_stored: int
    signed: bool
    row_spacing: float
    column_spacing: float
    slope: float
    intercept: float
    deferred: pydicom.Dataset | None


@dataclass(frozen=True)
class DicomSliceRecord(SliceRecord):
    descriptor: DicomDescriptor | None = None


def read_descriptor(path: Path, candidate: str):
    pixel_element = None
    with path.open("rb") as stream:

        def stop_at_pixels(tag, vr, length):
            nonlocal pixel_element
            if tag in {0x7FE00008, 0x7FE00009, 0x7FE00010}:
                pixel_element = (tag, vr, length, stream.tell())
                return True
            return False

        dataset = read_partial(stream, stop_when=stop_at_pixels)
        info = os.fstat(stream.fileno())
    try:
        if (
            str(dataset.file_meta.TransferSyntaxUID) != "1.2.840.10008.1.2.1"
            or dataset.original_encoding[:2] != (False, True)
            or pixel_element is None
        ):
            return dataset, None
        tag, vr, length, offset = pixel_element
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
        bits_stored = int(dataset.BitsStored)
        representation = int(dataset.PixelRepresentation)
        spacing = dataset.PixelSpacing
        if (
            tag != 0x7FE00010
            or vr not in {"OB", "OW"}
            or int(dataset.SamplesPerPixel) != 1
            or str(dataset.PhotometricInterpretation) != "MONOCHROME2"
            or int(getattr(dataset, "NumberOfFrames", 1)) != 1
            or int(dataset.BitsAllocated) != 16
            or not 1 <= bits_stored <= 16
            or int(dataset.HighBit) != bits_stored - 1
            or representation not in {0, 1}
            or rows < 1
            or columns < 1
            or length != rows * columns * 2
            or offset + length != info.st_size
            or len(spacing) != 2
        ):
            return dataset, None
        deferred = None
        if candidate == "dicom-deferred":
            deferred = dataset.copy()
            deferred._dict = {
                key: value
                for key, value in dataset._dict.items()
                if key.group == 0x0028
            }
            deferred[Tag(tag)] = RawDataElement(
                Tag(tag), vr, length, None, offset, False, True
            )
        return dataset, DicomDescriptor(
            offset,
            length,
            info.st_size,
            info.st_mtime_ns,
            rows,
            columns,
            bits_stored,
            representation == 1,
            float(spacing[0]),
            float(spacing[1]),
            float(getattr(dataset, "RescaleSlope", 1.0)),
            float(getattr(dataset, "RescaleIntercept", 0.0)),
            deferred,
        )
    except PREPARATION_ERRORS:
        return dataset, None


def descriptor_pixels(record: DicomSliceRecord):
    descriptor = record.descriptor
    if descriptor is None:
        return read_pixels(record.path)
    if descriptor.deferred is not None:
        dataset = descriptor.deferred.copy()
        dataset._dict = dataset._dict.copy()
        pixels = measured(
            "pixel_decode", lambda: np.asarray(dataset.pixel_array, dtype=np.float32)
        )
    else:
        started = time.perf_counter() if PROFILE_STAGES else 0.0
        with record.path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if (info.st_size, info.st_mtime_ns) != (
                descriptor.file_size,
                descriptor.modified_ns,
            ):
                return read_pixels(record.path)
            stream.seek(descriptor.offset)
            payload = stream.read(descriptor.length)
        if len(payload) != descriptor.rows * descriptor.columns * 2:
            return read_pixels(record.path)
        pixels = np.frombuffer(payload, dtype="<i2" if descriptor.signed else "<u2")
        shift = 16 - descriptor.bits_stored
        if shift:
            pixels = np.left_shift(pixels, shift)
            np.right_shift(pixels, shift, out=pixels)
        pixels = pixels.reshape(descriptor.rows, descriptor.columns).astype(np.float32)
        if PROFILE_STAGES:
            STAGE_TOTALS["pixel_seek_decode"] = (
                STAGE_TOTALS.get("pixel_seek_decode", 0.0)
                + time.perf_counter()
                - started
            )
    return (
        pixels,
        descriptor.row_spacing,
        descriptor.column_spacing,
        descriptor.slope,
        descriptor.intercept,
    )


def measured(name, function, *args, **kwargs):
    if not PROFILE_STAGES:
        return function(*args, **kwargs)
    started = time.perf_counter()
    value = function(*args, **kwargs)
    STAGE_TOTALS[name] = STAGE_TOTALS.get(name, 0.0) + time.perf_counter() - started
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_position(dataset: pydicom.Dataset) -> float | None:
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


def frozen_index_series(
    series_dir: Path,
    position_reference: Literal["origin", "center"] = "origin",
    candidate: str = "baseline",
) -> tuple[SliceRecord, ...]:
    records = []
    for fallback, path in enumerate(sorted(series_dir.glob("*.dcm"))):
        if path.stem in KNOWN_BAD_INSTANCES:
            continue
        try:
            descriptor = None
            if candidate in DICOM_CANDIDATES:
                dataset, descriptor = measured(
                    "header_parse", read_descriptor, path, candidate
                )
            else:
                dataset = measured(
                    "header_parse", pydicom.dcmread, path, stop_before_pixels=True
                )
            physical = physical_position(dataset)
            position = (
                physical
                if physical is not None
                else float(getattr(dataset, "InstanceNumber", fallback))
            )
            if math.isfinite(position):
                records.append((path, position, physical, descriptor))
        except PREPARATION_ERRORS:
            continue
    records.sort(key=lambda item: (item[1], item[0].name))
    origin = next((value for _, _, value, _ in records if value is not None), 0)
    if position_reference == "center":
        values = [value for _, _, value, _ in records if value is not None]
        if values:
            origin = (min(values) + max(values)) / 2
    return tuple(
        DicomSliceRecord(
            path,
            position,
            index,
            value - origin if value is not None else None,
            descriptor,
        )
        for index, (path, position, value, descriptor) in enumerate(records)
    )


def center_weights(
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


def weighted_validation_starts(
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


def frozen_select_slices(
    records: tuple[SliceRecord, ...],
    limit: int,
    group_size: int,
    sampling: str,
    center_weight: float,
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
        starts = weighted_validation_starts(
            center_weights(records, group_size, center_weight), groups, group_size
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
        selected = (
            torch.arange(groups)
            if compressed_count == groups
            else torch.linspace(0, compressed_count - 1, groups).round().to(torch.int64)
        )
        starts = selected + 2 * torch.arange(groups)
        return tuple(
            records[start + offset] for start in starts.tolist() for offset in range(3)
        )
    if count <= limit:
        return records
    selected = (
        [count // 2]
        if limit == 1
        else torch.linspace(0, count - 1, limit).round().to(torch.int64).tolist()
    )
    return tuple(records[index] for index in selected)


def validate_records(
    records: tuple[SliceRecord, ...], position_encoding: str, group_size: int
) -> None:
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


def crop_geometry(
    target_height: int, target_width: int, image_size: int
) -> tuple[int, int, int, int, int, int]:
    source_top = max(0, (target_height - image_size) // 2)
    source_left = max(0, (target_width - image_size) // 2)
    output_top = max(0, (image_size - target_height) // 2)
    output_left = max(0, (image_size - target_width) // 2)
    rows = min(target_height, image_size)
    cols = min(target_width, image_size)
    return source_top, source_left, output_top, output_left, rows, cols


def read_pixels(path: Path):
    started = time.perf_counter() if PROFILE_STAGES else 0.0
    dataset = pydicom.dcmread(path)
    parsed = time.perf_counter() if PROFILE_STAGES else 0.0
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if PROFILE_STAGES:
        STAGE_TOTALS["pixel_parse"] = (
            STAGE_TOTALS.get("pixel_parse", 0.0) + parsed - started
        )
        STAGE_TOTALS["pixel_decode"] = (
            STAGE_TOTALS.get("pixel_decode", 0.0) + time.perf_counter() - parsed
        )
    if pixels.ndim != 2:
        raise ValueError(f"Expected one two-dimensional frame, got {pixels.shape}")
    spacing = getattr(dataset, "PixelSpacing", None)
    if spacing is None or len(spacing) != 2:
        raise ValueError("Missing two-dimensional PixelSpacing")
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return pixels, float(spacing[0]), float(spacing[1]), slope, intercept


def baseline_decode(path: Path, spacing_mm: float, image_size: int):
    try:
        pixels, row_spacing, col_spacing, slope, intercept = read_pixels(path)
        started = time.perf_counter() if PROFILE_STAGES else 0.0
        pixels = pixels * np.float32(slope) + np.float32(intercept)
        if PROFILE_STAGES:
            STAGE_TOTALS["affine"] = (
                STAGE_TOTALS.get("affine", 0.0) + time.perf_counter() - started
            )
        target_height = max(1, round(pixels.shape[0] * row_spacing / spacing_mm))
        target_width = max(1, round(pixels.shape[1] * col_spacing / spacing_mm))
        started = time.perf_counter() if PROFILE_STAGES else 0.0
        image = F.interpolate(
            torch.from_numpy(pixels)[None, None],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, 0]
        if PROFILE_STAGES:
            STAGE_TOTALS["resample"] = (
                STAGE_TOTALS.get("resample", 0.0) + time.perf_counter() - started
            )
        started = time.perf_counter() if PROFILE_STAGES else 0.0
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
        image = image[top : top + image_size, left : left + image_size]
        valid = valid[top : top + image_size, left : left + image_size]
        if PROFILE_STAGES:
            STAGE_TOTALS["crop_mask"] = (
                STAGE_TOTALS.get("crop_mask", 0.0) + time.perf_counter() - started
            )
        if not torch.isfinite(image).all():
            return f"{path}: non-finite pixels"
        return image, valid
    except PREPARATION_ERRORS as error:
        return f"{path}: {type(error).__name__}: {error}"


def reference_quantiles(values: np.ndarray, lower_q: float, upper_q: float):
    quantiles = torch.tensor([lower_q, upper_q])
    ranks = quantiles * (values.size - 1)
    lower = ranks.floor().to(torch.int64)
    upper = ranks.ceil().to(torch.int64)
    indices = torch.stack((lower, upper)).numpy()
    values.partition(np.unique(indices))
    selected = torch.from_numpy(values[indices])
    return torch.lerp(selected[0], selected[1], ranks - lower)


def baseline_prepare(
    records: tuple[SliceRecord, ...],
    data: DataConfig,
) -> PreparedSeries:
    validate_records(records, data.position_encoding, data.slices_per_input)
    decoded = tuple(
        baseline_decode(record.path, data.spacing_mm, data.image_size)
        for record in records
    )
    images = []
    masks = []
    positions = []
    failures = []
    for start in range(0, len(records), data.slices_per_input):
        group_records = records[start : start + data.slices_per_input]
        group = decoded[start : start + data.slices_per_input]
        accepted = [value for value in group if not isinstance(value, str)]
        failures.extend(value for value in group if isinstance(value, str))
        if len(accepted) != data.slices_per_input:
            continue
        images.extend(value[0] for value in accepted)
        masks.extend(value[1] for value in accepted)
        center = group_records[data.slices_per_input // 2]
        positions.append(
            center.relative_position_mm
            if data.position_encoding == "physical_mm"
            else center.original_index
        )
    if not images:
        raise ValueError("No complete selected input group could be decoded")
    started = time.perf_counter() if PROFILE_STAGES else 0.0
    pixels = np.stack([image.numpy() for image in images], dtype=np.float32)
    validity = np.stack([mask.numpy() for mask in masks])
    values = pixels[validity]
    if PROFILE_STAGES:
        STAGE_TOTALS["collect"] = (
            STAGE_TOTALS.get("collect", 0.0) + time.perf_counter() - started
        )
    started = time.perf_counter() if PROFILE_STAGES else 0.0
    bounds = reference_quantiles(values, data.lower_quantile, data.upper_quantile)
    if PROFILE_STAGES:
        STAGE_TOTALS["percentile"] = (
            STAGE_TOTALS.get("percentile", 0.0) + time.perf_counter() - started
        )
    low, high = bounds.unbind()
    constant = (
        not math.isfinite(float(low))
        or not math.isfinite(float(high))
        or bool(high <= low)
    )
    started = time.perf_counter() if PROFILE_STAGES else 0.0
    if constant:
        output = torch.zeros_like(torch.from_numpy(pixels), dtype=torch.uint8)
    else:
        stacked = torch.from_numpy(pixels)
        valid = torch.from_numpy(validity)
        output = (
            ((stacked.clamp(low, high) - low) * (255.0 / (high - low)))
            .round()
            .to(torch.uint8)
        )
        output.masked_fill_(~valid, 0)
    if PROFILE_STAGES:
        STAGE_TOTALS["normalize"] = (
            STAGE_TOTALS.get("normalize", 0.0) + time.perf_counter() - started
        )
    if data.slices_per_input == 3:
        output = output.reshape(-1, 3, data.image_size, data.image_size)
    if CAPTURE_AUDIT:
        AUDIT_SERIES.append(
            {
                "paths": [str(record.path) for record in records],
                "float": pixels.copy(),
                "rects": [rectangle_from_mask(mask) for mask in validity],
                "bounds": [float(low), float(high)],
            }
        )
    return PreparedSeries(
        output,
        torch.tensor(positions, dtype=torch.float32),
        tuple(failures),
        constant,
    )


def rectangle_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0, 0, 0, 0
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max() - rows.min() + 1),
        int(cols.max() - cols.min() + 1),
    )


def torch_array_decode(
    path: Path, spacing_mm: float, image_size: int, late_affine: bool
) -> ArraySlice | str:
    try:
        pixels, row_spacing, col_spacing, slope, intercept = read_pixels(path)
        if not late_affine:
            started = time.perf_counter() if PROFILE_STAGES else 0.0
            pixels = pixels * np.float32(slope) + np.float32(intercept)
            if PROFILE_STAGES:
                STAGE_TOTALS["affine"] = (
                    STAGE_TOTALS.get("affine", 0.0) + time.perf_counter() - started
                )
        target_height = max(1, round(pixels.shape[0] * row_spacing / spacing_mm))
        target_width = max(1, round(pixels.shape[1] * col_spacing / spacing_mm))
        started = time.perf_counter() if PROFILE_STAGES else 0.0
        resized = F.interpolate(
            torch.from_numpy(pixels)[None, None],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, 0].numpy()
        if PROFILE_STAGES:
            STAGE_TOTALS["resample"] = (
                STAGE_TOTALS.get("resample", 0.0) + time.perf_counter() - started
            )
        source_top, source_left, output_top, output_left, rows, cols = crop_geometry(
            target_height, target_width, image_size
        )
        started = time.perf_counter() if PROFILE_STAGES else 0.0
        output = np.zeros((image_size, image_size), dtype=np.float32)
        region = resized[
            source_top : source_top + rows, source_left : source_left + cols
        ]
        if late_affine and (slope != 1.0 or intercept != 0.0):
            region = region * np.float32(slope) + np.float32(intercept)
        output[output_top : output_top + rows, output_left : output_left + cols] = (
            region
        )
        if PROFILE_STAGES:
            STAGE_TOTALS["crop_affine"] = (
                STAGE_TOTALS.get("crop_affine", 0.0) + time.perf_counter() - started
            )
        if not np.isfinite(region).all():
            return f"{path}: non-finite pixels"
        return ArraySlice(
            output,
            (output_top, output_left, rows, cols),
            tuple(pixels.shape),
            (target_height, target_width),
            slope,
            intercept,
        )
    except PREPARATION_ERRORS as error:
        return f"{path}: {type(error).__name__}: {error}"


def axis_filter(source: int, target: int, first: int, count: int) -> csr_matrix:
    key = (source, target, first, count)
    if key in FILTER_CACHE:
        return FILTER_CACHE[key]
    scale = source / target
    support = max(1.0, scale)
    rows = []
    cols = []
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
            cols.append(index)
            values.append(weight / total)
    matrix = csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=(count, source),
        dtype=np.float32,
    )
    FILTER_CACHE[key] = matrix
    return matrix


def direct_array_decode(
    path: Path, spacing_mm: float, image_size: int, record=None
) -> ArraySlice | str:
    try:
        pixels, row_spacing, col_spacing, slope, intercept = (
            read_pixels(path) if record is None else descriptor_pixels(record)
        )
        target_height = max(1, round(pixels.shape[0] * row_spacing / spacing_mm))
        target_width = max(1, round(pixels.shape[1] * col_spacing / spacing_mm))
        source_top, source_left, output_top, output_left, rows, cols = crop_geometry(
            target_height, target_width, image_size
        )
        started = time.perf_counter() if PROFILE_STAGES else 0.0
        vertical = axis_filter(pixels.shape[0], target_height, source_top, rows)
        horizontal = axis_filter(pixels.shape[1], target_width, source_left, cols)
        middle = vertical.dot(pixels)
        region = horizontal.dot(middle.T).T
        if slope != 1.0 or intercept != 0.0:
            region *= np.float32(slope)
            region += np.float32(intercept)
        output = np.zeros((image_size, image_size), dtype=np.float32)
        output[output_top : output_top + rows, output_left : output_left + cols] = (
            region
        )
        if PROFILE_STAGES:
            STAGE_TOTALS["direct_resample_affine"] = (
                STAGE_TOTALS.get("direct_resample_affine", 0.0)
                + time.perf_counter()
                - started
            )
        if not np.isfinite(region).all():
            return f"{path}: non-finite pixels"
        return ArraySlice(
            output,
            (output_top, output_left, rows, cols),
            tuple(pixels.shape),
            (target_height, target_width),
            slope,
            intercept,
        )
    except PREPARATION_ERRORS as error:
        return f"{path}: {type(error).__name__}: {error}"


def numpy_bounds(values: np.ndarray, lower_q: float, upper_q: float):
    quantiles = np.asarray([lower_q, upper_q], dtype=np.float32)
    ranks = quantiles * np.float32(values.size - 1)
    lower = np.floor(ranks).astype(np.int64)
    upper = np.ceil(ranks).astype(np.int64)
    values.partition(np.unique(np.concatenate((lower, upper))))
    low_values = values[lower]
    high_values = values[upper]
    bounds = low_values + (high_values - low_values) * (ranks - lower).astype(
        np.float32
    )
    return float(bounds[0]), float(bounds[1])


def native_normalize(
    pixels: np.ndarray, rects: np.ndarray, low: float, high: float
) -> np.ndarray:
    if NATIVE_LIBRARY is None:
        raise RuntimeError("Native normalization library was not loaded")
    output = np.zeros(pixels.shape, dtype=np.uint8)
    status = NATIVE_LIBRARY.normalize_rects(
        pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        rects.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        pixels.shape[0],
        pixels.shape[1],
        pixels.shape[2],
        np.float32(low),
        np.float32(high),
    )
    if status == 1:
        return output
    if status == 2:
        raise ValueError("Non-finite pixels reached native normalization")
    if status:
        raise RuntimeError(f"Native normalization failed with status {status}")
    return output


def numpy_normalize(
    pixels: np.ndarray, rects: np.ndarray, low: float, high: float
) -> np.ndarray:
    output = np.zeros(pixels.shape, dtype=np.uint8)
    low_value = np.float32(low)
    high_value = np.float32(high)
    if (
        not np.isfinite(low_value)
        or not np.isfinite(high_value)
        or high_value <= low_value
    ):
        return output
    gain = (np.float32(1.0) / (high_value - low_value)) * np.float32(255.0)
    np.clip(pixels, low_value, high_value, out=pixels)
    np.subtract(pixels, low_value, out=pixels)
    np.multiply(pixels, gain, out=pixels)
    np.rint(pixels, out=pixels)
    np.clip(pixels, np.float32(0.0), np.float32(255.0), out=pixels)
    output[:] = pixels
    for index, (top, left, rows, cols) in enumerate(rects):
        output[index, :top] = 0
        output[index, top + rows :] = 0
        output[index, top : top + rows, :left] = 0
        output[index, top : top + rows, left + cols :] = 0
    return output


def array_prepare(
    records: tuple[SliceRecord, ...], data: DataConfig, decoder, normalizer: str
) -> PreparedSeries:
    validate_records(records, data.position_encoding, data.slices_per_input)
    decoded = tuple(
        decoder(record.path, data.spacing_mm, data.image_size) for record in records
    )
    accepted = []
    positions = []
    failures = []
    accepted_records = []
    for start in range(0, len(records), data.slices_per_input):
        group_records = records[start : start + data.slices_per_input]
        group = decoded[start : start + data.slices_per_input]
        good = [value for value in group if not isinstance(value, str)]
        failures.extend(value for value in group if isinstance(value, str))
        if len(good) != data.slices_per_input:
            continue
        accepted.extend(good)
        accepted_records.extend(group_records)
        center = group_records[data.slices_per_input // 2]
        positions.append(
            center.relative_position_mm
            if data.position_encoding == "physical_mm"
            else center.original_index
        )
    if not accepted:
        raise ValueError("No complete selected input group could be decoded")
    started = time.perf_counter() if PROFILE_STAGES else 0.0
    pixels = np.empty(
        (len(accepted), data.image_size, data.image_size), dtype=np.float32
    )
    rects = np.empty((len(accepted), 4), dtype=np.int32)
    valid_count = sum(value.rect[2] * value.rect[3] for value in accepted)
    scratch = np.empty(valid_count, dtype=np.float32)
    offset = 0
    for index, value in enumerate(accepted):
        pixels[index] = value.image
        rects[index] = value.rect
        top, left, rows, cols = value.rect
        size = rows * cols
        scratch[offset : offset + size] = value.image[
            top : top + rows, left : left + cols
        ].reshape(-1)
        offset += size
    if PROFILE_STAGES:
        STAGE_TOTALS["collect"] = (
            STAGE_TOTALS.get("collect", 0.0) + time.perf_counter() - started
        )
    started = time.perf_counter() if PROFILE_STAGES else 0.0
    low, high = numpy_bounds(scratch, data.lower_quantile, data.upper_quantile)
    if PROFILE_STAGES:
        STAGE_TOTALS["percentile"] = (
            STAGE_TOTALS.get("percentile", 0.0) + time.perf_counter() - started
        )
    constant = not math.isfinite(low) or not math.isfinite(high) or high <= low
    audit_pixels = pixels.copy() if CAPTURE_AUDIT else None
    started = time.perf_counter() if PROFILE_STAGES else 0.0
    if normalizer == "native":
        output_array = native_normalize(pixels, rects, low, high)
        output = torch.from_numpy(output_array)
    elif normalizer == "numpy":
        native_reference = (
            native_normalize(pixels.copy(), rects, low, high) if CAPTURE_AUDIT else None
        )
        output_array = numpy_normalize(pixels, rects, low, high)
        if native_reference is not None:
            np.testing.assert_array_equal(native_reference, output_array)
        output = torch.from_numpy(output_array)
    elif normalizer == "torch":
        validity = np.zeros(pixels.shape, dtype=bool)
        for index, (top, left, rows, cols) in enumerate(rects):
            validity[index, top : top + rows, left : left + cols] = True
        low_tensor = torch.tensor(low)
        high_tensor = torch.tensor(high)
        stacked = torch.from_numpy(pixels)
        valid = torch.from_numpy(validity)
        if constant:
            output = torch.zeros_like(stacked, dtype=torch.uint8)
        else:
            output = (
                (
                    (stacked.clamp(low_tensor, high_tensor) - low_tensor)
                    * (255.0 / (high_tensor - low_tensor))
                )
                .round()
                .to(torch.uint8)
            )
            output.masked_fill_(~valid, 0)
    else:
        raise ValueError(f"Unknown normalizer {normalizer}")
    if PROFILE_STAGES:
        STAGE_TOTALS["normalize"] = (
            STAGE_TOTALS.get("normalize", 0.0) + time.perf_counter() - started
        )
    if data.slices_per_input == 3:
        output = output.reshape(-1, 3, data.image_size, data.image_size)
    if CAPTURE_AUDIT:
        AUDIT_SERIES.append(
            {
                "paths": [str(record.path) for record in accepted_records],
                "float": audit_pixels,
                "rects": [tuple(map(int, rect)) for rect in rects],
                "bounds": [low, high],
                "source_shapes": [value.source_shape for value in accepted],
                "target_shapes": [value.target_shape for value in accepted],
                "affines": [[value.slope, value.intercept] for value in accepted],
            }
        )
    return PreparedSeries(
        output,
        torch.tensor(positions, dtype=torch.float32),
        tuple(failures),
        constant,
    )


def prepare_candidate(
    records: tuple[SliceRecord, ...], data: DataConfig, candidate: str
):
    if candidate in DICOM_CANDIDATES:

        def decoder(selected, spacing, size):
            values = tuple(
                direct_array_decode(record.path, spacing, size, record)
                for record in selected
            )
            if CAPTURE_AUDIT:
                AUDIT_SERIES.append(
                    {
                        "paths": [str(record.path) for record in selected],
                        "rects": [
                            None if isinstance(value, str) else value.rect
                            for value in values
                        ],
                        "native_paths": sum(
                            record.descriptor is not None for record in selected
                        ),
                    }
                )
            return values

        return production_prepare_series(
            records,
            data.spacing_mm,
            data.image_size,
            data.lower_quantile,
            data.upper_quantile,
            data.position_encoding,
            data.slices_per_input,
            decoder=decoder,
        )
    if candidate == "baseline":
        return baseline_prepare(records, data)
    if candidate == "numpy-native":
        return array_prepare(
            records, data, lambda p, s, i: torch_array_decode(p, s, i, False), "native"
        )
    if candidate == "affine-late":
        return array_prepare(
            records, data, lambda p, s, i: torch_array_decode(p, s, i, True), "torch"
        )
    if candidate == "direct-region":
        return array_prepare(records, data, direct_array_decode, "torch")
    if candidate == "combined-numpy":
        return array_prepare(records, data, direct_array_decode, "numpy")
    if candidate == "combined":
        return array_prepare(records, data, direct_array_decode, "native")
    if candidate == "production":
        value = production_prepare_series(
            records,
            data.spacing_mm,
            data.image_size,
            data.lower_quantile,
            data.upper_quantile,
            data.position_encoding,
            data.slices_per_input,
        )
        if CAPTURE_AUDIT:
            rects = []
            for record in records:
                dataset = pydicom.dcmread(
                    record.path,
                    stop_before_pixels=True,
                    specific_tags=["Columns", "PixelSpacing", "Rows"],
                )
                spacing = dataset.PixelSpacing
                target_height = max(
                    1, round(int(dataset.Rows) * float(spacing[0]) / data.spacing_mm)
                )
                target_width = max(
                    1, round(int(dataset.Columns) * float(spacing[1]) / data.spacing_mm)
                )
                geometry = crop_geometry(target_height, target_width, data.image_size)
                rects.append((geometry[2], geometry[3], geometry[4], geometry[5]))
            AUDIT_SERIES.append(
                {
                    "paths": [str(record.path) for record in records],
                    "rects": rects,
                }
            )
        return value
    raise ValueError(f"Unknown candidate {candidate}")


class FrozenDataset(Dataset):
    def __init__(
        self,
        studies: pl.DataFrame,
        series_index: dict[str, StudySeries],
        data: DataConfig,
        targets: tuple[str, ...],
        seed: int,
        image_limit_multiplier: int,
        candidate: str,
    ):
        self.studies = studies
        self.series_index = series_index
        self.data = data
        self.targets = targets
        self.seed = seed
        self.image_limit_multiplier = image_limit_multiplier
        self.candidate = candidate
        self.slice_index = {}
        missing = set(studies[STUDY].to_list()) - set(series_index)
        if missing:
            raise ValueError(
                f"Series metadata is missing {len(missing)} requested studies"
            )

    def __len__(self):
        return self.studies.height

    def records(self, candidate: SeriesCandidate):
        value = self.slice_index.get(candidate.directory)
        if value is None:
            value = frozen_index_series(
                candidate.directory, self.data.position_reference, self.candidate
            )
            self.slice_index[candidate.directory] = value
        return value

    def try_prepare(self, candidate, limit, failures):
        try:
            records = frozen_select_slices(
                self.records(candidate),
                limit,
                self.data.slices_per_input,
                self.data.slice_sampling,
                self.data.center_sampling_weight,
            )
            value = prepare_candidate(records, self.data, self.candidate)
            failures.extend(value.failures)
            if value.constant:
                failures.append(f"{candidate.uid}: constant prepared series")
                return None
            return replace(value, uid=candidate.uid)
        except PREPARATION_ERRORS as error:
            failures.append(f"{candidate.uid}: {type(error).__name__}: {error}")
            return None

    def prepare_slot(self, candidates, limit, failures):
        if self.data.acquisition_sampling == "shortest":
            for candidate in candidates:
                value = self.try_prepare(candidate, limit, failures)
                if value is not None:
                    return (value,)
            return ()
        available = []
        group_size = self.data.slices_per_input
        for candidate in sorted(candidates, key=lambda item: item.uid):
            try:
                capacity = len(self.records(candidate)) // group_size
                if capacity:
                    available.append((candidate, capacity))
                else:
                    failures.append(f"{candidate.uid}: no complete input groups")
            except PREPARATION_ERRORS as error:
                failures.append(f"{candidate.uid}: {type(error).__name__}: {error}")
        cache = {}
        generator = torch.Generator().manual_seed(self.seed)
        while available:
            allocations = allocate_acquisition_inputs(
                tuple(capacity for _, capacity in available),
                limit // group_size,
                False,
                generator,
            )
            values = []
            failed = set()
            for (candidate, _), count in zip(available, allocations, strict=True):
                if not count:
                    continue
                key = (candidate.uid, count)
                if key not in cache:
                    cache[key] = self.try_prepare(
                        candidate, count * group_size, failures
                    )
                value = cache[key]
                if value is None:
                    failed.add(candidate.uid)
                else:
                    values.append(value)
            if not failed:
                return tuple(values)
            available = [item for item in available if item[0].uid not in failed]
        return ()

    def __getitem__(self, index):
        if CAPTURE_AUDIT:
            AUDIT_SERIES.clear()
        before = dict(STAGE_TOTALS)
        row = self.studies.row(index, named=True)
        study = self.series_index[row[STUDY]]
        prepared = []
        failures = []
        for slot, candidates in enumerate(study.slots):
            prepared.append(
                self.prepare_slot(
                    candidates,
                    self.data.series_types[slot].max_images
                    * self.image_limit_multiplier,
                    failures,
                )
            )
        if not any(prepared):
            raise RuntimeError(f"Study {study.uid} has no decodable supported series")
        target = torch.tensor(
            [
                float(row.get(f"target_{name}"))
                if row.get(f"target_{name}") is not None
                else 0.0
                for name in self.targets
            ],
            dtype=torch.float32,
        )
        observed = torch.tensor(
            [bool(row.get(f"observed_{name}", False)) for name in self.targets],
            dtype=torch.bool,
        )
        return {
            "study_uid": study.uid,
            "is_gold": bool(row.get("is_gold", False)),
            "series": tuple(prepared),
            "target": target,
            "target_observed": observed,
            "failures": tuple(failures),
            "audit": list(AUDIT_SERIES) if CAPTURE_AUDIT else [],
            "stage_seconds": {
                name: value - before.get(name, 0.0)
                for name, value in STAGE_TOTALS.items()
            },
        }


def frozen_collate(items):
    batch_size = len(items)
    type_count = len(items[0]["series"])
    entries = []
    presence = torch.zeros((batch_size, type_count), dtype=torch.bool)
    for study_index, item in enumerate(items):
        for type_index, acquisitions in enumerate(item["series"]):
            presence[study_index, type_index] = bool(acquisitions)
            for series in acquisitions:
                entries.append((study_index, type_index, series))
    max_slices = max(series.images.shape[0] for _, _, series in entries)
    image_size = entries[0][2].images.shape[-1]
    channels = entries[0][2].images.shape[1] if entries[0][2].images.ndim == 4 else 1
    images = torch.zeros(
        (len(entries), max_slices, channels, image_size, image_size), dtype=torch.uint8
    )
    positions = torch.zeros((len(entries), max_slices), dtype=torch.float32)
    slice_mask = torch.zeros((len(entries), max_slices), dtype=torch.bool)
    study_indices = torch.empty(len(entries), dtype=torch.int64)
    type_indices = torch.empty(len(entries), dtype=torch.int64)
    for series_index, (study_index, type_index, series) in enumerate(entries):
        count = series.images.shape[0]
        values = series.images if series.images.ndim == 4 else series.images[:, None]
        images[series_index, :count] = values
        positions[series_index, :count] = series.positions
        slice_mask[series_index, :count] = True
        study_indices[series_index] = study_index
        type_indices[series_index] = type_index
    stages = {}
    for item in items:
        for name, seconds in item["stage_seconds"].items():
            stages[name] = stages.get(name, 0.0) + seconds
    return {
        "study_uid": [item["study_uid"] for item in items],
        "is_gold": torch.tensor([item["is_gold"] for item in items], dtype=torch.bool),
        "images": images,
        "slice_positions": positions,
        "slice_mask": slice_mask,
        "series_presence": presence,
        "series_study_index": study_indices,
        "series_type_index": type_indices,
        "target": torch.stack([item["target"] for item in items]),
        "target_observed": torch.stack([item["target_observed"] for item in items]),
        "failures": [item["failures"] for item in items],
        "audit": [detail for item in items for detail in item["audit"]],
        "stage_seconds": stages,
        "ownership": [item["ownership"] for item in items if "ownership" in item],
    }


def build_series_index_raw(data_root: Path, data: DataConfig, split: str):
    raw_dir = resolve_raw_dir(data_root, split)
    metadata_path = raw_dir / f"{split}_series.csv"
    image_root = raw_dir / f"{split}_series"
    frame = pl.read_csv(metadata_path)
    type_to_slot = {
        (item.plane, item.fluid_sensitive, item.fat_suppression): slot
        for slot, item in enumerate(data.series_types)
    }
    grouped = {}
    for row in frame.iter_rows(named=True):
        key = (row["Anatomical_Plane"], row["Fluid_Sensitive"], row["Fat_Suppression"])
        if key not in type_to_slot:
            continue
        uid = row[SERIES]
        directory = image_root / row[STUDY] / uid
        count = sum(1 for _ in directory.glob("*.dcm"))
        if uid in KNOWN_BAD_SERIES:
            count -= 1
        slots = grouped.setdefault(row[STUDY], [[] for _ in data.series_types])
        slots[type_to_slot[key]].append(SeriesCandidate(uid, directory, count))
    return {
        uid: StudySeries(
            uid,
            tuple(
                tuple(sorted(values, key=lambda item: (item.image_count, item.uid)))
                for values in slots
            ),
        )
        for uid, slots in grouped.items()
    }


def load_rows(config: ExperimentConfig, fold: int, cohort: Path | None, samples: int):
    labels = load_fold_labels(config, fold)
    if cohort is None:
        if samples > labels.validation.height:
            raise ValueError(
                f"Requested {samples} studies from {labels.validation.height}"
            )
        indices = np.random.default_rng(config.seed).choice(
            labels.validation.height, samples, replace=False
        )
        rows = labels.validation[indices.tolist()]
    else:
        manifest = json.loads(cohort.read_text())
        order = manifest["study_uids"]
        by_uid = {row[STUDY]: row for row in labels.validation.iter_rows(named=True)}
        missing = set(order) - set(by_uid)
        if missing:
            raise ValueError(
                f"Cohort contains {len(missing)} studies outside validation"
            )
        rows = pl.DataFrame(
            [by_uid[uid] for uid in order], schema=labels.validation.schema
        )
    return rows, labels.targets


class AuditDatasetMixin:
    def _records(self, candidate):
        if candidate.directory not in self._slice_index:
            self.index_builds = getattr(self, "index_builds", 0) + 1
        return super()._records(candidate)

    def _prepare_candidate(self, candidate, limit, generator):
        module = self.audit_module
        original = module.prepare_series

        def capture(records, *args, **kwargs):
            decoder = original.__defaults__[-1]

            def decode(selected, spacing, size):
                values = decoder(selected, spacing, size)
                self.audit_details.append(
                    {
                        "paths": [str(record.path) for record in selected],
                        "records": [
                            (
                                str(record.path),
                                record.physical_position,
                                record.original_index,
                                record.relative_position_mm,
                            )
                            for record in selected
                        ],
                        "rects": [
                            None if isinstance(value, str) else value.rect
                            for value in values
                        ],
                        "native_paths": sum(
                            getattr(record, "descriptor", None) is not None
                            for record in selected
                        ),
                        "rng_state": hashlib.sha256(
                            generator.get_state().numpy().tobytes()
                        ).hexdigest(),
                    }
                )
                return values

            return original(records, *args, **kwargs, decoder=decode)

        module.prepare_series = capture
        try:
            return super()._prepare_candidate(candidate, limit, generator)
        finally:
            module.prepare_series = original

    def __getitem__(self, index):
        self.audit_details = []
        value = super().__getitem__(index)
        value["audit"] = self.audit_details
        value["stage_seconds"] = {}
        records = [
            record
            for group in self._slice_index.values()
            for record in getattr(group, "records", group)
        ]
        descriptor_bytes = []
        for record in records:
            descriptor = getattr(record, "descriptor", None)
            if descriptor is not None:
                fields = [getattr(descriptor, name) for name in descriptor.__slots__]
                if any(not isinstance(field, (int, float, bool)) for field in fields):
                    raise AssertionError("Descriptor retained non-scalar state")
                descriptor_bytes.append(
                    sys.getsizeof(descriptor)
                    + sum(sys.getsizeof(field) for field in fields)
                    + 8
                )
        handles = [
            item.path
            for item in psutil.Process().open_files()
            if item.path.endswith(".dcm")
        ]
        if handles:
            raise AssertionError(f"Retained DICOM handles: {handles}")
        value["ownership"] = {
            "pid": os.getpid(),
            "index_builds": self.index_builds,
            "records": len(records),
            "descriptor_count": len(descriptor_bytes),
            "incremental_bytes": sum(descriptor_bytes),
            "max_incremental_bytes_per_record": max(descriptor_bytes, default=0),
            "dicom_handles": handles,
        }
        return value


class AuditedProductionDataset(AuditDatasetMixin, production_dataset.KneeStudyDataset):
    audit_module = production_dataset


class AuditedReferenceDataset(AuditDatasetMixin, REFERENCE_DATASET.KneeStudyDataset):
    audit_module = REFERENCE_DATASET


def make_loader(config, rows, targets, candidate):
    index = build_series_index_raw(config.paths.data_root, config.data, "train")
    if candidate in {"production", "production-reference"}:
        module = production_dataset if candidate == "production" else REFERENCE_DATASET
        dataset_class = module.KneeStudyDataset
        if CAPTURE_AUDIT:
            dataset_class = (
                AuditedProductionDataset
                if candidate == "production"
                else AuditedReferenceDataset
            )
        dataset = dataset_class(
            rows,
            index,
            config.data,
            targets,
            False,
            config.seed,
            config.data.validation_image_multiplier,
        )
        collate = frozen_collate if CAPTURE_AUDIT else module.collate_studies
    else:
        dataset = FrozenDataset(
            rows,
            index,
            config.data,
            targets,
            config.seed,
            config.data.validation_image_multiplier,
            candidate,
        )
        collate = frozen_collate
    loader = DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=WORKERS,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=config.trainer.persistent_workers,
        prefetch_factor=config.trainer.prefetch_factor,
    )
    if loader.num_workers != WORKERS:
        raise RuntimeError(f"Constructed loader has {loader.num_workers} workers")
    if loader.batch_size != config.trainer.batch_size:
        raise RuntimeError(
            "Constructed loader batch size differs from the resolved configuration"
        )
    return loader


def configure(native_library: Path | None, profile: bool):
    global NATIVE_LIBRARY, PROFILE_STAGES
    PROFILE_STAGES = profile
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(THREADS)
    except RuntimeError:
        pass
    if native_library is not None:
        NATIVE_LIBRARY = ctypes.CDLL(str(native_library.resolve()))
        NATIVE_LIBRARY.normalize_rects.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
        ]
        NATIVE_LIBRARY.normalize_rects.restype = ctypes.c_int


def process_cpu(pid):
    try:
        value = psutil.Process(pid).cpu_times()
        return value.user + value.system
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def run_loader(args: BenchmarkCLI, profile: bool):
    if args.candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate {args.candidate}")
    configure(args.native_library, profile)
    config = load_config(args.config)
    rows, targets = load_rows(config, args.fold, args.cohort, args.samples)
    process = psutil.Process()
    parent_cpu_start = process_cpu(process.pid)
    setup_started = time.perf_counter()
    loader = make_loader(config, rows, targets, args.candidate)
    setup_seconds = time.perf_counter() - setup_started
    iterator_started = time.perf_counter()
    iterator = iter(loader)
    iterator_seconds = time.perf_counter() - iterator_started
    worker_pids = [worker.pid for worker in iterator._workers]
    worker_cpu_start = {pid: 0.0 for pid in worker_pids}
    peak_parent_rss = process.memory_info().rss
    peak_worker_rss = 0
    batches = []
    stages = {}
    loading_started = time.perf_counter()
    for batch_index in range(len(loader)):
        wait_started = time.perf_counter()
        batch = next(iterator)
        wait_seconds = time.perf_counter() - wait_started
        for name, seconds in batch.pop("stage_seconds", {}).items():
            stages[name] = stages.get(name, 0.0) + seconds
        peak_parent_rss = max(peak_parent_rss, process.memory_info().rss)
        worker_rss = 0
        for pid in worker_pids:
            try:
                worker_rss += psutil.Process(pid).memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak_worker_rss = max(peak_worker_rss, worker_rss)
        batches.append(
            {
                "batch": batch_index,
                "study_uids": batch["study_uid"],
                "wait_seconds": wait_seconds,
                "series": int(batch["images"].shape[0]),
                "inputs": int(batch["slice_mask"].sum()),
                "payload_bytes": int(batch["images"].numel()),
                "failures": sum(len(value) for value in batch["failures"]),
            }
        )
    loading_seconds = time.perf_counter() - loading_started
    worker_cpu = sum(
        max(0.0, process_cpu(pid) - worker_cpu_start[pid]) for pid in worker_pids
    )
    parent_cpu = max(0.0, process_cpu(process.pid) - parent_cpu_start)
    del iterator, loader
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "candidate": args.candidate,
        "profile_stages": profile,
        "samples": rows.height,
        "batch_size": config.trainer.batch_size,
        "workers": WORKERS,
        "prefetch_factor": config.trainer.prefetch_factor,
        "persistent_workers": config.trainer.persistent_workers,
        "torch_threads": torch.get_num_threads(),
        "setup_seconds": setup_seconds,
        "iterator_seconds": iterator_seconds,
        "loading_seconds": loading_seconds,
        "complete_seconds": setup_seconds + iterator_seconds + loading_seconds,
        "studies_per_second": rows.height / loading_seconds,
        "parent_cpu_seconds": parent_cpu,
        "worker_cpu_seconds": worker_cpu,
        "cpu_seconds_per_study": (parent_cpu + worker_cpu) / rows.height,
        "peak_parent_rss_bytes": peak_parent_rss,
        "peak_aggregate_worker_rss_bytes": peak_worker_rss,
        "stage_seconds": stages,
        "study_uids": [uid for batch in batches for uid in batch["study_uids"]],
        "selected_inputs": sum(batch["inputs"] for batch in batches),
        "series": sum(batch["series"] for batch in batches),
        "payload_bytes": sum(batch["payload_bytes"] for batch in batches),
        "failures": sum(batch["failures"] for batch in batches),
        "batches": batches,
    }


def compare_item(reference, candidate, reference_details, candidate_details):
    discrete = {
        "study_uid": reference["study_uid"] == candidate["study_uid"],
        "is_gold": reference["is_gold"] == candidate["is_gold"],
        "target": torch.equal(reference["target"], candidate["target"]),
        "target_observed": torch.equal(
            reference["target_observed"], candidate["target_observed"]
        ),
        "failures": reference["failures"] == candidate["failures"],
        "slot_counts": [len(value) for value in reference["series"]]
        == [len(value) for value in candidate["series"]],
    }
    changed = 0
    changed_gt1 = 0
    pixels = 0
    maximum = 0
    absolute_sum = 0
    position_equal = True
    uid_equal = True
    for reference_slot, candidate_slot in zip(
        reference["series"], candidate["series"], strict=True
    ):
        for reference_series, candidate_series in zip(
            reference_slot, candidate_slot, strict=True
        ):
            uid_equal &= reference_series.uid == candidate_series.uid
            position_equal &= torch.equal(
                reference_series.positions, candidate_series.positions
            )
            delta = reference_series.images.to(
                torch.int16
            ) - candidate_series.images.to(torch.int16)
            absolute = delta.abs()
            pixels += absolute.numel()
            changed += int((absolute > 0).sum())
            changed_gt1 += int((absolute > 1).sum())
            maximum = max(maximum, int(absolute.max()))
            absolute_sum += int(absolute.sum())
    discrete["series_uids"] = uid_equal
    discrete["positions"] = position_equal
    float_errors = []
    relative_float_errors = []
    bound_errors = []
    relative_bound_errors = []
    geometry_equal = True
    path_equal = True
    for reference_detail, candidate_detail in zip(reference_details, candidate_details):
        path_equal &= reference_detail["paths"] == candidate_detail["paths"]
        geometry_equal &= reference_detail["rects"] == candidate_detail["rects"]
        if (
            "float" in candidate_detail
            and reference_detail["float"].shape == candidate_detail["float"].shape
        ):
            error = np.abs(reference_detail["float"] - candidate_detail["float"])
            float_errors.append(error)
            dynamic_range = (
                reference_detail["bounds"][1] - reference_detail["bounds"][0]
            )
            if math.isfinite(dynamic_range) and dynamic_range > 0:
                relative_float_errors.append(
                    {
                        "max": float(error.max() / dynamic_range),
                        "mean": float(error.mean() / dynamic_range),
                        "p99": float(np.quantile(error, 0.99) / dynamic_range),
                    }
                )
        if "bounds" in candidate_detail:
            for a, b in zip(
                reference_detail["bounds"], candidate_detail["bounds"], strict=True
            ):
                error = abs(a - b)
                bound_errors.append(error)
                dynamic_range = (
                    reference_detail["bounds"][1] - reference_detail["bounds"][0]
                )
                if math.isfinite(dynamic_range) and dynamic_range > 0:
                    relative_bound_errors.append(error / dynamic_range)
    discrete["selected_paths"] = path_equal
    discrete["valid_rectangles"] = geometry_equal
    if float_errors:
        values = np.concatenate([value.reshape(-1) for value in float_errors])
        float_stats = {
            "max": float(values.max()),
            "mean": float(values.mean()),
            "p99": float(np.quantile(values, 0.99)),
        }
    else:
        float_stats = None
    return {
        "discrete": discrete,
        "byte_pixels": pixels,
        "byte_changed": changed,
        "byte_changed_gt1": changed_gt1,
        "byte_max_abs": maximum,
        "byte_mean_abs": absolute_sum / pixels if pixels else 0.0,
        "float_error": float_stats,
        "relative_float_error_worst": {
            name: max((value[name] for value in relative_float_errors), default=0.0)
            for name in ("max", "mean", "p99")
        },
        "bound_max_abs": max(bound_errors, default=0.0),
        "bound_max_relative": max(relative_bound_errors, default=0.0),
    }


def adversarial_indices(index, rows, count):
    scores = []
    for row_index, uid in enumerate(rows[STUDY].to_list()):
        study = index[uid]
        acquisitions = sum(len(slot) for slot in study.slots)
        images = sum(value.image_count for slot in study.slots for value in slot)
        maximum = max(
            (value.image_count for slot in study.slots for value in slot), default=0
        )
        scores.append((acquisitions, images, maximum, row_index))
    selected = []
    for axis in range(3):
        for value in sorted(scores, key=lambda item: item[axis], reverse=True):
            if value[3] not in selected:
                selected.append(value[3])
            if len(selected) >= count:
                return selected
    return selected


def format_audit(paths: list[str], spacing_mm: float, image_size: int):
    transfer_syntaxes = {}
    photometric = {}
    signed = 0
    non_identity_affine = 0
    shapes = {}
    target_area = 0
    retained_area = 0
    errors = []
    tags = [
        "BitsAllocated",
        "BitsStored",
        "Columns",
        "PhotometricInterpretation",
        "PixelRepresentation",
        "PixelSpacing",
        "RescaleIntercept",
        "RescaleSlope",
        "Rows",
        "SamplesPerPixel",
    ]
    for value in sorted(set(paths)):
        try:
            dataset = pydicom.dcmread(
                value, stop_before_pixels=True, specific_tags=tags
            )
            syntax = str(getattr(dataset.file_meta, "TransferSyntaxUID", "missing"))
            transfer_syntaxes[syntax] = transfer_syntaxes.get(syntax, 0) + 1
            photo = str(getattr(dataset, "PhotometricInterpretation", "missing"))
            photometric[photo] = photometric.get(photo, 0) + 1
            signed += int(getattr(dataset, "PixelRepresentation", 0) == 1)
            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            non_identity_affine += int(slope != 1.0 or intercept != 0.0)
            rows = int(dataset.Rows)
            cols = int(dataset.Columns)
            shape = f"{rows}x{cols}"
            shapes[shape] = shapes.get(shape, 0) + 1
            pixel_spacing = dataset.PixelSpacing
            target_rows = max(1, round(rows * float(pixel_spacing[0]) / spacing_mm))
            target_cols = max(1, round(cols * float(pixel_spacing[1]) / spacing_mm))
            target_area += target_rows * target_cols
            retained_area += min(target_rows, image_size) * min(target_cols, image_size)
        except PREPARATION_ERRORS as error:
            errors.append(f"{value}: {type(error).__name__}: {error}")
    return {
        "selected_paths": len(set(paths)),
        "transfer_syntaxes": transfer_syntaxes,
        "photometric_interpretations": photometric,
        "signed_paths": signed,
        "non_identity_affine_paths": non_identity_affine,
        "source_shapes": shapes,
        "full_target_area": target_area,
        "retained_valid_area": retained_area,
        "retained_fraction": retained_area / target_area if target_area else None,
        "errors": errors,
    }


def assert_item_output_equal(reference: dict, candidate: dict) -> None:
    if (
        reference["study_uid"] != candidate["study_uid"]
        or reference["is_gold"] != candidate["is_gold"]
        or reference["failures"] != candidate["failures"]
        or not torch.equal(reference["target"], candidate["target"])
        or not torch.equal(reference["target_observed"], candidate["target_observed"])
        or [len(value) for value in reference["series"]]
        != [len(value) for value in candidate["series"]]
    ):
        raise AssertionError("Production and experimental NumPy metadata differ")
    for reference_slot, candidate_slot in zip(
        reference["series"], candidate["series"], strict=True
    ):
        for reference_series, candidate_series in zip(
            reference_slot, candidate_slot, strict=True
        ):
            if reference_series.uid != candidate_series.uid or not torch.equal(
                reference_series.positions, candidate_series.positions
            ):
                raise AssertionError(
                    "Production and experimental NumPy series metadata differ"
                )
            np.testing.assert_array_equal(
                reference_series.images.numpy(), candidate_series.images.numpy()
            )


def audit(args: BenchmarkCLI):
    global CAPTURE_AUDIT
    CAPTURE_AUDIT = True
    configure(args.native_library, False)
    config = load_config(args.config)
    timing_rows, targets = load_rows(config, args.fold, args.cohort, args.samples)
    all_rows = load_fold_labels(config, args.fold).validation
    index = build_series_index_raw(config.paths.data_root, config.data, "train")
    representative_uids = timing_rows[STUDY].to_list()[:12]
    adversarial = adversarial_indices(index, all_rows, 12)
    adversarial_uids = all_rows[adversarial][STUDY].to_list()
    selected_uids = list(dict.fromkeys(representative_uids + adversarial_uids))
    by_uid = {row[STUDY]: row for row in all_rows.iter_rows(named=True)}
    rows = pl.DataFrame([by_uid[uid] for uid in selected_uids], schema=all_rows.schema)
    reference_dataset = FrozenDataset(
        rows,
        index,
        config.data,
        targets,
        config.seed,
        config.data.validation_image_multiplier,
        "baseline",
    )
    references = []
    selected_paths = []
    for row_index in range(rows.height):
        AUDIT_SERIES.clear()
        item = reference_dataset[row_index]
        details = list(AUDIT_SERIES)
        selected_paths.extend(path for detail in details for path in detail["paths"])
        references.append((item, details))
    records = {}
    candidate_names = (
        CANDIDATES[1:] if args.candidate == "baseline" else (args.candidate,)
    )
    for candidate_name in candidate_names:
        comparisons = []
        candidate_dataset = FrozenDataset(
            rows,
            index,
            config.data,
            targets,
            config.seed,
            config.data.validation_image_multiplier,
            candidate_name,
        )
        experimental_numpy_dataset = (
            FrozenDataset(
                rows,
                index,
                config.data,
                targets,
                config.seed,
                config.data.validation_image_multiplier,
                "combined-numpy",
            )
            if candidate_name == "production"
            else None
        )
        for row_index, (reference, reference_details) in enumerate(references):
            AUDIT_SERIES.clear()
            candidate = candidate_dataset[row_index]
            candidate_details = list(AUDIT_SERIES)
            comparison = compare_item(
                reference, candidate, reference_details, candidate_details
            )
            if experimental_numpy_dataset is not None:
                AUDIT_SERIES.clear()
                experimental_numpy = experimental_numpy_dataset[row_index]
                assert_item_output_equal(candidate, experimental_numpy)
                comparison["experimental_numpy_exact"] = True
            if not all(comparison["discrete"].values()):
                raise AssertionError(
                    f"{candidate_name} changed a discrete invariant for {reference['study_uid']}"
                )
            for reference_detail, candidate_detail in zip(
                reference_details, candidate_details, strict=True
            ):
                if "float" in candidate_detail:
                    np.testing.assert_allclose(
                        reference_detail["float"],
                        candidate_detail["float"],
                        rtol=1e-5,
                        atol=0.06,
                    )
                if "bounds" in candidate_detail:
                    np.testing.assert_allclose(
                        reference_detail["bounds"],
                        candidate_detail["bounds"],
                        rtol=1e-5,
                        atol=0.01,
                    )
            for reference_slot, candidate_slot in zip(
                reference["series"], candidate["series"], strict=True
            ):
                for reference_series, candidate_series in zip(
                    reference_slot, candidate_slot, strict=True
                ):
                    np.testing.assert_allclose(
                        reference_series.images.numpy(),
                        candidate_series.images.numpy(),
                        rtol=0,
                        atol=1,
                    )
                    if candidate_name == "numpy-native":
                        np.testing.assert_array_equal(
                            reference_series.images.numpy(),
                            candidate_series.images.numpy(),
                        )
            comparison["study_uid"] = reference["study_uid"]
            comparisons.append(comparison)
        records[candidate_name] = comparisons
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "representative_study_uids": representative_uids,
        "adversarial_study_uids": adversarial_uids,
        "study_uids": selected_uids,
        "format_coverage": format_audit(
            selected_paths, config.data.spacing_mm, config.data.image_size
        ),
        "candidates": records,
    }


def model_run(args: BenchmarkCLI):
    if args.checkpoint is None:
        raise ValueError("Model mode requires --checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    configure(args.native_library, False)
    config = load_config(args.config)
    rows, targets = load_rows(config, args.fold, args.cohort, args.samples)
    loader = make_loader(config, rows, targets, args.candidate)
    device = torch.device("cuda")
    started = time.perf_counter()
    model = load_checkpoint_model(
        config, targets, args.checkpoint, device, expected_fold=args.fold
    )
    model.eval()
    startup_seconds = time.perf_counter() - started
    logits = []
    predictions = []
    study_uids = []
    torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        for batch in loader:
            batch.pop("stage_seconds")
            study_uids.extend(batch["study_uid"])
            moved = move_batch(batch, device)
            output = model(moved).float().cpu()
            logits.append(output)
            predictions.append(torch.sigmoid(output))
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    values = torch.cat(predictions).numpy()
    logit_values = torch.cat(logits).numpy()
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "candidate": args.candidate,
        "samples": rows.height,
        "batch_size": config.trainer.batch_size,
        "workers": WORKERS,
        "precision": config.trainer.precision,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "startup_seconds": startup_seconds,
        "inference_seconds": inference_seconds,
        "complete_seconds": startup_seconds + inference_seconds,
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        "study_uids": study_uids,
        "logits": logit_values.tolist(),
        "probabilities": values.tolist(),
        "targets": list(targets),
    }


def compile_native(output_dir: Path):
    source = output_dir / "native_normalize.c"
    library = output_dir / "native_normalize.so"
    source.write_text(NATIVE_SOURCE)
    command = [
        "gcc",
        "-O3",
        "-march=native",
        "-fPIC",
        "-shared",
        "-fno-fast-math",
        "-ffp-contract=off",
        str(source),
        "-o",
        str(library),
        "-lm",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    seconds = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return library, command, seconds


def environment_record(config: ExperimentConfig):
    root = Path.cwd()
    packages = {}
    for distribution in ("numpy", "scipy", "torch", "pydicom", "polars", "psutil"):
        packages[distribution] = importlib.metadata.version(distribution)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "status", "--porcelain=v2", "--untracked-files=all"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines(),
        "source_hashes": {
            path: sha256(root / path) for path in CORE_FILES if (root / path).exists()
        },
        "script_sha256": sha256(Path(__file__)),
        "config": config.model_dump(mode="json"),
        "packages": packages,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def child_command(
    args, output, candidate, mode, config, cohort, library, checkpoint=None
):
    command = [
        sys.executable,
        "-m",
        "scripts.adhoc.benchmark_dataloader_optimizations",
        str(config),
        "--output-dir",
        str(output),
        "--mode",
        mode,
        "--candidate",
        candidate,
        "--cohort",
        str(cohort),
        "--samples",
        str(args.samples),
        "--fold",
        str(args.fold),
    ]
    if library is not None:
        command.extend(("--native-library", str(library)))
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    return command


def run_child(command, environment):
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False, env=environment
    )
    if completed.returncode:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return {"status": "ok", "stdout": completed.stdout, "stderr": completed.stderr}


def paired_summary(records):
    by_candidate = {}
    for candidate in CANDIDATES[1:]:
        pairs = []
        for repeat in sorted({row["repeat"] for row in records}):
            baseline = next(
                (
                    row
                    for row in records
                    if row["repeat"] == repeat
                    and row["candidate"] == "baseline"
                    and row.get("paired_with") == candidate
                    and row["status"] == "ok"
                ),
                None,
            )
            value = next(
                (
                    row
                    for row in records
                    if row["repeat"] == repeat
                    and row["candidate"] == candidate
                    and row["status"] == "ok"
                ),
                None,
            )
            if baseline and value:
                pairs.append(
                    {
                        "repeat": repeat,
                        "baseline_loading_seconds": baseline["loading_seconds"],
                        "candidate_loading_seconds": value["loading_seconds"],
                        "loading_ratio": baseline["loading_seconds"]
                        / value["loading_seconds"],
                        "baseline_complete_seconds": baseline["complete_seconds"],
                        "candidate_complete_seconds": value["complete_seconds"],
                        "complete_ratio": baseline["complete_seconds"]
                        / value["complete_seconds"],
                        "cpu_ratio": baseline["cpu_seconds_per_study"]
                        / value["cpu_seconds_per_study"],
                    }
                )
        ratios = [value["loading_ratio"] for value in pairs]
        by_candidate[candidate] = {
            "pairs": pairs,
            "median_loading_ratio": float(np.median(ratios)) if ratios else None,
            "min_loading_ratio": min(ratios, default=None),
            "max_loading_ratio": max(ratios, default=None),
        }
    return by_candidate


def suite(args: BenchmarkCLI):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    config = load_config(args.config)
    if config.trainer.batch_size != 8 or config.trainer.num_workers != WORKERS:
        raise ValueError(
            "Resolved configuration must specify batch size 8 and 8 workers"
        )
    rows, targets = load_rows(config, args.fold, None, args.samples)
    cohort = args.output_dir / "timing_cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "seed": config.seed,
                "fold": args.fold,
                "study_uids": rows[STUDY].to_list(),
                "target_names": list(targets),
                "row_count": rows.height,
            },
            indent=2,
        )
        + "\n"
    )
    source_dir = args.output_dir / "source"
    source_dir.mkdir()
    for path in CORE_FILES:
        source = Path(path)
        if source.exists():
            target = source_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copy2(Path(__file__), source_dir / "benchmark_dataloader_optimizations.py")
    shutil.copy2(args.config, source_dir / "effective_input_config.yaml")
    manifest = environment_record(config)
    manifest["cohort_sha256"] = sha256(cohort)
    manifest["fixed_conditions"] = {
        "workers": WORKERS,
        "batch_size": config.trainer.batch_size,
        "prefetch_factor": config.trainer.prefetch_factor,
        "persistent_workers": config.trainer.persistent_workers,
        "pin_memory": True,
        "torch_threads": THREADS,
        "raw_mount_without_count_parquet": True,
    }
    library, compile_command, compile_seconds = compile_native(args.output_dir)
    manifest["native_build"] = {
        "command": compile_command,
        "seconds": compile_seconds,
        "source_sha256": sha256(args.output_dir / "native_normalize.c"),
        "library_sha256": sha256(library),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = str(THREADS)
    records = []
    order_index = 0
    for candidate in CANDIDATES[1:]:
        for repeat in range(args.repeats):
            order = (
                ("baseline", candidate) if repeat % 2 == 0 else (candidate, "baseline")
            )
            for value in order:
                output = (
                    args.output_dir
                    / f"timing-{order_index:02d}-{value}-pair-{candidate}-repeat-{repeat}.json"
                )
                command = child_command(
                    args, output, value, "run", args.config, cohort, library
                )
                status = run_child(command, environment)
                if status["status"] == "ok":
                    record = json.loads(output.read_text())
                    record.update(
                        {
                            "status": "ok",
                            "repeat": repeat,
                            "paired_with": candidate,
                            "order": order_index,
                        }
                    )
                else:
                    record = {
                        **status,
                        "candidate": value,
                        "repeat": repeat,
                        "paired_with": candidate,
                        "order": order_index,
                    }
                records.append(record)
                order_index += 1
                (args.output_dir / "timing_progress.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        {
                            key: record.get(key)
                            for key in (
                                "candidate",
                                "paired_with",
                                "repeat",
                                "status",
                                "loading_seconds",
                                "cpu_seconds_per_study",
                            )
                        }
                    ),
                    flush=True,
                )
    (args.output_dir / "timing_results.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    summary = paired_summary(records)
    (args.output_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    audit_output = args.output_dir / "correctness.json"
    audit_command = child_command(
        args, audit_output, "baseline", "audit", args.config, cohort, library
    )
    audit_status = run_child(audit_command, environment)
    profile_records = []
    for candidate in CANDIDATES:
        output = args.output_dir / f"profile-{candidate}.json"
        command = child_command(
            args, output, candidate, "profile", args.config, cohort, library
        )
        status = run_child(command, environment)
        profile_records.append({"candidate": candidate, **status})
    (args.output_dir / "profile_status.json").write_text(
        json.dumps(profile_records, indent=2) + "\n"
    )
    final = {
        "timing": summary,
        "correctness_status": audit_status,
        "profile_status": profile_records,
    }
    (args.output_dir / "suite_status.json").write_text(
        json.dumps(final, indent=2) + "\n"
    )


def frame_values(frame):
    return (
        frame.pixels,
        frame.row_spacing,
        frame.column_spacing,
        frame.slope,
        frame.intercept,
    )


def dicom_edge_audit(output_dir: Path, data: DataConfig):
    evidence = Path("outputs/profiling/dicom-reader-20260912")
    shutil.copytree(evidence / "corrected-edge-fixtures", output_dir)
    expectations = json.loads((evidence / "corrected-edge-audit.json").read_text())
    cases = {}
    for case in expectations:
        name = case["case"]
        directory = output_dir / (Path(name).stem if name.endswith(".dcm") else name)
        path = directory / "1.dcm"
        if sha256(path) != case["sha256"]:
            raise AssertionError(f"Frozen edge fixture changed: {name}")
        cases[name] = (path, case["descriptor_valid"])
    for name in ("changed-size", "changed-mtime", "invalid-representation"):
        directory = output_dir / name
        shutil.copytree(output_dir / "rescale", directory)
        cases[name] = (directory / "1.dcm", name != "invalid-representation")
    results = []
    for name, (path, expected_descriptor) in cases.items():
        if name == "invalid-representation":
            dataset = pydicom.dcmread(path)
            dataset.PixelRepresentation = 2
            dataset.save_as(path, enforce_file_format=True)
        reference_records = REFERENCE_PREPROCESSING.index_series(
            path.parent, data.position_reference
        )
        records = production_dataset.index_series(
            path.parent, data.position_reference
        ).records
        ordering = lambda values: [
            (r.path, r.physical_position, r.original_index, r.relative_position_mm)
            for r in values
        ]
        if ordering(records) != ordering(reference_records):
            raise AssertionError(f"Ordering changed for {name}")
        record = next(r for r in records if r.path == path)
        if (record.descriptor is not None) != expected_descriptor:
            raise AssertionError(f"Unexpected descriptor for {name}")
        if name in {"changed-size", "changed-mtime"}:
            before = path.stat()
            dataset = pydicom.dcmread(path)
            dataset.RescaleIntercept = 12345.25 if name == "changed-size" else 14.25
            pixels = np.frombuffer(dataset.PixelData, dtype="<i2").copy()
            pixels[0] += 1
            dataset.PixelData = pixels.tobytes()
            dataset.save_as(path, enforce_file_format=True)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            if (path.stat().st_size == before.st_size) != (name == "changed-mtime"):
                raise AssertionError("Freshness fixture has incorrect size")
        try:
            reference_pixels = read_pixels(path)
            reference_error = None
        except PREPARATION_ERRORS as error:
            reference_pixels = None
            reference_error = f"{type(error).__name__}: {error}"
        fallback_calls = []
        original = production_dicom._read_pydicom_frame

        def fallback(selected_path, fallback_calls=fallback_calls, original=original):
            fallback_calls.append(str(selected_path))
            return original(selected_path)

        production_dicom._read_pydicom_frame = fallback
        try:
            try:
                pixels = frame_values(
                    production_dicom.read_frame(path, record.descriptor)
                )
                pixel_error = None
            except PREPARATION_ERRORS as error:
                pixels = None
                pixel_error = f"{type(error).__name__}: {error}"
        finally:
            production_dicom._read_pydicom_frame = original
        expected_fallback = not expected_descriptor or name in {
            "changed-size",
            "changed-mtime",
        }
        if len(fallback_calls) != int(expected_fallback):
            raise AssertionError(f"Unexpected fallback dispatch for {name}")
        if pixel_error != reference_error:
            raise AssertionError(
                f"Pixel failure changed for {name}: {pixel_error} != {reference_error}"
            )
        if pixels is not None:
            np.testing.assert_allclose(reference_pixels[0], pixels[0], rtol=0, atol=0)
            np.testing.assert_array_equal(reference_pixels[1:], pixels[1:])
        outcomes = []
        for module, selected in (
            (REFERENCE_PREPROCESSING, reference_records),
            (production_preprocessing, records),
        ):
            try:
                series = module.prepare_series(
                    selected,
                    data.spacing_mm,
                    data.image_size,
                    data.lower_quantile,
                    data.upper_quantile,
                    data.position_encoding,
                    data.slices_per_input,
                )
                outcomes.append((series, None))
            except PREPARATION_ERRORS as error:
                outcomes.append((None, f"{type(error).__name__}: {error}"))
        (reference_series, reference_series_error), (series, series_error) = outcomes
        if series_error != reference_series_error:
            raise AssertionError(f"Group failure changed for {name}: {outcomes}")
        if series is not None:
            np.testing.assert_allclose(
                reference_series.images.numpy(), series.images.numpy(), rtol=0, atol=0
            )
            np.testing.assert_array_equal(
                reference_series.positions.numpy(), series.positions.numpy()
            )
            if (reference_series.failures, reference_series.constant) != (
                series.failures,
                series.constant,
            ):
                raise AssertionError(f"Group metadata changed for {name}")
        results.append(
            {
                "case": name,
                "candidate": "production",
                "sha256": sha256(path),
                "descriptor_valid": record.descriptor is not None,
                "fallback_calls": len(fallback_calls),
                "pixel_error": pixel_error,
                "group_error": series_error,
                "exact": True,
            }
        )
    return results


def assert_batch_equal(reference, value):
    for key in reference:
        if key in {"audit", "stage_seconds", "ownership"}:
            continue
        if isinstance(reference[key], torch.Tensor):
            left, right = reference[key], value[key]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise AssertionError(f"Production changed {key} shape or dtype")
            left, right = left.numpy().reshape(-1), right.numpy().reshape(-1)
            for start in range(0, left.size, 1_000_000):
                np.testing.assert_allclose(
                    left[start : start + 1_000_000],
                    right[start : start + 1_000_000],
                    rtol=0,
                    atol=0,
                )
        elif reference[key] != value[key]:
            raise AssertionError(f"Production changed {key}")


def dicom_audit(args: BenchmarkCLI):
    global CAPTURE_AUDIT
    configure(None, False)
    config = load_config(args.config)
    rows, targets = load_rows(config, args.fold, args.cohort, args.samples)
    rows = rows.head(args.audit_samples)
    if (
        rows.height != 8
        or config.trainer.batch_size != 8
        or not config.trainer.persistent_workers
    ):
        raise ValueError(
            "Integration audit requires eight studies in one persistent-worker batch"
        )
    CAPTURE_AUDIT = True
    reference_loader = make_loader(config, rows, targets, "production-reference")
    references = list(reference_loader)
    del reference_loader
    results = {}
    for candidate in ("production",):
        loader = make_loader(config, rows, targets, candidate)
        paths = []
        descriptor_count = 0
        byte_pixels = 0
        for reference, value in zip(references, loader, strict=True):
            assert_batch_equal(reference, value)
            for expected, actual in zip(
                reference["audit"], value["audit"], strict=True
            ):
                if any(
                    expected[key] != actual[key]
                    for key in ("paths", "rects", "records", "rng_state")
                ):
                    raise AssertionError(
                        f"{candidate} changed selected paths or geometry"
                    )
                paths.extend(actual["paths"])
                descriptor_count += actual["native_paths"]
            byte_pixels += value["images"].numel()
        first_ownership = {item["pid"]: item for item in value["ownership"]}
        del value
        gc.collect()
        lifetime = []
        for pass_index in range(2):
            if pass_index:
                for reference, value in zip(references, loader, strict=True):
                    assert_batch_equal(reference, value)
                    if first_ownership != {
                        item["pid"]: item for item in value["ownership"]
                    }:
                        raise AssertionError(
                            "Repeated loading rebuilt index or changed ownership"
                        )
                del value
                gc.collect()
            workers = [
                psutil.Process(worker.pid) for worker in loader._iterator._workers
            ]
            handles = [
                [
                    item.path
                    for item in worker.open_files()
                    if item.path.endswith(".dcm")
                ]
                for worker in workers
            ]
            if any(handles):
                raise AssertionError("Repeated loader retained DICOM handles")
            lifetime.append(
                {
                    "pass": pass_index + 1,
                    "worker_fds": [worker.num_fds() for worker in workers],
                    "dicom_handles": handles,
                }
            )
        full_files_bound = sum(
            max(0, candidate.image_count) + 1
            for study in loader.dataset.series_index.values()
            for slot in study.slots
            for candidate in slot
        )
        per_record = max(
            (
                item["max_incremental_bytes_per_record"]
                for item in first_ownership.values()
            ),
            default=0,
        )
        del loader
        results[candidate] = {
            "byte_pixels": byte_pixels,
            "byte_changed": 0,
            "selected_paths": paths,
            "descriptor_paths": descriptor_count,
            "fallback_paths": len(paths) - descriptor_count,
            "ownership": first_ownership,
            "lifetime": lifetime,
            "full_worker_index_file_bound": full_files_bound,
            "conservative_incremental_bytes_per_worker": full_files_bound * per_record,
        }
    CAPTURE_AUDIT = False
    training = training_audit(config, rows.head(2), targets)
    edges = (
        dicom_edge_audit(args.output_dir.with_suffix(".fixtures"), config.data)
        if args.mode != "reference-audit"
        else []
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "workers": WORKERS,
        "batch_size": config.trainer.batch_size,
        "study_uids": rows[STUDY].to_list(),
        "candidates": results,
        "edge_cases": edges,
        "training": training,
        "environment": environment_record(config),
        "config_sha256": sha256(args.config),
        "format_coverage": format_audit(
            paths, config.data.spacing_mm, config.data.image_size
        ),
    }


def training_audit(config, rows, targets):
    index = build_series_index_raw(config.paths.data_root, config.data, "train")
    reference = AuditedReferenceDataset(rows, index, config.data, targets, True, 42, 1)
    production = AuditedProductionDataset(
        rows, index, config.data, targets, True, 42, 1
    )
    results = []
    for epoch in (0, 1):
        reference.set_epoch(epoch)
        production.set_epoch(epoch)
        for i in range(len(reference)):
            expected, actual = reference[i], production[i]
            assert_item_output_equal(expected, actual)
            for left, right in zip(expected["audit"], actual["audit"], strict=True):
                if any(
                    left[key] != right[key]
                    for key in ("paths", "rects", "records", "rng_state")
                ):
                    raise AssertionError("Training selection or RNG changed")
            tensors = []
            for slot in actual["series"]:
                for series in slot:
                    tensors.append(
                        {
                            "uid": series.uid,
                            "shape": list(series.images.shape),
                            "dtype": str(series.images.dtype),
                            "sha256": hashlib.sha256(
                                series.images.numpy().tobytes()
                            ).hexdigest(),
                        }
                    )
            results.append(
                {
                    "epoch": epoch,
                    "study_uid": actual["study_uid"],
                    "tensors": tensors,
                    "selection": actual["audit"],
                }
            )
    return results


def dicom_suite(args: BenchmarkCLI):
    config = load_config(args.config)
    rows, targets = load_rows(config, args.fold, args.cohort, args.samples)
    if (
        rows.height != 128
        or args.repeats != 3
        or config.seed != 42
        or args.fold != 0
        or config.trainer.batch_size != 8
        or not config.trainer.persistent_workers
        or config.trainer.prefetch_factor != 2
        or args.audit_samples != 8
    ):
        raise ValueError(
            "Integration timings require the frozen 128-study workload and three pairs"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cohort = args.output_dir / "timing_cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "seed": config.seed,
                "fold": args.fold,
                "study_uids": rows[STUDY].to_list(),
                "target_names": list(targets),
            },
            indent=2,
        )
        + "\n"
    )
    source_dir = args.output_dir / "source"
    for name in (*CORE_FILES, str(Path(__file__).relative_to(Path.cwd()))):
        source = Path(name)
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = environment_record(config)
    shutil.copy2(args.config, args.output_dir / "workload.yaml")
    manifest.update(
        {
            "cohort_sha256": sha256(cohort),
            "config_sha256": sha256(args.config),
            "reference": "production-reference",
            "workers": WORKERS,
            "batch_size": config.trainer.batch_size,
            "native_build_seconds": 0.0,
            "page_cache": "Uncontrolled OS page cache; fresh application process per run",
        }
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = str(THREADS)
    audit_output = args.output_dir / "correctness.json"
    command = child_command(
        args,
        audit_output,
        "production",
        "dicom-audit",
        args.config,
        Path("outputs/profiling/dicom-reader-20260912/comparison-v1/correctness.json"),
        None,
    )
    command.extend(("--audit-samples", str(args.audit_samples)))
    status = run_child(command, environment)
    if status["status"] != "ok":
        (args.output_dir / "audit_failure.json").write_text(
            json.dumps(status, indent=2) + "\n"
        )
        raise RuntimeError(status["stderr"])
    print(json.dumps({"audit": "passed", "samples": args.audit_samples}), flush=True)
    records = []
    for candidate in ("production",):
        for repeat in range(args.repeats):
            order = (
                ("production-reference", candidate)
                if repeat % 2 == 0
                else (candidate, "production-reference")
            )
            for value in order:
                output = args.output_dir / f"timing-{candidate}-{repeat}-{value}.json"
                command = child_command(
                    args, output, value, "run", args.config, cohort, None
                )
                started = time.perf_counter()
                status = run_child(command, environment)
                if status["status"] != "ok":
                    raise RuntimeError(status["stderr"])
                record = json.loads(output.read_text())
                record.update(
                    {
                        "repeat": repeat,
                        "paired_with": candidate,
                        "process_seconds": time.perf_counter() - started,
                    }
                )
                records.append(record)
                (args.output_dir / "timing_results.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in (
                                "candidate",
                                "repeat",
                                "loading_seconds",
                                "cpu_seconds_per_study",
                            )
                        }
                    ),
                    flush=True,
                )
    summary = {}
    for candidate in ("production",):
        pairs = []
        for repeat in range(args.repeats):
            pair = {
                r["candidate"]: r
                for r in records
                if r["repeat"] == repeat and r["paired_with"] == candidate
            }
            for key in (
                "study_uids",
                "selected_inputs",
                "series",
                "payload_bytes",
                "failures",
            ):
                if pair["production-reference"][key] != pair[candidate][key]:
                    raise AssertionError(f"{candidate} changed timed workload {key}")
            pairs.append(
                {
                    metric: pair["production-reference"][metric]
                    / pair[candidate][metric]
                    for metric in (
                        "loading_seconds",
                        "complete_seconds",
                        "cpu_seconds_per_study",
                        "process_seconds",
                    )
                }
            )
        summary[candidate] = {
            "pairs": pairs,
            "median_ratios": {
                key: float(np.median([p[key] for p in pairs])) for key in pairs[0]
            },
        }
    ratios = summary["production"]
    passed = (
        all(pair["loading_seconds"] > 1 for pair in ratios["pairs"])
        and ratios["median_ratios"]["complete_seconds"] > 1
        and ratios["median_ratios"]["cpu_seconds_per_study"] > 1
    )
    summary["performance_gate_passed"] = passed
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not passed:
        raise RuntimeError(
            "Production performance gate failed; retain all pairs and investigate before promotion"
        )


def main():
    args = BenchmarkCLI()
    if args.mode == "dicom-suite":
        dicom_suite(args)
        return
    if args.mode == "suite":
        suite(args)
        return
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.mode in {"dicom-audit", "reference-audit"}:
        result = dicom_audit(args)
    elif args.mode == "audit":
        result = audit(args)
    elif args.mode == "model":
        result = model_run(args)
    else:
        result = run_loader(args, args.mode == "profile")
    args.output_dir.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "candidate",
                    "samples",
                    "loading_seconds",
                    "complete_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
