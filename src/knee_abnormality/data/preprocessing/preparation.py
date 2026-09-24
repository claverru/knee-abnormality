import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from knee_abnormality.data.preprocessing.dicom import (
    PREPARATION_ERRORS,
    NativePixelDescriptor,
    io_pool,
    mask_stored_bits,
    read_frame,
    read_payload_into,
)
from knee_abnormality.data.preprocessing.geometry import RELIABILITY
from knee_abnormality.data.preprocessing.intensity import quantile_ranks
from knee_abnormality.data.preprocessing.series import SliceRecord
from knee_abnormality.data.preprocessing.spatial import (
    ResamplePlan,
    resample_crop,
    resample_plan,
    resample_targets,
)

NATIVE_DTYPES = frozenset(
    {np.dtype(np.uint8), np.dtype(np.int8), np.dtype(np.uint16), np.dtype(np.int16)}
)
NATIVE_RESCALE_LIMIT = 1e30
NEAR_EMPTY_FRACTION = 0.01


@dataclass(frozen=True)
class PreparedFrame:
    pixels: torch.Tensor
    offset: int
    shape: tuple[int, int]
    columns: tuple[int, int]
    signed: bool
    plan_key: tuple[int, int, int, int, int] | None
    slope: float
    intercept: float
    rect: tuple[int, int, int, int]
    output_to_native: np.ndarray


def shared_empty(count: int, dtype: torch.dtype) -> torch.Tensor:
    storage = torch.UntypedStorage._new_shared(
        count * torch.empty((), dtype=dtype).element_size()
    )
    return torch.empty(0, dtype=dtype).set_(storage, 0, (count,))


@dataclass(frozen=True)
class PreparedSeries:
    frames: tuple[PreparedFrame, ...]
    gather: np.ndarray
    quantile_ranks: tuple[np.ndarray, np.ndarray, np.ndarray]
    positions: torch.Tensor
    failures: tuple[str, ...]
    patient_maps: torch.Tensor
    valid_rects: torch.Tensor
    geometry_valid: torch.Tensor
    selected_paths: tuple[tuple[str, ...], ...]
    uid: str = ""
    failed_paths: tuple[Path, ...] = ()
    unique_slice_count: int = 0
    intensity_pixel_count: int = 0
    sampling_path: str = "physical"
    reflection_eligible: bool = False
    reflection_reason: str = "unavailable"
    knee_side_sign: float = 0.0
    knee_center_x_mm: float = 0.0

    @property
    def triplet_count(self) -> int:
        return len(self.gather) // 3

    @property
    def slice_count(self) -> int:
        return len(self.gather)


DecodedSlice = PreparedFrame | str
SeriesDecoder = Callable[
    [tuple[SliceRecord, ...], float, int], tuple[DecodedSlice, ...]
]


def _native_rescale(slope: float, intercept: float) -> bool:
    return all(
        math.isfinite(value) and abs(value) <= NATIVE_RESCALE_LIMIT
        for value in (slope, intercept)
    )


def _native_plan(
    rows: int,
    columns: int,
    row_spacing: float,
    column_spacing: float,
    slope: float,
    intercept: float,
    spacing_mm: float,
    image_size: int,
) -> ResamplePlan | None:
    if not _native_rescale(slope, intercept):
        return None
    return resample_plan(
        rows,
        columns,
        *resample_targets(rows, columns, row_spacing, column_spacing, spacing_mm),
        image_size,
    )


def _descriptor_plan(
    descriptor: NativePixelDescriptor | None, spacing_mm: float, image_size: int
) -> ResamplePlan | None:
    if descriptor is None:
        return None
    return _native_plan(
        descriptor.rows,
        descriptor.columns,
        descriptor.row_spacing,
        descriptor.column_spacing,
        descriptor.slope,
        descriptor.intercept,
        spacing_mm,
        image_size,
    )


def _stored_frame(
    rows: np.ndarray, plan: ResamplePlan, slope: float, intercept: float
) -> PreparedFrame:
    signed = rows.dtype.kind == "i"
    stored = np.array(rows, dtype=np.int16 if signed else np.uint16)
    return PreparedFrame(
        torch.from_numpy(stored.reshape(-1).view(np.int16)),
        0,
        stored.shape,
        (plan.column_start, plan.column_stop),
        signed,
        plan.key,
        slope,
        intercept,
        plan.rect,
        plan.output_to_native,
    )


def decode_slice(
    record: SliceRecord, spacing_mm: float, image_size: int
) -> DecodedSlice:
    path = record.path
    try:
        frame = read_frame(path, record.descriptor)
        if np.count_nonzero(frame.pixels) < NEAR_EMPTY_FRACTION * frame.pixels.size:
            return f"{path}: near-empty frame"
        plan = (
            _native_plan(
                *frame.pixels.shape,
                frame.row_spacing,
                frame.column_spacing,
                frame.slope,
                frame.intercept,
                spacing_mm,
                image_size,
            )
            if frame.pixels.dtype in NATIVE_DTYPES
            else None
        )
        if plan is not None:
            return _stored_frame(
                frame.pixels[plan.row_start : plan.row_stop],
                plan,
                frame.slope,
                frame.intercept,
            )
        value = resample_crop(frame, spacing_mm, image_size)
    except PREPARATION_ERRORS as error:
        return f"{path}: {type(error).__name__}: {error}"
    top, left, rows, columns = value.rect
    region = value.image[top : top + rows, left : left + columns]
    if not np.isfinite(region).all():
        return f"{path}: non-finite pixels"
    return PreparedFrame(
        torch.from_numpy(np.ascontiguousarray(region).reshape(-1)),
        0,
        region.shape,
        (0, columns),
        False,
        None,
        1.0,
        0.0,
        value.rect,
        value.output_to_native,
    )


def _read_rows(
    record: SliceRecord, plan: ResamplePlan | None, target: np.ndarray
) -> bool:
    if plan is None:
        return False
    try:
        return read_payload_into(record.path, record.descriptor, plan.row_start, target)
    except PREPARATION_ERRORS:
        return False


def decode_series(
    records: tuple[SliceRecord, ...],
    spacing_mm: float,
    image_size: int,
    threads: int = 1,
) -> tuple[DecodedSlice, ...]:
    plans = [
        _descriptor_plan(record.descriptor, spacing_mm, image_size)
        for record in records
    ]
    sizes = [
        (plan.row_stop - plan.row_start) * record.descriptor.columns
        if plan is not None
        else 0
        for record, plan in zip(records, plans, strict=True)
    ]
    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    pixels = shared_empty(int(offsets[-1]), torch.int16)
    values = pixels.numpy()
    targets = [
        (values if plan is None or record.descriptor.signed else values.view("u2"))[
            offsets[index] : offsets[index + 1]
        ]
        for index, (record, plan) in enumerate(zip(records, plans, strict=True))
    ]
    arguments = (records, plans, targets)
    read = (
        tuple(io_pool(threads).map(_read_rows, *arguments))
        if threads > 1 and len(records) > 1
        else tuple(map(_read_rows, *arguments))
    )
    decoded = []
    for index, (record, plan, target, ok) in enumerate(
        zip(records, plans, targets, read, strict=True)
    ):
        if ok:
            mask_stored_bits(target, record.descriptor)
        if not ok or np.count_nonzero(target) < NEAR_EMPTY_FRACTION * (
            record.descriptor.rows * record.descriptor.columns
        ):
            decoded.append(decode_slice(record, spacing_mm, image_size))
            continue
        decoded.append(
            PreparedFrame(
                pixels,
                int(offsets[index]),
                (plan.row_stop - plan.row_start, record.descriptor.columns),
                (plan.column_start, plan.column_stop),
                record.descriptor.signed,
                plan.key,
                record.descriptor.slope,
                record.descriptor.intercept,
                plan.rect,
                plan.output_to_native,
            )
        )
    return tuple(decoded)


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
            (),
            np.empty(0, dtype=np.int64),
            quantile_ranks(0, lower_quantile, upper_quantile),
            torch.empty(0),
            failures,
            torch.empty((0, group_size, 3, 3)),
            torch.empty((0, group_size, 4)),
            torch.empty(0, dtype=torch.bool),
            (),
            failed_paths=failed_paths,
            reflection_reason="empty_prepared_series",
        )
    lookup = {r.path: i for i, r in enumerate(retained)}
    successful_records = [r for group in groups for r in group]
    gather = np.asarray([lookup[r.path] for r in successful_records], dtype=np.int64)
    frames = tuple(cache[r.path] for r in retained)
    rects = np.asarray([frame.rect for frame in frames], dtype=np.int32)
    pixel_count = int((rects[:, 2].astype(np.int64) * rects[:, 3]).sum())
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
        frames,
        gather,
        quantile_ranks(pixel_count, lower_quantile, upper_quantile),
        torch.tensor(positions, dtype=torch.float32),
        tuple(failures),
        torch.tensor(np.asarray(patient_maps), dtype=torch.float32),
        torch.tensor(rects.reshape(-1, group_size, 4), dtype=torch.float32),
        torch.tensor(geometry_valid, dtype=torch.bool),
        tuple(selected_paths),
        failed_paths=failed_paths,
        unique_slice_count=len(retained),
        intensity_pixel_count=pixel_count,
    )
