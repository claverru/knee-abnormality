from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.filereader import read_partial

from knee_abnormality.preprocessing.geometry import NativePose, PoseLayout

PREPARATION_ERRORS = (
    AttributeError,
    EOFError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class DicomHeader:
    physical_position: float | None
    instance_number: float | None
    descriptor: NativePixelDescriptor | None
    position_source: str
    pose: NativePose | None


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    pixels: np.ndarray
    row_spacing: float
    column_spacing: float
    slope: float
    intercept: float


def _physical_position(dataset: pydicom.Dataset) -> tuple[float | None, str]:
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
            return position, "patient"
    if hasattr(dataset, "SliceLocation"):
        position = float(dataset.SliceLocation)
        if math.isfinite(position):
            return position, "slice_location"
    return None, "filename"


def _native_pose(dataset: pydicom.Dataset) -> NativePose | None:
    try:
        position = tuple(float(value) for value in dataset.ImagePositionPatient)
        orientation = tuple(float(value) for value in dataset.ImageOrientationPatient)
        spacing = tuple(float(value) for value in dataset.PixelSpacing)
        if len(position) != 3 or len(orientation) != 6 or len(spacing) != 2:
            return None
        return NativePose(
            position,
            PoseLayout(
                orientation,
                spacing,
                int(dataset.Rows),
                int(dataset.Columns),
                str(dataset.FrameOfReferenceUID)
                if hasattr(dataset, "FrameOfReferenceUID")
                else None,
            ),
        )
    except PREPARATION_ERRORS:
        return None


@dataclass(frozen=True, slots=True)
class NativePixelDescriptor:
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


def _native_descriptor(dataset, pixel_element, info) -> NativePixelDescriptor | None:
    try:
        if (
            str(dataset.file_meta.TransferSyntaxUID) != "1.2.840.10008.1.2.1"
            or dataset.original_encoding[:2] != (False, True)
            or pixel_element is None
        ):
            return None
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
            return None
        return NativePixelDescriptor(
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
        )
    except PREPARATION_ERRORS:
        return None


def read_header(path: Path) -> DicomHeader:
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
    physical, source = _physical_position(dataset)
    instance = (
        float(dataset.InstanceNumber)
        if physical is None and hasattr(dataset, "InstanceNumber")
        else None
    )
    return DicomHeader(
        physical,
        instance,
        _native_descriptor(dataset, pixel_element, info),
        "instance_number" if instance is not None else source,
        _native_pose(dataset),
    )


def read_frame(path: Path, descriptor: NativePixelDescriptor | None) -> DecodedFrame:
    if descriptor is not None:
        payload = None
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                (info.st_size, info.st_mtime_ns)
                == (descriptor.file_size, descriptor.modified_ns)
                and descriptor.offset >= 0
                and descriptor.length == descriptor.rows * descriptor.columns * 2
                and descriptor.offset + descriptor.length == info.st_size
            ):
                stream.seek(descriptor.offset)
                payload = stream.read(descriptor.length)
        if payload is not None and len(payload) == descriptor.length:
            pixels = np.frombuffer(payload, dtype="<i2" if descriptor.signed else "<u2")
            shift = 16 - descriptor.bits_stored
            if shift:
                pixels = np.left_shift(pixels, shift)
                np.right_shift(pixels, shift, out=pixels)
            pixels = pixels.reshape(descriptor.rows, descriptor.columns).astype(
                np.float32
            )
            return DecodedFrame(
                pixels,
                descriptor.row_spacing,
                descriptor.column_spacing,
                descriptor.slope,
                descriptor.intercept,
            )
    return _read_pydicom_frame(path)


def _read_pydicom_frame(path: Path) -> DecodedFrame:
    dataset = pydicom.dcmread(path)
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"Expected one two-dimensional frame, got {pixels.shape}")
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    source_spacing = getattr(dataset, "PixelSpacing", None)
    if source_spacing is None or len(source_spacing) != 2:
        raise ValueError("Missing two-dimensional PixelSpacing")
    return DecodedFrame(
        pixels, float(source_spacing[0]), float(source_spacing[1]), slope, intercept
    )
