from __future__ import annotations

import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.charset import convert_encodings, decode_bytes
from pydicom.filereader import read_partial
from pydicom.multival import MultiValue

from knee_abnormality.data.preprocessing.geometry import NativePose, PoseLayout
from knee_abnormality.data.preprocessing.laterality import HeaderSide, header_side

PREFIX_BYTES = 12288
EXTEND_ATTEMPTS = 3
PREPARATION_ERRORS = (
    AttributeError,
    EOFError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class SeriesTags:
    series_number: float | None
    image_type: tuple[str, ...]
    description: str | None
    repetition_time: float | None
    echo_time: float | None
    echo_train_length: float | None
    flip_angle: float | None
    acquisition_type: str | None
    sequence_variant: tuple[str, ...]
    scan_options: tuple[str, ...]
    acquisition_matrix: tuple[float, ...]
    receive_coil: str | None
    scanning_sequence: tuple[str, ...] = ()
    sequence_name: str | None = None
    inversion_time: float | None = None


@dataclass(frozen=True, slots=True)
class DicomHeader:
    physical_position: float | None
    instance_number: float | None
    descriptor: NativePixelDescriptor | None
    position_source: str
    pose: NativePose | None
    tags: SeriesTags
    side: HeaderSide


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
        normal = np.array(
            [
                row[1] * column[2] - row[2] * column[1],
                row[2] * column[0] - row[0] * column[2],
                row[0] * column[1] - row[1] * column[0],
            ]
        )
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


def _values(dataset: pydicom.Dataset, keyword: str) -> tuple[str, ...]:
    try:
        value = dataset.get(keyword)
        if value is None:
            return ()
        items = value if isinstance(value, MultiValue | list | tuple) else (value,)
        return tuple(text for item in items if (text := str(item).strip()))
    except PREPARATION_ERRORS:
        return ()


def _text(dataset: pydicom.Dataset, keyword: str) -> str | None:
    values = _values(dataset, keyword)
    return "\\".join(values) if values else None


def _number(dataset: pydicom.Dataset, keyword: str) -> float | None:
    values = _values(dataset, keyword)
    try:
        value = float(values[0]) if len(values) == 1 else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) else None


def _numbers(dataset: pydicom.Dataset, keyword: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in _values(dataset, keyword))
    except ValueError:
        return ()
    return values if all(math.isfinite(value) for value in values) else ()


def _series_tags(dataset: pydicom.Dataset) -> SeriesTags:
    return SeriesTags(
        _number(dataset, "SeriesNumber"),
        tuple(value.upper() for value in _values(dataset, "ImageType")),
        _text(dataset, "SeriesDescription"),
        _number(dataset, "RepetitionTime"),
        _number(dataset, "EchoTime"),
        _number(dataset, "EchoTrainLength"),
        _number(dataset, "FlipAngle"),
        _text(dataset, "MRAcquisitionType"),
        _values(dataset, "SequenceVariant"),
        _values(dataset, "ScanOptions"),
        _numbers(dataset, "AcquisitionMatrix"),
        _text(dataset, "ReceiveCoilName"),
        _values(dataset, "ScanningSequence"),
        _text(dataset, "SequenceName"),
        _number(dataset, "InversionTime"),
    )


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


EXPLICIT_LE = "1.2.840.10008.1.2.1"
LONG_VR = {b"OB", b"OW", b"OF", b"OD", b"OL", b"OV", b"SQ", b"UT", b"UN", b"OK"}
SHORT_VR = frozenset(
    {
        b"AE",
        b"AS",
        b"AT",
        b"CS",
        b"DA",
        b"DS",
        b"DT",
        b"FD",
        b"FL",
        b"IS",
        b"LO",
        b"LT",
        b"PN",
        b"SH",
        b"SL",
        b"SS",
        b"ST",
        b"TM",
        b"UI",
        b"UL",
        b"US",
    }
)
ELEMENT = struct.Struct("<HH2sH")
LONG_LENGTH = struct.Struct("<I")
TAGS_CACHE_LIMIT = 4096
_TAGS_CACHE: dict[tuple, SeriesTags] = {}
_POOL: ThreadPoolExecutor | None = None
_POOL_IDENTITY: tuple[int, int] | None = None


def io_pool(threads: int) -> ThreadPoolExecutor:
    global _POOL, _POOL_IDENTITY
    identity = (os.getpid(), threads)
    if _POOL is None or _POOL_IDENTITY != identity:
        _POOL = ThreadPoolExecutor(threads, thread_name_prefix="knee-io")
        _POOL_IDENTITY = identity
    return _POOL


TAG_FIELDS = (
    0x00200011,
    0x00080008,
    0x0008103E,
    0x00180080,
    0x00180081,
    0x00180091,
    0x00181314,
    0x00180023,
    0x00180021,
    0x00180022,
    0x00181310,
    0x00181250,
    0x00180020,
    0x00180024,
    0x00180082,
)
SIDE_FIELDS = {
    0x00200060: "Laterality",
    0x00200062: "ImageLaterality",
    0x00081030: "StudyDescription",
    0x0008103E: "SeriesDescription",
    0x00181030: "ProtocolName",
    0x00180015: "BodyPartExamined",
    0x00080070: "Manufacturer",
    0x00081090: "ManufacturerModelName",
}
WANTED = {
    0x00080005,
    *SIDE_FIELDS,
    0x00080008,
    0x0008103E,
    0x00180020,
    0x00180021,
    0x00180022,
    0x00180023,
    0x00180024,
    0x00180080,
    0x00180081,
    0x00180082,
    0x00180091,
    0x00181250,
    0x00181310,
    0x00181314,
    0x00200011,
    0x00200013,
    0x00200032,
    0x00200037,
    0x00200052,
    0x00201041,
    0x00280002,
    0x00280004,
    0x00280008,
    0x00280010,
    0x00280011,
    0x00280030,
    0x00280100,
    0x00280101,
    0x00280102,
    0x00280103,
    0x00281052,
    0x00281053,
}
PIXEL_TAGS = {0x7FE00008, 0x7FE00009, 0x7FE00010}


def _strings(raw: bytes) -> tuple:
    text = raw.decode("ascii", "replace").rstrip("\x00 ")
    return tuple(part for item in text.split("\\") if (part := item.strip()))


def _floats(raw: bytes) -> tuple:
    try:
        return tuple(float(value) for value in _strings(raw))
    except ValueError:
        return ()


def _scan(buffer: memoryview, start: int, limit: int, values: dict) -> int | None:
    offset = start
    while offset + 8 <= limit:
        group, element, vr, short_length = ELEMENT.unpack_from(buffer, offset)
        tag = (group << 16) | element
        if vr in SHORT_VR:
            length = short_length
            body = offset + 8
        elif vr in LONG_VR:
            if offset + 12 > limit:
                return None
            length = LONG_LENGTH.unpack_from(buffer, offset + 8)[0]
            body = offset + 12
        elif vr.isalpha() and vr.isupper():
            length = short_length
            body = offset + 8
        else:
            raise Unsupported("implicit VR")
        if tag in PIXEL_TAGS:
            values["pixel"] = (tag, vr.decode(), length, body)
            return body
        if length == 0xFFFFFFFF:
            raise Unsupported("undefined length")
        if tag in WANTED:
            if body + length > limit:
                return None
            values[tag] = (vr, bytes(buffer[body : body + length]))
        offset = body + length
        if group > 0x7FE0:
            return offset
    return None


def _meta_end(buffer: memoryview, values: dict) -> int:
    if bytes(buffer[128:132]) != b"DICM":
        raise Unsupported("no preamble")
    group = int.from_bytes(buffer[132:134], "little")
    element = int.from_bytes(buffer[134:136], "little")
    if (group, element) != (0x0002, 0x0000):
        raise Unsupported("no group length")
    length = int.from_bytes(buffer[140:144], "little")
    end = 144 + length
    offset = 144
    while offset + 8 <= end:
        group = int.from_bytes(buffer[offset : offset + 2], "little")
        element = int.from_bytes(buffer[offset + 2 : offset + 4], "little")
        vr = bytes(buffer[offset + 4 : offset + 6])
        if vr in LONG_VR:
            size = int.from_bytes(buffer[offset + 8 : offset + 12], "little")
            body = offset + 12
        else:
            size = int.from_bytes(buffer[offset + 6 : offset + 8], "little")
            body = offset + 8
        if (group, element) == (0x0002, 0x0010):
            values["syntax"] = (
                bytes(buffer[body : body + size]).decode("ascii").rstrip("\x00 ")
            )
        offset = body + size
    return end


class Unsupported(Exception):
    pass


def parse_header(buffer: bytes, size: int, modified_ns: int) -> DicomHeader:
    view = memoryview(buffer)
    values: dict = {}
    end = _meta_end(view, values)
    if values.get("syntax") != EXPLICIT_LE:
        raise Unsupported("transfer syntax")
    if _scan(view, end, len(buffer), values) is None:
        raise Unsupported("truncated prefix")

    def numbers(tag):
        item = values.get(tag)
        return _floats(item[1]) if item else ()

    def one(tag):
        found = numbers(tag)
        value = found[0] if len(found) == 1 else None
        return value if value is not None and math.isfinite(value) else None

    def strings(tag):
        item = values.get(tag)
        return _strings(item[1]) if item else ()

    def text(tag):
        found = strings(tag)
        return "\\".join(found) if found else None

    def integer(tag):
        item = values.get(tag)
        if not item:
            return None
        if item[0] == b"US":
            return int.from_bytes(item[1][:2], "little")
        found = numbers(tag)
        return int(found[0]) if len(found) == 1 else None

    orientation = numbers(0x00200037)
    location = numbers(0x00200032)
    physical = None
    source = "filename"
    if len(orientation) == 6 and len(location) == 3:
        row = np.asarray(orientation[:3], dtype=np.float64)
        column = np.asarray(orientation[3:], dtype=np.float64)
        normal = np.array(
            [
                row[1] * column[2] - row[2] * column[1],
                row[2] * column[0] - row[0] * column[2],
                row[0] * column[1] - row[1] * column[0],
            ]
        )
        candidate = float(np.dot(np.asarray(location, dtype=np.float64), normal))
        if math.isfinite(candidate) and np.linalg.norm(normal) > 0:
            physical, source = candidate, "patient"
    if physical is None and 0x00201041 in values:
        found = numbers(0x00201041)
        if found and math.isfinite(found[0]):
            physical, source = found[0], "slice_location"
    instance = None
    if physical is None and 0x00200013 in values:
        found = numbers(0x00200013)
        if len(found) == 1:
            instance = found[0]

    spacing = numbers(0x00280030)
    rows = integer(0x00280010)
    columns = integer(0x00280011)
    pose = None
    if len(location) == 3 and len(orientation) == 6 and len(spacing) == 2:
        uid = values.get(0x00200052)
        pose = NativePose(
            location,
            PoseLayout(
                orientation,
                spacing,
                int(rows),
                int(columns),
                uid[1].decode("ascii").rstrip("\x00") if uid else None,
            ),
        )

    descriptor = None
    pixel = values.get("pixel")
    if pixel is not None:
        tag, vr, length, offset = pixel
        bits_stored = integer(0x00280101)
        representation = integer(0x00280103)
        frames = integer(0x00280008)
        photometric = strings(0x00280004)
        if (
            tag == 0x7FE00010
            and vr in {"OB", "OW"}
            and integer(0x00280002) == 1
            and photometric
            and photometric[0] == "MONOCHROME2"
            and (frames if frames is not None else 1) == 1
            and integer(0x00280100) == 16
            and bits_stored is not None
            and 1 <= bits_stored <= 16
            and integer(0x00280102) == bits_stored - 1
            and representation in {0, 1}
            and rows is not None
            and columns is not None
            and rows >= 1
            and columns >= 1
            and length == rows * columns * 2
            and offset + length == size
            and len(spacing) == 2
        ):
            slope = one(0x00281053)
            intercept = one(0x00281052)
            descriptor = NativePixelDescriptor(
                offset,
                length,
                size,
                modified_ns,
                rows,
                columns,
                bits_stored,
                representation == 1,
                float(spacing[0]),
                float(spacing[1]),
                1.0 if slope is None else slope,
                0.0 if intercept is None else intercept,
            )

    signature = tuple(values.get(tag) for tag in TAG_FIELDS)
    tags = _TAGS_CACHE.get(signature)
    if tags is None:
        matrix = numbers(0x00181310)
        if values.get(0x00181310) and values[0x00181310][0] == b"US":
            payload = values[0x00181310][1]
            matrix = tuple(
                float(int.from_bytes(payload[index : index + 2], "little"))
                for index in range(0, len(payload) - 1, 2)
            )
        tags = SeriesTags(
            one(0x00200011),
            tuple(value.upper() for value in strings(0x00080008)),
            text(0x0008103E),
            one(0x00180080),
            one(0x00180081),
            one(0x00180091),
            one(0x00181314),
            text(0x00180023),
            strings(0x00180021),
            strings(0x00180022),
            matrix,
            text(0x00181250),
            strings(0x00180020),
            text(0x00180024),
            one(0x00180082),
        )
        if len(_TAGS_CACHE) < TAGS_CACHE_LIMIT:
            _TAGS_CACHE[signature] = tags
    encodings = convert_encodings(list(strings(0x00080005)) or None)
    side_values = {
        name: decode_bytes(values[tag][1], encodings, {92}).rstrip("\x00 ")
        for tag, name in SIDE_FIELDS.items()
        if tag in values
    }
    return DicomHeader(
        physical,
        instance,
        descriptor,
        "instance_number" if instance is not None else source,
        pose,
        tags,
        header_side(side_values),
    )


def read_header(path: Path) -> DicomHeader:
    with path.open("rb") as stream:
        buffer = stream.read(PREFIX_BYTES)
        info = os.fstat(stream.fileno())
        for _ in range(EXTEND_ATTEMPTS):
            try:
                return parse_header(buffer, info.st_size, info.st_mtime_ns)
            except Unsupported as error:
                if str(error) != "truncated prefix" or len(buffer) >= info.st_size:
                    break
                stream.seek(0)
                buffer = stream.read(min(len(buffer) * 8, info.st_size))
            except PREPARATION_ERRORS:
                break
    return pydicom_read_header(path)


def parse_prefix(
    path: Path, prefix: tuple[bytes, os.stat_result] | None
) -> DicomHeader:
    if prefix is None:
        return read_header(path)
    buffer, info = prefix
    try:
        return parse_header(buffer, info.st_size, info.st_mtime_ns)
    except Unsupported:
        return read_header(path)
    except PREPARATION_ERRORS:
        return pydicom_read_header(path)


def read_prefix(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        return os.pread(descriptor, PREFIX_BYTES, 0), os.fstat(descriptor)
    finally:
        os.close(descriptor)


def pydicom_read_header(path: Path) -> DicomHeader:
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
        _series_tags(dataset),
        header_side({name: _text(dataset, name) for name in SIDE_FIELDS.values()}),
    )


def read_payload(path: Path, descriptor: NativePixelDescriptor) -> bytes | None:
    stream = os.open(path, os.O_RDONLY)
    try:
        info = os.fstat(stream)
        if (
            (info.st_size, info.st_mtime_ns)
            != (descriptor.file_size, descriptor.modified_ns)
            or descriptor.offset < 0
            or descriptor.length != descriptor.rows * descriptor.columns * 2
            or descriptor.offset + descriptor.length != info.st_size
        ):
            return None
        payload = os.pread(stream, descriptor.length, descriptor.offset)
    finally:
        os.close(stream)
    return payload if len(payload) == descriptor.length else None


def frame_from_payload(
    payload: bytes, descriptor: NativePixelDescriptor
) -> DecodedFrame:
    pixels = np.frombuffer(payload, dtype="<i2" if descriptor.signed else "<u2")
    shift = 16 - descriptor.bits_stored
    if shift:
        if descriptor.signed:
            pixels = np.left_shift(pixels, shift)
            np.right_shift(pixels, shift, out=pixels)
        else:
            pixels = np.bitwise_and(
                pixels, np.uint16((1 << descriptor.bits_stored) - 1)
            )
    pixels = pixels.reshape(descriptor.rows, descriptor.columns)
    return DecodedFrame(
        pixels,
        descriptor.row_spacing,
        descriptor.column_spacing,
        descriptor.slope,
        descriptor.intercept,
    )


def read_frame(path: Path, descriptor: NativePixelDescriptor | None) -> DecodedFrame:
    if descriptor is not None:
        payload = read_payload(path, descriptor)
        if payload is not None:
            return frame_from_payload(payload, descriptor)
    return _read_pydicom_frame(path)


def _read_pydicom_frame(path: Path) -> DecodedFrame:
    dataset = pydicom.dcmread(path)
    pixels = np.asarray(dataset.pixel_array)
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
