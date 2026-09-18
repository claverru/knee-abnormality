import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.filereader import read_partial
from pydicom.multival import MultiValue

CELLS = [
    ("pydicom", 1, False),
    ("pydicom", 4, False),
    ("pydicom", 16, False),
    ("raw", 1, False),
    ("raw", 4, False),
    ("raw", 16, False),
    ("raw", 16, True),
    ("raw", 64, True),
]
FILES_PER_CELL = 1200
AUDIT_FILES = 600
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
class PoseLayout:
    orientation: tuple
    spacing: tuple
    rows: int
    columns: int
    frame_of_reference: str | None


@dataclass(frozen=True, slots=True)
class NativePose:
    position: tuple
    layout: PoseLayout


@dataclass(frozen=True, slots=True)
class SeriesTags:
    series_number: float | None
    image_type: tuple
    description: str | None
    repetition_time: float | None
    echo_time: float | None
    echo_train_length: float | None
    flip_angle: float | None
    acquisition_type: str | None
    sequence_variant: tuple
    scan_options: tuple
    acquisition_matrix: tuple
    receive_coil: str | None


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


@dataclass(frozen=True, slots=True)
class DicomHeader:
    physical_position: float | None
    instance_number: float | None
    descriptor: NativePixelDescriptor | None
    position_source: str
    pose: NativePose | None
    tags: SeriesTags


def _physical_position(dataset):
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


def _native_pose(dataset):
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


def _values(dataset, keyword):
    try:
        value = dataset.get(keyword)
        if value is None:
            return ()
        items = value if isinstance(value, MultiValue | list | tuple) else (value,)
        return tuple(text for item in items if (text := str(item).strip()))
    except PREPARATION_ERRORS:
        return ()


def _text(dataset, keyword):
    values = _values(dataset, keyword)
    return "\\".join(values) if values else None


def _number(dataset, keyword):
    values = _values(dataset, keyword)
    try:
        value = float(values[0]) if len(values) == 1 else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) else None


def _numbers(dataset, keyword):
    try:
        values = tuple(float(value) for value in _values(dataset, keyword))
    except ValueError:
        return ()
    return values if all(math.isfinite(value) for value in values) else ()


def _series_tags(dataset):
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
    )


def _native_descriptor(dataset, pixel_element, info):
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
        _series_tags(dataset),
    )


EXPLICIT_LE = "1.2.840.10008.1.2.1"
LONG_VR = {b"OB", b"OW", b"OF", b"OD", b"OL", b"OV", b"SQ", b"UT", b"UN", b"OK"}
WANTED = {
    0x00080008,
    0x0008103E,
    0x00180021,
    0x00180022,
    0x00180023,
    0x00180080,
    0x00180081,
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


class Unsupported(Exception):
    pass


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
        group = int.from_bytes(buffer[offset : offset + 2], "little")
        element = int.from_bytes(buffer[offset + 2 : offset + 4], "little")
        tag = (group << 16) | element
        vr = bytes(buffer[offset + 4 : offset + 6])
        if vr in LONG_VR:
            if offset + 12 > limit:
                return None
            length = int.from_bytes(buffer[offset + 8 : offset + 12], "little")
            body = offset + 12
        elif vr.isalpha() and vr.isupper():
            length = int.from_bytes(buffer[offset + 6 : offset + 8], "little")
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
            values["syntax"] = bytes(buffer[body : body + size]).decode("ascii").rstrip("\x00 ")
        offset = body + size
    return end


def parse_header(path: Path, buffer: bytes, size: int, modified_ns: int) -> DicomHeader:
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
        normal = np.cross(row, column)
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
    )
    return DicomHeader(
        physical,
        instance,
        descriptor,
        "instance_number" if instance is not None else source,
        pose,
        tags,
    )


FALLBACKS: dict[str, int] = {}
EXTENSIONS: dict[int, int] = {}


def raw_read_header(path: Path) -> DicomHeader:
    with path.open("rb") as stream:
        buffer = stream.read(PREFIX_BYTES)
        info = os.fstat(stream.fileno())
        for attempt in range(EXTEND_ATTEMPTS):
            try:
                return parse_header(path, buffer, info.st_size, info.st_mtime_ns)
            except Unsupported as error:
                if str(error) != "truncated prefix" or len(buffer) >= info.st_size:
                    FALLBACKS[str(error)] = FALLBACKS.get(str(error), 0) + 1
                    break
                EXTENSIONS[attempt] = EXTENSIONS.get(attempt, 0) + 1
                stream.seek(0)
                buffer = stream.read(min(len(buffer) * 8, info.st_size))
            except PREPARATION_ERRORS as error:
                name = type(error).__name__
                FALLBACKS[name] = FALLBACKS.get(name, 0) + 1
                break
        else:
            FALLBACKS["extend exhausted"] = FALLBACKS.get("extend exhausted", 0) + 1
    return read_header(path)


def read_prefix(path: Path):
    with path.open("rb") as stream:
        return path, stream.read(PREFIX_BYTES), os.fstat(stream.fileno())


def run_cell(implementation: str, threads: int, split: bool, paths: list[Path]) -> dict:
    reader = read_header if implementation == "pydicom" else raw_read_header
    failures = 0
    with Timer() as timer:
        if split:
            with ThreadPoolExecutor(threads) as pool:
                prefixes = list(pool.map(read_prefix, paths))
            for path, buffer, info in prefixes:
                try:
                    parse_header(path, buffer, info.st_size, info.st_mtime_ns)
                except (Unsupported, *PREPARATION_ERRORS):
                    failures += 1
        elif threads == 1:
            for path in paths:
                try:
                    reader(path)
                except PREPARATION_ERRORS:
                    failures += 1
        else:
            def safe(path):
                try:
                    reader(path)
                    return 0
                except PREPARATION_ERRORS:
                    return 1

            with ThreadPoolExecutor(threads) as pool:
                failures = sum(pool.map(safe, paths))
    return {
        "implementation": implementation,
        "threads": threads,
        "split_read_parse": split,
        "files": len(paths),
        "failures": failures,
        "wall_seconds": round(timer.wall, 3),
        "cpu_seconds": round(timer.cpu, 3),
        "files_per_second": round(len(paths) / timer.wall, 1),
        "milliseconds_per_file": round(1000 * timer.wall / len(paths), 3),
    }


def audit(paths: list[Path]) -> dict:
    mismatches = []
    compared = 0
    for path in paths:
        try:
            expected = read_header(path)
        except PREPARATION_ERRORS:
            continue
        try:
            actual = raw_read_header(path)
        except PREPARATION_ERRORS:
            mismatches.append({"path": path.name, "field": "exception"})
            continue
        compared += 1
        for field, value in asdict(expected).items():
            other = asdict(actual)[field]
            if value != other:
                mismatches.append(
                    {
                        "path": path.name,
                        "field": field,
                        "expected": str(value)[:200],
                        "actual": str(other)[:200],
                    }
                )
    return {
        "compared": compared,
        "mismatched_files": len({item["path"] for item in mismatches}),
        "examples": mismatches[:20],
    }


def main() -> None:
    blocks = file_blocks(len(CELLS) + 1, FILES_PER_CELL)
    audit_paths = blocks[0][:AUDIT_FILES]
    result = {
        "pydicom_version": pydicom.__version__,
        "prefix_bytes": PREFIX_BYTES,
        "audit": audit(audit_paths),
    }
    result["cells"] = (
        []
        if AUDIT_ONLY
        else [
            run_cell(implementation, threads, split, block[:FILES_PER_CELL])
            for (implementation, threads, split), block in zip(CELLS, blocks[1:])
        ]
    )
    result["fallbacks"] = dict(FALLBACKS)
    result["extensions"] = dict(EXTENSIONS)
    report("header", result)


main()
