import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

CELLS = [1, 4, 16, 64]
FILES_PER_CELL = 900
AUDIT_FILES = 300

PREPARATION_ERRORS = (AttributeError, EOFError, OSError, RuntimeError, TypeError, ValueError)


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
class DecodedFrame:
    pixels: np.ndarray
    row_spacing: float
    column_spacing: float
    slope: float
    intercept: float


def describe(path: Path) -> NativePixelDescriptor | None:
    pixel_element = None
    with path.open("rb") as stream:

        def stop_at_pixels(tag, vr, length):
            nonlocal pixel_element
            if tag in {0x7FE00008, 0x7FE00009, 0x7FE00010}:
                pixel_element = (tag, vr, length, stream.tell())
                return True
            return False

        dataset = pydicom.filereader.read_partial(stream, stop_when=stop_at_pixels)
        info = os.fstat(stream.fileno())
    try:
        if (
            str(dataset.file_meta.TransferSyntaxUID) != "1.2.840.10008.1.2.1"
            or dataset.original_encoding[:2] != (False, True)
            or pixel_element is None
        ):
            return None
        tag, vr, length, offset = pixel_element
        rows, columns = int(dataset.Rows), int(dataset.Columns)
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


def decode(payload: bytes, descriptor: NativePixelDescriptor) -> DecodedFrame:
    pixels = np.frombuffer(payload, dtype="<i2" if descriptor.signed else "<u2")
    shift = 16 - descriptor.bits_stored
    if shift:
        pixels = np.left_shift(pixels, shift)
        np.right_shift(pixels, shift, out=pixels)
    pixels = pixels.reshape(descriptor.rows, descriptor.columns).astype(np.float32)
    return DecodedFrame(
        pixels,
        descriptor.row_spacing,
        descriptor.column_spacing,
        descriptor.slope,
        descriptor.intercept,
    )


def read_payload(item):
    path, descriptor = item
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if (info.st_size, info.st_mtime_ns) != (descriptor.file_size, descriptor.modified_ns):
            return None
        stream.seek(descriptor.offset)
        return stream.read(descriptor.length)


def read_frame(item) -> DecodedFrame | None:
    payload = read_payload(item)
    return None if payload is None else decode(payload, item[1])


def run_cell(threads: int, items: list) -> dict:
    megabytes = sum(descriptor.length for _, descriptor in items) / 2**20
    failures = 0
    with Timer() as timer:
        if threads == 1:
            frames = [read_frame(item) for item in items]
        else:
            with ThreadPoolExecutor(threads) as pool:
                payloads = list(pool.map(read_payload, items))
            frames = [
                None if payload is None else decode(payload, item[1])
                for payload, item in zip(payloads, items)
            ]
        failures = sum(1 for frame in frames if frame is None)
    return {
        "threads": threads,
        "frames": len(items),
        "failures": failures,
        "megabytes": round(megabytes, 1),
        "wall_seconds": round(timer.wall, 3),
        "cpu_seconds": round(timer.cpu, 3),
        "frames_per_second": round(len(items) / timer.wall, 1),
        "megabytes_per_second": round(megabytes / timer.wall, 1),
    }


def describe_block(paths: list[Path]) -> list:
    items = []
    for path in paths:
        try:
            descriptor = describe(path)
        except PREPARATION_ERRORS:
            continue
        if descriptor is not None:
            items.append((path, descriptor))
    return items


def matched_blocks(cells: int, frames_per_cell: int, studies) -> list[list]:
    buckets: list[dict[int, list]] = [{} for _ in range(cells)]
    while min(sum(len(v) for v in bucket.values()) for bucket in buckets) < frames_per_cell:
        items = describe_block(sorted(series_paths(next(studies))))
        for item in items:
            length = item[1].length
            target = min(
                range(cells),
                key=lambda index: (len(buckets[index].get(length, ())), index),
            )
            buckets[target].setdefault(length, []).append(item)
    lengths = set.intersection(*(set(bucket) for bucket in buckets))
    shared = {
        length: min(len(bucket[length]) for bucket in buckets) for length in lengths
    }
    blocks = []
    for bucket in buckets:
        block = []
        for length in sorted(shared):
            block.extend(bucket[length][: shared[length]])
        blocks.append(block)
    return blocks


def audit(items: list) -> dict:
    serial = [read_frame(item) for item in items]
    with ThreadPoolExecutor(16) as pool:
        payloads = list(pool.map(read_payload, items))
    threaded = [
        None if payload is None else decode(payload, item[1])
        for payload, item in zip(payloads, items)
    ]
    mismatches = sum(
        1
        for left, right in zip(serial, threaded)
        if (left is None) != (right is None)
        or (left is not None and not np.array_equal(left.pixels, right.pixels))
    )
    return {"compared": len(items), "mismatched_frames": mismatches}


def main() -> None:
    studies = iter(series_directories())
    audit_paths: list[Path] = []
    while len(audit_paths) < AUDIT_FILES:
        audit_paths.extend(sorted(series_paths(next(studies))))
    result = {"audit": audit(describe_block(audit_paths[:AUDIT_FILES]))}
    if AUDIT_ONLY:
        result["cells"] = []
    else:
        blocks = matched_blocks(len(CELLS), FILES_PER_CELL, studies)
        result["matched"] = {
            "frames": [len(block) for block in blocks],
            "bytes": [sum(d.length for _, d in block) for block in blocks],
        }
        result["cells"] = [
            run_cell(threads, block) for threads, block in zip(CELLS, blocks)
        ]
    report("frames", result)


main()
