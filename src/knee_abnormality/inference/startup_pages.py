import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STARTUP_PAGES = Path(__file__).with_name("kaggle_startup_pages.json")
STARTUP_READ_THREADS = 16
READ_BYTES = 1 << 23


def _read_ranges(item: tuple[str, list[list[int]]]) -> None:
    path, ranges = item
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        for offset, length in ranges:
            while length > 0:
                chunk = os.pread(descriptor, min(length, READ_BYTES), offset)
                if not chunk:
                    break
                offset += len(chunk)
                length -= len(chunk)
    finally:
        os.close(descriptor)


def _warm() -> None:
    pages = json.loads(STARTUP_PAGES.read_text())
    items = sorted(
        pages.items(), key=lambda item: -sum(length for _, length in item[1])
    )
    with ThreadPoolExecutor(STARTUP_READ_THREADS) as pool:
        for _ in pool.map(_read_ranges, items):
            pass


def warm_startup_pages() -> None:
    threading.Thread(target=_warm, daemon=True).start()
