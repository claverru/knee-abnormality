import hashlib
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import polars as pl
import pydicom

CATALOG = "/workspace/data/fastmri/series_catalog.parquet"
SAMPLE = "/root/uncompress_sample.json"
MINIMUM_FREE_GB = 80


def digest(path):
    dataset = pydicom.dcmread(path)
    if "PixelData" not in dataset:
        return None
    if dataset.file_meta.TransferSyntaxUID.is_compressed:
        dataset.pixel_array_options(decoding_plugin="pillow")
    pixels = np.asarray(dataset.pixel_array)
    return [str(pixels.dtype), list(pixels.shape), hashlib.sha256(pixels.tobytes()).hexdigest(), str(dataset.SOPInstanceUID)]


def convert(directory):
    done = skipped = failed = 0
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if name.endswith(".uncompressing"):
            os.remove(path)
            continue
        try:
            dataset = pydicom.dcmread(path)
            if "PixelData" not in dataset or not dataset.file_meta.TransferSyntaxUID.is_compressed:
                skipped += 1
                continue
            dataset.decompress(decoding_plugin="pillow", generate_instance_uid=False)
            temporary = path + ".uncompressing"
            dataset.save_as(temporary)
            os.replace(temporary, path)
            done += 1
        except Exception as error:
            failed += 1
            print("FAILED", path, type(error).__name__, str(error)[:120], flush=True)
    return done, skipped, failed


def files(directories, count):
    random.seed(11)
    chosen = random.sample(directories, min(len(directories), count))
    return [os.path.join(directory, random.choice(sorted(os.listdir(directory)))) for directory in chosen]


directories = sorted(set(pl.read_parquet(CATALOG)["directory"].to_list()))
mode = sys.argv[1]
if mode == "sample":
    paths = files(directories, 4000)
    with ProcessPoolExecutor(60) as pool:
        values = list(pool.map(digest, paths, chunksize=50))
    json.dump(dict(zip(paths, values)), open(SAMPLE, "w"))
    print("sampled", len(paths), "with pixels", sum(value is not None for value in values), flush=True)
elif mode == "convert":
    started = time.time()
    totals = [0, 0, 0]
    position = 0
    while position < len(directories):
        if shutil.disk_usage("/").free / 1e9 < MINIMUM_FREE_GB:
            print("STOPPED: free disk below", MINIMUM_FREE_GB, "GB", flush=True)
            break
        block = directories[position : position + 600]
        try:
            with ProcessPoolExecutor(60) as pool:
                for done, skipped, failed in pool.map(convert, block, chunksize=4):
                    totals[0] += done
                    totals[1] += skipped
                    totals[2] += failed
        except BrokenProcessPool:
            print("worker crash in block at", position, "; block will be retried once", flush=True)
            with ProcessPoolExecutor(8) as pool:
                for done, skipped, failed in pool.map(convert, block, chunksize=1):
                    totals[0] += done
                    totals[1] += skipped
                    totals[2] += failed
        position += len(block)
        print(f"{position}/{len(directories)} series, converted {totals[0]}, already raw {totals[1]}, failed {totals[2]}, free {shutil.disk_usage('/').free / 1e9:.0f} GB, {time.time() - started:.0f} s", flush=True)
    print("CONVERT_DONE", totals, flush=True)
elif mode == "verify":
    sample = json.load(open(SAMPLE))
    with ProcessPoolExecutor(60) as pool:
        values = list(pool.map(digest, list(sample), chunksize=50))
    different = [path for path, value in zip(sample, values) if value != sample[path]]
    compressed = 0
    for path in list(sample)[:1500]:
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        compressed += bool(dataset.file_meta.TransferSyntaxUID.is_compressed)
    print("verified", len(sample), "different", len(different), different[:3], "| still compressed among 1500:", compressed, flush=True)
