import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

INPUT = Path(sys.argv[1])
WORKERS = int(sys.argv[2])
DOWNLOADS = Path("/workspace/downloads/oai")
RAW = Path("/workspace/data/raw/oai")
DONE = Path("/workspace/logs/oai_extracted.txt")
ERRORS = Path("/workspace/logs/oai_extract_errors.log")
lock = threading.Lock()
lines = INPUT.read_text().splitlines()
jobs = [
    (lines[index].strip(), lines[index + 1].split("out=", 1)[1].strip())
    for index in range(0, len(lines) - 1, 2)
    if lines[index].startswith("http")
]
done = set(DONE.read_text().split()) if DONE.exists() else set()
jobs = [job for job in jobs if job[1] not in done]
print(f"{len(jobs)} files to fetch, {len(done)} already extracted", flush=True)
session = requests.Session()
session.mount(
    "https://",
    requests.adapters.HTTPAdapter(pool_connections=WORKERS, pool_maxsize=WORKERS),
)
counter = {"files": 0, "bytes": 0, "failed": 0}
started = time.time()


def fetch(job):
    url, relative = job
    archive = DOWNLOADS / relative
    target = RAW / relative.removesuffix(".tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            with session.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with archive.open("wb") as stream:
                    for block in response.iter_content(1 << 20):
                        stream.write(block)
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["tar", "-xzf", str(archive), "-C", str(target)],
                check=True,
                capture_output=True,
            )
            size = archive.stat().st_size
            archive.unlink()
            with lock:
                with DONE.open("a") as stream:
                    stream.write(relative + "\n")
                counter["files"] += 1
                counter["bytes"] += size
                if counter["files"] % 500 == 0:
                    elapsed = time.time() - started
                    print(
                        f"{counter['files']}/{len(jobs)} files, {counter['bytes'] / 1e9:.1f} GB compressed, {counter['bytes'] / elapsed / 1e6:.0f} MB/s, {elapsed / 60:.1f} min",
                        flush=True,
                    )
            return
        except Exception as error:
            last = f"{type(error).__name__}: {error}"[:200]
            time.sleep(5 * (attempt + 1))
    with lock:
        counter["failed"] += 1
        with ERRORS.open("a") as stream:
            stream.write(f"FAILED {relative} {last}\n")


with ThreadPoolExecutor(WORKERS) as pool:
    list(pool.map(fetch, jobs))
elapsed = time.time() - started
print(
    f"DONE {counter['files']} files, {counter['failed']} failed, {counter['bytes'] / 1e9:.1f} GB compressed in {elapsed / 60:.1f} min",
    flush=True,
)
