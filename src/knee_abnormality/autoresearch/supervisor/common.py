import subprocess
import time
import uuid
from pathlib import Path

from knee_abnormality.autoresearch.runtime.storage import (
    read_json,
)

RETRY_DELAYS = (60, 300, 900)
REJECTION_LIMIT = 3
REJECTION_PAUSE = "Report rejected"
SEED_TERMINAL = {"completed", "blocked", "retired"}


def seed_files(docs: Path):
    return sorted(
        docs.glob("seed-ideas*.yaml"),
        key=lambda path: (path.name != "seed-ideas.yaml", path.name),
    )


def identifier(prefix):
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


def gpu_available(gpu: str):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Cannot inspect selected GPU: {result.stderr.strip()}")
    return not result.stdout.strip()


def work_item(job):
    payload = job["payload"]
    return f"{job['kind']}:{payload.get('experiment') or payload.get('event', {}).get('id', '')}"


def trained_epochs(directory: Path):
    path = directory / "training/fold_0/triplet_accounting.json"
    try:
        return len(read_json(path)["train"]) if path.exists() else 0
    except (OSError, ValueError, KeyError, TypeError):
        return None
