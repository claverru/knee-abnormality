import time
import uuid
from pathlib import Path

from knee_abnormality.autoresearch.runtime.storage import (
    read_json,
)

RETRY_DELAYS = (60, 300, 900)
REJECTION_LIMIT = 3


def seed_files(docs: Path):
    return sorted(
        docs.glob("seed-ideas*.yaml"),
        key=lambda path: (path.name != "seed-ideas.yaml", path.name),
    )


def identifier(prefix):
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


def work_item(job):
    payload = job["payload"]
    return f"{job['kind']}:{payload.get('experiment') or payload.get('event', {}).get('id', '')}"


def trained_epochs(directory: Path, fold: int):
    path = directory / f"training/fold_{fold}/triplet_accounting.json"
    try:
        return len(read_json(path)["train"]) if path.exists() else 0
    except (OSError, ValueError, KeyError, TypeError):
        return None
