import fcntl
import hashlib
import json
import os
import signal
import tempfile
from contextlib import contextmanager
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text())


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def tail(path: Path, size: int = 4000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode(errors="replace")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


def process_identity(pid: int) -> dict | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {
            "pid": pid,
            "start": stat[19],
            "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(identity: dict | None) -> bool:
    return bool(identity and process_identity(identity["pid"]) == identity)


def terminate(identity: dict | None) -> None:
    if alive(identity):
        try:
            os.kill(identity["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
