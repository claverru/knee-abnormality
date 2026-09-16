import hashlib
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_sources(paths: Iterable[Path], destination: Path) -> dict[str, str]:
    root = Path.cwd().resolve()
    files = []
    for path in paths:
        path = Path(path).resolve()
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    copied = {}
    for path in files:
        relative = (
            path.relative_to(root)
            if path.is_relative_to(root)
            else Path("external") / path.relative_to(path.anchor)
        )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied[relative.as_posix()] = sha256_file(target)
    return copied


def git_revision(root: Path) -> tuple[str | None, str | None]:
    def output(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return output("rev-parse", "HEAD"), output("status", "--short")
