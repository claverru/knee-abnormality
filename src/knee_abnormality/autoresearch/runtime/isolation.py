import os
import shutil
from pathlib import Path

from knee_abnormality.autoresearch.campaign.definition import CampaignPolicy
from knee_abnormality.autoresearch.runtime.storage import sha256

SOURCE_ROOTS = ("src", "scripts", "configs", "docs", "autoresearch", "labels")
SOURCE_FILES = ("AGENTS.md", "README.md", "pyproject.toml", "uv.lock", "justfile")


def manifest(root: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not {"__pycache__", ".ruff_cache"}.intersection(path.parts)
    }


def snapshot(source: Path, target: Path) -> dict:
    target.mkdir(parents=True, exist_ok=False)
    for name in (*SOURCE_ROOTS, *SOURCE_FILES):
        path = source / name
        if not path.exists():
            continue
        if path.is_dir():
            shutil.copytree(
                path,
                target / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ruff_cache"),
            )
        else:
            shutil.copy2(path, target / name)
    return manifest(target)


def changed_files(before: dict, after: dict) -> list[str]:
    return sorted(
        key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
    )


def check_scope(
    source: Path, before: dict, policy: CampaignPolicy
) -> tuple[dict, list[str]]:
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Candidate source contains a symlink")
    after = manifest(source)
    changed = changed_files(before, after)
    forbidden = [
        path
        for path in changed
        if path not in policy.editable_files
        and not any(
            path.startswith(f"{directory}/")
            for directory in policy.editable_directories
        )
    ]
    if forbidden:
        raise ValueError(f"Changes outside the campaign workspace: {forbidden}")
    return after, changed


def sandbox(
    command: list[str],
    cwd: Path,
    writable: list[Path],
    *,
    gpu=False,
    readonly=(),
    network=False,
) -> list[str]:
    result = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
    ]
    result += (
        ["--dev-bind", "/dev", "/dev"]
        if gpu
        else ["--dev", "/dev", "--tmpfs", "/dev/shm"]
    )
    if not network:
        result.append("--unshare-net")
    for path in writable:
        result += ["--bind", str(path), str(path)]
    for path in readonly:
        if path.exists():
            result += ["--ro-bind", str(path), str(path)]
    result += ["--chdir", str(cwd), "--", *command]
    return result


def environment(source: Path, *, gpu: str = "") -> dict:
    value = dict(os.environ)
    for name in list(value):
        if (
            name.startswith(("OPENAI_", "ANTHROPIC_", "CODEX_", "CLAUDE_", "KAGGLE_"))
            or name == "CLAUDECODE"
        ):
            value.pop(name)
    value.update(
        PYTHONPATH=f"{source / 'src'}:{source}",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES=gpu,
        POLARS_MAX_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        KNEE_PROJECT_ROOT=str(source),
        HF_HUB_OFFLINE="1",
    )
    return value
