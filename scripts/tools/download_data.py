import fcntl
import io
import json
import logging
import os
import shutil
import sqlite3
import stat
import subprocess
import time
import zipfile
import zlib
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import requests
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.kaggle import COMPETITION

GIB = 1024**3
CHUNK_BYTES = 128 * 1024**2
COPY_BYTES = 1024**2
LOG = logging.getLogger("download")


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class KaggleSource:
    def __init__(self, retries: int):
        self.retries = retries
        self.url = None
        self.session = requests.Session()
        self.session.headers["Accept-Encoding"] = "identity"
        for attempt in range(retries):
            try:
                with self.session.head(
                    self.signed_url(), allow_redirects=True, timeout=(20, 60)
                ) as response:
                    response.raise_for_status()
                    self.size = int(response.headers["Content-Length"])
                    self.etag = response.headers["ETag"]
                    if response.headers.get("Accept-Ranges") != "bytes":
                        raise RuntimeError(
                            "The archive server does not support byte ranges"
                        )
                    if self.size <= 0 or self.etag.startswith("W/"):
                        raise RuntimeError(
                            "The archive has no usable size or strong ETag"
                        )
                    return
            except requests.RequestException:
                self.retry(attempt)

    def signed_url(self) -> str:
        if self.url is None:
            try:
                from kaggle import api
                from kagglesdk.competitions.types.competition_api_service import (
                    ApiDownloadDataFilesRequest,
                )

                request = ApiDownloadDataFilesRequest()
                request.competition_name = COMPETITION
                with api.build_kaggle_client() as client:
                    self.url = (
                        client.competitions.competition_api_client.download_data_files(
                            request
                        ).url
                    )
            except (ImportError, OSError, ValueError, requests.RequestException):
                raise requests.ConnectionError(
                    "Could not obtain the Kaggle archive URL"
                ) from None
        return self.url

    def retry(self, attempt: int) -> None:
        self.url = None
        if attempt + 1 == self.retries:
            raise RuntimeError(
                "Network retries exhausted. Check connectivity, Kaggle credentials and "
                "accepted competition rules, then rerun the same command to resume."
            ) from None
        delay = min(60, 2 ** min(attempt + 1, 6))
        LOG.warning(
            "Network request failed; refreshing access and retrying in %s s", delay
        )
        time.sleep(delay)

    def download_range(self, start: int, end: int, destination: Path) -> None:
        part = destination.with_suffix(".part")
        if part.is_symlink():
            raise RuntimeError(f"Unexpected symlink in download state: {part}")
        expected = end - start + 1
        for attempt in range(self.retries):
            present = part.stat().st_size if part.exists() else 0
            if present > expected:
                raise RuntimeError(f"Oversized partial chunk: {part}")
            if present == expected:
                part.replace(destination)
                return
            first = start + present
            try:
                with self.session.get(
                    self.signed_url(),
                    headers={"Range": f"bytes={first}-{end}", "If-Match": self.etag},
                    stream=True,
                    timeout=(20, 120),
                ) as response:
                    if response.status_code == 412:
                        raise RuntimeError(
                            "The Kaggle archive changed; use a fresh destination"
                        )
                    response.raise_for_status()
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {first}-{end}/{self.size}"
                        or response.headers.get("ETag") != self.etag
                        or response.headers.get("Content-Encoding", "identity")
                        != "identity"
                    ):
                        raise RuntimeError(
                            "Server did not confirm the requested archive range; refusing to append"
                        )
                    with part.open("ab") as output:
                        for block in response.iter_content(COPY_BYTES):
                            if present + len(block) > expected:
                                raise RuntimeError(
                                    "Server returned more bytes than requested"
                                )
                            output.write(block)
                            present += len(block)
                if present != expected:
                    raise requests.ConnectionError("Incomplete archive range")
                part.replace(destination)
                LOG.info(
                    "Fetched ZIP chunk at %.2f GiB (%s MiB)",
                    start / GIB,
                    expected // 1024**2,
                )
                return
            except requests.RequestException:
                self.retry(attempt)


class ChunkedArchive(io.RawIOBase):
    def __init__(self, source: KaggleSource, cache: Path):
        self.source = source
        self.cache = cache
        self.position = 0
        self.recent = OrderedDict(
            (int(path.stem), path)
            for path in sorted(
                cache.glob("*.chunk"), key=lambda item: item.stat().st_mtime_ns
            )
        )

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        bases = {
            os.SEEK_SET: 0,
            os.SEEK_CUR: self.position,
            os.SEEK_END: self.source.size,
        }
        position = bases[whence] + offset
        if position < 0:
            raise ValueError("Negative archive offset")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self.source.size - self.position)
        size = remaining if size < 0 else min(size, remaining)
        output = bytearray()
        while len(output) < size:
            index, offset = divmod(self.position, CHUNK_BYTES)
            start = index * CHUNK_BYTES
            end = min(start + CHUNK_BYTES, self.source.size) - 1
            path = self.cache / f"{index}.chunk"
            if path.is_symlink():
                raise RuntimeError(f"Unexpected symlink in download state: {path}")
            if not path.exists():
                self.source.download_range(start, end, path)
            if path.stat().st_size != end - start + 1:
                raise RuntimeError(f"Wrong cached chunk size: {path}")
            self.recent[index] = path
            self.recent.move_to_end(index)
            while len(self.recent) > 3:
                _, old = self.recent.popitem(last=False)
                old.unlink()
            with path.open("rb") as cached:
                cached.seek(offset)
                block = cached.read(min(size - len(output), end - start + 1 - offset))
            if not block:
                raise RuntimeError("Unexpected end of cached chunk")
            output.extend(block)
            self.position += len(block)
        return bytes(output)


def windows_free_bytes(destination: Path) -> int | None:
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        return None
    if shutil.which("powershell.exe") is None:
        raise RuntimeError(
            "PowerShell is required to check WSL's physical host free space"
        )
    parts = destination.parts
    if (
        len(parts) >= 3
        and parts[1] == "mnt"
        and len(parts[2]) == 1
        and parts[2].isalpha()
    ):
        locate = f"$letter = '{parts[2]}'; "
    else:
        if destination.stat().st_dev != Path("/").stat().st_dev:
            raise RuntimeError(
                "Cannot identify the backing drive for this WSL destination"
            )
        escaped = distro.replace("'", "''")
        locate = (
            r"$d = Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' | "
            r"ForEach-Object { Get-ItemProperty $_.PSPath } | "
            f"Where-Object {{ $_.DistributionName -eq '{escaped}' }}; "
            r"if (-not $d) { throw 'Distribution location unavailable' }; "
            r"$letter = [regex]::Match($d.BasePath, '([A-Za-z]):').Groups[1].Value; "
        )
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$ErrorActionPreference = 'Stop'; "
            + locate
            + r"[System.IO.DriveInfo]::new($letter + ':\').AvailableFreeSpace",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        if result.returncode:
            raise ValueError
        return int(result.stdout.strip())
    except ValueError:
        raise RuntimeError(
            "Could not determine Windows host free space; check WSL/PowerShell access"
        ) from None


def check_space(
    destination: Path, remaining: int, reserve: int, missing: int = 0
) -> None:
    available = shutil.disk_usage(destination).free
    host_available = windows_free_bytes(destination)
    if host_available is not None:
        LOG.info("Windows host free space: %.1f GiB", host_available / GIB)
        available = min(available, host_available)
    needed = remaining + reserve + 2 * GIB + missing * 4096
    LOG.info(
        "Space: %.1f GiB available; %.1f GiB required including reserve",
        available / GIB,
        needed / GIB,
    )
    if available < needed:
        raise RuntimeError(
            "Insufficient space for the remaining extracted files and reserve"
        )
    filesystem = os.statvfs(destination)
    if filesystem.f_files and filesystem.f_favail < missing + 100_000:
        raise RuntimeError(
            "Insufficient free filesystem inodes for the remaining images/directories"
        )


def destination_for(root: Path, member: zipfile.ZipInfo) -> Path:
    relative = PurePosixPath(member.filename)
    mode = stat.S_IFMT(member.external_attr >> 16)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or "\\" in member.filename
        or "\x00" in member.orig_filename
        or member.filename.rstrip("/") != relative.as_posix()
        or relative.parts[0] == ".download"
        or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
        or member.compress_type
        not in (
            zipfile.ZIP_STORED,
            zipfile.ZIP_DEFLATED,
            zipfile.ZIP_BZIP2,
            zipfile.ZIP_LZMA,
        )
        or member.flag_bits & 1
    ):
        raise RuntimeError(f"Unsupported archive entry: {member.filename}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise RuntimeError(
            f"Archive destination escapes through a symlink: {member.filename}"
        )
    return path


def verify_existing(path: Path, member: zipfile.ZipInfo) -> None:
    crc = 0
    count = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(COPY_BYTES), b""):
            count += len(block)
            crc = zlib.crc32(block, crc)
    if count != member.file_size or crc != member.CRC:
        raise RuntimeError(f"Existing file differs from Kaggle; left untouched: {path}")


def extract(archive: zipfile.ZipFile, destination: Path, state: Path, args) -> None:
    members = archive.infolist()
    if len({member.filename for member in members}) != len(members):
        raise RuntimeError("Duplicate archive paths are unsupported")
    files = sorted(
        (member for member in members if not member.is_dir()),
        key=lambda member: member.header_offset,
    )
    total_bytes = sum(member.file_size for member in files)
    remaining = total_bytes
    missing = 0
    LOG.info(
        "ZIP inventory: %s files, %.1f GiB extracted; inspecting destination",
        len(files),
        total_bytes / GIB,
    )
    for member in members:
        path = destination_for(destination, member)
        if member.is_dir():
            continue
        if path.exists():
            if not path.is_file() or path.stat().st_size != member.file_size:
                raise RuntimeError(
                    f"Existing file has the wrong type or size; left untouched: {path}"
                )
            remaining -= member.file_size
        else:
            missing += 1
    reserve = int(args.reserve_gib * GIB)
    check_space(destination, remaining, reserve, missing)
    save_json(
        state / "inventory.json", {"files": len(files), "extracted_bytes": total_bytes}
    )
    (state / "complete.json").unlink(missing_ok=True)
    temporary = state / "extract.part"
    if temporary.is_symlink():
        raise RuntimeError("Unexpected symlink at extraction temporary path")
    last_log = last_space = time.monotonic()
    complete_bytes = 0
    with sqlite3.connect(state / "progress.sqlite") as journal:
        journal.execute(
            "CREATE TABLE IF NOT EXISTS files (name TEXT PRIMARY KEY, size INTEGER, crc INTEGER, mtime INTEGER)"
        )
        for index, member in enumerate(files, start=1):
            path = destination_for(destination, member)
            if path.exists():
                current = path.stat()
                recorded = journal.execute(
                    "SELECT size, crc, mtime FROM files WHERE name = ?",
                    (member.filename,),
                ).fetchone()
                if args.verify_existing or recorded != (
                    current.st_size,
                    member.CRC,
                    current.st_mtime_ns,
                ):
                    verify_existing(path, member)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, COPY_BYTES)
                if temporary.stat().st_size != member.file_size:
                    raise RuntimeError(f"Wrong extracted size: {member.filename}")
                os.link(temporary, path)
                temporary.unlink()
                remaining -= member.file_size
                missing -= 1
            current = path.stat()
            journal.execute(
                "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?)",
                (member.filename, current.st_size, member.CRC, current.st_mtime_ns),
            )
            complete_bytes += member.file_size
            now = time.monotonic()
            if index % 500 == 0 or now - last_log >= 30:
                journal.commit()
            if now - last_log >= 30:
                LOG.info(
                    "Extracted/confirmed %s/%s files, %.1f/%.1f GiB",
                    index,
                    len(files),
                    complete_bytes / GIB,
                    total_bytes / GIB,
                )
                last_log = now
            if now - last_space >= 300:
                check_space(destination, remaining, reserve, missing)
                last_space = now
        journal.commit()
    temporary.unlink(missing_ok=True)
    save_json(
        state / "complete.json",
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "files": len(files),
            "extracted_bytes": total_bytes,
            "source": json.loads((state / "source.json").read_text()),
            "rechecked_existing_crc": args.verify_existing,
        },
    )
    LOG.info(
        "COMPLETE: %s files, %.1f GiB in %s", len(files), total_bytes / GIB, destination
    )


class DownloadCLI(CLISettings):
    model_config = describe(
        "Download and extract the Kaggle MRI archive with bounded disk use and "
        "resumable chunks; rerun the same command to resume"
    )

    data_dir: Path = Field(
        default=Path(__file__).resolve().parents[2] / "data/raw",
        description="Extraction directory",
    )
    reserve_gib: float = Field(
        default=50, ge=0, allow_inf_nan=False, description="Space to leave free in GiB"
    )
    retries: int = Field(default=20, gt=0, description="Attempts per network operation")
    verify_existing: bool = Field(
        default=False,
        description="Re-read CRCs of all existing files; otherwise reuse journal entries with matching size/mtime",
    )


def main() -> None:
    args = DownloadCLI()
    destination = args.data_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    state = destination / ".download"
    state.mkdir(exist_ok=True)
    with (state / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another downloader is already using this destination"
            ) from None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(state / "download.log"),
            ],
        )
        cache = state / "chunks"
        cache.mkdir(exist_ok=True)
        check_space(destination, 0, int(args.reserve_gib * GIB))
        source = KaggleSource(args.retries)
        identity = {
            "competition": COMPETITION,
            "bytes": source.size,
            "etag": source.etag,
            "chunk_bytes": CHUNK_BYTES,
        }
        manifest = state / "source.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise RuntimeError(
                    "The archive identity changed; preserve this directory and choose a fresh --data-dir"
                )
        else:
            if any(cache.iterdir()) or (state / "progress.sqlite").exists():
                raise RuntimeError(
                    "Download state is missing its source identity; use a fresh destination"
                )
            save_json(manifest, identity)
        LOG.info("Archive: %.1f GiB; destination: %s", source.size / GIB, destination)
        with (
            ChunkedArchive(source, cache) as reader,
            zipfile.ZipFile(reader) as archive,
        ):
            extract(archive, destination, state, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.warning("Interrupted. Rerun the same command to resume.")
        raise SystemExit(130)
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
        zipfile.BadZipFile,
        zlib.error,
    ) as error:
        if isinstance(error, requests.RequestException):
            LOG.error("Network operation failed; rerun to resume")
        else:
            LOG.error("%s: %s", type(error).__name__, error)
        raise SystemExit(1)
