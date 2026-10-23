import os
import shutil
import tempfile
from pathlib import Path

COMPILE_CACHE = "compile-cache.bin"
COMPILE_CACHE_OUTPUT = "KNEE_COMPILE_CACHE_OUTPUT"


def use_compile_cache(bundle: Path) -> Path | None:
    shipped = bundle / COMPILE_CACHE
    if not shipped.is_file() and COMPILE_CACHE_OUTPUT not in os.environ:
        return None
    local = Path(tempfile.mkdtemp(prefix="knee-inductor-"))
    if shipped.is_file():
        shutil.unpack_archive(shipped, local, "tar")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(local)
    os.environ["TRITON_CACHE_DIR"] = str(local / "triton")
    return local


def pack_compile_cache(local: Path, destination: Path) -> None:
    archive = shutil.make_archive(
        str(destination.with_name(destination.name + ".part")), "tar", local
    )
    Path(archive).replace(destination)
