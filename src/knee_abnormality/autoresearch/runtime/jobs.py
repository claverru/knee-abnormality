import os
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from knee_abnormality.autoresearch.runtime.isolation import environment
from knee_abnormality.autoresearch.runtime.storage import (
    lock,
    process_identity,
    read_json,
    write_json,
)
from knee_abnormality.autoresearch.runtime.usage import capture_usage


def run_job(directory: Path):
    try:
        with lock(directory / "lease.lock"):
            if (directory / "receipt.json").exists():
                return
            if (directory / "started.json").exists():
                write_json(
                    directory / "receipt.json",
                    {
                        "exit_code": -1,
                        "error": "Interrupted job; refusing automatic replay",
                        "seconds": None,
                    },
                )
                return
            spec = read_json(directory / "spec.json")
            write_json(directory / "owner.json", process_identity(os.getpid()))
            process = None
            cancelled = False

            def stop(signum, frame):
                nonlocal cancelled
                cancelled = True
                if process is not None:
                    process.terminate()

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            started = time.monotonic()
            error = None
            code = -1
            env = environment(Path(spec["source"]), gpu=spec.get("gpu", ""))
            env.update(spec.get("environment", {}))
            resources = ExitStack()
            try:
                if spec.get("gpu_lock"):
                    try:
                        resources.enter_context(lock(Path(spec["gpu_lock"])))
                    except BlockingIOError:
                        write_json(
                            directory / "receipt.json",
                            {
                                "exit_code": 75,
                                "error": "GPU lease occupied",
                                "seconds": 0,
                            },
                        )
                        return
                    from knee_abnormality.autoresearch.supervisor.common import (
                        gpu_available,
                    )

                    if not gpu_available(spec["gpu"]):
                        write_json(
                            directory / "receipt.json",
                            {"exit_code": 75, "error": "GPU busy", "seconds": 0},
                        )
                        return
                with (
                    (directory / "stdin.txt").open("rb") as stdin,
                    (directory / "stdout.log").open("wb") as stdout,
                    (directory / "stderr.log").open("wb") as stderr,
                ):
                    process = subprocess.Popen(
                        spec["command"],
                        cwd=spec["source"],
                        env=env,
                        stdin=stdin,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    write_json(
                        directory / "started.json",
                        {
                            "time": time.time(),
                            "child": process_identity(process.pid),
                        },
                    )
                    try:
                        code = process.wait(timeout=spec["timeout_seconds"])
                    except subprocess.TimeoutExpired:
                        process.kill()
                        code = process.wait()
                        error = "Fixed job runtime ceiling exceeded"
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as exception:
                error = f"{type(exception).__name__}: {exception}"
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
            finally:
                resources.close()
            write_json(
                directory / "receipt.json",
                {
                    "exit_code": code,
                    "error": error,
                    "cancelled": cancelled,
                    "seconds": time.monotonic() - started,
                },
            )
            if spec.get("provider"):
                capture_usage(spec["provider"], directory)
    except BlockingIOError:
        return
