import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

FORWARDED = (
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "KNEE_PROJECT_ROOT",
    "KNEE_AUTORESEARCH_LIMITS",
    "HF_HUB_OFFLINE",
)
POLL_SECONDS = 20
HEARTBEAT_LIMIT_SECONDS = 300
OPEN_FILES = 1048576
RANGE_CEILING = 1.25
RESTART_MARGIN = 1.05
RESTART_RESERVE_SECONDS = 300
UPLOAD_EXCLUDES = (".git", ".venv", "__pycache__", "*.ckpt")
DOWNLOAD_EXCLUDES = ("*.ckpt", "/source", "/workspace")
GUARD = """#!/bin/bash
run={run}
touch "$run/heartbeat"
setsid bash "$run/job.sh" > "$run/log" 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  age=$(( $(date +%s) - $(stat -c %Y "$run/heartbeat") ))
  if [ $age -gt {limit} ] || [ -e "$run/cancel" ]; then
    kill -TERM -$pid 2>/dev/null
    sleep 20
    kill -KILL -$pid 2>/dev/null
  fi
  sleep 5
done
wait $pid
echo $? > "$run/exit"
"""


class TrainingHost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    address: str = Field(min_length=1)
    port: int = Field(gt=0, lt=65536)
    user: str = Field(min_length=1)
    key: str = Field(min_length=1)
    root: str = Field(pattern=r"^/[\w./-]+$")


class Launch(BaseModel):
    host: TrainingHost
    name: str = Field(pattern=r"^[\w.-]+$")
    command: list[str]
    cwd: str
    uploads: list[str]
    downloads: list[str]
    lock: str


def transport(host: TrainingHost) -> list[str]:
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-i",
        str(Path(host.key).expanduser()),
        "-p",
        str(host.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=15",
    ]


def remote(host: TrainingHost, command: list[str]) -> list[str]:
    return [*transport(host), f"{host.user}@{host.address}", shlex.join(command)]


def shell(
    host: TrainingHost, script: str, check: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*transport(host), f"{host.user}@{host.address}", "bash -s"],
        input=script,
        capture_output=True,
        text=True,
        check=check,
    )


def copy(host: TrainingHost, source: str, target: str, excludes: tuple[str, ...]):
    command = ["rsync", "-a", "--delete", "-e", shlex.join(transport(host))]
    for pattern in excludes:
        command += ["--exclude", pattern]
    subprocess.run([*command, source, target], check=True, capture_output=True)


def interpreter(host: dict | None) -> str:
    return (
        sys.executable
        if host is None
        else f"{host['root']}/environment/.venv/bin/python"
    )


def launcher(
    host: dict,
    name: str,
    command: list[str],
    cwd: Path,
    uploads: list[Path],
    downloads: list[Path],
    lock: Path,
) -> list[str]:
    launch = Launch(
        host=host,
        name=name,
        command=command,
        cwd=str(cwd),
        uploads=[str(path) for path in uploads],
        downloads=[str(path) for path in downloads],
        lock=str(lock),
    )
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        launch.model_dump_json(),
    ]


def prepare(launch: Launch) -> str:
    host = launch.host
    address = f"{host.user}@{host.address}"
    environment = f"{host.root}/environment"
    run = f"{host.root}/runs/{launch.name}"
    lock = Path(launch.lock)
    parents = sorted(
        {str(Path(path).parent) for path in [*launch.uploads, *launch.downloads]}
    )
    current = shell(
        host,
        f"mkdir -p {shlex.join([environment, run, *parents])}\n"
        f"rm -f {shlex.quote(run)}/exit {shlex.quote(run)}/cancel\n"
        f"sha256sum {shlex.quote(environment)}/uv.lock 2>/dev/null | cut -d' ' -f1\n",
        check=True,
    ).stdout.strip()
    digest = subprocess.run(
        ["sha256sum", str(lock)], check=True, capture_output=True, text=True
    ).stdout.split()[0]
    if current != digest:
        for name in ("pyproject.toml", "uv.lock"):
            copy(host, str(lock.parent / name), f"{address}:{environment}/{name}", ())
        shell(
            host,
            f"cd {shlex.quote(environment)} && ~/.local/bin/uv sync --locked --no-install-project\n",
            check=True,
        )
    for path in launch.uploads:
        source = f"{path}/" if Path(path).is_dir() else path
        target = f"{path}/" if Path(path).is_dir() else path
        copy(host, source, f"{address}:{target}", UPLOAD_EXCLUDES)
    exports = "".join(
        f"export {name}={shlex.quote(os.environ[name])}\n"
        for name in FORWARDED
        if name in os.environ
    )
    job = (
        f"#!/bin/bash\n{exports}export PYTHONFAULTHANDLER=1\n"
        f"ulimit -n {OPEN_FILES}\n"
        f"cd {shlex.quote(launch.cwd)}\nexec {shlex.join(launch.command)}\n"
    )
    guard = GUARD.format(run=shlex.quote(run), limit=HEARTBEAT_LIMIT_SECONDS)
    shell(
        host,
        f"cat > {shlex.quote(run)}/job.sh <<'KNEE_JOB'\n{job}KNEE_JOB\n"
        f"cat > {shlex.quote(run)}/guard.sh <<'KNEE_GUARD'\n{guard}KNEE_GUARD\n"
        f": > {shlex.quote(run)}/log\n"
        f"nohup setsid bash {shlex.quote(run)}/guard.sh > /dev/null 2>&1 < /dev/null &\n",
        check=True,
    )
    return run


def collect(launch: Launch) -> None:
    address = f"{launch.host.user}@{launch.host.address}"
    for path in launch.downloads:
        command = ["rsync", "-a", "-e", shlex.join(transport(launch.host))]
        for pattern in DOWNLOAD_EXCLUDES:
            command += ["--exclude", pattern]
        subprocess.run(
            [*command, f"{address}:{path}/", f"{path}/"],
            check=False,
            capture_output=True,
        )


def follow(launch: Launch, run: str) -> int:
    cancelled = []
    signal.signal(signal.SIGTERM, lambda *_: cancelled.append(True))
    signal.signal(signal.SIGINT, lambda *_: cancelled.append(True))
    quoted = shlex.quote(run)
    offset = 1
    silent_since = None
    while True:
        mark = f"touch {quoted}/cancel\n" if cancelled else ""
        poll = shell(
            launch.host,
            f"{mark}touch {quoted}/heartbeat\n"
            f"if [ -e {quoted}/exit ]; then echo EXIT $(cat {quoted}/exit); else echo RUNNING; fi\n"
            f"tail -c +{offset} {quoted}/log\n",
        )
        if poll.returncode:
            silent_since = silent_since or time.monotonic()
            if time.monotonic() - silent_since > 2 * HEARTBEAT_LIMIT_SECONDS:
                print("Training host unreachable; the run was ended there", flush=True)
                return 1
        else:
            silent_since = None
            status, _, output = poll.stdout.partition("\n")
            sys.stdout.write(output)
            sys.stdout.flush()
            offset += len(output.encode())
            collect(launch)
            if status.startswith("EXIT"):
                return int(status.split()[1])
        time.sleep(POLL_SECONDS)


def worker_crashed(launch: Launch, run: str) -> bool:
    log = shlex.quote(f"{run}/log")
    found = shell(
        launch.host,
        f"grep -a -q 'DataLoader worker' {log} && grep -a -q 'Segmentation fault' {log} && echo CRASH\n",
    )
    return "CRASH" in found.stdout


def fresh_run_fits(elapsed: float) -> bool:
    limits = os.environ.get("KNEE_AUTORESEARCH_LIMITS")
    if not launch_is_training() or limits is None:
        return False
    values = json.loads(Path(limits).read_text())
    training = values["max_training_seconds"] / RANGE_CEILING * RESTART_MARGIN
    return elapsed + training + RESTART_RESERVE_SECONDS < values["max_run_seconds"]


def launch_is_training() -> bool:
    return "scripts.pipeline.train" in sys.argv[1]


def discard(launch: Launch) -> None:
    shell(launch.host, f"rm -rf {shlex.join(launch.downloads)}\n", check=True)
    for path in launch.downloads:
        shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    launch = Launch.model_validate_json(sys.argv[1])
    started = time.monotonic()
    run = prepare(launch)
    code = follow(launch, run)
    if (
        code
        and worker_crashed(launch, run)
        and fresh_run_fits(time.monotonic() - started)
    ):
        print(
            "TRAINING HOST LAUNCHER: a data-loader worker died of a segmentation fault "
            f"{time.monotonic() - started:.0f} s into the job; the partial run is discarded and the "
            "same command starts again from scratch, once",
            flush=True,
        )
        discard(launch)
        code = follow(launch, prepare(launch))
    collect(launch)
    sys.exit(code)


if __name__ == "__main__":
    main()
