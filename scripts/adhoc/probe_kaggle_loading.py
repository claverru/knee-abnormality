import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from kaggle.api.kaggle_api_extended import KaggleApi
from pydantic import Field
from pydantic_settings import CliPositionalArg

from knee_abnormality.cli import CLISettings
from scripts.pipeline.submit import _jsonable, _kernel_status

PROBES = ("header", "frames", "resample", "quantiles", "indexing")
SOURCE = Path(__file__).parent / "kaggle_loading_probes"
TERMINAL = ("complete", "error", "fail", "cancel")


class ProbeCLI(CLISettings):
    command: CliPositionalArg[Literal["launch", "poll", "collect"]]
    output_dir: Path
    owner: str = "claverru"
    probes: list[str] = Field(default_factory=lambda: list(PROBES))
    suffix: str = datetime.now(UTC).strftime("%Y%m%d")
    concurrency: int = Field(default=5, gt=0, le=5)
    poll_seconds: int = Field(default=45, gt=0)
    timeout_seconds: int = Field(default=32400, gt=0, le=32400)


def build_script(probe: str, directory: Path) -> Path:
    prelude = (SOURCE / "common.py").read_text()
    body = (SOURCE / f"{probe}.py").read_text()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "probe.py"
    path.write_text(f"{prelude}\n\n{body}")
    return path


def write_metadata(probe: str, slug: str, directory: Path, owner: str) -> None:
    (directory / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{owner}/{slug}",
                "title": slug.replace("-", " ").title(),
                "code_file": "probe.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": False,
                "competition_sources": ["rsna-knee-abnormality-detection"],
                "dataset_sources": [],
            },
            indent=2,
        )
        + "\n"
    )


def safe_status(api: KaggleApi, notebook: str) -> tuple[str | None, object]:
    for attempt in range(5):
        try:
            return _kernel_status(api, notebook)
        except Exception as error:
            if attempt == 4:
                return None, {"poll_error": str(error)}
            time.sleep(5 * (attempt + 1))
    return None, None


def harvest(api: KaggleApi, args: ProbeCLI, status: dict, status_path: Path, active: dict) -> None:
    output = args.output_dir.expanduser().resolve()
    while active:
        time.sleep(args.poll_seconds)
        for probe, item in list(active.items()):
            state, response = safe_status(api, item["notebook"])
            if state is None:
                continue
            if not any(value in state for value in TERMINAL):
                item["seen_active"] = True
                status["runs"][probe]["state"] = state
                status_path.write_text(json.dumps(status, indent=2) + "\n")
                continue
            if not item["seen_active"] and time.monotonic() - item["pushed"] < 600:
                continue
            run_dir = output / "runs" / probe
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)
            try:
                api.kernels_output(item["notebook"], path=str(run_dir), force=True, quiet=True)
                (run_dir / "kernel.log").write_text(api.kernels_logs(item["notebook"]))
            except Exception as error:
                status["runs"][probe]["download_error"] = str(error)
            status["runs"][probe].update(
                state=state,
                finished_at=datetime.now(UTC).isoformat(),
                kernel_status=_jsonable(response),
            )
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            del active[probe]


def poll(args: ProbeCLI) -> dict:
    output = args.output_dir.expanduser().resolve()
    status_path = output / "status.json"
    status = json.loads(status_path.read_text())
    api = KaggleApi()
    api.authenticate()
    active = {
        probe: {"notebook": run["notebook"], "seen_active": True, "pushed": 0.0}
        for probe, run in status.get("runs", {}).items()
        if run.get("notebook") and "finished_at" not in run
    }
    harvest(api, args, status, status_path, active)
    return status


def launch(args: ProbeCLI) -> dict:
    unknown = sorted(set(args.probes) - set(PROBES))
    if unknown:
        raise ValueError(f"Unknown probes: {unknown}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"runs": {}}
    api = KaggleApi()
    api.authenticate()
    pending = list(args.probes)
    active: dict[str, dict] = {}
    while pending or active:
        while pending and len(active) < args.concurrency:
            probe = pending[0]
            slug = f"knee-loadprobe-{probe}-{args.suffix}"
            directory = output / "kernels" / probe
            build_script(probe, directory)
            write_metadata(probe, slug, directory, args.owner)
            response = api.kernels_push(str(directory), str(args.timeout_seconds), None)
            error = getattr(response, "error", None) if response is not None else "empty"
            if error and "Maximum batch" in str(error):
                break
            pending.pop(0)
            run = {
                "notebook": f"{args.owner}/{slug}",
                "pushed_at": datetime.now(UTC).isoformat(),
                "push_response": _jsonable(response),
            }
            if error:
                run["state"] = "push_failed"
            else:
                run["state"] = "pushed"
                active[probe] = {
                    "notebook": f"{args.owner}/{slug}",
                    "seen_active": False,
                    "pushed": time.monotonic(),
                }
            status["runs"][probe] = run
            status_path.write_text(json.dumps(status, indent=2) + "\n")
        harvest(api, args, status, status_path, active)
    return status


def collect(args: ProbeCLI) -> dict:
    output = args.output_dir.expanduser().resolve()
    summary = {}
    for probe in args.probes:
        path = output / "runs" / probe / f"{probe}.json"
        summary[probe] = (
            json.loads(path.read_text()) if path.exists() else {"missing": str(path)}
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = ProbeCLI()
    handlers = {"launch": launch, "poll": poll, "collect": collect}
    result = handlers[args.command](args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
