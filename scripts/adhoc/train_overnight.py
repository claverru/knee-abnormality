import csv
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import CliImplicitFlag, SettingsError

from knee_abnormality.cli import NonExitingCLISettings
from knee_abnormality.config import ExperimentConfig, load_config

ROOT = Path(__file__).resolve().parents[2]


class OvernightCLI(NonExitingCLISettings):
    config: Path = ROOT / "configs/baseline.yaml"
    prepare_only: CliImplicitFlag[bool] = Field(
        default=False, description="Write validated experiment configs without training"
    )


def prepare_queue(config_path: Path) -> tuple[Path, list[Path]]:
    base = load_config(config_path).model_dump(mode="json")
    base["seed"] = 42
    base["validation"] = {"split": "all_5fold_v3", "folds": [0]}
    base["trainer"].update(
        max_epochs=10, patience=10, batch_size=8, accumulate_grad_batches=1
    )
    base["model"]["series_dropout"] = 0.2
    base["data"]["validation_image_multiplier"] = 2
    base["optimizer"].update(backbone_lr=1e-4, head_lr=3e-4, weight_decay=0.01)
    experiments = (
        ("01-control", {}),
        ("02-longer", {"trainer": {"max_epochs": 20, "patience": 20}}),
        ("03-lower-lr", {"optimizer": {"backbone_lr": 5e-5, "head_lr": 1.5e-4}}),
        ("04-higher-lr", {"optimizer": {"backbone_lr": 2e-4, "head_lr": 6e-4}}),
        ("05-faster-head", {"optimizer": {"head_lr": 1e-3}}),
        ("06-lower-decay", {"optimizer": {"weight_decay": 0.003}}),
        ("07-higher-decay", {"optimizer": {"weight_decay": 0.03}}),
    )
    configs = []
    for name, changes in experiments:
        values = deepcopy(base)
        values["name"] = name
        for section, settings in changes.items():
            values[section].update(settings)
        config = ExperimentConfig.model_validate(values)
        if not (config.split_dir / "protocol.json").is_file():
            raise FileNotFoundError(f"Missing prepared split: {config.split_dir}")
        configs.append(config)
    output_root = Path(base["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"overnight-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
    queue = Path(tempfile.mkdtemp(prefix=prefix, dir=output_root))
    for directory in ("configs", "logs", "runs"):
        (queue / directory).mkdir()
    paths = []
    for config in configs:
        values = config.model_dump(mode="json")
        values["paths"]["output_root"] = str(queue / "runs")
        path = queue / "configs" / f"{config.name}.yaml"
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return queue, paths


def run_command(command: list[str], log_path: Path) -> int:
    with (
        log_path.open("w", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            return process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise


def run_queue(queue: Path, configs: list[Path]) -> int:
    failed = False
    completed = []
    with (queue / "status.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["experiment", "exit_code", "run_dir", "log"])
        stream.flush()
        for config in configs:
            name = config.stem
            log_path = queue / "logs" / f"{name}.log"
            run_dir = queue / "runs" / name
            print(
                f"[{datetime.now(UTC).isoformat()}] Starting {name}: {log_path}",
                flush=True,
            )
            status = run_command(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.pipeline.train",
                    str(config),
                    "--run-id",
                    name,
                ],
                log_path,
            )
            writer.writerow([name, status, run_dir, log_path])
            stream.flush()
            print(f"Finished {name} with exit code {status}", flush=True)
            if status == 0:
                completed.append(str(run_dir))
            else:
                failed = True
    if completed:
        status = run_command(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.tools.plot_validation_metrics",
                *completed,
                "--output",
                str(queue / "validation_metrics.png"),
            ],
            queue / "logs" / "plot.log",
        )
        if status != 0:
            print(f"Comparison plot failed; see {queue / 'logs/plot.log'}", flush=True)
            failed = True
    print(f"Queue complete. Run status: {queue / 'status.tsv'}", flush=True)
    return int(failed)


def stop_queue(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def main() -> None:
    try:
        args = OvernightCLI()
        signal.signal(signal.SIGINT, stop_queue)
        signal.signal(signal.SIGTERM, stop_queue)
        limits_before = resource.getrlimit(resource.RLIMIT_NOFILE)
        if not args.prepare_only:
            resource.setrlimit(
                resource.RLIMIT_NOFILE, (limits_before[1], limits_before[1])
            )
        queue, configs = prepare_queue(args.config)
        print(f"Prepared {len(configs)} experiments: {queue}", flush=True)
        if args.prepare_only:
            return
        limits_after = resource.getrlimit(resource.RLIMIT_NOFILE)
        (queue / "runtime.json").write_text(
            json.dumps(
                {
                    "open_file_limits_before": limits_before,
                    "open_file_limits": limits_after,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Open-file limit: {limits_after[0]} (was {limits_before[0]})", flush=True
        )
        status = run_queue(queue, configs)
    except (OSError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Overnight queue error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(status)


if __name__ == "__main__":
    main()
