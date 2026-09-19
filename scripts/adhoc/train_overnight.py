import csv
import hashlib
import json
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import CliImplicitFlag, SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import ExperimentConfig, load_config
from knee_abnormality.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("src", "scripts", "configs")
SUMMARY_HEADER = [
    "experiment",
    "selected_epoch",
    "best_val_macro_auc",
    "last_val_macro_auc",
    "delta_best_vs_control",
    "delta_last_vs_control",
    "live_best_val_macro_auc",
    "live_last_val_macro_auc",
    "soft_target_bce_at_selected",
    "gold_macro_auc_at_selected",
    "training_seconds",
    "batch_norm_refresh_seconds",
    "optimizer_steps",
    "peak_cuda_memory_gib",
]

EXPERIMENTS = (
    (
        "00-control",
        {
            "seed": 927831896,
        },
    ),
    (
        "01-eps-1e-6",
        {
            "seed": 1491446251,
            "optimizer": {"epsilon": 0.000001},
        },
    ),
    (
        "02-intensity-p-0.25",
        {
            "seed": 788474613,
            "data": {"intensity_augmentation": {"probability": 0.25}},
        },
    ),
    (
        "03-beta1-0.95",
        {
            "seed": 663550682,
            "optimizer": {"beta1": 0.95},
        },
    ),
    (
        "04-noise-std-0.01-0.04",
        {
            "seed": 10764577,
            "data": {"noise_augmentation": {"std_range": [0.01, 0.04]}},
        },
    ),
)


class OvernightCLI(CLISettings):
    config: Path = ROOT / "configs/training.yaml"
    prepare_only: CliImplicitFlag[bool] = Field(
        default=False, description="Write validated experiment configs without training"
    )


def merge(values: dict, changes: dict) -> None:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(values.get(key), dict):
            merge(values[key], value)
        else:
            values[key] = value


def override_keys(base: dict, changes: dict, prefix: str = "") -> list[str]:
    keys = []
    for key, value in changes.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            keys.extend(override_keys(base[key], value, f"{path}."))
        else:
            keys.append(path)
    return keys


def source_fingerprint() -> str:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *SOURCE_DIRS,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    digest = hashlib.sha256()
    for relative in sorted(result.stdout.splitlines()):
        digest.update(relative.encode())
        digest.update(sha256_file(ROOT / relative).encode())
    return digest.hexdigest()


def gpu_processes() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def prepare_queue(
    config_path: Path,
) -> tuple[Path, list[Path], dict[str, list[str]]]:
    base = load_config(config_path).model_dump(mode="json")
    base["seed"] = 42
    base["validation"] = {"split": "all_5fold_v3", "folds": [0]}
    base["data"]["validation_image_multiplier"] = 1
    configs = []
    overrides = {}
    for name, changes in EXPERIMENTS:
        values = deepcopy(base)
        values["name"] = name
        merge(values, changes)
        overrides[name] = override_keys(base, changes)
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
    return queue, paths, overrides


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


def read_epoch_metrics(fold_dir: Path, prefix: str) -> list[tuple[int, dict]]:
    pattern = re.compile(rf"^{re.escape(prefix)}metrics_epoch_(\d+)\.json$")
    records = []
    for path in sorted((fold_dir / "epochs").glob(f"{prefix}metrics_epoch_*.json")):
        match = pattern.match(path.name)
        if match is not None:
            records.append((int(match.group(1)), json.loads(path.read_text())))
    return sorted(records, key=lambda item: item[0])


def run_summary(run_dir: Path) -> dict | None:
    folds = sorted(run_dir.glob("fold_*"))
    if len(folds) != 1:
        return None
    fold_dir = folds[0]
    selected_path = fold_dir / "selected_checkpoint.json"
    timing_path = fold_dir / "timing.json"
    if not selected_path.is_file() or not timing_path.is_file():
        return None
    selected = json.loads(selected_path.read_text())
    timing = json.loads(timing_path.read_text())
    records = read_epoch_metrics(fold_dir, "")
    if not records:
        return None
    by_epoch = dict(records)
    match = re.search(r"epoch=(\d+)", Path(selected["path"]).name)
    selected_index = int(match.group(1)) if match is not None else records[-1][0]
    selected_record = by_epoch.get(selected_index, records[-1][1])
    _, last_record = records[-1]
    live_records = read_epoch_metrics(fold_dir, "live_")
    live_best = (
        max(record["macro_auc"] for _, record in live_records) if live_records else None
    )
    live_last = live_records[-1][1]["macro_auc"] if live_records else None
    peak_bytes = timing.get("peak_cuda_allocated_bytes")
    return {
        "selected_epoch": selected_index + 1,
        "best_val_macro_auc": selected["score"],
        "last_val_macro_auc": last_record["macro_auc"],
        "live_best_val_macro_auc": live_best,
        "live_last_val_macro_auc": live_last,
        "soft_target_bce_at_selected": selected_record["soft_target_bce"],
        "gold_macro_auc_at_selected": selected_record["gold"]["macro_auc"],
        "training_seconds": timing["training_seconds"],
        "batch_norm_refresh_seconds": timing.get("batch_norm_refresh_seconds", 0.0),
        "optimizer_steps": timing["optimizer_steps"],
        "peak_cuda_memory_gib": peak_bytes / 2**30 if peak_bytes is not None else None,
    }


def write_summary(queue: Path, configs: list[Path]) -> None:
    summaries = {}
    for config in configs:
        summary = run_summary(queue / "runs" / config.stem)
        if summary is not None:
            summaries[config.stem] = summary
    control = summaries.get("00-control")
    with (queue / "summary.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(SUMMARY_HEADER)
        for config in configs:
            summary = summaries.get(config.stem)
            if summary is None:
                continue
            delta_best = (
                summary["best_val_macro_auc"] - control["best_val_macro_auc"]
                if control is not None
                else None
            )
            delta_last = (
                summary["last_val_macro_auc"] - control["last_val_macro_auc"]
                if control is not None
                else None
            )
            writer.writerow(
                [
                    config.stem,
                    summary["selected_epoch"],
                    summary["best_val_macro_auc"],
                    summary["last_val_macro_auc"],
                    delta_best,
                    delta_last,
                    summary["live_best_val_macro_auc"],
                    summary["live_last_val_macro_auc"],
                    summary["soft_target_bce_at_selected"],
                    summary["gold_macro_auc_at_selected"],
                    summary["training_seconds"],
                    summary["batch_norm_refresh_seconds"],
                    summary["optimizer_steps"],
                    summary["peak_cuda_memory_gib"],
                ]
            )


def experiment_folds(name: str) -> list[int]:
    for experiment, changes in EXPERIMENTS:
        if experiment == name:
            return changes.get("validation", {}).get("folds", [0])
    return [0]


def print_comparison_commands(
    queue: Path,
    configs: list[Path],
    completed: set[str],
    overrides: dict[str, list[str]],
) -> None:
    control = EXPERIMENTS[0][0]
    if control not in completed:
        print(f"Skipping comparison commands: {control} did not complete", flush=True)
        return
    control_folds = experiment_folds(control)
    control_dir = queue / "runs" / control
    for config in configs:
        name = config.stem
        if name == control or name not in completed:
            continue
        if experiment_folds(name) != control_folds:
            continue
        allowed = ",".join(["name", *overrides[name]])
        candidate_dir = queue / "runs" / name
        output_dir = (
            ROOT / "outputs" / "analysis" / f"{queue.name}-{name}-vs-{control}"
        )
        print(
            "rtk proxy uv run --locked python -m scripts.tools.compare_fold_runs "
            f"{control_dir} {candidate_dir} --output-dir {output_dir} "
            f"--allowed-config-differences {allowed}",
            flush=True,
        )


def run_queue(
    queue: Path,
    configs: list[Path],
    overrides: dict[str, list[str]],
    fingerprint: str,
) -> int:
    failed = False
    stopped = False
    completed_dirs = []
    completed_names = set()
    with (queue / "status.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "experiment",
                "started_at",
                "finished_at",
                "wall_seconds",
                "exit_code",
                "run_dir",
                "log",
            ]
        )
        stream.flush()
        for config in configs:
            if source_fingerprint() != fingerprint:
                print(
                    "Source under src/, scripts/ or configs/ changed since queue "
                    "start; stopping the queue.",
                    file=sys.stderr,
                    flush=True,
                )
                stopped = True
                break
            name = config.stem
            log_path = queue / "logs" / f"{name}.log"
            run_dir = queue / "runs" / name
            started = datetime.now(UTC)
            print(
                f"[{started.isoformat()}] Starting {name}: {log_path}",
                flush=True,
            )
            wall_started = time.perf_counter()
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
            wall_seconds = time.perf_counter() - wall_started
            finished = datetime.now(UTC)
            writer.writerow(
                [
                    name,
                    started.isoformat(),
                    finished.isoformat(),
                    f"{wall_seconds:.3f}",
                    status,
                    run_dir,
                    log_path,
                ]
            )
            stream.flush()
            print(f"Finished {name} with exit code {status}", flush=True)
            if status == 0:
                completed_dirs.append(str(run_dir))
                completed_names.add(name)
            else:
                failed = True
            write_summary(queue, configs)
    if completed_dirs:
        status = run_command(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.tools.plot_validation_metrics",
                *completed_dirs,
                "--output",
                str(queue / "validation_metrics.png"),
            ],
            queue / "logs" / "plot.log",
        )
        if status != 0:
            print(f"Comparison plot failed; see {queue / 'logs/plot.log'}", flush=True)
            failed = True
    print(f"Queue complete. Run status: {queue / 'status.tsv'}", flush=True)
    print(f"Run summary: {queue / 'summary.tsv'}", flush=True)
    print_comparison_commands(queue, configs, completed_names, overrides)
    if stopped:
        return 3
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
            busy = gpu_processes()
            if busy:
                print(
                    f"Refusing to start: {len(busy)} active GPU compute process(es)",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            resource.setrlimit(
                resource.RLIMIT_NOFILE, (limits_before[1], limits_before[1])
            )
        queue, configs, overrides = prepare_queue(args.config)
        print(f"Prepared {len(configs)} experiments: {queue}", flush=True)
        if args.prepare_only:
            return
        fingerprint = source_fingerprint()
        limits_after = resource.getrlimit(resource.RLIMIT_NOFILE)
        (queue / "runtime.json").write_text(
            json.dumps(
                {
                    "open_file_limits_before": limits_before,
                    "open_file_limits": limits_after,
                    "source_fingerprint": fingerprint,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Open-file limit: {limits_after[0]} (was {limits_before[0]})", flush=True
        )
        status = run_queue(queue, configs, overrides, fingerprint)
    except (
        OSError,
        ValueError,
        yaml.YAMLError,
        SettingsError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Overnight queue error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(status)


if __name__ == "__main__":
    main()
