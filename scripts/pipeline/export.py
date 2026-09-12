import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from email.parser import Parser
from importlib.metadata import PackageNotFoundError, metadata, version
from pathlib import Path
from typing import Literal

import torch
import yaml
from packaging.utils import parse_wheel_filename
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from pydantic_settings import CliImplicitFlag, SettingsError
from pytorch_lightning.plugins.io import TorchCheckpointIO

from knee_abnormality.bundle import (
    BUNDLE_SCHEMA_VERSION,
    bundle_hashes,
    sha256_file,
    verify_bundle,
)
from knee_abnormality.checkpoints import validate_checkpoint
from knee_abnormality.cli import NonExitingCLISettings
from knee_abnormality.config import ExperimentConfig, load_config
from knee_abnormality.inference import load_checkpoint_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DISTRIBUTIONS = (
    "kornia",
    "pytorch-lightning",
    "numpy",
    "polars",
    "pydantic",
    "pydantic-settings",
    "pydicom",
    "pyyaml",
    "timm",
    "torch",
    "torchvision",
)
CODEC_DISTRIBUTIONS = (
    "pylibjpeg",
    "pylibjpeg-libjpeg",
    "pylibjpeg-openjpeg",
)


class ExportCLI(NonExitingCLISettings):
    checkpoint: Path
    output: Path
    wheel_dir: Path | None = None
    target_python: str | None = Field(default=None, pattern=r"^\d+\.\d+$")
    target_platform: str | None = None
    verification_record: Path | None = None
    image_multiplier: int = Field(default=2, gt=0)
    inference_batch_size: int = Field(default=2, gt=0)
    quick_verify: CliImplicitFlag[bool] = False
    verification_samples: int = Field(default=16, ge=4)
    codec_reference_bundle: Path | None = None


class VerificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundled_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_max_abs_difference: FiniteFloat = Field(ge=0)
    validation_macro_auc: FiniteFloat | None = Field(default=None, ge=0, le=1)
    validation_scope: Literal["full_fold", "sample"] = "full_fold"
    validation_studies: int | None = Field(default=None, gt=0)
    inference_image_multiplier: int | None = Field(default=None, gt=0)
    details: dict = Field(default_factory=dict)
    example_submission_rows: int = Field(gt=0)
    example_submission_valid: bool
    jpeg_lossless_decoded: bool
    jpeg_2000_decoded: bool


def _command(args: list[str]) -> str | None:
    result = subprocess.run(
        args, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _portable_config(config: ExperimentConfig, fold: int) -> dict:
    values = config.model_dump(mode="json")
    values["paths"] = {
        "data_root": "data",
        "output_root": "output",
        "label_manifest": "label-manifest-not-required-for-test.json",
    }
    values["validation"]["folds"] = [fold]
    ExperimentConfig.model_validate(values, context={"config_dir": Path.cwd()})
    return values


def _inference_checkpoint(checkpoint: dict, config: dict) -> dict:
    allowed = ("state_dict", "epoch", "global_step", "pytorch-lightning_version")
    result = {name: checkpoint[name] for name in allowed if name in checkpoint}
    result["hyper_parameters"] = deepcopy(checkpoint["hyper_parameters"])
    result["hyper_parameters"]["config"] = config
    return result


def _scalar(value: object) -> int | float | str | None:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def _artifact_record(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "record": json.loads(path.read_text(encoding="utf-8")),
    }


def _checkpoint_results(checkpoint: dict, checkpoint_path: Path) -> dict:
    scores = []
    for state in checkpoint.get("callbacks", {}).values():
        if not isinstance(state, dict):
            continue
        for name in ("best_model_score", "current_score"):
            if name in state and state[name] is not None:
                scores.append({"name": name, "value": _scalar(state[name])})
    epoch = checkpoint.get("epoch")
    epoch_metrics = (
        _artifact_record(
            checkpoint_path.parent / "epochs" / f"metrics_epoch_{int(epoch):03d}.json"
        )
        if epoch is not None
        else None
    )
    selected = _artifact_record(checkpoint_path.parent / "selected_checkpoint.json")
    return {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "checkpoint_callback_scores": scores,
        "selected_checkpoint": selected,
        "epoch_metrics": epoch_metrics,
        "metrics_status": "recorded"
        if scores or selected or epoch_metrics
        else "not_present_with_checkpoint",
    }


def _training_source(checkpoint_path: Path) -> dict | None:
    for parent in checkpoint_path.parents:
        path = parent / "provenance.json"
        if path.is_file():
            result = {
                "path": str(path),
                "sha256": sha256_file(path),
                "record": json.loads(path.read_text(encoding="utf-8")),
            }
            snapshot = parent / "source_snapshot.zip"
            result["source_snapshot"] = (
                {"path": str(snapshot), "sha256": sha256_file(snapshot)}
                if snapshot.is_file()
                else None
            )
            return result
        if parent == PROJECT_ROOT:
            break
    return None


def _copy_code(destination: Path) -> None:
    package = destination / "knee_abnormality"
    shutil.copytree(
        PROJECT_ROOT / "src" / "knee_abnormality",
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    scripts = destination / "scripts"
    pipeline = scripts / "pipeline"
    pipeline.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "scripts" / "__init__.py", scripts / "__init__.py")
    shutil.copy2(
        PROJECT_ROOT / "scripts" / "pipeline" / "__init__.py",
        pipeline / "__init__.py",
    )
    shutil.copy2(
        PROJECT_ROOT / "scripts" / "pipeline" / "predict.py",
        pipeline / "predict.py",
    )
    utilities = scripts / "tools"
    utilities.mkdir()
    shutil.copy2(
        PROJECT_ROOT / "scripts" / "tools" / "verify_inference.py",
        utilities / "verify_inference.py",
    )


def _matches_training_snapshot(training_source: dict | None) -> bool:
    if training_source is None or training_source.get("source_snapshot") is None:
        return False
    snapshot = Path(training_source["source_snapshot"]["path"])
    source_paths = sorted((PROJECT_ROOT / "src" / "knee_abnormality").rglob("*.py"))
    source_paths.extend(
        [
            PROJECT_ROOT / "scripts" / "__init__.py",
            PROJECT_ROOT / "scripts" / "pipeline" / "__init__.py",
            PROJECT_ROOT / "scripts" / "pipeline" / "predict.py",
            PROJECT_ROOT / "scripts" / "tools" / "verify_inference.py",
        ]
    )
    with zipfile.ZipFile(snapshot) as archive:
        names = set(archive.namelist())
        for path in source_paths:
            name = path.relative_to(PROJECT_ROOT).as_posix()
            if name not in names or archive.read(name) != path.read_bytes():
                return False
    return True


def _source_hash(code: Path) -> str:
    inventory = json.dumps(
        bundle_hashes(code), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(inventory).hexdigest()


def _verification(
    path: Path | None, checkpoint_hash: str, source_hash: str, image_multiplier: int
) -> dict | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    value = VerificationRecord.model_validate_json(path.read_text(encoding="utf-8"))
    if value.checkpoint_sha256 != checkpoint_hash:
        raise ValueError("Verification record checkpoint hash does not match")
    if value.bundled_source_sha256 != source_hash:
        raise ValueError("Verification record bundled source hash does not match")
    if value.inference_image_multiplier != image_multiplier:
        raise ValueError(
            "Verification record must identify this inference image multiplier"
        )
    if not all(
        (
            value.example_submission_valid,
            value.jpeg_lossless_decoded,
            value.jpeg_2000_decoded,
        )
    ):
        raise ValueError("Verification record contains an unsuccessful required check")
    return {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        **value.model_dump(mode="json"),
    }


def _quick_verification(
    bundle: Path,
    args: ExportCLI,
    config: ExperimentConfig,
    epoch: int,
    checkpoint_hash: str,
    source_hash: str,
    image_multiplier: int,
) -> dict:
    if args.codec_reference_bundle is None:
        raise ValueError("Quick verification requires --codec-reference-bundle")
    reference = args.codec_reference_bundle.expanduser().resolve()
    previous = verify_bundle(reference)
    evidence = previous.get("verification") or {}
    if not previous.get("ready") or not all(
        evidence.get(name) for name in ("jpeg_lossless_decoded", "jpeg_2000_decoded")
    ):
        raise ValueError("Codec reference bundle has no successful decoding evidence")
    wheels = bundle_hashes(bundle / "wheels")
    if not wheels or wheels != bundle_hashes(reference / "wheels"):
        raise ValueError("Codec wheels differ from the verified reference bundle")
    request = {
        "config": config.model_dump(mode="json"),
        "reference_csv": str(
            args.checkpoint.expanduser().resolve().parent
            / "epochs"
            / f"validation_epoch_{epoch:03d}.csv"
        ),
        "samples": args.verification_samples,
        "image_multiplier": image_multiplier,
        "batch_size": args.inference_batch_size,
        "checkpoint_sha256": checkpoint_hash,
        "bundled_source_sha256": source_hash,
        "codec_evidence": {
            "bundle": str(reference),
            "manifest_sha256": sha256_file(reference / "manifest.json"),
            "wheel_hashes": wheels,
        },
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(bundle / "code")
    with tempfile.TemporaryDirectory(prefix="knee-verification-") as name:
        request_file = Path(name) / "request.json"
        request_file.write_text(json.dumps(request))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.tools.verify_inference",
                "--bundle",
                str(bundle),
                "--request-file",
                str(request_file),
            ],
            cwd=bundle,
            env=environment,
            check=True,
        )
    path = bundle / "verification.json"
    value = VerificationRecord.model_validate_json(path.read_text())
    return value.model_dump(mode="json")


def _wheel_requirements(
    wheel_dir: Path | None, destination: Path
) -> tuple[list[str], list[str], list[dict]]:
    requirements = []
    copied = []
    dependencies = []
    versions = {}
    if wheel_dir is not None:
        wheel_dir = wheel_dir.expanduser().resolve()
        if not wheel_dir.is_dir():
            raise FileNotFoundError(f"Wheel directory does not exist: {wheel_dir}")
        for source in sorted(wheel_dir.glob("*.whl")):
            name, wheel_version, _, _ = parse_wheel_filename(source.name)
            normalized_name = str(name)
            normalized_version = str(wheel_version)
            if (
                normalized_name in versions
                and versions[normalized_name] != normalized_version
            ):
                raise ValueError(
                    f"Wheel directory contains multiple {normalized_name} versions"
                )
            versions[normalized_name] = normalized_version
            requirements.append(f"{normalized_name}=={normalized_version}")
            shutil.copy2(source, destination / source.name)
            copied.append(source.name)
            with zipfile.ZipFile(source) as archive:
                metadata_names = [
                    value
                    for value in archive.namelist()
                    if value.endswith(".dist-info/METADATA")
                ]
                if len(metadata_names) != 1:
                    raise ValueError(f"Wheel has no unique METADATA file: {source}")
                info = Parser().parsestr(
                    archive.read(metadata_names[0]).decode("utf-8", errors="replace")
                )
            dependencies.append(
                {
                    "name": normalized_name,
                    "version": normalized_version,
                    "license": info.get("License-Expression")
                    or info.get("License")
                    or "not declared",
                    "license_classifiers": [
                        value
                        for value in info.get_all("Classifier", [])
                        if "License" in value
                    ],
                    "project_url": info.get("Home-page") or info.get("Project-URL"),
                    "wheel": source.name,
                }
            )
    return sorted(set(requirements)), copied, dependencies


def _dependency_record(name: str) -> dict:
    try:
        info = metadata(name)
        classifiers = [
            value for value in info.get_all("Classifier", []) if "License" in value
        ]
        return {
            "name": name,
            "version": version(name),
            "license": info.get("License-Expression")
            or info.get("License")
            or "not declared",
            "license_classifiers": classifiers,
            "project_url": info.get("Home-page") or info.get("Project-URL"),
        }
    except PackageNotFoundError:
        return {"name": name, "version": None, "license": "not installed"}


def _notebook() -> dict:
    setup = """import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

started = time.perf_counter()
configured = os.environ.get("KNEE_BUNDLE_DIR")
if configured:
    bundle = Path(configured).expanduser().resolve()
else:
    input_root = Path("/kaggle/input")
    manifests = sorted({path for pattern in ("*/manifest.json", "*/*/manifest.json", "*/*/*/manifest.json") for path in input_root.glob(pattern)}) if input_root.is_dir() else []
    archives = sorted({path for pattern in ("*/bundle.zip", "*/*/bundle.zip", "*/*/*/bundle.zip") for path in input_root.glob(pattern)}) if input_root.is_dir() else []
    if len(manifests) == 1:
        bundle = manifests[0].parent
    elif len(archives) == 1:
        bundle = Path("/kaggle/working/knee_bundle")
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True)
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(bundle)
    else:
        raise RuntimeError(f"Expected one mounted bundle, manifests={manifests}, archives={archives}")
wheels = sorted((bundle / "wheels").glob("*.whl"))
if wheels:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *map(str, wheels)], check=True)
manifest = json.loads((bundle / "manifest.json").read_text())
versions = {}
for item in manifest["dependencies"]:
    try:
        versions[item["name"]] = version(item["name"])
    except PackageNotFoundError:
        versions[item["name"]] = None
print(json.dumps({"bundle": str(bundle), "dependency_versions": versions, "setup_seconds": time.perf_counter() - started}))"""
    inference = """data_root = os.environ.get("KNEE_DATA_ROOT")
output = Path(os.environ.get("KNEE_OUTPUT", "/kaggle/working/submission.csv" if Path("/kaggle/working").is_dir() else "submission.csv")).expanduser().resolve()
command = [
    sys.executable,
    "-m",
    "scripts.pipeline.predict",
    str(bundle / "config.json"),
    "--bundle",
    str(bundle),
    "--output",
    str(output),
    "--device",
    os.environ.get("KNEE_DEVICE", "auto"),
    "--precision",
    os.environ.get("KNEE_PRECISION", "auto"),
    "--batch-size",
    os.environ.get("KNEE_BATCH_SIZE", str(manifest["runtime"].get("batch_size", 2))),
    "--num-workers",
    os.environ.get("KNEE_NUM_WORKERS", "8"),
]
if data_root:
    command.extend(["--data-root", data_root])
environment = os.environ.copy()
environment["PYTHONPATH"] = str(bundle / "code")
run_started = time.perf_counter()
subprocess.run(command, check=True, env=environment)
print(json.dumps({"inference_command_seconds": time.perf_counter() - run_started, "output": str(output)}))"""
    validate = """sys.path.insert(0, str(bundle / "code"))
import math
import csv
from knee_abnormality.bundle import discover_kaggle_data_root
from knee_abnormality.dataset import resolve_raw_dir

root = Path(data_root).expanduser().resolve() if data_root else discover_kaggle_data_root()
template_path = resolve_raw_dir(root, "test") / "sample_submission.csv"
with template_path.open(newline="") as stream:
    template = list(csv.reader(stream))
with output.open(newline="") as stream:
    submission = list(csv.reader(stream))
if not submission or submission[0] != template[0] or len(submission) != len(template):
    raise RuntimeError("Submission schema or row count differs from the mounted template")
if [row[0] for row in submission[1:]] != [row[0] for row in template[1:]]:
    raise RuntimeError("Submission study order differs from the mounted template")
for row in submission[1:]:
    values = [float(value) for value in row[1:]]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise RuntimeError(f"Invalid probabilities for study {row[0]}")
print(json.dumps({"validated_rows": len(submission) - 1, "columns": submission[0], "total_seconds": time.perf_counter() - started}))"""
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": setup,
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": inference,
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": validate,
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def export_bundle(args: ExportCLI) -> dict:
    started = time.perf_counter()
    if args.quick_verify and args.verification_record is not None:
        raise ValueError("Choose quick verification or an existing verification record")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("hyper_parameters")
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
        raise TypeError("Checkpoint must include saved configuration metadata")
    targets = tuple(saved.get("targets", ()))
    fold = saved.get("fold")
    if (
        not targets
        or len(set(targets)) != len(targets)
        or not all(isinstance(target, str) and target for target in targets)
        or type(fold) is not int
    ):
        raise ValueError("Checkpoint must include ordered targets and producing fold")
    saved_config = ExperimentConfig.model_validate(saved["config"])
    image_multiplier = args.image_multiplier
    validate_checkpoint(checkpoint, saved_config, targets, fold)
    config = _portable_config(saved_config, fold)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "code").mkdir()
        (temporary / "wheels").mkdir()
        _copy_code(temporary / "code")
        source_hash = _source_hash(temporary / "code")
        verification = _verification(
            args.verification_record, checkpoint_hash, source_hash, image_multiplier
        )
        if verification is not None:
            (temporary / "verification.json").write_text(
                json.dumps(verification, indent=2) + "\n", encoding="utf-8"
            )
        requirements, wheels, wheel_dependencies = _wheel_requirements(
            args.wheel_dir, temporary / "wheels"
        )
        (temporary / "requirements.txt").write_text(
            "".join(f"{requirement}\n" for requirement in requirements),
            encoding="utf-8",
        )
        bundle_config = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "experiment": config,
            "targets": list(targets),
            "fold": fold,
            "inference": {"image_multiplier": image_multiplier},
        }
        (temporary / "config.json").write_text(
            json.dumps(bundle_config, indent=2) + "\n", encoding="utf-8"
        )
        TorchCheckpointIO().save_checkpoint(
            _inference_checkpoint(checkpoint, config), temporary / "model.ckpt"
        )
        portable_model_config = load_config(temporary / "config.json")
        load_checkpoint_model(
            portable_model_config,
            targets,
            temporary / "model.ckpt",
            torch.device("cpu"),
        )
        if args.quick_verify:
            verification = _quick_verification(
                temporary,
                args,
                saved_config,
                int(checkpoint["epoch"]),
                checkpoint_hash,
                source_hash,
                image_multiplier,
            )
        dependencies_by_name = {
            record["name"]: record
            for record in (
                [_dependency_record(name) for name in INFERENCE_DISTRIBUTIONS]
                + wheel_dependencies
            )
        }
        dependencies = list(dependencies_by_name.values())
        third_party = {"dependencies": dependencies, "project_license": "not declared"}
        (temporary / "THIRD_PARTY.json").write_text(
            json.dumps(third_party, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "inference.ipynb").write_text(
            json.dumps(_notebook(), indent=1) + "\n", encoding="utf-8"
        )
        included_commit = _command(["git", "rev-parse", "HEAD"])
        included_status = _command(["git", "status", "--short"])
        training_source = _training_source(checkpoint_path)
        training_commit = (
            training_source.get("record", {}).get("git_commit")
            if training_source
            else None
        )
        source_matches = _matches_training_snapshot(training_source) or bool(
            training_commit
            and included_commit
            and training_commit == included_commit
            and not included_status
        )
        supplied_names = {
            requirement.split("==", 1)[0].lower() for requirement in requirements
        }
        unmet = []
        if args.target_python is None:
            unmet.append("Target Python version was not supplied")
        if args.target_platform is None:
            unmet.append("Target platform was not supplied")
        missing_codecs = [
            name for name in CODEC_DISTRIBUTIONS if name not in supplied_names
        ]
        if missing_codecs:
            unmet.append(
                f"Offline codec wheels were not supplied: {', '.join(missing_codecs)}"
            )
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "checkpoint": {
                "source_path": str(checkpoint_path),
                "source_sha256": checkpoint_hash,
                "exported_sha256": sha256_file(temporary / "model.ckpt"),
                "fold": fold,
                "targets": list(targets),
                "results": _checkpoint_results(checkpoint, checkpoint_path),
                "input_identity": saved.get("input_identity"),
                "label_provenance": _artifact_record(
                    checkpoint_path.parent / "label_provenance.json"
                ),
                "timing": _artifact_record(checkpoint_path.parent / "timing.json"),
            },
            "source": {
                "training": training_source,
                "included_git_commit": included_commit,
                "included_git_status": included_status,
                "bundled_source_sha256": source_hash,
                "matches_training_source": source_matches,
            },
            "target_environment": {
                "python": args.target_python,
                "platform": args.target_platform,
                "wheel_files": wheels,
            },
            "dependencies": dependencies,
            "verification": verification,
            "checks": {
                "checkpoint_metadata": "passed",
                "strict_model_load": "passed",
                "local_inference": "passed" if verification else "not_requested",
            },
            "runtime": {
                "device": "auto",
                "precision": "auto",
                "image_multiplier": image_multiplier,
                "batch_size": args.inference_batch_size,
                "automatic_precision_policy": "native CUDA BF16, otherwise FP32",
            },
            "unmet_prerequisites": unmet,
            "ready": not unmet,
        }
        manifest["files"] = bundle_hashes(temporary)
        manifest["preparation_seconds"] = time.perf_counter() - started
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    try:
        args = ExportCLI()
        manifest = export_bundle(args)
        print(
            json.dumps(
                {
                    "bundle": str(args.output.expanduser().resolve()),
                    "checkpoint_sha256": manifest["checkpoint"]["source_sha256"],
                    "fold": manifest["checkpoint"]["fold"],
                    "ready": manifest["ready"],
                    "local_inference": manifest["checks"]["local_inference"],
                    "preparation_seconds": manifest["preparation_seconds"],
                    "unmet_prerequisites": manifest["unmet_prerequisites"],
                }
            )
        )
    except (
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
        yaml.YAMLError,
        SettingsError,
    ) as error:
        print(f"Export error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
