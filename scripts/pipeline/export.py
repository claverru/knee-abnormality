import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from email.parser import Parser
from importlib.metadata import PackageNotFoundError, metadata, version
from pathlib import Path

import torch
import yaml
from packaging.utils import parse_wheel_filename
from pydantic import Field
from pydantic_settings import SettingsError
from pytorch_lightning.plugins.io import TorchCheckpointIO

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import (
    INFERENCE_BATCH_SIZE,
    INFERENCE_NUM_WORKERS,
    TEST_IMAGE_MULTIPLIER,
    ExperimentConfig,
    load_config,
)
from knee_abnormality.identity import validate_preprocessing_identity
from knee_abnormality.inference.bundle import BUNDLE_SCHEMA_VERSION, bundle_hashes
from knee_abnormality.inference.checkpoint import (
    checkpoint_model,
    read_checkpoint,
    validate_checkpoint,
)
from knee_abnormality.provenance import git_revision, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = Path(__file__).resolve().with_name("inference.ipynb")
BUNDLED_SCRIPTS = (
    "scripts/__init__.py",
    "scripts/pipeline/__init__.py",
    "scripts/pipeline/predict.py",
    "scripts/pipeline/predict_worker.py",
)
INFERENCE_DISTRIBUTIONS = (
    "kornia",
    "pytorch-lightning",
    "numpy",
    "scipy",
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


class ExportCLI(CLISettings):
    checkpoint: Path
    output: Path
    wheel_dir: Path | None = None
    target_python: str | None = Field(default=None, pattern=r"^\d+\.\d+$")
    target_platform: str | None = None
    image_multiplier: int = Field(default=TEST_IMAGE_MULTIPLIER, gt=0)
    inference_batch_size: int = Field(default=INFERENCE_BATCH_SIZE, gt=0)
    inference_num_workers: int = Field(default=INFERENCE_NUM_WORKERS, ge=0)


def _bundled_files() -> list[str]:
    package = PROJECT_ROOT / "src" / "knee_abnormality"
    return [
        *(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in sorted(package.rglob("*.py"))
        ),
        *BUNDLED_SCRIPTS,
    ]


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
            result = _artifact_record(path)
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
    for name in _bundled_files():
        relative = Path(name)
        target = destination / (
            relative.relative_to("src") if relative.parts[0] == "src" else relative
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / name, target)


def _matches_training_snapshot(training_source: dict | None) -> bool:
    if training_source is None or training_source.get("source_snapshot") is None:
        return False
    snapshot = Path(training_source["source_snapshot"]["path"])
    with zipfile.ZipFile(snapshot) as archive:
        names = set(archive.namelist())
        for name in _bundled_files():
            if (
                name not in names
                or archive.read(name) != (PROJECT_ROOT / name).read_bytes()
            ):
                return False
    return True


def _source_hash(code: Path) -> str:
    inventory = json.dumps(
        bundle_hashes(code), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(inventory).hexdigest()


def _license_record(info) -> dict:
    return {
        "license": info.get("License-Expression")
        or info.get("License")
        or "not declared",
        "license_classifiers": [
            value for value in info.get_all("Classifier", []) if "License" in value
        ],
        "project_url": info.get("Home-page") or info.get("Project-URL"),
    }


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
                    **_license_record(info),
                    "wheel": source.name,
                }
            )
    return sorted(set(requirements)), copied, dependencies


def _dependency_record(name: str) -> dict:
    try:
        return {
            "name": name,
            "version": version(name),
            **_license_record(metadata(name)),
        }
    except PackageNotFoundError:
        return {"name": name, "version": None, "license": "not installed"}


def export_bundle(args: ExportCLI) -> dict:
    started = time.perf_counter()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = read_checkpoint(checkpoint_path)
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
    validate_checkpoint(checkpoint, saved_config, targets, fold)
    config = _portable_config(saved_config, fold)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "code").mkdir()
        (temporary / "wheels").mkdir()
        _copy_code(temporary / "code")
        bundled_preprocessing = validate_preprocessing_identity(
            saved.get("preprocessing_identity"),
            saved_config.model.backbone,
            temporary / "code" / "knee_abnormality",
        )
        source_hash = _source_hash(temporary / "code")
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
            "inference": {"image_multiplier": args.image_multiplier},
        }
        (temporary / "config.json").write_text(
            json.dumps(bundle_config, indent=2) + "\n", encoding="utf-8"
        )
        exported = _inference_checkpoint(checkpoint, config)
        TorchCheckpointIO().save_checkpoint(exported, temporary / "model.ckpt")
        portable_config = load_config(temporary / "config.json")
        validate_checkpoint(exported, portable_config, targets)
        checkpoint_model(exported, portable_config, targets, torch.device("cpu"))
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
        shutil.copy2(NOTEBOOK, temporary / "inference.ipynb")
        included_commit, included_status = git_revision(PROJECT_ROOT)
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
            "preprocessing_identity": bundled_preprocessing,
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
            "checks": {
                "checkpoint_metadata": "passed",
                "strict_model_load": "passed",
            },
            "runtime": {
                "device": "auto",
                "precision": "auto",
                "image_multiplier": args.image_multiplier,
                "batch_size": args.inference_batch_size,
                "batch_size_scope": "per_device",
                "num_workers": args.inference_num_workers,
                "num_workers_scope": "total",
                "parallelism": "one_process_per_visible_gpu",
                "automatic_precision_policy": "native CUDA BF16, other CUDA FP16 with fixed backbone chunks warmed at the input size, CPU FP32",
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
                    "preparation_seconds": manifest["preparation_seconds"],
                    "unmet_prerequisites": manifest["unmet_prerequisites"],
                }
            )
        )
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
        SettingsError,
    ) as error:
        print(f"Export error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
