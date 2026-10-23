import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from email.parser import Parser
from importlib.metadata import PackageNotFoundError, metadata, version
from pathlib import Path
from typing import NamedTuple

import torch
from packaging.utils import parse_wheel_filename
from pydantic import Field
from pytorch_lightning.plugins.io import TorchCheckpointIO

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import (
    ATTENTION_TEMPERATURE,
    INFERENCE_BATCH_SIZE,
    INFERENCE_NUM_WORKERS,
    TEST_IMAGE_MULTIPLIER,
    ExperimentConfig,
    PositiveFiniteFloat,
    load_config,
)
from knee_abnormality.identity import validate_preprocessing_identity
from knee_abnormality.inference.bundle import BUNDLE_SCHEMA_VERSION, bundle_hashes
from knee_abnormality.inference.checkpoint import (
    bake_attention_temperature,
    checkpoint_model,
    ensemble_member_path,
    half_precision_weights,
    read_checkpoint,
    validate_checkpoint,
)
from knee_abnormality.inference.processes import BackboneMode
from knee_abnormality.kaggle import (
    BACKBONE_WHEELS,
    KAGGLE_PLATFORM,
    KAGGLE_PYTHON,
    KAGGLE_WHEELS,
    OFFLINE_WHEELS,
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
BUNDLED_DATA = ("src/knee_abnormality/inference/kaggle_startup_pages.json",)
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
OFFLINE_DISTRIBUTIONS = tuple(wheel.split("==")[0] for wheel in OFFLINE_WHEELS)


class ExportCLI(CLISettings):
    model_config = describe(
        "Export one or more scored training checkpoints as an atomic, portable "
        "inference bundle and notebook; several checkpoints are averaged as an ensemble"
    )

    checkpoint: list[Path] = Field(
        min_length=1, description="Training checkpoints, one per ensemble member"
    )
    output: Path = Field(description="New bundle directory; must not exist")
    wheel_dir: Path = Field(
        default=KAGGLE_WHEELS,
        description="Offline wheels to bundle, from prepare-codecs",
    )
    target_python: str = Field(
        default=KAGGLE_PYTHON,
        pattern=r"^\d+\.\d+$",
        description="Kaggle Python version",
    )
    target_platform: str = Field(
        default=KAGGLE_PLATFORM, description="Kaggle wheel platform tag"
    )
    image_multiplier: int = Field(
        default=TEST_IMAGE_MULTIPLIER, gt=0, description="Test-time image multiplier"
    )
    attention_temperature: list[PositiveFiniteFloat] = Field(
        default_factory=lambda: [ATTENTION_TEMPERATURE],
        description="Pooling softmax temperature: one value for all members or one per checkpoint",
    )
    inference_batch_size: int = Field(
        default=INFERENCE_BATCH_SIZE,
        gt=0,
        description="Studies per GPU batch on Kaggle",
    )
    inference_num_workers: int = Field(
        default=INFERENCE_NUM_WORKERS,
        ge=0,
        description="Total loader workers across GPUs on Kaggle",
    )
    reflection_tta: bool = Field(
        default=False,
        description="Also run each member on the mirrored batch and average per study by reflection eligibility",
    )
    backbone: BackboneMode = Field(
        default="graph",
        description="FP16 backbone on Kaggle: graph compiles it and captures one CUDA graph per member, with a compile cache the submission job builds; eager runs it uncaptured, for ensembles whose graph pools do not fit the GPU",
    )


class Member(NamedTuple):
    path: Path
    hash: str
    checkpoint: dict
    saved: dict
    targets: tuple[str, ...]
    fold: int
    config: ExperimentConfig


def _bundled_files() -> list[str]:
    package = PROJECT_ROOT / "src" / "knee_abnormality"
    return [
        *(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in sorted(package.rglob("*.py"))
        ),
        *BUNDLED_DATA,
        *BUNDLED_SCRIPTS,
    ]


def _portable_config(config: ExperimentConfig, fold: int) -> dict:
    values = config.model_dump(mode="json")
    values["paths"] = {
        "data_root": "data",
        "output_root": "output",
        "label_manifest": "label-manifest-not-required-for-test.json",
        "anatomy_targets": "anatomy-targets-not-required-for-test",
        "meniscus_targets": "meniscus-targets-not-required-for-test",
    }
    values["validation"]["folds"] = [fold]
    ExperimentConfig.model_validate(values, context={"config_dir": Path.cwd()})
    return values


def _inference_checkpoint(checkpoint: dict, config: dict) -> dict:
    allowed = ("epoch", "global_step", "pytorch-lightning_version")
    result = {name: checkpoint[name] for name in allowed if name in checkpoint}
    result["state_dict"] = {
        key: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
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


def _source_matches(
    training_source: dict | None, included_commit, included_status
) -> bool:
    training_commit = (
        training_source.get("record", {}).get("git_commit") if training_source else None
    )
    return _matches_training_snapshot(training_source) or bool(
        training_commit
        and included_commit
        and training_commit == included_commit
        and not included_status
    )


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
    wheel_dirs: list[Path], destination: Path
) -> tuple[list[str], list[str], list[dict]]:
    requirements = []
    copied = []
    dependencies = []
    versions = {}
    wheel_dirs = [wheel_dir.expanduser().resolve() for wheel_dir in wheel_dirs]
    for wheel_dir in wheel_dirs:
        if not wheel_dir.is_dir():
            raise FileNotFoundError(f"Wheel directory does not exist: {wheel_dir}")
    for source in sorted(
        source for wheel_dir in wheel_dirs for source in wheel_dir.glob("*.whl")
    ):
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


def _read_member(checkpoint_path: Path) -> Member:
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
    return Member(
        checkpoint_path, checkpoint_hash, checkpoint, saved, targets, fold, saved_config
    )


def _validate_members(members: list[Member]) -> None:
    reference = members[0]
    for other in members[1:]:
        if other.targets != reference.targets:
            raise ValueError("Checkpoints disagree on target names or order")
        if other.config.validation.split != reference.config.validation.split:
            raise ValueError("Checkpoints disagree on validation.split")
        reference_data = reference.config.data.model_dump(mode="json")
        other_data = other.config.data.model_dump(mode="json")
        for name, value in reference_data.items():
            if other_data[name] != value:
                raise ValueError(f"Checkpoints disagree on data.{name}")


def _resolve_temperatures(values: list[float], count: int) -> list[float]:
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(
            "attention_temperature must supply exactly one value or one value per checkpoint"
        )
    return values


def export_bundle(args: ExportCLI) -> dict:
    started = time.perf_counter()
    checkpoint_paths = [path.expanduser().resolve() for path in args.checkpoint]
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ValueError("Checkpoint paths must be distinct")
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temperatures = _resolve_temperatures(
        args.attention_temperature, len(checkpoint_paths)
    )
    members = [_read_member(path) for path in checkpoint_paths]
    _validate_members(members)
    fold = members[0].fold
    targets = members[0].targets
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "code").mkdir()
        (temporary / "wheels").mkdir()
        _copy_code(temporary / "code")
        source_hash = _source_hash(temporary / "code")
        requirements, wheels, wheel_dependencies = _wheel_requirements(
            [
                args.wheel_dir,
                *sorted(
                    {
                        BACKBONE_WHEELS[member.config.model.backbone]
                        for member in members
                        if member.config.model.backbone in BACKBONE_WHEELS
                    }
                ),
            ],
            temporary / "wheels",
        )
        (temporary / "requirements.txt").write_text(
            "".join(f"{requirement}\n" for requirement in requirements),
            encoding="utf-8",
        )
        included_commit, included_status = git_revision(PROJECT_ROOT)

        if len(members) == 1:
            member = members[0]
            config = _portable_config(member.config, fold)
            bundled_preprocessing = validate_preprocessing_identity(
                member.saved.get("preprocessing_identity"),
                member.config.model.backbone,
                temporary / "code" / "knee_abnormality",
            )
            bundle_config = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "experiment": config,
                "targets": list(targets),
                "fold": fold,
                "inference": {
                    "image_multiplier": args.image_multiplier,
                    "attention_temperature": temperatures[0],
                    "reflection_tta": args.reflection_tta,
                    "backbone": args.backbone,
                },
            }
            (temporary / "config.json").write_text(
                json.dumps(bundle_config, indent=2) + "\n", encoding="utf-8"
            )
            exported = _inference_checkpoint(member.checkpoint, config)
            portable_config = load_config(temporary / "config.json")
            validate_checkpoint(exported, portable_config, targets)
            exported = half_precision_weights(
                exported,
                checkpoint_model(
                    exported,
                    portable_config,
                    targets,
                    torch.device("cpu"),
                    temperatures[0],
                ),
            )
            TorchCheckpointIO().save_checkpoint(exported, temporary / "model.ckpt")
            checkpoint_model(
                exported,
                portable_config,
                targets,
                torch.device("cpu"),
                temperatures[0],
            )
            training_source = _training_source(member.path)
            source_matches = _source_matches(
                training_source, included_commit, included_status
            )
            checkpoint_record = {
                "source_path": str(member.path),
                "source_sha256": member.hash,
                "exported_sha256": sha256_file(temporary / "model.ckpt"),
                "fold": fold,
                "targets": list(targets),
                "results": _checkpoint_results(member.checkpoint, member.path),
                "input_identity": member.saved.get("input_identity"),
                "label_provenance": _artifact_record(
                    member.path.parent / "label_provenance.json"
                ),
                "timing": _artifact_record(member.path.parent / "timing.json"),
            }
            manifest_checkpoint_fields = {
                "preprocessing_identity": bundled_preprocessing,
                "checkpoint": checkpoint_record,
                "source": {
                    "training": training_source,
                    "included_git_commit": included_commit,
                    "included_git_status": included_status,
                    "bundled_source_sha256": source_hash,
                    "matches_training_source": source_matches,
                },
            }
            runtime_attention_temperature = temperatures[0]
        else:
            representative = _portable_config(members[0].config, fold)
            bundle_config = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "experiment": representative,
                "targets": list(targets),
                "fold": fold,
                "inference": {
                    "image_multiplier": args.image_multiplier,
                    "attention_temperature": 1.0,
                    "reflection_tta": args.reflection_tta,
                    "backbone": args.backbone,
                },
            }
            (temporary / "config.json").write_text(
                json.dumps(bundle_config, indent=2) + "\n", encoding="utf-8"
            )
            (temporary / "models").mkdir()
            checkpoint_records = []
            for index, (member, temperature) in enumerate(
                zip(members, temperatures, strict=True)
            ):
                portable = _portable_config(member.config, member.fold)
                validated = ExperimentConfig.model_validate(
                    portable, context={"config_dir": Path.cwd()}
                )
                bundled_preprocessing = validate_preprocessing_identity(
                    member.saved.get("preprocessing_identity"),
                    member.config.model.backbone,
                    temporary / "code" / "knee_abnormality",
                )
                exported = _inference_checkpoint(member.checkpoint, portable)
                validate_checkpoint(exported, validated, targets)
                model = checkpoint_model(
                    exported, validated, targets, torch.device("cpu"), temperature
                )
                baked = half_precision_weights(
                    bake_attention_temperature(
                        exported, validated, targets, temperature
                    ),
                    model,
                )
                checkpoint_model(baked, validated, targets, torch.device("cpu"), 1.0)
                member_path = ensemble_member_path(temporary / "models", index)
                TorchCheckpointIO().save_checkpoint(baked, member_path)
                training_source = _training_source(member.path)
                source_matches = _source_matches(
                    training_source, included_commit, included_status
                )
                checkpoint_records.append(
                    {
                        "member_index": index,
                        "checkpoint_path": member_path.relative_to(
                            temporary
                        ).as_posix(),
                        "source_path": str(member.path),
                        "source_sha256": member.hash,
                        "exported_sha256": sha256_file(member_path),
                        "fold": member.fold,
                        "targets": list(targets),
                        "model": member.config.model.model_dump(mode="json"),
                        "attention_temperature": temperature,
                        "preprocessing_identity": bundled_preprocessing,
                        "results": _checkpoint_results(member.checkpoint, member.path),
                        "input_identity": member.saved.get("input_identity"),
                        "label_provenance": _artifact_record(
                            member.path.parent / "label_provenance.json"
                        ),
                        "timing": _artifact_record(member.path.parent / "timing.json"),
                        "training_source": training_source,
                        "matches_training_source": source_matches,
                    }
                )
            manifest_checkpoint_fields = {
                "checkpoints": checkpoint_records,
                "member_count": len(members),
                "source": {
                    "included_git_commit": included_commit,
                    "included_git_status": included_status,
                    "bundled_source_sha256": source_hash,
                },
            }
            runtime_attention_temperature = 1.0

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
        supplied_names = {
            requirement.split("==", 1)[0].lower() for requirement in requirements
        }
        unmet = []
        missing_wheels = [
            name for name in OFFLINE_DISTRIBUTIONS if name not in supplied_names
        ]
        if missing_wheels:
            unmet.append(
                f"Offline wheels were not supplied: {', '.join(missing_wheels)}"
            )
        manifest = {
            **manifest_checkpoint_fields,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
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
                "attention_temperature": runtime_attention_temperature,
                "reflection_tta": args.reflection_tta,
                "backbone": args.backbone,
                "batch_size": args.inference_batch_size,
                "batch_size_scope": "per_device",
                "num_workers": args.inference_num_workers,
                "num_workers_scope": "total",
                "parallelism": "one_process_per_visible_gpu",
                "automatic_precision_policy": "native CUDA BF16, other CUDA FP16 with the backbone in FP16 and, for the graph backbone, compiled fixed chunks captured at the input size, CPU FP32",
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
    args = ExportCLI()
    manifest = export_bundle(args)
    summary = {
        "bundle": str(args.output.expanduser().resolve()),
        "member_count": manifest.get("member_count", 1),
        "fold": manifest["checkpoints"][0]["fold"]
        if "checkpoints" in manifest
        else manifest["checkpoint"]["fold"],
        "ready": manifest["ready"],
        "preparation_seconds": manifest["preparation_seconds"],
        "unmet_prerequisites": manifest["unmet_prerequisites"],
    }
    if "checkpoint" in manifest:
        summary["checkpoint_sha256"] = manifest["checkpoint"]["source_sha256"]
    else:
        summary["checkpoint_sha256"] = [
            entry["source_sha256"] for entry in manifest["checkpoints"]
        ]
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
