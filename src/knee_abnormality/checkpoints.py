import hashlib
import shutil
from pathlib import Path

import polars as pl
from pytorch_lightning.callbacks import Callback

from knee_abnormality.config import ExperimentConfig
from knee_abnormality.labels import FoldLabels
from knee_abnormality.preprocessing_identity import (
    preprocessing_identity,
    validate_preprocessing_identity,
)
from knee_abnormality.provenance import sha256_file


class CheckpointRelocation(Callback):
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: dict) -> None:
        relocate_checkpoint_paths(
            checkpoint, self.directory, Path(trainer.ckpt_path).parent
        )


def relocate_checkpoint_paths(
    checkpoint: dict, directory: Path, source_directory: Path
) -> None:
    directory = directory.resolve()
    for state in checkpoint.get("callbacks", {}).values():
        if not isinstance(state, dict) or "best_k_models" not in state:
            continue
        if state.get("dirpath") == str(directory):
            continue
        paths = set(state["best_k_models"])
        fields = ("best_model_path", "kth_best_model_path", "last_model_path")
        paths.update(state.get(name) for name in fields if state.get(name))
        relocated = {path: str(directory / Path(path).name) for path in paths}
        if len(set(relocated.values())) != len(relocated):
            raise ValueError("Checkpoint paths collide in the resumed run directory")
        for source, destination in relocated.items():
            source_path = source_directory / Path(source).name
            if not source_path.is_file():
                source_path = Path(source)
            if source_path.resolve() != Path(destination):
                directory.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
        state["dirpath"] = str(directory)
        state["best_k_models"] = {
            relocated[path]: score for path, score in state["best_k_models"].items()
        }
        for name in fields:
            if state.get(name):
                state[name] = relocated[state[name]]


def checkpoint_sampling_epoch(checkpoint: dict) -> int:
    return int(
        checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["processed"]
    )


def training_input_identity(
    config: ExperimentConfig,
    labels: FoldLabels,
    training: pl.DataFrame,
    validation: pl.DataFrame,
) -> dict:
    counts_path = (
        config.paths.data_root / "metadata" / "train_series_image_counts.parquet"
    )
    return {
        "version": 1,
        "preprocessing_identity": preprocessing_identity(),
        "label_manifest_sha256": labels.provenance["manifest_sha256"],
        "split_protocol_sha256": labels.provenance["split_protocol_sha256"],
        "membership_sha256": labels.provenance["membership_sha256"],
        "label_source": {
            key: value
            for key, value in labels.provenance["selected"].items()
            if key != "path"
        },
        "training_rows_sha256": hashlib.sha256(
            training.sort("StudyInstanceUID").write_csv().encode()
        ).hexdigest(),
        "validation_rows_sha256": hashlib.sha256(
            validation.sort("StudyInstanceUID").write_csv().encode()
        ).hexdigest(),
        "series_metadata_sha256": sha256_file(config.raw_dir / "train_series.csv"),
        "series_counts_sha256": sha256_file(counts_path)
        if counts_path.exists()
        else None,
    }


def validate_checkpoint(
    checkpoint: dict,
    config: ExperimentConfig,
    targets: tuple[str, ...],
    expected_fold: int | None = None,
) -> ExperimentConfig:
    saved = checkpoint.get("hyper_parameters")
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
        raise TypeError("Checkpoint must include its saved configuration and targets")
    validate_preprocessing_identity(saved.get("preprocessing_identity"))
    saved_config = ExperimentConfig.model_validate(saved["config"])
    if saved.get("targets") != list(targets):
        raise ValueError(
            "Checkpoint target names or order differ from prediction targets"
        )
    if type(saved.get("fold")) is not int:
        raise ValueError("Checkpoint must identify its producing fold")
    if expected_fold is not None and saved["fold"] != expected_fold:
        raise ValueError(
            f"Checkpoint was trained for fold {saved['fold']}, not fold {expected_fold}"
        )
    if (
        expected_fold is not None
        and saved_config.validation.split != config.validation.split
    ):
        raise ValueError("Checkpoint validation split differs from the requested split")
    identity = saved.get("input_identity")
    if expected_fold is not None and isinstance(identity, dict):
        for name, path in (
            ("membership_sha256", config.split_dir / "studies.csv"),
            ("split_protocol_sha256", config.split_dir / "protocol.json"),
        ):
            if identity.get(name) != sha256_file(path):
                raise ValueError(f"Checkpoint validation input mismatch: {name}")
    differences = []
    if config.seed != saved_config.seed:
        differences.append("seed")
    for section in ("data", "model"):
        actual = getattr(config, section).model_dump(mode="json")
        expected = getattr(saved_config, section).model_dump(mode="json")
        for name, value in actual.items():
            if section == "model" and name in {
                "pretrained",
                "channels_last",
                "checkpoint_backbone",
            }:
                continue
            if value != expected[name]:
                differences.append(f"{section}.{name}")
    if differences:
        raise ValueError(
            "Checkpoint configuration mismatch: "
            + ", ".join(differences)
            + ". Use the run's effective_config.json for its original input semantics."
        )
    return saved_config


def _resume_settings(config: ExperimentConfig) -> dict:
    result = config.model_dump(mode="json", exclude={"paths", "name"})
    result["validation"].pop("folds")
    result["model"].pop("checkpoint_backbone")
    for name in (
        "accelerator",
        "devices",
        "num_workers",
        "persistent_workers",
        "prefetch_factor",
    ):
        result["trainer"].pop(name)
    return result


def validate_resume_checkpoint(
    checkpoint: dict,
    config: ExperimentConfig,
    targets: tuple[str, ...],
    fold: int,
    training_studies: int,
    input_identity: dict,
) -> None:
    saved_config = validate_checkpoint(checkpoint, config, targets, fold)
    saved = checkpoint["hyper_parameters"]
    if _resume_settings(saved_config) != _resume_settings(config):
        raise ValueError(
            "Resume checkpoint training settings differ from the requested run"
        )
    if saved.get("training_studies") != training_studies:
        raise ValueError(
            "Resume checkpoint training study count differs from the requested run"
        )
    recorded_identity = saved.get("input_identity")
    if not isinstance(recorded_identity, dict) or recorded_identity.get("version") != 1:
        raise ValueError(
            "Resume requires a checkpoint with recorded input identities; start a fresh run"
        )
    if recorded_identity != input_identity:
        raise ValueError(
            "Resume checkpoint input identities differ from the requested run"
        )
