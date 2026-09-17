from pathlib import Path

import torch

from knee_abnormality.config import ExperimentConfig
from knee_abnormality.identity import validate_preprocessing_identity
from knee_abnormality.modeling.model import KneeAbnormalityModel
from knee_abnormality.provenance import sha256_file


def read_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_checkpoint(
    checkpoint: dict,
    config: ExperimentConfig,
    targets: tuple[str, ...],
    expected_fold: int | None = None,
) -> ExperimentConfig:
    saved = checkpoint.get("hyper_parameters")
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
        raise TypeError("Checkpoint must include its saved configuration and targets")
    saved_config = ExperimentConfig.model_validate(saved["config"])
    validate_preprocessing_identity(
        saved.get("preprocessing_identity"),
        saved_config.model.backbone,
        saved_config.data.robust_center_outlier_mm,
    )
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


def checkpoint_model(
    checkpoint: dict,
    config: ExperimentConfig,
    targets: tuple[str, ...],
    device: torch.device,
) -> KneeAbnormalityModel:
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model = KneeAbnormalityModel(
        config.model, config.data, len(targets), load_pretrained=False
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_checkpoint_model(
    config: ExperimentConfig,
    targets: tuple[str, ...],
    checkpoint_path: Path,
    device: torch.device,
    *,
    expected_fold: int | None = None,
) -> KneeAbnormalityModel:
    checkpoint = read_checkpoint(checkpoint_path)
    validate_checkpoint(checkpoint, config, targets, expected_fold)
    return checkpoint_model(checkpoint, config, targets, device)
