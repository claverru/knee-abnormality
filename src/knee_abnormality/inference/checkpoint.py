from pathlib import Path

import torch

from knee_abnormality.config import (
    ATTENTION_TEMPERATURE,
    ExperimentConfig,
    validate_attention_temperature,
)
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
                "pretrained_weights",
                "channels_last",
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


def apply_attention_temperature(
    model: KneeAbnormalityModel, temperature: float
) -> float:
    temperature = validate_attention_temperature(temperature)
    scorer = model.attention[-1]
    with torch.no_grad():
        scorer.weight.div_(temperature)
        scorer.bias.div_(temperature)
        model.view_label_bias.div_(temperature)
        model.position_prior.weight.div_(temperature)
        model.position_prior.bias.div_(temperature)
        model.routing_temperature.mul_(temperature)
    return temperature


def checkpoint_model(
    checkpoint: dict,
    config: ExperimentConfig,
    targets: tuple[str, ...],
    device: torch.device,
    attention_temperature: float = ATTENTION_TEMPERATURE,
) -> KneeAbnormalityModel:
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model = KneeAbnormalityModel(
        config.model, config.data, targets, load_pretrained=False
    )
    model.load_state_dict(state, strict=True)
    apply_attention_temperature(model, attention_temperature)
    model.to(device).eval()
    return model


def load_checkpoint_model(
    config: ExperimentConfig,
    targets: tuple[str, ...],
    checkpoint_path: Path,
    device: torch.device,
    *,
    expected_fold: int | None = None,
    attention_temperature: float = ATTENTION_TEMPERATURE,
) -> KneeAbnormalityModel:
    checkpoint = read_checkpoint(checkpoint_path)
    validate_checkpoint(checkpoint, config, targets, expected_fold)
    return checkpoint_model(checkpoint, config, targets, device, attention_temperature)


def bake_attention_temperature(
    checkpoint: dict,
    config: ExperimentConfig,
    targets: tuple[str, ...],
    temperature: float,
) -> dict:
    model = checkpoint_model(
        checkpoint, config, targets, torch.device("cpu"), temperature
    )
    baked = dict(checkpoint)
    baked["state_dict"] = {
        f"model.{name}": value for name, value in model.state_dict().items()
    }
    return baked


ENSEMBLE_MEMBER_GLOB = "member-*.ckpt"


def ensemble_member_path(directory: Path, index: int) -> Path:
    return directory / f"member-{index:03d}.ckpt"


def load_ensemble_models(
    config: ExperimentConfig,
    targets: tuple[str, ...],
    checkpoint_dir: Path,
    device: torch.device,
    *,
    expected_fold: int | None = None,
) -> list[KneeAbnormalityModel]:
    paths = sorted(checkpoint_dir.glob(ENSEMBLE_MEMBER_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No ensemble member checkpoints found in {checkpoint_dir}"
        )
    models = []
    for path in paths:
        checkpoint = read_checkpoint(path)
        saved = checkpoint.get("hyper_parameters")
        if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
            raise TypeError(
                f"Ensemble member {path.name} is missing saved configuration"
            )
        member_config = ExperimentConfig.model_validate(saved["config"])
        if member_config.data.model_dump(mode="json") != config.data.model_dump(
            mode="json"
        ):
            raise ValueError(
                f"Ensemble member {path.name} data config differs from the bundle"
            )
        validate_checkpoint(checkpoint, member_config, targets, expected_fold)
        models.append(checkpoint_model(checkpoint, member_config, targets, device, 1.0))
    return models


def load_models(
    config: ExperimentConfig,
    targets: tuple[str, ...],
    checkpoint_path: Path,
    device: torch.device,
    *,
    expected_fold: int | None = None,
    attention_temperature: float = ATTENTION_TEMPERATURE,
) -> list[KneeAbnormalityModel]:
    if checkpoint_path.is_dir():
        return load_ensemble_models(
            config, targets, checkpoint_path, device, expected_fold=expected_fold
        )
    return [
        load_checkpoint_model(
            config,
            targets,
            checkpoint_path,
            device,
            expected_fold=expected_fold,
            attention_temperature=attention_temperature,
        )
    ]
