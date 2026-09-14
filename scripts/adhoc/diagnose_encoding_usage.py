import json
import os
import shutil
import time
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from pydantic import Field
from pydantic_settings import CliPositionalArg
from sklearn.metrics import roc_auc_score

import knee_abnormality
from knee_abnormality.checkpoints import training_input_identity
from knee_abnormality.cli import CLISettings
from knee_abnormality.config import load_config
from knee_abnormality.inference import (
    load_checkpoint_model,
    make_prediction_frame,
    prediction_data,
    prediction_loader,
    resolve_device,
    resolve_precision,
)
from knee_abnormality.labels import load_fold_labels
from knee_abnormality.preprocessing_identity import preprocessing_identity
from knee_abnormality.provenance import sha256_file
from knee_abnormality.runtime import configure_cuda_allocator, move_batch


class DiagnoseCLI(CLISettings):
    run_dir: CliPositionalArg[Path]
    output_dir: Path
    checkpoint: Path | None = None
    fold: int = Field(default=0, ge=0)
    batch_size: int | None = Field(default=None, ge=1)
    workers: int = Field(default=8, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["auto", "fp32", "bf16"] = "auto"
    bootstrap_resamples: int = Field(default=2000, ge=1)
    bootstrap_seed: int = 20260913


VARIANTS = {
    "intact": frozenset(),
    "without_1d": frozenset(("position_1d",)),
    "without_2d": frozenset(("position_2d",)),
    "without_3d": frozenset(("position_3d",)),
    "without_plane": frozenset(("plane",)),
    "without_contrast": frozenset(("contrast",)),
    "without_position_information": frozenset(
        ("position_1d", "position_2d", "position_3d")
    ),
}


def binary_cross_entropy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    values = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(targets * np.log(values) + (1 - targets) * np.log1p(-values)).mean())


def auc_values(
    targets: np.ndarray, probabilities: np.ndarray, names: tuple[str, ...]
) -> dict[str, float | None]:
    hard = (targets >= 0.5).astype(np.int8)
    return {
        name: (
            float(roc_auc_score(hard[:, index], probabilities[:, index]))
            if len(np.unique(hard[:, index])) == 2
            else None
        )
        for index, name in enumerate(names)
    }


def metrics(
    targets: np.ndarray, probabilities: np.ndarray, names: tuple[str, ...]
) -> dict:
    per_target_auc = auc_values(targets, probabilities, names)
    defined = [value for value in per_target_auc.values() if value is not None]
    return {
        "studies": len(targets),
        "macro_auc": float(np.mean(defined)) if len(defined) == len(names) else None,
        "auc_targets": len(defined),
        "per_target_auc": per_target_auc,
        "soft_target_bce": binary_cross_entropy(targets, probabilities),
    }


def interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "valid_replicates": len(array),
        "low": float(np.quantile(array, 0.025)) if len(array) else None,
        "high": float(np.quantile(array, 0.975)) if len(array) else None,
    }


def paired_bootstrap(
    targets: np.ndarray,
    intact: np.ndarray,
    variant: np.ndarray,
    names: tuple[str, ...],
    resamples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    values = {
        "macro_auc_delta": [],
        "soft_target_bce_delta": [],
        "mean_abs_probability_change": [],
        **{f"auc_delta:{name}": [] for name in names},
    }
    hard = (targets >= 0.5).astype(np.int8)
    for _ in range(resamples):
        sampled = rng.integers(0, len(targets), size=len(targets))
        before = intact[sampled]
        after = variant[sampled]
        sampled_targets = targets[sampled]
        values["soft_target_bce_delta"].append(
            binary_cross_entropy(sampled_targets, after)
            - binary_cross_entropy(sampled_targets, before)
        )
        values["mean_abs_probability_change"].append(
            float(np.abs(after - before).mean())
        )
        target_deltas = []
        for index, name in enumerate(names):
            if len(np.unique(hard[sampled, index])) != 2:
                continue
            delta = float(
                roc_auc_score(hard[sampled, index], after[:, index])
                - roc_auc_score(hard[sampled, index], before[:, index])
            )
            values[f"auc_delta:{name}"].append(delta)
            target_deltas.append(delta)
        if len(target_deltas) == len(names):
            values["macro_auc_delta"].append(float(np.mean(target_deltas)))
    return {name: interval(measurements) for name, measurements in values.items()}


def pooled_logits(model, encoded: dict, omitted: frozenset[str]) -> torch.Tensor:
    features = encoded["features"]
    tokens = features
    if "position_1d" not in omitted:
        tokens = tokens + encoded["position_1d"][:, None]
    if "position_2d" not in omitted:
        tokens = tokens + encoded["position_2d"][None]
    if "position_3d" not in omitted:
        tokens = tokens + encoded["position_3d"]
    else:
        position_3d = torch.where(
            encoded["available_3d"][..., None],
            model.spatial_tokens.unknown_3d.to(dtype=features.dtype),
            encoded["position_3d"],
        )
        tokens = tokens + position_3d
    if "plane" not in omitted:
        tokens = tokens + encoded["plane"][:, None]
    if "contrast" not in omitted:
        tokens = tokens + encoded["contrast"][:, None]
    tokens = model.mil_norm(tokens).flatten(0, 1)
    token_study = encoded["study"][:, None].expand(-1, features.shape[1]).flatten()
    return model._attention_pool(
        tokens,
        token_study,
        encoded["visual_valid"].flatten(),
        encoded["batch_size"],
    )


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def batch_identity(batch: dict) -> dict:
    return {
        "study_uid": batch["study_uid"],
        "selected_paths": batch["selected_paths"],
        "failures": batch["failures"],
        "tensor_sha256": {
            name: tensor_hash(value)
            for name, value in batch.items()
            if isinstance(value, torch.Tensor)
        },
    }


def main() -> None:
    args = DiagnoseCLI()
    started = time.perf_counter()
    config_path = args.run_dir / "effective_config.json"
    config = load_config(config_path)
    allocator = configure_cuda_allocator()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.trainer.deterministic)
    torch.backends.cudnn.benchmark = False
    device = resolve_device(args.device)
    precision, dtype = resolve_precision(args.precision, device)
    fold_dir = args.run_dir / f"fold_{args.fold}"
    selected_path = fold_dir / "selected_checkpoint.json"
    selected = json.loads(selected_path.read_text())
    selected_checkpoint = fold_dir / Path(selected["path"]).name
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else selected_checkpoint
    )
    saved_path = fold_dir / "validation_predictions.csv"
    saved = pl.read_csv(saved_path).sort("StudyInstanceUID")
    dataset, targets, studies = prediction_data(config, "validation", args.fold)
    fold_labels = load_fold_labels(config, args.fold)
    input_identity_path = fold_dir / "input_identity.json"
    saved_input_identity = json.loads(input_identity_path.read_text())
    current_input_identity = training_input_identity(
        config, fold_labels, fold_labels.training, fold_labels.validation
    )
    if saved_input_identity != current_input_identity:
        raise ValueError(
            "Run input identity differs from current frozen training inputs"
        )
    checkpoint_metadata = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    ).get("hyper_parameters")
    if not isinstance(checkpoint_metadata, dict):
        raise TypeError("Checkpoint lacks saved hyperparameters")
    if checkpoint_metadata.get("input_identity") != current_input_identity:
        raise ValueError("Checkpoint input identity differs from current frozen inputs")
    if saved["StudyInstanceUID"].to_list() != studies["StudyInstanceUID"].to_list():
        raise ValueError("Saved predictions must match the complete validation fold")
    if saved["fold"].to_list() != [args.fold] * len(studies):
        raise ValueError("Saved predictions have an unexpected producing fold")
    model = load_checkpoint_model(
        config, targets, checkpoint, device, expected_fold=args.fold
    )
    if model.training:
        raise ValueError("Encoding diagnosis requires evaluation mode")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = args.output_dir / "source"
    shutil.copytree(
        Path(knee_abnormality.__file__).parent,
        snapshot / "src/knee_abnormality",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    script_snapshot = snapshot / "scripts/adhoc" / Path(__file__).name
    script_snapshot.parent.mkdir(parents=True)
    shutil.copy2(__file__, script_snapshot)
    shutil.copy2(config_path, args.output_dir / "effective_config.json")
    loader = prediction_loader(
        config, dataset, batch_size=args.batch_size, num_workers=args.workers
    )
    collected = {name: [] for name in VARIANTS}
    ids = []
    geometry = []
    inputs = []
    preparation_conditions = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader:
            ids.extend(batch["study_uid"])
            inputs.append(batch_identity(batch))
            preparation_conditions += sum(len(values) for values in batch["failures"])
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=dtype is not None
            ):
                encoded = model.encode_batch(move_batch(batch, device))
                if encoded["batch_size"] != len(batch["study_uid"]):
                    raise ValueError("Encoded batch size differs from study membership")
                for name, omitted in VARIANTS.items():
                    collected[name].append(
                        torch.sigmoid(
                            pooled_logits(model, encoded, omitted).float()
                        ).cpu()
                    )
            visual = encoded["visual_valid"]
            available = encoded["available_3d"]
            token_study = encoded["study"]
            if (
                visual.shape != available.shape
                or visual.shape[0] != len(token_study)
                or token_study.min() < 0
                or token_study.max() >= encoded["batch_size"]
            ):
                raise ValueError("Encoded geometry masks do not match study membership")
            for study_index, uid in enumerate(batch["study_uid"]):
                membership = token_study == study_index
                valid = visual[membership]
                three_d = available[membership]
                geometry.append(
                    {
                        "StudyInstanceUID": uid,
                        "visual_tokens": int(valid.sum().item()),
                        "available_3d_tokens": int((valid & three_d).sum().item()),
                        "receives_valid_3d": bool((valid & three_d).any().item()),
                    }
                )
            if len(ids) % 128 == 0 or len(ids) == len(dataset):
                print(
                    f"Processed {len(ids)}/{len(dataset)} validation studies",
                    flush=True,
                )
    if ids != studies["StudyInstanceUID"].to_list():
        raise ValueError("Inference order differs from frozen validation membership")
    predictions = {
        name: torch.cat(values).numpy() for name, values in collected.items()
    }
    labels = studies.select([f"target_{target}" for target in targets]).to_numpy()
    saved_values = saved.select(targets).to_numpy()
    metric_values = {
        name: metrics(labels, values, targets) for name, values in predictions.items()
    }
    for name, values in predictions.items():
        metric_values[name]["mean_abs_probability_change"] = float(
            np.abs(values - predictions["intact"]).mean()
        )
    for name, values in predictions.items():
        make_prediction_frame(ids, values, targets, args.fold).write_csv(
            args.output_dir / f"{name}.csv"
        )
    geometry_frame = pl.DataFrame(geometry)
    geometry_frame.write_parquet(args.output_dir / "geometry_availability.parquet")
    (args.output_dir / "inputs.json").write_text(
        json.dumps(inputs, indent=2, default=str) + "\n"
    )
    eligible = geometry_frame["receives_valid_3d"].to_numpy()
    subset = {}
    subset_probability_change = None
    if eligible.any():
        for name in ("intact", "without_3d"):
            subset[name] = metrics(
                labels[eligible], predictions[name][eligible], targets
            )
        changes = np.abs(
            predictions["without_3d"][eligible] - predictions["intact"][eligible]
        )
        subset_probability_change = {
            "mean_abs": float(changes.mean()),
            "max_abs": float(changes.max()),
        }
    per_target = pl.DataFrame(
        {
            "target": targets,
            **{
                f"{name}_auc": [
                    metric_values[name]["per_target_auc"][target] for target in targets
                ]
                for name in VARIANTS
            },
            **{
                f"{name}_mean_abs_probability_change": np.abs(
                    predictions[name] - predictions["intact"]
                ).mean(0)
                for name in VARIANTS
                if name != "intact"
            },
        }
    )
    per_target.write_csv(args.output_dir / "per_target.csv")
    bootstrap = {
        name: paired_bootstrap(
            labels,
            predictions["intact"],
            values,
            targets,
            args.bootstrap_resamples,
            args.bootstrap_seed,
        )
        for name, values in predictions.items()
        if name != "intact"
    }
    if eligible.any():
        bootstrap["without_3d_valid_3d_subset"] = paired_bootstrap(
            labels[eligible],
            predictions["intact"][eligible],
            predictions["without_3d"][eligible],
            targets,
            args.bootstrap_resamples,
            args.bootstrap_seed,
        )
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "options": args.model_dump(mode="json"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_checkpoint": selected,
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "config_sha256": sha256_file(config_path),
        "saved_predictions_sha256": sha256_file(saved_path),
        "input_identity": saved_input_identity,
        "input_identity_sha256": sha256_file(input_identity_path),
        "current_input_identity": current_input_identity,
        "checkpoint_input_identity": checkpoint_metadata["input_identity"],
        "actual_inputs_sha256": sha256_file(args.output_dir / "inputs.json"),
        "preprocessing_identity": preprocessing_identity(config.model.backbone),
        "label_split_provenance": load_fold_labels(config, args.fold).provenance,
        "source_sha256": {
            str(path.relative_to(snapshot)): sha256_file(path)
            for path in sorted(snapshot.rglob("*.py"))
        },
        "versions": {
            name: version(name)
            for name in ("torch", "timm", "polars", "numpy", "scikit-learn")
        },
        "runtime": {
            "device": str(device),
            "precision": precision,
            "batch_size": loader.batch_size,
            "workers": args.workers,
            "seed": config.seed,
            "strict_determinism": config.trainer.deterministic,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_allocator": allocator,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else None,
            "total_seconds": time.perf_counter() - started,
        },
        "interventions": {name: sorted(omitted) for name, omitted in VARIANTS.items()},
        "metrics": metric_values,
        "valid_3d_subset_metrics": subset,
        "valid_3d_subset_without_3d_probability_change": subset_probability_change,
        "paired_bootstrap": {
            "seed": args.bootstrap_seed,
            "resamples": args.bootstrap_resamples,
            "unit": "study; all targets for each sampled study are retained together",
            "interval": "percentile 95% interval for variant minus intact on this held-out checkpoint comparison",
            "variants": bootstrap,
        },
        "geometry_availability": {
            "studies": geometry_frame.height,
            "visual_tokens": int(geometry_frame["visual_tokens"].sum()),
            "available_3d_tokens": int(geometry_frame["available_3d_tokens"].sum()),
            "studies_receiving_valid_3d": int(
                geometry_frame["receives_valid_3d"].sum()
            ),
        },
        "reproduction_probability_difference": {
            "mean_abs": float(np.abs(predictions["intact"] - saved_values).mean()),
            "max_abs": float(np.abs(predictions["intact"] - saved_values).max()),
        },
        "preparation_conditions": preparation_conditions,
        "interpretation": "These are removal-based reliance measurements for one held-out checkpoint. Embeddings overlap and interact, effects do not add to unique percentages, and omitted embeddings can create inputs absent from training. The 3D unknown intervention has explicit training support. Saved prediction disagreement must be investigated before interpreting small ablation effects.",
        "per_target": per_target.to_dicts(),
    }
    result["artifact_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file()
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "reproduction_probability_difference": result[
                    "reproduction_probability_difference"
                ],
                "geometry_availability": result["geometry_availability"],
                "intact": metric_values["intact"],
                "without_3d": metric_values["without_3d"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
