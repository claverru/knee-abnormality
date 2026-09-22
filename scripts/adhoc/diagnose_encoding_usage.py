import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
import yaml
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsError
from sklearn.metrics import roc_auc_score

import knee_abnormality
from knee_abnormality.cli import CLISettings
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD, load_fold_labels
from knee_abnormality.data.preprocessing.device import prepare_images
from knee_abnormality.data.sources import prediction_image_multiplier
from knee_abnormality.evaluation.metrics import (
    COMPARTMENT_PAIRS,
    compartment_metrics,
    logit_margin,
    margin_auc,
)
from knee_abnormality.evaluation.predictions import make_prediction_frame
from knee_abnormality.identity import preprocessing_identity
from knee_abnormality.inference.checkpoint import (
    checkpoint_model,
    read_checkpoint,
    validate_checkpoint,
)
from knee_abnormality.inference.engine import (
    move_batch,
    prediction_dataset,
    resolve_device,
    resolve_precision,
)
from knee_abnormality.modeling.spatial_tokens import SPATIAL_TOKEN_IDENTITY
from knee_abnormality.provenance import copy_sources, sha256_file
from knee_abnormality.training.checkpoints import training_input_identity
from knee_abnormality.training.cuda import configure_cuda_allocator


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


PLANES = tuple(SPATIAL_TOKEN_IDENTITY["plane_order"])
EMBEDDINGS = ("position_1d", "position_2d", "plane", "contrast")
FULL_VARIANTS = {
    "intact": frozenset(),
    "without_1d": frozenset(("position_1d",)),
    "without_2d": frozenset(("position_2d",)),
    "without_plane": frozenset(("plane",)),
    "without_contrast": frozenset(("contrast",)),
    "without_anatomical_bias": frozenset(("medial_axis",)),
    "without_position_information": frozenset(
        ("position_1d", "position_2d", "medial_axis")
    ),
}
PLANE_VARIANTS = {f"{plane.lower()}_only": index for index, plane in enumerate(PLANES)}
VARIANTS = (*FULL_VARIANTS, *PLANE_VARIANTS)


def binary_cross_entropy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    values = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(targets * np.log(values) + (1 - targets) * np.log1p(-values)).mean())


def auc_values(
    targets: np.ndarray, probabilities: np.ndarray, names: tuple[str, ...]
) -> dict[str, float | None]:
    hard = (targets >= TARGET_THRESHOLD).astype(np.int8)
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
        "compartments": compartment_metrics(targets, probabilities, names),
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
        **{f"margin_auc_delta:{family}": [] for family in COMPARTMENT_PAIRS},
    }
    hard = (targets >= TARGET_THRESHOLD).astype(np.int8)
    pairs = {
        family: tuple(names.index(name) for name in pair)
        for family, pair in COMPARTMENT_PAIRS.items()
    }
    margins = {
        family: (logit_margin(intact, *pair), logit_margin(variant, *pair))
        for family, pair in pairs.items()
    }
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
        for family, pair in pairs.items():
            reference, changed = (
                margin_auc(hard[sampled], margin[sampled], *pair)
                for margin in margins[family]
            )
            if reference is not None:
                values[f"margin_auc_delta:{family}"].append(changed - reference)
    return {name: interval(measurements) for name, measurements in values.items()}


def variant_tokens(
    model, encoded: dict, omitted: frozenset[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    variant = {
        **encoded,
        **{
            name: torch.zeros_like(encoded[name])
            for name in omitted
            if encoded[name] is not None
        },
    }
    tokens, values = model._pooling_tokens(variant)
    shape = encoded["visual_valid"].shape
    return (
        tokens.unflatten(0, shape),
        values.unflatten(0, shape),
        variant["medial_axis"],
    )


def pooled_logits(
    model,
    encoded: dict,
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    images: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    attention_tokens, value_tokens, medial_axis = tokens
    valid = encoded["visual_valid"][images]
    study = encoded["study"][images]
    studies = torch.arange(encoded["batch_size"], device=study.device)
    present = (studies[:, None] == study[valid.any(-1)][None]).any(-1)
    if not present.any():
        return None, present
    remap = present.cumsum(0) - 1
    logits = model._attention_pool(
        attention_tokens[images].flatten(0, 1),
        value_tokens[images].flatten(0, 1),
        remap[study][:, None].expand_as(valid).flatten(),
        valid.flatten(),
        int(present.sum()),
        medial_axis[images].flatten() if medial_axis is not None else None,
    )
    return logits, present


def plane_records(batch: dict, encoded: dict) -> list[dict]:
    visual = encoded["visual_valid"]
    study = encoded["study"]
    planes = encoded["image_plane"]
    if (
        visual.shape[0] != len(study)
        or len(planes) != len(study)
        or study.min() < 0
        or study.max() >= encoded["batch_size"]
    ):
        raise ValueError("Encoded tokens do not match evaluation study membership")
    return [
        {
            STUDY: uid,
            **{
                f"{plane.lower()}_visual_tokens": int(
                    visual[(study == study_index) & (planes == plane_index)]
                    .sum()
                    .item()
                )
                for plane_index, plane in enumerate(PLANES)
            },
        }
        for study_index, uid in enumerate(batch["study_uid"])
    ]


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


def diagnose(args: DiagnoseCLI) -> None:
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
    saved = pl.read_csv(saved_path).sort(STUDY)
    fold_labels = load_fold_labels(config, args.fold)
    targets = fold_labels.targets
    studies = fold_labels.validation
    dataset = prediction_dataset(
        config,
        targets,
        studies,
        "train",
        prediction_image_multiplier(config, "validation"),
    )
    input_identity_path = fold_dir / "input_identity.json"
    saved_input_identity = json.loads(input_identity_path.read_text())
    current_input_identity = training_input_identity(
        config, fold_labels, fold_labels.training, fold_labels.validation
    )
    if saved_input_identity != current_input_identity:
        raise ValueError(
            "Run input identity differs from current frozen training inputs"
        )
    checkpoint_data = read_checkpoint(checkpoint)
    validate_checkpoint(checkpoint_data, config, targets, args.fold)
    checkpoint_metadata = checkpoint_data["hyper_parameters"]
    if checkpoint_metadata.get("input_identity") != current_input_identity:
        raise ValueError("Checkpoint input identity differs from current frozen inputs")
    if saved[STUDY].to_list() != studies[STUDY].to_list():
        raise ValueError("Saved predictions must match the complete validation fold")
    if saved["fold"].to_list() != [args.fold] * len(studies):
        raise ValueError("Saved predictions have an unexpected producing fold")
    model = checkpoint_model(checkpoint_data, config, targets, device)
    del checkpoint_data
    if model.training:
        raise ValueError("Encoding diagnosis requires evaluation mode")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_sha256 = copy_sources(
        [Path(knee_abnormality.__file__).parent, Path(__file__)],
        args.output_dir / "source",
    )
    shutil.copy2(config_path, args.output_dir / "effective_config.json")
    loader = build_data_loader(
        dataset,
        config.trainer,
        config.trainer.batch_size if args.batch_size is None else args.batch_size,
        num_workers=args.workers,
    )
    collected = {name: [] for name in VARIANTS}
    membership = {name: [] for name in VARIANTS}
    ids = []
    plane_rows = []
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
                encoded = model.encode_batch(prepare_images(move_batch(batch, device)))
                if encoded["batch_size"] != len(batch["study_uid"]):
                    raise ValueError("Encoded batch size differs from study membership")
                every_image = torch.ones_like(encoded["study"], dtype=torch.bool)
                for name, omitted in FULL_VARIANTS.items():
                    tokens = variant_tokens(model, encoded, omitted)
                    if name == "intact":
                        intact_tokens = tokens
                    logits, present = pooled_logits(model, encoded, tokens, every_image)
                    if not present.all():
                        raise ValueError("A study has no visually valid spatial tokens")
                    collected[name].append(torch.sigmoid(logits.float()).cpu())
                    membership[name].extend(present.tolist())
                for name, plane in PLANE_VARIANTS.items():
                    logits, present = pooled_logits(
                        model, encoded, intact_tokens, encoded["image_plane"] == plane
                    )
                    if logits is not None:
                        collected[name].append(torch.sigmoid(logits.float()).cpu())
                    membership[name].extend(present.tolist())
            plane_rows.extend(plane_records(batch, encoded))
            if len(ids) % 128 == 0 or len(ids) == len(dataset):
                print(
                    f"Processed {len(ids)}/{len(dataset)} validation studies",
                    flush=True,
                )
    if ids != studies[STUDY].to_list():
        raise ValueError("Inference order differs from frozen validation membership")
    predictions = {
        name: torch.cat(values).numpy() for name, values in collected.items()
    }
    masks = {
        name: np.asarray(values, dtype=bool) for name, values in membership.items()
    }
    plane_frame = pl.DataFrame(plane_rows)
    for name, plane in PLANE_VARIANTS.items():
        expected = plane_frame[f"{PLANES[plane].lower()}_visual_tokens"].to_numpy() > 0
        if not np.array_equal(masks[name], expected):
            raise ValueError(
                f"Plane variant membership differs from token counts: {name}"
            )
    for name in VARIANTS:
        if len(predictions[name]) != masks[name].sum():
            raise ValueError(f"Variant predictions differ from membership: {name}")
    labels = studies.select([f"target_{target}" for target in targets]).to_numpy()
    saved_values = saved.select(targets).to_numpy()
    metric_values = {}
    for name in VARIANTS:
        mask = masks[name]
        metric_values[name] = metrics(labels[mask], predictions[name], targets)
        metric_values[name]["mean_abs_probability_change"] = float(
            np.abs(predictions[name] - predictions["intact"][mask]).mean()
        )
        if name in PLANE_VARIANTS:
            metric_values[name]["intact_on_same_studies"] = metrics(
                labels[mask], predictions["intact"][mask], targets
            )
    for name in VARIANTS:
        make_prediction_frame(
            [uid for uid, kept in zip(ids, masks[name], strict=True) if kept],
            predictions[name],
            targets,
            args.fold,
        ).write_csv(args.output_dir / f"{name}.csv")
    (args.output_dir / "inputs.json").write_text(
        json.dumps(inputs, indent=2, default=str) + "\n"
    )
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
                f"{name}_intact_on_same_studies_auc": [
                    metric_values[name]["intact_on_same_studies"]["per_target_auc"][
                        target
                    ]
                    for target in targets
                ]
                for name in PLANE_VARIANTS
            },
            **{
                f"{name}_mean_abs_probability_change": np.abs(
                    predictions[name] - predictions["intact"][masks[name]]
                ).mean(0)
                for name in VARIANTS
                if name != "intact"
            },
        }
    )
    per_target.write_csv(args.output_dir / "per_target.csv")
    bootstrap = {
        name: paired_bootstrap(
            labels[masks[name]],
            predictions["intact"][masks[name]],
            predictions[name],
            targets,
            args.bootstrap_resamples,
            args.bootstrap_seed,
        )
        for name in VARIANTS
        if name != "intact"
    }
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
        "label_split_provenance": fold_labels.provenance,
        "source_sha256": source_sha256,
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
        "interventions": {
            **{
                name: {
                    "omitted_embeddings": sorted(omitted),
                    "pooled_planes": PLANES,
                }
                for name, omitted in FULL_VARIANTS.items()
            },
            **{
                name: {
                    "omitted_embeddings": [],
                    "pooled_planes": [PLANES[plane]],
                }
                for name, plane in PLANE_VARIANTS.items()
            },
        },
        "variant_membership": {
            name: {
                "studies": int(masks[name].sum()),
                "rule": f"studies_with_visually_valid_{PLANES[PLANE_VARIANTS[name]]}_tokens"
                if name in PLANE_VARIANTS
                else "complete_validation_fold",
            }
            for name in VARIANTS
        },
        "compartment_metrics": {
            "pairs": COMPARTMENT_PAIRS,
            "target_threshold": TARGET_THRESHOLD,
            "margin_auc": "AUC of logit(medial) minus logit(lateral), medial-only positive against lateral-only positive studies",
            "lateral_cross_talk": "top-k lateral probabilities with k equal to lateral positives among the variant's studies; medial-positive share of lateral false positives, with the medial-positive base rate among lateral negatives",
        },
        "metrics": metric_values,
        "paired_bootstrap": {
            "seed": args.bootstrap_seed,
            "resamples": args.bootstrap_resamples,
            "unit": "study; all targets for each sampled study are retained together",
            "interval": "percentile 95% interval for variant minus intact on the variant's studies for this held-out checkpoint comparison; plane-only variants compare with intact on studies having that plane",
            "variants": bootstrap,
        },
        "reproduction_probability_difference": {
            "mean_abs": float(np.abs(predictions["intact"] - saved_values).mean()),
            "max_abs": float(np.abs(predictions["intact"] - saved_values).max()),
        },
        "preparation_conditions": preparation_conditions,
        "interpretation": "These are removal-based reliance measurements for one held-out checkpoint. Embeddings and planes overlap and interact, effects do not add to unique percentages, and removals can create inputs absent from training. All position information removal drops the 1D and 2D embeddings. Plane-only variants pool one plane's intact tokens for studies having that plane; series dropout gives such inputs only partial training support, and their metrics describe a study subset. Saved prediction disagreement must be investigated before interpreting small ablation effects.",
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
                "variants": {
                    name: {
                        "studies": values["studies"],
                        "macro_auc": values["macro_auc"],
                        **{
                            f"{family}_{key}": compartment[key]
                            for family, compartment in values["compartments"].items()
                            for key in ("margin_auc", "lateral_cross_talk")
                        },
                    }
                    for name, values in metric_values.items()
                },
            },
            indent=2,
        )
    )


def main() -> None:
    try:
        diagnose(DiagnoseCLI())
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
        SettingsError,
    ) as error:
        print(f"Encoding diagnosis error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
