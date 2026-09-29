import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import INFERENCE_BATCH_SIZE, load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.data.splits import STUDY
from knee_abnormality.evaluation.metrics import validation_metrics
from knee_abnormality.evaluation.predictions import make_prediction_frame
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import (
    predict_loader,
    prediction_data,
    resolve_device,
)
from knee_abnormality.provenance import copy_sources, sha256_file
from knee_abnormality.training.cuda import configure_cuda_allocator

RESERVOIR_CAPACITY = 200_000
STATS_SEED = 20260922
MODES = ("full", "fine_zeroed", "deep_zeroed")


class DiagnoseCLI(CLISettings):
    run: Path = Path("outputs/training/fusion-sum")
    fold: int = Field(default=0, ge=0)
    image_multiplier: int = Field(default=4, gt=0)
    output_dir: Path = Field(description="Fresh directory for outputs")


class Reservoir:
    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.values = np.empty(0, dtype=np.float64)
        self.keys = np.empty(0, dtype=np.float64)

    def extend(self, values: np.ndarray) -> None:
        if len(values) == 0:
            return
        keys = self.rng.random(len(values))
        values = np.concatenate([self.values, values])
        keys = np.concatenate([self.keys, keys])
        if len(values) > self.capacity:
            top = np.argpartition(-keys, self.capacity - 1)[: self.capacity]
            values = values[top]
            keys = keys[top]
        self.values = values
        self.keys = keys

    def median(self) -> float | None:
        return float(np.median(self.values)) if len(self.values) else None


class BranchStats:
    def __init__(
        self, capacity: int = RESERVOIR_CAPACITY, seed: int = STATS_SEED
    ) -> None:
        rng = np.random.default_rng(seed)
        self.count = 0
        self.sum_fine_norm = 0.0
        self.sum_deep_norm = 0.0
        self.sum_sum_norm = 0.0
        self.sum_cosine = 0.0
        self.sum_fine_share = 0.0
        self.fine_norm_reservoir = Reservoir(capacity, rng)
        self.deep_norm_reservoir = Reservoir(capacity, rng)
        self.sum_norm_reservoir = Reservoir(capacity, rng)

    def update(
        self, fine: torch.Tensor, deep: torch.Tensor, valid: torch.Tensor
    ) -> None:
        fine_flat = fine[valid].double()
        deep_flat = deep[valid].double()
        if fine_flat.numel() == 0:
            return
        fine_norm = fine_flat.norm(dim=-1)
        deep_norm = deep_flat.norm(dim=-1)
        sum_norm = (fine_flat + deep_flat).norm(dim=-1)
        dot = (fine_flat * deep_flat).sum(-1)
        cosine = dot / (fine_norm * deep_norm).clamp_min(1e-12)
        share = fine_norm / (fine_norm + deep_norm).clamp_min(1e-12)
        self.count += fine_flat.shape[0]
        self.sum_fine_norm += float(fine_norm.sum())
        self.sum_deep_norm += float(deep_norm.sum())
        self.sum_sum_norm += float(sum_norm.sum())
        self.sum_cosine += float(cosine.sum())
        self.sum_fine_share += float(share.sum())
        self.fine_norm_reservoir.extend(fine_norm.cpu().numpy())
        self.deep_norm_reservoir.extend(deep_norm.cpu().numpy())
        self.sum_norm_reservoir.extend(sum_norm.cpu().numpy())

    def summary(self) -> dict:
        if self.count == 0:
            raise ValueError(
                "No visually valid spatial tokens were recorded during the full pass"
            )
        return {
            "valid_tokens": self.count,
            "fine_norm": {
                "mean": self.sum_fine_norm / self.count,
                "median": self.fine_norm_reservoir.median(),
            },
            "deep_norm": {
                "mean": self.sum_deep_norm / self.count,
                "median": self.deep_norm_reservoir.median(),
            },
            "sum_norm": {
                "mean": self.sum_sum_norm / self.count,
                "median": self.sum_norm_reservoir.median(),
            },
            "mean_cosine_similarity": self.sum_cosine / self.count,
            "mean_fine_norm_share": self.sum_fine_share / self.count,
            "reservoir_capacity": self.fine_norm_reservoir.capacity,
            "reservoir_seed": STATS_SEED,
            "token_scope": (
                "every visually valid spatial token (encode_batch's visual_valid mask)"
                " across the whole fold, not per label"
            ),
        }


def install_encode_images(
    model, mode: str, terms: dict, recorder: BranchStats | None
) -> None:
    def encode_images(images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if model.checkpoint_backbone:
            model.backbone.set_grad_checkpointing(
                model.training and torch.is_grad_enabled()
            )
        shallow, deep = model.backbone_maps(images)
        shallow_tokens = model.projection(shallow.flatten(2).transpose(1, 2))
        deep_tokens = model.deep_projection(deep.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        grid = model._resample_deep_grid(
            shallow.shape[-2:], deep.shape[-2:], deep_tokens.device
        )
        with torch.autocast(device_type=deep_tokens.device.type, enabled=False):
            resampled = F.grid_sample(
                deep_tokens.float(),
                grid.unsqueeze(0).expand(deep_tokens.shape[0], -1, -1, -1),
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )
        resampled_tokens = resampled.to(shallow_tokens.dtype).flatten(2).transpose(1, 2)
        if recorder is not None:
            terms["fine"] = shallow_tokens.detach()
            terms["deep"] = resampled_tokens.detach()
        fine_term = (
            torch.zeros_like(shallow_tokens)
            if mode == "fine_zeroed"
            else shallow_tokens
        )
        deep_term = (
            torch.zeros_like(resampled_tokens)
            if mode == "deep_zeroed"
            else resampled_tokens
        )
        tokens = fine_term + deep_term
        return tokens, shallow.shape[-2:]

    model._encode_images = encode_images


def install_encode_batch_recorder(model, terms: dict, recorder: BranchStats) -> None:
    original_encode_batch = model.encode_batch

    def encode_batch(batch: dict) -> dict:
        encoded = original_encode_batch(batch)
        recorder.update(terms["fine"], terms["deep"], encoded["visual_valid"])
        return encoded

    model.encode_batch = encode_batch


def restore_model(model) -> None:
    model.__dict__.pop("_encode_images", None)
    model.__dict__.pop("encode_batch", None)


def run_pass(
    model, loader, device, mode: str, terms: dict, recorder: BranchStats | None
) -> tuple[list[str], torch.Tensor, float]:
    install_encode_images(model, mode, terms, recorder)
    if recorder is not None:
        install_encode_batch_recorder(model, terms, recorder)
    try:
        started = time.perf_counter()
        study_ids, predictions, _ = predict_loader([model], loader, device)
        seconds = time.perf_counter() - started
    finally:
        restore_model(model)
    return study_ids, predictions, seconds


def score_pass(config, fold: int, targets: tuple[str, ...], path: Path) -> dict:
    frame = (
        pl.read_csv(config.split_dir / "studies.csv")
        .filter(pl.col("fold") == fold)
        .join(pl.read_csv(config.split_dir / "targets.csv"), on=STUDY, validate="1:1")
        .sort(STUDY)
    )
    prediction = pl.read_csv(path).sort(STUDY)
    if prediction[STUDY].to_list() != frame[STUDY].to_list():
        raise ValueError(f"Prediction membership differs from the fold: {path}")
    return validation_metrics(
        frame.select(targets).to_numpy().astype(float),
        prediction.select(targets).to_numpy().astype(float),
        targets,
        frame["is_gold"].to_numpy(),
    )


def diagnose(settings: DiagnoseCLI) -> dict:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory already exists: {settings.output_dir}")
    run = settings.run.expanduser().resolve()
    fold_dir = run / f"fold_{settings.fold}"
    selected = json.loads((fold_dir / "selected_checkpoint.json").read_text())
    checkpoint_path = fold_dir / Path(selected["path"]).name
    config_path = run / "effective_config.json"
    config = load_config(config_path)
    if not config.model.stride16_fusion:
        raise ValueError("Run's model.stride16_fusion is not enabled")
    if config.model.fusion_readout != "token_sum":
        raise ValueError("This diagnosis only supports fusion_readout=token_sum")
    device = resolve_device("auto")
    dataset, targets, _ = prediction_data(
        config, "validation", settings.fold, image_multiplier=settings.image_multiplier
    )
    loader = build_data_loader(dataset, config.trainer, INFERENCE_BATCH_SIZE)
    model = load_checkpoint_model(
        config, targets, checkpoint_path, device, expected_fold=settings.fold
    )
    settings.output_dir.mkdir(parents=True)
    source_sha256 = copy_sources([Path(__file__)], settings.output_dir / "source")
    results = {}
    terms: dict = {}
    branch_stats = None
    for mode in MODES:
        recorder = BranchStats() if mode == "full" else None
        study_ids, predictions, seconds = run_pass(
            model, loader, device, mode, terms, recorder
        )
        path = settings.output_dir / f"predictions_{mode}.csv"
        make_prediction_frame(study_ids, predictions, targets, settings.fold).write_csv(
            path
        )
        metrics = score_pass(config, settings.fold, targets, path)
        results[mode] = {
            "macro_auc": metrics["macro_auc"],
            "per_target_auc": metrics["per_target_auc"],
            "wall_seconds": seconds,
        }
        if recorder is not None:
            branch_stats = recorder.summary()
    deltas = {
        mode: {
            target: results[mode]["per_target_auc"][target]
            - results["full"]["per_target_auc"][target]
            for target in targets
        }
        for mode in ("fine_zeroed", "deep_zeroed")
    }
    macro_deltas = {
        mode: results[mode]["macro_auc"] - results["full"]["macro_auc"]
        for mode in ("fine_zeroed", "deep_zeroed")
    }
    reproduction = None
    if settings.image_multiplier == 4:
        reproduction = {
            "checked": True,
            "recorded_prediction_macro_auc": selected["prediction_macro_auc"],
            "computed_macro_auc": results["full"]["macro_auc"],
            "difference": results["full"]["macro_auc"]
            - selected["prediction_macro_auc"],
        }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "run": str(run),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "fold": settings.fold,
            "image_multiplier": settings.image_multiplier,
            "config_sha256": sha256_file(config_path),
        },
        "device": str(device),
        "targets": list(targets),
        "results": results,
        "per_target_deltas": deltas,
        "macro_deltas": macro_deltas,
        "branch_statistics": branch_stats,
        "attention_weighted_statistics": (
            "skipped: getting per-label attention weights needs duplicating "
            "_attention_pool's private masked-softmax and anatomical-bias logic "
            "outside model.py, since the production inference path returns only "
            "probabilities; that duplication risks silently drifting from the real "
            "computation"
        ),
        "reproduction_check": reproduction,
        "source_sha256": source_sha256,
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    header = f"{'label':<20}{'full_auc':>10}{'d_fine_zeroed':>16}{'d_deep_zeroed':>16}"
    print(header)
    for target in targets:
        print(
            f"{target:<20}{results['full']['per_target_auc'][target]:>10.4f}"
            f"{deltas['fine_zeroed'][target]:>16.4f}{deltas['deep_zeroed'][target]:>16.4f}"
        )
    print(
        f"{'macro':<20}{results['full']['macro_auc']:>10.4f}"
        f"{macro_deltas['fine_zeroed']:>16.4f}{macro_deltas['deep_zeroed']:>16.4f}"
    )
    if reproduction is not None and abs(reproduction["difference"]) > 1e-6:
        raise ValueError(
            f"Full pass did not reproduce the recorded x4 macro AUC: {reproduction}"
        )
    return summary


def main() -> None:
    try:
        settings = DiagnoseCLI()
        configure_cuda_allocator()
        diagnose(settings)
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Fusion branch diagnosis error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
