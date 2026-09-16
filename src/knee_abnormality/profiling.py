import json
import os
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
import pytorch_lightning as L
import torch

from knee_abnormality.config import SLICES_PER_INPUT, ExperimentConfig
from knee_abnormality.data.dataset import KneeStudyDataset
from knee_abnormality.data.labels import STUDY, FoldLabels, load_fold_labels
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.provenance import copy_sources, sha256_file
from knee_abnormality.training.cuda import configure_cuda_allocator
from knee_abnormality.training.lightning import masked_binary_cross_entropy_with_logits


def sample_dataset(
    config: ExperimentConfig, count: int
) -> tuple[KneeStudyDataset, FoldLabels]:
    labels = load_fold_labels(config, config.validation.folds[0])
    index = build_series_index(config.paths.data_root, config.data, "train")
    ranked = sorted(
        labels.training[STUDY].to_list(),
        key=lambda uid: (
            -sum(
                min(
                    sum(max(c.image_count - 2, 0) for c in values),
                    kind.max_images // SLICES_PER_INPUT,
                )
                for values, kind in zip(
                    index[uid].slots, config.data.series_types, strict=True
                )
            ),
            uid,
        ),
    )
    rng = np.random.default_rng(config.seed)
    heaviest = min(count, 8)
    selected = (
        ranked[:heaviest]
        + rng.choice(ranked[heaviest:], count - heaviest, replace=False).tolist()
    )
    rows = pl.DataFrame({STUDY: selected}).join(
        labels.training, on=STUDY, how="left", maintain_order="left"
    )
    dataset = KneeStudyDataset(
        rows, index, config.data, labels.targets, False, config.seed
    )
    return dataset, labels


def training_step(model, optimizer, batch: dict, config: ExperimentConfig) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        logits = model(batch)
    loss = masked_binary_cross_entropy_with_logits(
        logits.float(), batch["target"], batch["target_observed"]
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.gradient_clip_val)
    optimizer.step()
    return float(loss.detach())


def setup_profile(
    config: ExperimentConfig,
    output_dir: Path,
    threads: int,
    source_paths: list[Path],
) -> tuple[str, Path]:
    allocator = configure_cuda_allocator()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    L.seed_everything(config.seed, workers=True)
    torch.use_deterministic_algorithms(config.trainer.deterministic)
    torch.backends.cudnn.benchmark = False
    snapshot = output_dir / "source"
    copy_sources([Path("src/knee_abnormality"), *source_paths], snapshot)
    return allocator, snapshot


def profile_manifest(
    config: ExperimentConfig,
    options: dict,
    allocator: str,
    snapshot: Path,
    labels: FoldLabels,
    setup_seconds: float,
    extra: dict | None = None,
) -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "options": options,
        "cuda_allocator_config": allocator,
        "versions": {
            name: version(name)
            for name in (
                "torch",
                "numpy",
                "pydicom",
                "pytorch-lightning",
                "timm",
            )
        },
        "source_sha256": {
            str(path.relative_to(snapshot)): sha256_file(path)
            for path in snapshot.rglob("*")
            if path.is_file()
        },
        "label_provenance": labels.provenance,
        "setup_seconds": setup_seconds,
        **(extra or {}),
    }


def write_profile_manifest(output_dir: Path, result: dict) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


def finish_profile(output_dir: Path, result: dict, started: float) -> None:
    result["total_seconds"] = time.perf_counter() - started
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
