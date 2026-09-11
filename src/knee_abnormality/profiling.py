import json
import os
import shutil
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
import pytorch_lightning as L
import torch

from knee_abnormality.config import ExperimentConfig
from knee_abnormality.dataset import KneeStudyDataset, build_series_index
from knee_abnormality.labels import FoldLabels, load_fold_labels
from knee_abnormality.provenance import sha256_file
from knee_abnormality.runtime import configure_cuda_allocator
from knee_abnormality.training import masked_binary_cross_entropy_with_logits


def sample_dataset(
    config: ExperimentConfig, count: int
) -> tuple[KneeStudyDataset, FoldLabels]:
    labels = load_fold_labels(config, config.validation.folds[0])
    index = build_series_index(config.paths.data_root, config.data, "train")
    ranked = sorted(
        labels.training["StudyInstanceUID"].to_list(),
        key=lambda uid: (
            -sum(
                min(values[0].image_count, kind.max_images) if values else 0
                for values, kind in zip(
                    index[uid].slots, config.data.series_types, strict=True
                )
            ),
            uid,
        ),
    )
    rng = np.random.default_rng(config.seed)
    selected = ranked[:4] + rng.choice(ranked[4:], count - 4, replace=False).tolist()
    rows = pl.DataFrame({"StudyInstanceUID": selected}).join(
        labels.training, on="StudyInstanceUID", how="left", maintain_order="left"
    )
    dataset = KneeStudyDataset(
        rows, index, config.data, labels.targets, False, config.seed
    )
    return dataset, labels


def training_step(model, optimizer, batch: dict, config: ExperimentConfig) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
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
    snapshot.mkdir()
    for path in [*Path("src/knee_abnormality").glob("*.py"), *source_paths]:
        shutil.copy2(path, snapshot / path.name)
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
        "source_sha256": {path.name: sha256_file(path) for path in snapshot.iterdir()},
        "label_provenance": labels.provenance,
        "setup_seconds": setup_seconds,
        **(extra or {}),
    }


def write_profile_manifest(output_dir: Path, result: dict) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


def finish_profile(output_dir: Path, result: dict, started: float) -> None:
    result["total_seconds"] = time.perf_counter() - started
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
