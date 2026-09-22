import cProfile
import gc
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pytorch_lightning as L
import torch
import yaml
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsError
from torch.utils.data import Subset

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import SLICES_PER_INPUT, ExperimentConfig, load_config
from knee_abnormality.data.dataset import (
    KneeStudyDataset,
    build_data_loader,
    collate_prepared,
)
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.modeling.model import KneeAbnormalityModel
from knee_abnormality.profiling import (
    finish_profile,
    profile_manifest,
    sample_dataset,
    setup_profile,
    training_step,
    write_profile_manifest,
)
from knee_abnormality.training.lightning import (
    PRECISION,
    KneeModule,
    build_optimizer,
    fixed_class_balance,
    validated_class_balance_weights,
)


class ProfileCLI(CLISettings):
    config: CliPositionalArg[Path]
    output_dir: Path
    mode: Literal["preprocessing", "gpu", "loader", "lightning"] = "preprocessing"
    samples: int = Field(default=8, ge=4, le=64)
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 2, 3])
    workers: list[int] = Field(default_factory=lambda: [0, 2, 4])
    warmup: int = Field(default=2, ge=1, le=4)
    steps: int = Field(default=4, ge=1, le=16)
    threads: int = Field(default=1, ge=1, le=16)


def preprocessing_profile(dataset: KneeStudyDataset, args: ProfileCLI) -> dict:
    passes = []
    for repeat in range(2):
        profiler = cProfile.Profile()
        rows = []
        for index in range(len(dataset)):
            start = time.perf_counter()
            profiler.enable()
            item = dataset[index]
            profiler.disable()
            seconds = time.perf_counter() - start
            prepared = [
                value for acquisitions in item["series"] for value in acquisitions
            ]
            rows.append(
                {
                    "study_uid": item["study_uid"],
                    "seconds": seconds,
                    "channel_appearances": sum(value.slice_count for value in prepared),
                    "unique_source_slices": sum(
                        value.unique_slice_count for value in prepared
                    ),
                    "inputs": sum(value.triplet_count for value in prepared),
                    "series": len(prepared),
                }
            )
        profiler.dump_stats(args.output_dir / f"preprocessing_{repeat}.pstats")
        with (args.output_dir / f"preprocessing_{repeat}.txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                "cumulative"
            ).print_stats(30)
        seconds = sum(row["seconds"] for row in rows)
        passes.append(
            {
                "pass": repeat,
                "seconds": seconds,
                "studies_per_second": len(rows) / seconds,
                "studies": rows,
            }
        )
        print(
            json.dumps(
                {
                    "pass": repeat,
                    "seconds": seconds,
                    "studies_per_second": len(rows) / seconds,
                }
            ),
            flush=True,
        )
    return {"passes": passes}


def gpu_candidate(
    config: ExperimentConfig,
    dataset: KneeStudyDataset,
    items: list[dict],
    size: int,
    args: ProfileCLI,
    class_balance_weights: torch.Tensor | None,
) -> dict:
    model = (
        KneeAbnormalityModel(
            config.model, config.data, dataset.targets, load_pretrained=False
        )
        .cuda()
        .train()
    )
    model.series_dropout = 0
    optimizer = build_optimizer(model, config.optimizer)
    batch = move_batch(collate_prepared(items[:size]), torch.device("cuda"))
    torch.cuda.reset_peak_memory_stats()
    durations = []
    losses = []
    for step in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = training_step(model, optimizer, batch, config, class_balance_weights)
        torch.cuda.synchronize()
        if step >= args.warmup:
            durations.append(time.perf_counter() - start)
            losses.append(loss)
    return {
        "batch_size": size,
        "status": "ok",
        "study_ids": batch["study_uid"],
        "inputs": int(batch["slice_mask"].sum()),
        "slices": int(batch["slice_mask"].sum()) * SLICES_PER_INPUT,
        "step_seconds": durations,
        "studies_per_second": size / float(np.mean(durations)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "losses": losses,
        "batchnorm_batches_tracked": sorted(
            {
                int(module.num_batches_tracked)
                for module in model.backbone.modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                and module.num_batches_tracked is not None
            }
        ),
    }


def gpu_profile(
    config: ExperimentConfig,
    dataset: KneeStudyDataset,
    args: ProfileCLI,
    class_balance: dict | None,
) -> dict:
    sizes = sorted(set(args.batch_sizes))
    if min(sizes) < 1 or max(sizes) > min(8, len(dataset)):
        raise ValueError(
            "GPU profiling supports batch sizes 1 through 8 using distinct maximum-load studies"
        )
    items = [dataset[index] for index in range(max(sizes))]
    class_balance_weights = validated_class_balance_weights(
        config, dataset.targets, class_balance
    )
    if class_balance_weights is not None:
        class_balance_weights = class_balance_weights.to("cuda")
    rows = []
    for size in sizes:
        torch.manual_seed(config.seed)
        try:
            row = gpu_candidate(
                config, dataset, items, size, args, class_balance_weights
            )
        except torch.OutOfMemoryError as error:
            row = {"batch_size": size, "status": "out_of_memory", "error": str(error)}
        gc.collect()
        torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
        (args.output_dir / "gpu_progress.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
        if (
            row["status"] == "out_of_memory"
            or row.get("peak_reserved_gib", 0)
            > 0.95 * torch.cuda.get_device_properties(0).total_memory / 2**30
        ):
            break
    return {
        "gpu": torch.cuda.get_device_name(),
        "gpu_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "series_dropout_for_capacity": 0,
        "candidates": rows,
    }


def loader_candidate(
    config: ExperimentConfig,
    dataset: KneeStudyDataset,
    size: int,
    workers: int,
    args: ProfileCLI,
) -> dict:
    dataset._slice_index.clear()
    dataset._study_references.clear()
    dataset.training_seed = config.seed
    count = (args.warmup + args.steps) * size
    if count > len(dataset):
        raise ValueError(f"Need --samples at least {count} for this loader profile")
    sample = Subset(
        dataset,
        np.random.default_rng(config.seed).permutation(len(dataset))[:count].tolist(),
    )
    class_balance = (
        fixed_class_balance(
            dataset.studies[sample.indices],
            dataset.targets,
            config.trainer.class_balance_exponent,
        )
        if config.trainer.class_balanced_bce
        else None
    )
    loader = build_data_loader(
        sample,
        config.trainer,
        size,
        num_workers=workers,
        persistent_workers=False,
    )
    iterator = iter(loader)
    class_balance_weights = validated_class_balance_weights(
        config, dataset.targets, class_balance
    )
    if class_balance_weights is not None:
        class_balance_weights = class_balance_weights.to("cuda")
    model = (
        KneeAbnormalityModel(
            config.model, config.data, dataset.targets, load_pretrained=False
        )
        .cuda()
        .train()
    )
    optimizer = build_optimizer(model, config.optimizer)
    waits = []
    durations = []
    slices = []
    inputs = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        batch = next(iterator)
        waited = time.perf_counter() - started
        batch = with_images(move_batch(batch, torch.device("cuda")))
        training_step(model, optimizer, batch, config, class_balance_weights)
        torch.cuda.synchronize()
        if step >= args.warmup:
            waits.append(waited)
            durations.append(time.perf_counter() - started)
            count = int(batch["slice_mask"].sum())
            inputs.append(count)
            slices.append(count * SLICES_PER_INPUT)
    return {
        "batch_size": size,
        "workers": workers,
        "step_seconds": durations,
        "loader_wait_seconds": waits,
        "studies_per_second": size / float(np.mean(durations)),
        "loader_wait_fraction": sum(waits) / sum(durations),
        "slices": slices,
        "inputs": inputs,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "class_balance": class_balance,
    }


def loader_profile(
    config: ExperimentConfig, dataset: KneeStudyDataset, args: ProfileCLI
) -> dict:
    rows = []
    for size in args.batch_sizes:
        for workers in args.workers:
            if size < 1 or workers < 0:
                raise ValueError(
                    "Batch sizes must be positive and worker counts nonnegative"
                )
            torch.manual_seed(config.seed)
            row = loader_candidate(config, dataset, size, workers, args)
            gc.collect()
            torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.output_dir / "loader_progress.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
    return {"gpu": torch.cuda.get_device_name(), "candidates": rows}


class BatchTimings(L.Callback):
    def __init__(self, warmup_batches: int) -> None:
        self.warmup_batches = warmup_batches
        self.rows = []
        self.previous_end = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        torch.cuda.synchronize()
        self.started = time.perf_counter()
        self.wait = (
            self.started - self.previous_end if self.previous_end is not None else None
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        torch.cuda.synchronize()
        ended = time.perf_counter()
        loss = float(outputs["loss"].detach())
        if not np.isfinite(loss):
            raise RuntimeError("Non-finite training loss during profiling")
        if batch_idx >= self.warmup_batches:
            self.rows.append(
                {
                    "batch": batch_idx,
                    "compute_seconds": ended - self.started,
                    "between_batches_seconds": self.wait,
                    "loss": loss,
                    "studies": len(batch["study_uid"]),
                    "inputs": int(batch["slice_mask"].sum()),
                    "slices": int(batch["slice_mask"].sum()) * SLICES_PER_INPUT,
                }
            )
        self.previous_end = ended


def lightning_profile(
    config: ExperimentConfig,
    dataset: KneeStudyDataset,
    args: ProfileCLI,
    class_balance: dict | None,
) -> dict:
    settings = config.trainer
    steps = args.warmup + args.steps
    if steps * settings.accumulate_grad_batches * settings.batch_size >= len(dataset):
        raise ValueError(
            "Increase --samples so the bounded Lightning run stops before exhausting its study subset"
        )
    dataset.training_seed = config.seed
    loader = build_data_loader(
        dataset,
        settings,
        settings.batch_size,
        shuffle=True,
    )
    module = KneeModule(
        config,
        dataset.targets,
        config.validation.folds[0],
        len(dataset),
        0,
        args.output_dir,
        class_balance,
        load_pretrained=False,
    )
    timing = BatchTimings(args.warmup * settings.accumulate_grad_batches)
    trainer = L.Trainer(
        max_steps=steps,
        max_epochs=settings.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=PRECISION,
        accumulate_grad_batches=settings.accumulate_grad_batches,
        gradient_clip_val=settings.gradient_clip_val,
        deterministic=settings.deterministic,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[timing],
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(module, train_dataloaders=loader)
    seconds = sum(
        row["compute_seconds"] + row["between_batches_seconds"] for row in timing.rows
    )
    result = {
        "batches": timing.rows,
        "optimizer_steps": trainer.global_step,
        "studies_per_second": sum(row["studies"] for row in timing.rows) / seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    print(json.dumps(result), flush=True)
    return result


def run(args: ProfileCLI) -> None:
    config = load_config(args.config)
    allocator_config, source = setup_profile(
        config, args.output_dir, args.threads, [Path(__file__), args.config]
    )
    started = time.perf_counter()
    dataset, labels = sample_dataset(config, args.samples)
    class_balance = (
        fixed_class_balance(
            dataset.studies, dataset.targets, config.trainer.class_balance_exponent
        )
        if config.trainer.class_balanced_bce and args.mode in ("gpu", "lightning")
        else None
    )
    result = profile_manifest(
        config,
        args.model_dump(mode="json"),
        allocator_config,
        source,
        labels,
        time.perf_counter() - started,
        {"class_balance": class_balance},
    )
    write_profile_manifest(args.output_dir, result)
    if args.mode == "preprocessing":
        result.update(preprocessing_profile(dataset, args))
    elif args.mode == "gpu":
        result.update(gpu_profile(config, dataset, args, class_balance))
    elif args.mode == "loader":
        result.update(loader_profile(config, dataset, args))
    else:
        result.update(lightning_profile(config, dataset, args, class_balance))
    finish_profile(args.output_dir, result, started)


def main() -> None:
    try:
        run(ProfileCLI())
    except (OSError, RuntimeError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Training profile error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
