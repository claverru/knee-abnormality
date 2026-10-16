import json
import multiprocessing
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytorch_lightning as L
import torch
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.training.checkpoints import training_input_identity
from knee_abnormality.training.cuda import configure_cuda_allocator
from knee_abnormality.training.lightning import (
    PRECISION,
    KneeDataModule,
    KneeModule,
    fixed_class_balance,
)
from scripts.pipeline.train import _state_sha256

WARMUP_STEPS = 15
VALIDATION_WARMUP = 3
EPOCH_OVERHEAD = 1.13
VALIDATION_INCOMPLETE = "Validation epoch did not gather every study exactly once"
GIB = 2**30


class PreflightCLI(CLISettings):
    model_config = describe(
        "Before an autoresearch training, run the candidate's real training loop on a few dozen batches with a "
        "GPU memory cap, project its training time from the measured steps, and exec the training only when the "
        "peak memory and the projected time fit the campaign's limits"
    )

    config: Path = Field(description="Candidate experiment config")
    fold: int = Field(ge=0, description="Campaign fold")
    output: Path = Field(description="Writable directory for the preflight report")
    train_batches: int = Field(
        default=60, gt=WARMUP_STEPS, description="Training batches to run"
    )
    validation_batches: int = Field(
        default=20, gt=0, description="Validation batches to time"
    )
    memory_cap_gib: float = Field(
        default=22.5, gt=0, description="Allowed peak allocated GPU memory"
    )
    count_margin: float = Field(
        default=1.08,
        ge=1,
        description="Largest batch assumed, relative to the largest seen",
    )
    command: list[str] = Field(
        description="Training command executed after a passing preflight"
    )
    measure_only: bool = Field(
        default=False, description="Run the measurement in this process and exit"
    )


class Measure(L.Callback):
    def __init__(self, module: KneeModule) -> None:
        self.steps: list[dict] = []
        self.validation: list[float] = []
        self.count = 0
        stages = module.model.backbone_stages
        original = stages.to_stride_8

        def counted(images, *args, **kwargs):
            self.count = max(self.count, images.shape[0])
            return original(images, *args, **kwargs)

        stages.to_stride_8 = counted

    def on_train_batch_start(self, trainer, module, batch, batch_idx) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.count = 0
        self.started = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        torch.cuda.synchronize()
        self.steps.append(
            {
                "seconds": time.perf_counter() - self.started,
                "images": self.count,
                "peak_gib": torch.cuda.max_memory_allocated() / GIB,
            }
        )

    def on_validation_batch_start(
        self, trainer, module, batch, batch_idx, dataloader_idx=0
    ) -> None:
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_validation_batch_end(
        self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        torch.cuda.synchronize()
        self.validation.append(time.perf_counter() - self.started)


def measure(settings: PreflightCLI) -> dict:
    config = load_config(settings.config.expanduser().resolve())
    limits = json.loads(Path(os.environ["KNEE_AUTORESEARCH_LIMITS"]).read_text())
    L.seed_everything(config.seed, workers=True)
    data = KneeDataModule(config, settings.fold)
    data.setup("fit")
    labels = data.labels
    class_balance = fixed_class_balance(
        data.train_dataset.studies,
        labels.targets,
        config.trainer.class_balance_exponent,
    )
    fold_dir = settings.output / "fold"
    fold_dir.mkdir(parents=True, exist_ok=True)
    module = KneeModule(
        config,
        labels.targets,
        settings.fold,
        len(data.train_dataset),
        labels.validation.height,
        fold_dir / "epochs",
        class_balance,
        input_identity=training_input_identity(
            config, labels, data.train_dataset.studies, data.validation_dataset.studies
        ),
    )
    if config.model.pretrained_weights != "radimagenet":
        json.dumps(module.model.backbone.pretrained_cfg, default=str)
    _state_sha256(module.model.backbone.state_dict())
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, settings.memory_cap_gib * GIB / total)
    )
    callback = Measure(module)
    trainer = L.Trainer(
        default_root_dir=fold_dir,
        max_epochs=1,
        limit_train_batches=settings.train_batches,
        limit_val_batches=settings.validation_batches,
        precision=PRECISION,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.trainer.gradient_clip_val,
        deterministic=config.trainer.deterministic,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    error = None
    try:
        trainer.fit(module, datamodule=data)
    except (RuntimeError, ValueError) as exception:
        if VALIDATION_INCOMPLETE not in str(exception):
            error = f"{type(exception).__name__}: {str(exception)[:400]}"
    steps = callback.steps
    report = {
        "train_batches_run": len(steps),
        "validation_batches_run": len(callback.validation),
        "memory_cap_gib": settings.memory_cap_gib,
        "error": error,
        "limits": {
            key: limits.get(key)
            for key in ("epochs", "min_training_seconds", "max_training_seconds")
        },
    }
    if (
        len(steps) < settings.train_batches
        or len(callback.validation) < settings.validation_batches
    ):
        report["passed"] = False
        report["reason"] = f"The candidate failed during the preflight batches: {error}"
        return report
    measured = steps[WARMUP_STEPS:]
    step_seconds = statistics.median(step["seconds"] for step in measured)
    train_steps = len(data.train_dataloader())
    validation_steps = len(data.val_dataloader())
    validation_seconds = (
        statistics.median(callback.validation[VALIDATION_WARMUP:]) * validation_steps
    )
    epochs = config.trainer.max_epochs
    first_epoch_extra = sum(
        max(step["seconds"] - step_seconds, 0.0) for step in steps
    ) + sum(
        max(seconds - validation_seconds / validation_steps, 0.0)
        for seconds in callback.validation
    )
    projected = (
        EPOCH_OVERHEAD * epochs * (train_steps * step_seconds + validation_seconds)
        + first_epoch_extra
    )
    images = [step["images"] for step in steps]
    peaks = [step["peak_gib"] for step in steps]
    mean_images = statistics.fmean(images)
    slope = sum(
        (count - mean_images) * (peak - statistics.fmean(peaks))
        for count, peak in zip(images, peaks)
    ) / max(sum((count - mean_images) ** 2 for count in images), 1e-9)
    largest = max(images) * settings.count_margin
    predicted_peak = max(peaks) + max(slope, 0.0) * (
        largest - images[peaks.index(max(peaks))]
    )
    report.update(
        {
            "median_step_seconds": step_seconds,
            "epoch_overhead": EPOCH_OVERHEAD,
            "first_epoch_extra_seconds": first_epoch_extra,
            "steps": steps,
            "validation_batch_seconds": callback.validation,
            "train_steps_per_epoch": train_steps,
            "validation_seconds_per_epoch": validation_seconds,
            "epochs": epochs,
            "projected_training_seconds": projected,
            "largest_batch_images_seen": max(images),
            "assumed_largest_batch_images": largest,
            "measured_peak_gib": max(peaks),
            "gib_per_image": slope,
            "predicted_peak_gib": predicted_peak,
        }
    )
    reasons = []
    if predicted_peak > settings.memory_cap_gib:
        reasons.append(
            f"predicted peak GPU memory {predicted_peak:.1f} GiB at {largest:.0f} images per batch exceeds the "
            f"{settings.memory_cap_gib:.1f} GiB cap (measured {max(peaks):.1f} GiB at up to {max(images)} images, "
            f"{slope:.3f} GiB per image); reduce activation memory, e.g. activation checkpointing of early stages"
        )
    if (
        limits.get("max_training_seconds") is not None
        and projected > limits["max_training_seconds"]
    ):
        reasons.append(
            f"projected training time {projected:.0f} s ({step_seconds:.3f} s per step x {train_steps} steps x "
            f"{epochs} epochs plus validation {validation_seconds:.0f} s per epoch) exceeds the upper bound of "
            f"{limits['max_training_seconds']:.0f} s"
        )
    if (
        limits.get("min_training_seconds") is not None
        and projected < limits["min_training_seconds"]
    ):
        reasons.append(
            f"projected training time {projected:.0f} s is below the lower bound of {limits['min_training_seconds']:.0f} s"
        )
    report["passed"] = not reasons
    report["reason"] = "; ".join(reasons) or None
    return report


def main() -> None:
    settings = PreflightCLI()
    settings.output.mkdir(parents=True, exist_ok=True)
    if not settings.measure_only:
        child = subprocess.run(
            [sys.executable, "-P", *sys.argv, "--measure-only"], check=False
        )
        if child.returncode != 0:
            sys.exit(child.returncode)
        os.execv(settings.command[0], settings.command)
    configure_cuda_allocator()
    multiprocessing.set_start_method("fork", force=True)
    torch.set_float32_matmul_precision("high")
    report = measure(settings)
    (settings.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PREFLIGHT {json.dumps(report)}", file=sys.stderr, flush=True)
    if not report["passed"]:
        print(f"PREFLIGHT FAILED: {report['reason']}", file=sys.stderr, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
