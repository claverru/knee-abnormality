import cProfile
import json
import tempfile
import time
from pathlib import Path
from typing import Literal

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


class Settings(CLISettings):
    model_config = describe(
        "Time the training and validation steps of a recipe under the real trainer on "
        "the GPU: mean wait on the loader and mean step in milliseconds, steps per "
        "second, peak memory and the first training losses, with an optional breakdown "
        "per function (synchronised timers) or a cProfile dump. Validation limited to "
        "a few batches ends on the module's completeness check, which is ignored"
    )
    config: Path = Field(description="Experiment configuration")
    output: Path = Field(description="JSON report path")
    steps: int = Field(default=110, gt=10, description="Training steps to run")
    validation_steps: int = Field(default=40, ge=0, description="Validation steps")
    mode: Literal["time", "profile", "cprofile"] = Field(
        default="time",
        description="time only, per-function synchronised timers, or a cProfile dump",
    )


class Timer(L.Callback):
    def __init__(self, skip: int) -> None:
        self.skip = skip
        self.waits, self.steps, self.losses = [], [], []
        self.validation_waits, self.validation_steps = [], []
        self.mark = None

    def _now(self) -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    def on_train_batch_start(self, trainer, module, batch, batch_index) -> None:
        now = self._now()
        if self.mark is not None and batch_index >= self.skip:
            self.waits.append(now - self.mark)
        self.mark = now

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_index) -> None:
        now = self._now()
        if batch_index >= self.skip:
            self.steps.append(now - self.mark)
        self.losses.append(float(outputs["loss"]))
        self.mark = now

    def on_validation_epoch_start(self, trainer, module) -> None:
        self.mark = None

    def on_validation_batch_start(
        self, trainer, module, batch, batch_index, dataloader_idx=0
    ) -> None:
        now = self._now()
        if self.mark is not None and batch_index >= self.skip:
            self.validation_waits.append(now - self.mark)
        self.mark = now

    def on_validation_batch_end(
        self, trainer, module, outputs, batch, batch_index, dataloader_idx=0
    ) -> None:
        now = self._now()
        if batch_index >= self.skip:
            self.validation_steps.append(now - self.mark)
        self.mark = now


PROFILED = {
    "module": (
        "training_step",
        "validation_step",
        "_record_sampling_sources",
        "_log_crop_frames",
        "_log_augmentation_activity",
        "_log_views",
        "_side_step",
        "_report_attribute_step",
        "_anatomy_step",
        "_scanner_confusion_step",
        "on_before_optimizer_step",
    ),
    "model": (
        "encode_batch",
        "first_look_pose",
        "_encode_crops",
        "_encode_images",
        "_augment_images",
        "_mix_contrast",
        "pool_encoded",
        "study_frames",
        "anatomical_frames",
        "frame",
        "_regions",
        "routing_weights",
    ),
    "view_windows": ("forward", "first_look_batch", "frame_pose", "jitter"),
}


def profile(target, names: tuple[str, ...], label: str, totals: dict) -> None:
    for name in names:
        if not hasattr(target, name):
            continue
        original = getattr(target, name)

        def timed(*args, _original=original, _key=f"{label}.{name}", **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = _original(*args, **kwargs)
            torch.cuda.synchronize()
            entry = totals.setdefault(_key, [0.0, 0])
            entry[0] += time.perf_counter() - started
            entry[1] += 1
            return result

        setattr(target, name, timed)


def main() -> None:
    settings = Settings()
    config_path, steps, validation_steps, output = (
        settings.config,
        settings.steps,
        settings.validation_steps,
        settings.output,
    )
    mode = settings.mode
    configure_cuda_allocator()
    config = load_config(config_path.resolve(), overrides={})
    torch.set_float32_matmul_precision("high")
    fold = config.validation.folds[0]
    L.seed_everything(config.seed, workers=True)
    data = KneeDataModule(config, fold)
    data.setup("fit")
    labels = data.labels
    class_balance = fixed_class_balance(
        data.train_dataset.studies,
        labels.targets,
        config.trainer.class_balance_exponent,
    )
    identity = training_input_identity(
        config, labels, data.train_dataset.studies, data.validation_dataset.studies
    )
    with tempfile.TemporaryDirectory() as scratch:
        module = KneeModule(
            config,
            labels.targets,
            fold,
            len(data.train_dataset),
            labels.validation.height,
            Path(scratch) / "epochs",
            class_balance,
            input_identity=identity,
        )
        timer = Timer(skip=10)
        totals: dict = {}
        if mode == "profile":
            profile(module, PROFILED["module"], "module", totals)
            profile(module.model, PROFILED["model"], "model", totals)
            profile(
                module.model.view_windows,
                PROFILED["view_windows"],
                "view_windows",
                totals,
            )
        trainer = L.Trainer(
            default_root_dir=scratch,
            max_epochs=1,
            limit_train_batches=steps,
            limit_val_batches=validation_steps,
            precision=PRECISION,
            accelerator="auto",
            devices=1,
            accumulate_grad_batches=config.trainer.accumulate_grad_batches,
            gradient_clip_val=config.trainer.gradient_clip_val,
            deterministic=config.trainer.deterministic,
            callbacks=[timer],
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        torch.cuda.reset_peak_memory_stats()
        profiler = cProfile.Profile() if mode == "cprofile" else None
        try:
            if profiler is not None:
                profiler.enable()
            trainer.fit(module, datamodule=data)
        except RuntimeError as error:
            if "every study exactly once" not in str(error):
                raise
        finally:
            if profiler is not None:
                profiler.disable()
                profiler.dump_stats(str(output.with_suffix(".prof")))

    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    result = {
        "train_wait_ms": 1000 * mean(timer.waits),
        "train_step_ms": 1000 * mean(timer.steps),
        "train_it_per_s": 1 / max(mean(timer.waits) + mean(timer.steps), 1e-9),
        "validation_wait_ms": 1000 * mean(timer.validation_waits),
        "validation_step_ms": 1000 * mean(timer.validation_steps),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
        "losses": timer.losses,
    }
    steps_run = max(len(timer.losses), 1)
    result["profile_ms_per_train_step"] = {
        key: [round(1000 * total / steps_run, 2), count]
        for key, (total, count) in sorted(totals.items(), key=lambda item: -item[1][0])
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "losses"}, indent=2))
    print("losses", [round(v, 6) for v in timer.losses[:8]])


if __name__ == "__main__":
    main()
