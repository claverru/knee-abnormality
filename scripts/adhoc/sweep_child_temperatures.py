import json
import os
import time
from pathlib import Path

import polars as pl
import torch
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import (
    INFERENCE_IO_THREADS,
    INFERENCE_LOADER_NICENESS,
    INFERENCE_NUM_WORKERS,
    PositiveFiniteFloat,
    load_config,
)
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import (
    apply_attention_temperature,
    load_checkpoint_model,
)
from knee_abnormality.inference.engine import move_batch, resolve_precision
from knee_abnormality.inference.reflection import reflect_batch, sagittal_types


class SweepCLI(CLISettings):
    model_config = describe(
        "Predict each run's validation folds at several image multipliers and attention "
        "temperatures from one backbone pass per fold and multiplier, with and without "
        "reflection TTA"
    )

    run: list[Path] = Field(min_length=1, description="Training run directories")
    family: str = Field(description="Family name written to the output files")
    multiplier: list[int] = Field(default=[4, 5, 6], min_length=1)
    tau: list[PositiveFiniteFloat] = Field(
        default=[0.7, 0.8, 0.9, 1.0, 1.1], min_length=1
    )
    output_dir: Path = Field(description="Directory for the prediction tables")
    batch_size: int = Field(default=2, gt=0, description="Studies per batch")


def lower_priority(worker_id: int) -> None:
    os.nice(INFERENCE_LOADER_NICENESS)


def pooled(
    model, batch: dict, taus: list[float], dtype: torch.dtype | None
) -> list[torch.Tensor]:
    device = next(model.parameters()).device.type
    with torch.autocast(device, dtype=dtype, enabled=dtype is not None):
        encoded = model.encode_batch(batch)
    outputs = []
    for tau in taus:
        apply_attention_temperature(model, tau)
        with torch.autocast(device, dtype=dtype, enabled=dtype is not None):
            outputs.append(torch.sigmoid(model.pool_encoded(encoded).to(torch.float32)))
        apply_attention_temperature(model, 1 / tau)
    return outputs


def sweep_fold(
    settings: SweepCLI, run: Path, fold: int, multiplier: int, output: Path
) -> dict:
    started = time.perf_counter()
    config = load_config(run / "effective_config.json")
    targets, studies, metadata_split = prediction_studies(config, "validation", fold)
    selected = json.loads(
        (run / f"fold_{fold}" / "selected_checkpoint.json").read_text()
    )
    device = torch.device("cuda")
    model = load_checkpoint_model(
        config, targets, Path(selected["path"]), device, expected_fold=fold
    )
    loader = build_data_loader(
        KneeStudyDataset(
            studies,
            build_series_index(
                config.paths.data_root,
                config.data,
                metadata_split,
                studies[STUDY].to_list(),
            ),
            config.data,
            targets,
            None,
            multiplier,
            INFERENCE_IO_THREADS,
        ),
        config.trainer,
        settings.batch_size,
        num_workers=INFERENCE_NUM_WORKERS,
        pin_memory=False,
        prepare_on_loader=False,
    )
    loader.worker_init_fn = lower_priority
    precision, dtype = resolve_precision("auto", device)
    sagittal = sagittal_types(model.augmentation_settings.series_types)
    study_ids = []
    plain = [[] for _ in settings.tau]
    mirrored = [[] for _ in settings.tau]
    with torch.inference_mode():
        for batch in loader:
            study_ids.extend(batch["study_uid"])
            batch = with_images(move_batch(batch, device))
            reflected = reflect_batch(batch, sagittal, model.repeated_middle_slice)
            eligible = batch["reflection_eligible"][:, None]
            direct = pooled(model, batch, settings.tau, dtype)
            flipped = pooled(model, reflected, settings.tau, dtype)
            for index, (a, b) in enumerate(zip(direct, flipped, strict=True)):
                plain[index].append(a.cpu())
                mirrored[index].append(torch.where(eligible, (a + b) / 2, a).cpu())
            del batch, reflected, direct, flipped
            torch.cuda.empty_cache()
    frames = []
    for index, tau in enumerate(settings.tau):
        for tta, parts in ((False, plain[index]), (True, mirrored[index])):
            values = torch.cat(parts).numpy()
            frames.append(
                pl.DataFrame(
                    {
                        STUDY: study_ids,
                        "fold": fold,
                        "family": settings.family,
                        "multiplier": multiplier,
                        "tau": tau,
                        "tta": tta,
                        **{t: values[:, i] for i, t in enumerate(targets)},
                    }
                )
            )
    frame = pl.concat(frames)
    if (
        frame.filter((pl.col("tau") == settings.tau[0]) & ~pl.col("tta"))[
            STUDY
        ].n_unique()
        != studies.height
    ):
        raise ValueError(f"{run} fold {fold} did not predict every validation study")
    frame.write_parquet(output)
    return {
        "run": str(run),
        "fold": fold,
        "multiplier": multiplier,
        "checkpoint": selected["path"],
        "precision": precision,
        "studies": studies.height,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    settings = SweepCLI()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(0.85)
    for run in settings.run:
        folds = json.loads((run / "effective_config.json").read_text())["validation"][
            "folds"
        ]
        for fold in folds:
            for multiplier in settings.multiplier:
                output = (
                    settings.output_dir
                    / f"{settings.family}-fold{fold}-x{multiplier}.parquet"
                )
                if output.exists():
                    continue
                record = sweep_fold(settings, run, fold, multiplier, output)
                output.with_suffix(".json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
