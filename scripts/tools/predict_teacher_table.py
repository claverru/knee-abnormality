import json
import os
from pathlib import Path

import polars as pl
import torch
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import (
    INFERENCE_IO_THREADS,
    INFERENCE_LOADER_NICENESS,
    INFERENCE_NUM_WORKERS,
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
from knee_abnormality.provenance import sha256_file

IMAGE_MULTIPLIER = 6
ATTENTION_TEMPERATURE = 0.85
PROBABILITY_FLOOR = 1e-6


class TeacherTableCLI(CLISettings):
    model_config = describe(
        "Write the teacher table of a distillation: each teacher run's joint and "
        "single-view logits for the studies of its validation fold, at x6 with "
        "reflection TTA and attention temperature 0.85"
    )

    run: list[Path] = Field(
        min_length=1, description="Teacher training run directories, one fold each"
    )
    output: Path = Field(description="Teacher table to write (parquet)")
    batch_size: int = Field(default=2, gt=0, description="Studies per batch")


def lower_priority(worker_id: int) -> None:
    os.nice(INFERENCE_LOADER_NICENESS)


def readings(
    model, batch: dict, views: int, dtype: torch.dtype | None
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device.type
    with torch.autocast(device, dtype=dtype, enabled=dtype is not None):
        encoded = model.encode_batch(batch)
        probabilities = torch.stack(
            [
                torch.sigmoid(model.pool_encoded(encoded, view).to(torch.float32))
                for view in (None, *range(views))
            ]
        )
    seen = encoded["visual_valid"].any(-1)
    present = torch.stack(
        [
            torch.zeros(
                encoded["batch_size"], dtype=torch.bool, device=seen.device
            ).index_fill_(
                0, encoded["study"][seen & (encoded["view_index"] == view)], True
            )
            for view in range(views)
        ]
    )
    return probabilities, present


def predict_fold(settings: TeacherTableCLI, run: Path, fold: int) -> pl.DataFrame:
    config = load_config(run / "effective_config.json")
    targets, studies, metadata_split = prediction_studies(config, "validation", fold)
    selected = json.loads(
        (run / f"fold_{fold}" / "selected_checkpoint.json").read_text()
    )
    device = torch.device("cuda")
    model = load_checkpoint_model(
        config, targets, Path(selected["path"]), device, expected_fold=fold
    )
    apply_attention_temperature(model, ATTENTION_TEMPERATURE)
    names = config.data.view_names
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
            IMAGE_MULTIPLIER,
            INFERENCE_IO_THREADS,
        ),
        config.trainer,
        settings.batch_size,
        num_workers=INFERENCE_NUM_WORKERS,
        pin_memory=False,
        prepare_on_loader=False,
    )
    loader.worker_init_fn = lower_priority
    _, dtype = resolve_precision("auto", device)
    sagittal = sagittal_types(model.augmentation_settings.series_types)
    study_ids: list[str] = []
    values = []
    available = []
    with torch.inference_mode():
        for batch in loader:
            study_ids.extend(batch["study_uid"])
            batch = with_images(move_batch(batch, device))
            reflected = reflect_batch(batch, sagittal, model.repeated_middle_slice)
            direct, present = readings(model, batch, len(names), dtype)
            flipped, _ = readings(model, reflected, len(names), dtype)
            eligible = batch["reflection_eligible"][None, :, None]
            values.append(torch.where(eligible, (direct + flipped) / 2, direct).cpu())
            available.append(present.cpu())
            del batch, reflected, direct, flipped
            torch.cuda.empty_cache()
    if sorted(study_ids) != sorted(studies[STUDY].to_list()):
        raise ValueError(f"{run} fold {fold} did not predict every validation study")
    probabilities = torch.cat(values, dim=1).clamp(
        PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR
    )
    logits = torch.logit(probabilities).numpy()
    present = torch.cat(available, dim=1).numpy()
    frames = []
    for index, reading in enumerate(("joint", *names)):
        frame = pl.DataFrame(
            {
                STUDY: study_ids,
                "fold": fold,
                "reading": reading,
                **{target: logits[index, :, i] for i, target in enumerate(targets)},
            }
        )
        frames.append(frame if index == 0 else frame.filter(present[index - 1]))
    return pl.concat(frames)


def main() -> None:
    settings = TeacherTableCLI()
    if settings.output.exists():
        raise FileExistsError(f"Refusing to replace {settings.output}")
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(0.85)
    frames = []
    teachers = []
    for run in settings.run:
        validation = json.loads((run / "effective_config.json").read_text())[
            "validation"
        ]
        for fold in validation["folds"]:
            if not (run / f"fold_{fold}" / "selected_checkpoint.json").exists():
                continue
            frames.append(predict_fold(settings, run, fold))
            teachers.append({"run": str(run), "fold": fold})
    table = pl.concat(frames).sort("reading", STUDY)
    joint = table.filter(pl.col("reading") == "joint")
    if joint[STUDY].n_unique() != joint.height:
        raise ValueError("Two teachers predicted the same study")
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(settings.output)
    record = {
        "table": str(settings.output),
        "sha256": sha256_file(settings.output),
        "image_multiplier": IMAGE_MULTIPLIER,
        "attention_temperature": ATTENTION_TEMPERATURE,
        "reflection_tta": True,
        "teachers": teachers,
        "studies": joint.height,
        "rows_per_reading": dict(
            table.group_by("reading").len().sort("reading").iter_rows()
        ),
    }
    settings.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
