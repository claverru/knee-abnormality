import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from knee_abnormality.cli import CLISettings, RunFold, describe
from knee_abnormality.config import INFERENCE_BATCH_SIZE, load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.evaluation.predictions import (
    fold_prediction_macro_auc,
    make_prediction_frame,
)
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import (
    predict_loader,
    prediction_data,
    resolve_device,
)
from knee_abnormality.training.cuda import configure_cuda_allocator

REPLACED = (
    "*.ckpt",
    "validation_predictions.csv",
    "oof_predictions.csv",
    "selected_checkpoint.json",
    "effective_config.json",
    "timing.json",
)


class RepredictCLI(CLISettings, RunFold):
    model_config = describe(
        "Derive a new run directory whose selected checkpoint's validation predictions "
        "are remade at another image multiplier, at attention temperature 1"
    )

    run: Path = Field(description="Completed training run directory")
    output: Path = Field(description="New run directory to create")
    image_multiplier: int = Field(gt=0, description="Validation prediction multiplier")


def repredict(settings: RepredictCLI) -> dict:
    source, output = settings.run, settings.output
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    effective = json.loads((source / "effective_config.json").read_text())
    fold = settings.run_fold(effective["validation"]["folds"])
    fold_name = f"fold_{fold}"
    selected = json.loads((source / fold_name / "selected_checkpoint.json").read_text())
    checkpoint = source / fold_name / Path(selected["path"]).name
    effective["validation"]["prediction_image_multiplier"] = settings.image_multiplier
    shutil.copytree(source, output, ignore=shutil.ignore_patterns(*REPLACED))
    linked = output / fold_name / checkpoint.name
    os.link(checkpoint, linked)
    (output / "effective_config.json").write_text(
        json.dumps(effective, indent=2) + "\n"
    )
    config = load_config(output / "effective_config.json")
    dataset, targets, _ = prediction_data(
        config, "validation", fold, image_multiplier=settings.image_multiplier
    )
    loader = build_data_loader(dataset, config.trainer, INFERENCE_BATCH_SIZE)
    device = resolve_device("auto")
    model = load_checkpoint_model(config, targets, linked, device, expected_fold=fold)
    started = time.perf_counter()
    study_ids, predictions, _ = predict_loader([model], loader, device)
    seconds = time.perf_counter() - started
    path = output / fold_name / "validation_predictions.csv"
    make_prediction_frame(study_ids, predictions, targets, fold).write_csv(path)
    selected.update(
        path=str(linked.resolve()),
        prediction_image_multiplier=settings.image_multiplier,
        prediction_macro_auc=fold_prediction_macro_auc(
            config.split_dir, fold, targets, path
        ),
    )
    (output / fold_name / "selected_checkpoint.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    timing = json.loads((source / fold_name / "timing.json").read_text())
    timing["validation_prediction_seconds"] = seconds
    (output / fold_name / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    record = {
        "source_run": str(source.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "fold": fold,
        "image_multiplier": settings.image_multiplier,
        "attention_temperature": 1.0,
        "prediction_macro_auc": selected["prediction_macro_auc"],
        "epoch_macro_auc": selected["score"],
        "prediction_seconds": seconds,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (output / "repredicted.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    settings = RepredictCLI()
    configure_cuda_allocator()
    print(json.dumps(repredict(settings), indent=2))


if __name__ == "__main__":
    main()
