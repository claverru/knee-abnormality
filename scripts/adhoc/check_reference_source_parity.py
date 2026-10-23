import json
from pathlib import Path

import numpy as np
import torch
from pydantic import Field

import knee_abnormality.inference.checkpoint as checkpoint_module
from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import (
    INFERENCE_BATCH_SIZE,
    INFERENCE_IO_THREADS,
    load_config,
)
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader
from knee_abnormality.data.sources import build_series_index, prediction_studies
from knee_abnormality.identity import preprocessing_identity
from knee_abnormality.inference.engine import predict_loader

try:
    from knee_abnormality.data.splits import STUDY
except ImportError:
    from knee_abnormality.data.labels import STUDY


class SourceParityCLI(CLISettings):
    model_config = describe(
        "Predict the first validation studies of a run's selected checkpoint on CPU with whatever "
        "knee_abnormality source is on PYTHONPATH, skipping the preprocessing-identity check, so "
        "two source trees can be compared bitwise"
    )

    run: Path = Field(description="Completed training run directory")
    studies: int = Field(
        default=16, gt=0, description="Validation studies, in study-ID order"
    )
    output: Path = Field(description="Fresh .npy output path")


def run(settings: SourceParityCLI) -> None:
    if settings.output.exists():
        raise ValueError(f"Output already exists: {settings.output}")
    checkpoint_module.validate_preprocessing_identity = lambda *args, **kwargs: None
    config = load_config(settings.run / "effective_config.json")
    (fold,) = config.validation.folds
    selected = json.loads(
        (settings.run / f"fold_{fold}/selected_checkpoint.json").read_text()
    )
    checkpoint = settings.run / f"fold_{fold}" / Path(selected["path"]).name
    targets, studies, split = prediction_studies(config, "validation", fold)
    studies = studies.sort(STUDY).head(settings.studies)
    index = build_series_index(
        config.paths.data_root, config.data, split, studies[STUDY].to_list()
    )
    dataset = KneeStudyDataset(
        studies, index, config.data, targets, None, 1, INFERENCE_IO_THREADS
    )
    loader = build_data_loader(
        dataset, config.trainer, INFERENCE_BATCH_SIZE, num_workers=2
    )
    device = torch.device("cpu")
    model = checkpoint_module.load_checkpoint_model(
        config, targets, checkpoint, device, expected_fold=fold
    )
    _, predictions, _ = predict_loader([model], loader, device)
    np.save(settings.output, np.asarray(predictions, dtype=np.float64))
    print(
        json.dumps(
            {
                "source": checkpoint_module.__file__,
                "sources_sha256": preprocessing_identity(config.model.backbone)[
                    "sources_sha256"
                ],
                "shape": list(np.asarray(predictions).shape),
            }
        )
    )


if __name__ == "__main__":
    run(SourceParityCLI())
