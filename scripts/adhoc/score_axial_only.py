import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import (
    INFERENCE_BATCH_SIZE,
    INFERENCE_IO_THREADS,
    load_config,
)
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader
from knee_abnormality.data.sources import (
    StudySeries,
    build_series_index,
    prediction_studies,
)
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD
from knee_abnormality.evaluation.metrics import bootstrap_auc, validation_metrics
from knee_abnormality.evaluation.predictions import make_prediction_frame
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import predict_loader, resolve_device
from knee_abnormality.training.cuda import configure_cuda_allocator


class AxialOnlyCLI(CLISettings):
    model_config = describe(
        "Predict a run's validation fold from axial series only with its selected "
        "checkpoint and compare with its saved all-series predictions on the same studies"
    )

    run: Path = Field(
        default=Path("outputs/training/scanner-confusion-control"),
        description="Completed training run directory",
    )
    output_dir: Path = Field(description="Fresh output directory")
    plane: str = Field(default="Axial", description="Plane whose series are kept")
    image_multiplier: int = Field(default=4, gt=0, description="Prediction multiplier")
    workers: int = Field(default=8, gt=0, description="Loader workers")
    draws: int = Field(default=2000, gt=0, description="Paired bootstrap resamples")
    seed: int = Field(default=20260914, description="Bootstrap seed")


def keep_plane(study: StudySeries, keep: tuple[bool, ...]) -> StudySeries:
    return StudySeries(
        study.uid,
        tuple(values if kept else () for values, kept in zip(study.slots, keep)),
    )


def run(settings: AxialOnlyCLI) -> dict:
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(settings.run / "effective_config.json")
    (fold,) = config.validation.folds
    fold_dir = settings.run / f"fold_{fold}"
    selected = json.loads((fold_dir / "selected_checkpoint.json").read_text())
    checkpoint = fold_dir / Path(selected["path"]).name
    targets, studies, metadata_split = prediction_studies(config, "validation", fold)
    index = build_series_index(
        config.paths.data_root, config.data, metadata_split, studies[STUDY].to_list()
    )
    keep = tuple(kind.plane == settings.plane for kind in config.data.series_types)
    filtered = {uid: keep_plane(study, keep) for uid, study in index.items()}
    kept = [uid for uid, study in filtered.items() if any(study.slots)]
    subset = studies.filter(pl.col(STUDY).is_in(kept))
    dataset = KneeStudyDataset(
        subset,
        filtered,
        config.data,
        targets,
        None,
        settings.image_multiplier,
        INFERENCE_IO_THREADS,
    )
    loader = build_data_loader(
        dataset, config.trainer, INFERENCE_BATCH_SIZE, num_workers=settings.workers
    )
    device = resolve_device("auto")
    model = load_checkpoint_model(
        config, targets, checkpoint, device, expected_fold=fold
    )
    started = time.perf_counter()
    study_ids, predictions, _ = predict_loader([model], loader, device)
    seconds = time.perf_counter() - started
    frame = make_prediction_frame(study_ids, predictions, targets, fold)
    frame.write_csv(settings.output_dir / "axial_only_predictions.csv")

    truth = (
        pl.read_csv(config.split_dir / "studies.csv")
        .join(pl.read_csv(config.split_dir / "targets.csv"), on=STUDY, validate="1:1")
        .filter(pl.col(STUDY).is_in(kept))
        .sort(STUDY)
    )
    reference = pl.read_csv(fold_dir / "validation_predictions.csv")
    full_reference = (
        pl.read_csv(config.split_dir / "studies.csv")
        .filter(pl.col("fold") == fold)
        .join(pl.read_csv(config.split_dir / "targets.csv"), on=STUDY, validate="1:1")
        .sort(STUDY)
    )
    full_prediction = reference.sort(STUDY)
    reference = reference.filter(pl.col(STUDY).is_in(kept)).sort(STUDY)
    candidate = frame.sort(STUDY)
    if not (
        truth[STUDY].to_list()
        == reference[STUDY].to_list()
        == candidate[STUDY].to_list()
    ):
        raise ValueError("Study membership differs between truth and predictions")
    y = truth.select(targets).to_numpy().astype(float)
    gold = truth["is_gold"].to_numpy()
    all_series = reference.select(targets).to_numpy().astype(float)
    axial = candidate.select(targets).to_numpy().astype(float)
    all_metrics = validation_metrics(y, all_series, targets, gold)
    axial_metrics = validation_metrics(y, axial, targets, gold)
    sanity = validation_metrics(
        full_reference.select(targets).to_numpy().astype(float),
        full_prediction.select(targets).to_numpy().astype(float),
        targets,
        full_reference["is_gold"].to_numpy(),
    )["macro_auc"]

    hard = (y >= TARGET_THRESHOLD).astype(np.int8)
    rng = np.random.default_rng(settings.seed)
    counts = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=settings.draws)
    difference = bootstrap_auc(hard, axial, counts) - bootstrap_auc(
        hard, all_series, counts
    )
    record = {
        "run": str(settings.run.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "fold": fold,
        "plane": settings.plane,
        "image_multiplier": settings.image_multiplier,
        "attention_temperature": 1.0,
        "fold_studies": studies.height,
        "studies_with_plane": len(kept),
        "saved_full_fold_macro_auc": sanity,
        "all_series_macro_auc": all_metrics["macro_auc"],
        "plane_only_macro_auc": axial_metrics["macro_auc"],
        "delta_macro_auc": axial_metrics["macro_auc"] - all_metrics["macro_auc"],
        "delta_95_interval_study_bootstrap": [
            float(np.percentile(difference, 2.5)),
            float(np.percentile(difference, 97.5)),
        ],
        "all_series_soft_macro_auc": all_metrics["soft_macro_auc"],
        "plane_only_soft_macro_auc": axial_metrics["soft_macro_auc"],
        "per_target": {
            target: {
                "all_series": all_metrics["per_target_auc"][target],
                "plane_only": axial_metrics["per_target_auc"][target],
                "delta": axial_metrics["per_target_auc"][target]
                - all_metrics["per_target_auc"][target],
                "positives": int(hard[:, index].sum()),
            }
            for index, target in enumerate(targets)
        },
        "prediction_seconds": seconds,
    }
    (settings.output_dir / "results.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


if __name__ == "__main__":
    configure_cuda_allocator()
    print(json.dumps(run(AxialOnlyCLI()), indent=2))
