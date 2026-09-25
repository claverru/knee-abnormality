import hashlib
import json
import logging
import os
import shutil
import time
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import polars as pl
import pytorch_lightning as L
import torch
from pydantic import ConfigDict, Field
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from knee_abnormality.autoresearch.scoring.accounting import TripletAccounting
from knee_abnormality.config import (
    INFERENCE_BATCH_SIZE,
    ConfigOptions,
    ExperimentConfig,
    load_config,
    parse_experiment_arguments,
)
from knee_abnormality.data.dataset import KneeStudyDataset, build_data_loader
from knee_abnormality.data.labels import STUDY
from knee_abnormality.evaluation.predictions import (
    fold_prediction_macro_auc,
    make_prediction_frame,
)
from knee_abnormality.evaluation.scoring import score
from knee_abnormality.identity import preprocessing_identity
from knee_abnormality.inference.checkpoint import (
    load_checkpoint_model,
    read_checkpoint,
)
from knee_abnormality.inference.engine import predict_loader, resolve_device
from knee_abnormality.provenance import git_revision, sha256_file
from knee_abnormality.training.checkpoints import (
    CheckpointRelocation,
    training_input_identity,
    validate_resume_checkpoint,
)
from knee_abnormality.training.cuda import configure_cuda_allocator
from knee_abnormality.training.lightning import (
    PRECISION,
    KneeDataModule,
    KneeModule,
    fixed_class_balance,
)


class TrainOptions(ConfigOptions):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Train the experiment on its selected folds and save validation "
            "predictions. A dotted flag overrides that field of the YAML; the defaults "
            "and required marks shown are the schema's, not the YAML's"
        }
    )

    run_id: str | None = Field(
        default=None,
        description="Run directory name under paths.output_root; defaults to a UTC timestamp and the experiment name",
    )
    resume: Path | None = Field(
        default=None,
        description="Checkpoint to resume from; requires exactly one selected fold and reuses its seed",
    )
    limit_train: int | None = Field(
        default=None,
        gt=0,
        description="Train on only the first N training studies of each fold; validation stays complete",
    )


def _snapshot(
    run_dir: Path, config_path: Path, config: ExperimentConfig, overrides: dict
) -> None:
    shutil.copy2(config_path, run_dir / "input_config.yaml")
    (run_dir / "effective_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n"
    )
    commit, status = git_revision(Path.cwd())
    provenance = {
        "overrides": overrides,
        "preprocessing_identity": preprocessing_identity(config.model.backbone),
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "git_status": status,
        "cuda_allocator_config": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF")
        ),
        "versions": {
            name: version(name)
            for name in (
                "pytorch-lightning",
                "numpy",
                "polars",
                "pydicom",
                "timm",
                "torch",
            )
        },
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    roots = [Path("src"), Path("scripts"), Path("configs"), Path("docs")]
    files = [
        Path("AGENTS.md"),
        Path("README.md"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    with zipfile.ZipFile(
        run_dir / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for root in roots:
            for path in root.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path)
        for path in files:
            archive.write(path)


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _check_resume(
    checkpoint: dict,
    config: ExperimentConfig,
    fold: int,
    targets: tuple[str, ...],
    training_studies: int,
    input_identity: dict,
    class_balance: dict,
) -> None:
    validate_resume_checkpoint(
        checkpoint,
        config,
        targets,
        fold,
        training_studies,
        input_identity,
        class_balance,
    )


def run_fold(
    config: ExperimentConfig,
    fold: int,
    fold_dir: Path,
    resume: Path | None,
    resume_checkpoint: dict | None,
    limit_train: int | None,
) -> None:
    L.seed_everything(config.seed, workers=True)
    setup_started = time.perf_counter()
    data = KneeDataModule(config, fold, limit_train=limit_train)
    data.setup("fit")
    labels = data.labels
    training_studies = len(data.train_dataset)
    class_balance = fixed_class_balance(
        data.train_dataset.studies,
        labels.targets,
        config.trainer.class_balance_exponent,
    )
    input_identity = training_input_identity(
        config, labels, data.train_dataset.studies, data.validation_dataset.studies
    )
    setup_seconds = time.perf_counter() - setup_started
    (fold_dir / "label_provenance.json").write_text(
        json.dumps(labels.provenance, indent=2) + "\n"
    )
    if resume_checkpoint is not None:
        _check_resume(
            resume_checkpoint,
            config,
            fold,
            labels.targets,
            training_studies,
            input_identity,
            class_balance,
        )
    (fold_dir / "class_balance.json").write_text(
        json.dumps(class_balance, indent=2) + "\n"
    )
    (fold_dir / "input_identity.json").write_text(
        json.dumps(input_identity, indent=2) + "\n"
    )
    module = KneeModule(
        config,
        labels.targets,
        fold,
        training_studies,
        labels.validation.height,
        fold_dir / "epochs",
        class_balance,
        load_pretrained=False if resume is not None else None,
        input_identity=input_identity,
    )
    initialized_identity = (
        {
            "pretrained_config": module.model.backbone.pretrained_cfg,
            "initialized_state_sha256": _state_sha256(
                module.model.backbone.state_dict()
            ),
        }
        if resume is None
        else {
            "resume_checkpoint": str(resume.expanduser().resolve()),
            "resume_checkpoint_sha256": sha256_file(resume),
        }
    )
    (fold_dir / "backbone_identity.json").write_text(
        json.dumps(
            {
                "name": config.model.backbone,
                "pretrained": config.model.pretrained,
                "feature_mapping": module.model.feature_mapping_identity,
                **initialized_identity,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    checkpoint = ModelCheckpoint(
        dirpath=fold_dir,
        filename="best-{epoch:02d}-{val_macro_auc:.5f}",
        monitor="val_macro_auc",
        mode="max",
        save_last=False,
        save_top_k=1,
    )
    latest_checkpoint = ModelCheckpoint(
        dirpath=fold_dir,
        filename="last",
        monitor=None,
        save_top_k=1,
        save_last=False,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    early_stopping = EarlyStopping(
        monitor="val_macro_auc",
        mode="max",
        patience=config.trainer.patience,
        min_delta=config.trainer.min_delta,
        check_finite=True,
    )
    trainer = L.Trainer(
        default_root_dir=fold_dir,
        max_epochs=config.trainer.max_epochs,
        precision=PRECISION,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.trainer.gradient_clip_val,
        deterministic=config.trainer.deterministic,
        callbacks=[
            *(
                [TripletAccounting(fold_dir / "triplet_accounting.json")]
                if os.environ.get("KNEE_AUTORESEARCH_LIMITS")
                else []
            ),
            CheckpointRelocation(fold_dir),
            early_stopping,
            checkpoint,
            latest_checkpoint,
        ],
        logger=CSVLogger(fold_dir, name="logs"),
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    trainer.fit(
        module, datamodule=data, ckpt_path=str(resume) if resume is not None else None
    )
    training_seconds = time.perf_counter() - training_started
    if not checkpoint.best_model_path or checkpoint.best_model_score is None:
        raise RuntimeError("Training did not produce or restore a scored checkpoint")
    best_path = Path(checkpoint.best_model_path)
    device = resolve_device("auto")
    model = load_checkpoint_model(
        config, labels.targets, best_path, device, expected_fold=fold
    )
    prediction_loader = build_data_loader(
        KneeStudyDataset(
            data.validation_dataset.studies,
            data.validation_dataset.series_index,
            config.data,
            labels.targets,
            None,
            config.validation.prediction_image_multiplier,
        ),
        config.trainer,
        INFERENCE_BATCH_SIZE,
    )
    prediction_started = time.perf_counter()
    study_ids, predictions, _ = predict_loader([model], prediction_loader, device)
    prediction_seconds = time.perf_counter() - prediction_started
    prediction_path = fold_dir / "validation_predictions.csv"
    make_prediction_frame(study_ids, predictions, labels.targets, fold).write_csv(
        prediction_path
    )
    (fold_dir / "selected_checkpoint.json").write_text(
        json.dumps(
            {
                "path": str(best_path),
                "score": float(checkpoint.best_model_score),
                "prediction_image_multiplier": config.validation.prediction_image_multiplier,
                "prediction_macro_auc": fold_prediction_macro_auc(
                    config.split_dir, fold, labels.targets, prediction_path
                ),
            },
            indent=2,
        )
        + "\n"
    )
    (fold_dir / "timing.json").write_text(
        json.dumps(
            {
                "setup_seconds": setup_seconds,
                "training_seconds": training_seconds,
                "validation_prediction_seconds": prediction_seconds,
                "training_studies": training_studies,
                "validation_studies": labels.validation.height,
                "optimizer_steps": trainer.global_step,
                "batch_norm_refresh_seconds": module.batch_norm_refresh_seconds,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else None,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    args, overrides = parse_experiment_arguments(TrainOptions)
    for module in (KneeStudyDataset.__module__, predict_loader.__module__):
        logging.getLogger(module).setLevel(logging.ERROR)
    configure_cuda_allocator()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path, overrides=overrides)
    torch.set_float32_matmul_precision("high")
    if args.resume is not None and len(config.validation.folds) != 1:
        raise ValueError("--resume requires exactly one selected fold")
    resume_checkpoint = (
        read_checkpoint(args.resume) if args.resume is not None else None
    )
    if resume_checkpoint is not None:
        config = config.model_copy(
            update={"seed": resume_checkpoint["hyper_parameters"]["config"]["seed"]}
        )
    run_id = (
        args.run_id or f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{config.name}"
    )
    run_dir = config.paths.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _snapshot(run_dir, config_path, config, overrides)
    (run_dir / "execution.json").write_text(
        json.dumps(
            {
                "folds": list(config.validation.folds),
                "limit_train": args.limit_train,
                "resume": str(args.resume.expanduser().resolve())
                if args.resume is not None
                else None,
            },
            indent=2,
        )
        + "\n"
    )
    for fold in config.validation.folds:
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir()
        run_fold(
            config,
            fold,
            fold_dir,
            args.resume,
            resume_checkpoint,
            args.limit_train,
        )
    predictions = pl.concat(
        [
            pl.read_csv(run_dir / f"fold_{fold}" / "validation_predictions.csv")
            for fold in config.validation.folds
        ]
    ).sort(STUDY)
    oof_path = run_dir / "oof_predictions.csv"
    predictions.write_csv(oof_path)
    protocol = json.loads((config.split_dir / "protocol.json").read_text())
    if set(config.validation.folds) == set(range(protocol["n_splits"])):
        metrics = score(config.raw_dir, config.split_dir, oof_path)
        (run_dir / "oof_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
