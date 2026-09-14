import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import polars as pl
import pytorch_lightning as L
import torch
import yaml
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsError
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from knee_abnormality.checkpoints import (
    CheckpointRelocation,
    training_input_identity,
    validate_resume_checkpoint,
)
from knee_abnormality.cli import NonExitingCLISettings
from knee_abnormality.config import ExperimentConfig, Fold, load_config
from knee_abnormality.inference import (
    load_checkpoint_model,
    make_prediction_frame,
    predict_loader,
)
from knee_abnormality.preprocessing_identity import preprocessing_identity
from knee_abnormality.provenance import sha256_file
from knee_abnormality.runtime import configure_cuda_allocator
from knee_abnormality.training import BaselineModule, KneeDataModule
from scripts.tools.validation import score


class TrainCLI(NonExitingCLISettings):
    config: CliPositionalArg[Path] = Field(description="Experiment YAML file")
    data_root: Path | None = None
    output_root: Path | None = None
    folds: list[Fold] | None = None
    run_id: str | None = None
    resume: Path | None = None
    limit_train: int | None = Field(default=None, gt=0)


def _command(args: list[str]) -> str | None:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _snapshot(run_dir: Path, config_path: Path, config: ExperimentConfig) -> None:
    shutil.copy2(config_path, run_dir / "input_config.yaml")
    (run_dir / "effective_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n"
    )
    provenance = {
        "preprocessing_identity": preprocessing_identity(config.model.backbone),
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
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
        Path("plan.md"),
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
            if path.exists():
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
    path: Path,
    config: ExperimentConfig,
    fold: int,
    targets: tuple[str, ...],
    training_studies: int,
    input_identity: dict,
) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(
        checkpoint, config, targets, fold, training_studies, input_identity
    )


def run_fold(
    config: ExperimentConfig,
    fold: int,
    fold_dir: Path,
    resume: Path | None,
    limit_train: int | None,
) -> None:
    L.seed_everything(config.seed, workers=True)
    setup_started = time.perf_counter()
    data = KneeDataModule(config, fold, limit_train=limit_train)
    data.setup("fit")
    labels = data.labels
    training_studies = len(data.train_dataset)
    input_identity = training_input_identity(
        config, labels, data.train_dataset.studies, data.validation_dataset.studies
    )
    setup_seconds = time.perf_counter() - setup_started
    (fold_dir / "label_provenance.json").write_text(
        json.dumps(labels.provenance, indent=2) + "\n"
    )
    if resume is not None:
        _check_resume(
            resume, config, fold, labels.targets, training_studies, input_identity
        )
    (fold_dir / "input_identity.json").write_text(
        json.dumps(input_identity, indent=2) + "\n"
    )
    module = BaselineModule(
        config,
        labels.targets,
        fold,
        training_studies,
        labels.validation.height,
        fold_dir / "epochs",
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
        precision=config.trainer.precision,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.trainer.gradient_clip_val,
        deterministic=config.trainer.deterministic,
        callbacks=[
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
    (fold_dir / "selected_checkpoint.json").write_text(
        json.dumps(
            {
                "path": str(best_path),
                "score": float(checkpoint.best_model_score),
            },
            indent=2,
        )
        + "\n"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(
        config, labels.targets, best_path, device, expected_fold=fold
    )
    prediction_started = time.perf_counter()
    study_ids, predictions = predict_loader(model, data.val_dataloader(), device)
    prediction_seconds = time.perf_counter() - prediction_started
    make_prediction_frame(study_ids, predictions, labels.targets, fold).write_csv(
        fold_dir / "validation_predictions.csv"
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
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else None,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    try:
        args = TrainCLI()
        configure_cuda_allocator()
        config_path = args.config.expanduser().resolve()
        config = load_config(
            config_path,
            data_root=args.data_root,
            output_root=args.output_root,
            folds=args.folds,
        )
        torch.set_float32_matmul_precision("high")
        if config.model.checkpoint_backbone and not config.model.backbone_eval:
            if args.resume is None:
                raise ValueError(
                    "Training with activation checkpointing requires backbone_eval=true; "
                    "disable checkpoint_backbone to preserve training-mode BatchNorm"
                )
            config = config.model_copy(
                update={
                    "model": config.model.model_copy(
                        update={"checkpoint_backbone": False}
                    )
                }
            )
            print(
                "Resuming with activation checkpointing disabled to preserve "
                "training-mode BatchNorm; GPU memory use may increase.",
                file=sys.stderr,
            )
        if args.resume is not None and len(config.validation.folds) != 1:
            raise ValueError("--resume requires exactly one selected fold")
        run_id = (
            args.run_id
            or f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{config.name}"
        )
        run_dir = config.paths.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        _snapshot(run_dir, config_path, config)
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
            run_fold(config, fold, fold_dir, args.resume, args.limit_train)
        predictions = pl.concat(
            [
                pl.read_csv(run_dir / f"fold_{fold}" / "validation_predictions.csv")
                for fold in config.validation.folds
            ]
        ).sort("StudyInstanceUID")
        oof_path = run_dir / "oof_predictions.csv"
        predictions.write_csv(oof_path)
        protocol = json.loads((config.split_dir / "protocol.json").read_text())
        if set(config.validation.folds) == set(range(protocol["n_splits"])):
            metrics = score(config.raw_dir, config.split_dir, oof_path)
            (run_dir / "oof_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
        SettingsError,
    ) as error:
        print(f"Training error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
