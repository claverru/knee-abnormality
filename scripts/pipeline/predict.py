import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal

import polars as pl
import torch
import yaml
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsError

from knee_abnormality.cli import NonExitingCLISettings
from knee_abnormality.config import Fold, load_config
from knee_abnormality.dataset import resolve_raw_dir
from knee_abnormality.inference import (
    load_checkpoint_model,
    make_prediction_frame,
    predict_loader,
    prediction_data,
    prediction_loader,
    resolve_device,
    resolve_precision,
)


class PredictCLI(NonExitingCLISettings):
    config: CliPositionalArg[Path] = Field(
        description="Experiment YAML or saved effective_config.json file"
    )
    bundle: Path | None = None
    checkpoint: Path | None = None
    output: Path
    split: Literal["test", "validation"] = "test"
    fold: Fold | None = None
    data_root: Path | None = None
    label_manifest: Path | None = None
    batch_size: int | None = Field(default=None, gt=0)
    num_workers: int | None = Field(default=None, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    image_multiplier: int | None = Field(default=None, gt=0)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        args = PredictCLI()
        bundle = args.bundle.expanduser().resolve() if args.bundle is not None else None
        manifest = None
        if bundle is not None:
            from knee_abnormality.bundle import verify_bundle

            manifest = verify_bundle(bundle)
        if args.checkpoint is not None and bundle is not None:
            raise ValueError("A bundle supplies its own checkpoint")
        config_path = args.config.expanduser().resolve()
        if bundle is not None and config_path != bundle / "config.json":
            raise ValueError("Bundle inference must use the bundle's config.json")
        if bundle is None and args.checkpoint is None:
            raise ValueError("Repository inference requires --checkpoint")
        checkpoint_path = (
            args.checkpoint if args.checkpoint is not None else bundle / "model.ckpt"
        )
        data_root = args.data_root
        if data_root is None and bundle is not None and Path("/kaggle/input").is_dir():
            from knee_abnormality.bundle import discover_kaggle_data_root

            data_root = discover_kaggle_data_root()
        config = load_config(
            config_path,
            data_root=data_root,
            label_manifest=args.label_manifest,
        )
        torch.set_float32_matmul_precision("high")
        image_multiplier = args.image_multiplier
        if manifest is not None:
            recorded = manifest.get("runtime", {}).get("image_multiplier")
            if recorded is not None:
                if image_multiplier is not None and image_multiplier != recorded:
                    raise ValueError(
                        "Export a new bundle to change its image multiplier"
                    )
                image_multiplier = recorded
        fold = args.fold if args.fold is not None else config.validation.folds[0]
        dataset, targets, studies = prediction_data(
            config, args.split, fold, image_multiplier=image_multiplier
        )
        image_multiplier = dataset.image_limit_multiplier
        batch_size = args.batch_size
        if batch_size is None and manifest is not None:
            batch_size = manifest.get("runtime", {}).get("batch_size")
        loader = prediction_loader(
            config,
            dataset,
            batch_size=batch_size,
            num_workers=args.num_workers,
        )
        device = resolve_device(args.device)
        actual_precision, _ = resolve_precision(args.precision, device)
        started = time.perf_counter()
        model = load_checkpoint_model(
            config,
            targets,
            checkpoint_path,
            device,
            expected_fold=fold if args.split == "validation" else None,
        )
        loaded = time.perf_counter()
        study_ids, predictions = predict_loader(model, loader, device, args.precision)
        predicted = time.perf_counter()
        frame = make_prediction_frame(
            study_ids,
            predictions,
            targets,
            fold if args.split == "validation" else None,
        )
        if args.split == "test":
            template = pl.read_csv(
                resolve_raw_dir(config.paths.data_root, "test")
                / "sample_submission.csv"
            )
            if (
                frame.columns != template.columns
                or set(frame["StudyInstanceUID"]) != set(studies["StudyInstanceUID"])
                or set(template["StudyInstanceUID"]) != set(studies["StudyInstanceUID"])
            ):
                raise ValueError(
                    "Test predictions do not match the submission schema and studies"
                )
            frame = template.select("StudyInstanceUID").join(
                frame, on="StudyInstanceUID", how="left", validate="1:1"
            )
            if (
                frame.height != studies.height
                or frame.select(
                    pl.any_horizontal(
                        pl.col(targets).is_null()
                        | ~pl.col(targets).is_finite()
                        | ~pl.col(targets).is_between(0, 1)
                    ).any()
                ).item()
            ):
                raise ValueError("Submission contains missing or invalid predictions")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(args.output)
        finished = time.perf_counter()
        print(
            json.dumps(
                {
                    "output": str(args.output.expanduser().resolve()),
                    "studies": frame.height,
                    "device": str(device),
                    "precision": actual_precision,
                    "image_multiplier": image_multiplier,
                    "batch_size": loader.batch_size,
                    "model_load_seconds": loaded - started,
                    "prediction_seconds": predicted - loaded,
                    "write_seconds": finished - predicted,
                    "studies_per_second": frame.height / (predicted - loaded),
                }
            )
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
        SettingsError,
    ) as error:
        print(f"Prediction error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
