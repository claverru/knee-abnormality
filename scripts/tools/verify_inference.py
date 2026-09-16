import gc
import json
import math
import sys
import time
from pathlib import Path

import polars as pl
import torch
from pydantic_settings import SettingsError

import knee_abnormality
from knee_abnormality.cli import CLISettings
from knee_abnormality.config import SLICES_PER_INPUT, ExperimentConfig
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.data.labels import STUDY
from knee_abnormality.evaluation.predictions import make_prediction_frame
from knee_abnormality.inference.checkpoint import load_checkpoint_model
from knee_abnormality.inference.engine import (
    predict_loader,
    prediction_data,
    resolve_device,
    resolve_precision,
)
from knee_abnormality.provenance import sha256_file
from knee_abnormality.training.cuda import configure_cuda_allocator


class VerifyInferenceCLI(CLISettings):
    bundle: Path
    request_file: Path


def _sample_batches(dataset, count: int, batch_size: int) -> list[int]:
    groups = [
        list(range(start, min(start + batch_size, len(dataset))))
        for start in range(0, len(dataset), batch_size)
    ]

    def complexity(indices: list[int]) -> tuple[int, int]:
        inputs = acquisitions = 0
        for index in indices:
            uid = dataset.studies[STUDY][index]
            for slot, kind in zip(
                dataset.series_index[uid].slots, dataset.data.series_types, strict=True
            ):
                inputs += min(
                    sum(max(candidate.image_count - 2, 0) for candidate in slot),
                    kind.max_images
                    * dataset.image_limit_multiplier
                    // SLICES_PER_INPUT,
                )
                acquisitions += len(slot)
        return inputs, acquisitions

    ordered = sorted(range(len(groups)), key=lambda i: complexity(groups[i]))
    number = min(len(groups), max(1, math.ceil(count / batch_size)))
    selected = (
        [ordered[-1]]
        if number == 1
        else [
            ordered[round(i * (len(ordered) - 1) / (number - 1))] for i in range(number)
        ]
    )
    return [index for group in sorted(selected) for index in groups[group]]


def verify_inference(bundle: Path, request: dict) -> None:
    started = time.perf_counter()
    if Path(knee_abnormality.__file__).resolve().parent != bundle / "code" / (
        "knee_abnormality"
    ):
        raise RuntimeError(
            "Verification must import the exported source; set PYTHONPATH to the bundle's code directory"
        )
    configure_cuda_allocator()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    config = ExperimentConfig.model_validate(request["config"])
    fold = config.validation.folds[0]
    device = resolve_device("auto")
    precision, _ = resolve_precision("auto", device)
    if precision != "bf16":
        raise RuntimeError("Saved BF16 parity requires native BF16 CUDA support")
    dataset, targets, _ = prediction_data(config, "validation", fold)
    indices = _sample_batches(dataset, request["samples"], config.trainer.batch_size)
    dataset.studies = dataset.studies[indices]
    model = load_checkpoint_model(
        config, targets, bundle / "model.ckpt", device, expected_fold=fold
    )
    ids, values, _ = predict_loader(
        model,
        build_data_loader(
            dataset, config.trainer, config.trainer.batch_size, num_workers=4
        ),
        device,
        precision,
    )
    frame = make_prediction_frame(ids, values, targets).sort(STUDY)
    reference_path = Path(request["reference_csv"])
    reference = pl.read_csv(reference_path).filter(pl.col(STUDY).is_in(ids)).sort(STUDY)
    if reference[STUDY].to_list() != frame[STUDY].to_list():
        raise ValueError(
            "Saved predictions do not cover the parity sample exactly once"
        )
    difference = max(
        float((frame[target] - reference[target]).abs().max()) for target in targets
    )
    tolerance = 1e-5
    if difference > tolerance:
        raise ValueError(f"Sample parity difference {difference} exceeds {tolerance}")
    frame.write_csv(bundle / "verification_sample.csv")
    parity_seconds = time.perf_counter() - started
    print(
        json.dumps({"parity_studies": len(ids), "max_abs_difference": difference}),
        flush=True,
    )

    dataset.image_limit_multiplier = request["image_multiplier"]
    capacities = {
        uid: sum(
            min(
                sum(max(candidate.image_count - 2, 0) for candidate in slot),
                kind.max_images * dataset.image_limit_multiplier // SLICES_PER_INPUT,
            )
            for slot, kind in zip(
                dataset.series_index[uid].slots, dataset.data.series_types, strict=True
            )
        )
        for uid in ids
    }
    heaviest = sorted(ids, key=capacities.get, reverse=True)[: request["batch_size"]]
    dataset.studies = dataset.studies.filter(pl.col(STUDY).is_in(heaviest))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    predict_loader(
        model,
        build_data_loader(
            dataset, config.trainer, request["batch_size"], num_workers=2
        ),
        device,
        "fp32",
    )
    memory = {
        "precision": "fp32",
        "gpu": torch.cuda.get_device_name(),
        "studies": len(dataset),
        "batch_size": request["batch_size"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    print(
        json.dumps(
            {
                "inference_image_multiplier": request["image_multiplier"],
                "memory": memory,
            }
        ),
        flush=True,
    )
    dataset, example_targets, template_studies = prediction_data(
        config, "test", fold, image_multiplier=request["image_multiplier"]
    )
    if example_targets != targets:
        raise ValueError("Example target order differs from the checkpoint")
    example_ids, example_values, _ = predict_loader(
        model,
        build_data_loader(
            dataset, config.trainer, request["batch_size"], num_workers=2
        ),
        device,
        "fp32",
    )
    example = make_prediction_frame(example_ids, example_values, targets)
    if set(example_ids) != set(template_studies[STUDY]):
        raise ValueError("Example predictions differ from the mounted template studies")
    example = template_studies.select(STUDY).join(
        example,
        on=STUDY,
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    example.write_csv(bundle / "verification_example.csv")
    record = {
        "checkpoint_sha256": request["checkpoint_sha256"],
        "bundled_source_sha256": request["bundled_source_sha256"],
        "validation_scope": "sample",
        "validation_studies": len(ids),
        "validation_max_abs_difference": difference,
        "validation_macro_auc": None,
        "inference_image_multiplier": request["image_multiplier"],
        "example_submission_rows": example.height,
        "example_submission_valid": True,
        "jpeg_lossless_decoded": True,
        "jpeg_2000_decoded": True,
        "details": {
            "sample_method": "Whole original validation batches spanning low/high input counts",
            "study_ids": ids,
            "reference_predictions_sha256": sha256_file(reference_path),
            "parity_image_multiplier": config.data.validation_image_multiplier,
            "parity_precision": precision,
            "parity_tolerance": tolerance,
            "parity_seconds": parity_seconds,
            "inference_memory": memory,
            "codec_evidence": request["codec_evidence"],
            "total_seconds": time.perf_counter() - started,
        },
    }
    (bundle / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verified_studies": len(ids),
                "example_rows": example.height,
                "total_seconds": record["details"]["total_seconds"],
            }
        ),
        flush=True,
    )


def main() -> None:
    try:
        args = VerifyInferenceCLI()
        verify_inference(
            args.bundle.expanduser().resolve(),
            json.loads(args.request_file.expanduser().read_text()),
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        SettingsError,
    ) as error:
        print(f"Inference verification error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
