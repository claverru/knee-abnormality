import copy
import json
import os
import time
from pathlib import Path

import numpy as np
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
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD
from knee_abnormality.evaluation.metrics import aucs, bootstrap_auc
from knee_abnormality.inference.checkpoint import (
    apply_attention_temperature,
    load_checkpoint_model,
)
from knee_abnormality.inference.engine import move_batch, resolve_precision
from knee_abnormality.inference.reflection import reflect_batch, sagittal_types

VARIANTS = ("plain", "tta_unmirrored_detail", "tta_mirrored_detail")
GEOMETRY = ("position_2d", "position_1d", "medial_axis", "anterior_axis")
AUGMENTATIONS = (
    "contrast_mixing_augmentation",
    "intensity_augmentation",
    "affine_augmentation",
    "noise_augmentation",
    "smoothing_augmentation",
    "resolution_augmentation",
    "gin_augmentation",
)


class ScoreCLI(CLISettings):
    model_config = describe(
        "Score reflection TTA with the detail-to-wide map left as loaded and mirrored "
        "with the image, from one prepared batch per study, and check both against the "
        "model's own training reflection"
    )

    run: list[Path] = Field(min_length=1, description="Training run directories")
    data_root: Path = Field(description="Data root holding raw/ and folds/")
    label_manifest: Path = Field(description="Label manifest the runs trained on")
    split_dir: Path = Field(description="Split directory with studies and targets")
    multiplier: int = Field(default=6, gt=0, description="Image multiplier")
    tau: list[PositiveFiniteFloat] = Field(default=[0.8], min_length=1)
    output_dir: Path = Field(description="Fresh output directory")
    batch_size: int = Field(default=2, gt=0, description="Studies per batch")
    draws: int = Field(default=2000, ge=200, description="Bootstrap resamples")
    seed: int = Field(default=20261005, description="Bootstrap seed")


def lower_priority(worker_id: int) -> None:
    os.nice(INFERENCE_LOADER_NICENESS)


def pooled(model, batch, taus, dtype) -> list[torch.Tensor]:
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


def mirrored_view_to_wide(batch: dict, sagittal_lookup: torch.Tensor) -> torch.Tensor:
    reflected = batch["reflection_eligible"][batch["series_study_index"]]
    sagittal = sagittal_lookup.to(reflected.device)[batch["series_type_index"]]
    mirror = reflected & ~sagittal
    view_to_wide = batch["view_to_wide"].clone()
    reversal = torch.eye(3, device=view_to_wide.device, dtype=view_to_wide.dtype)
    reversal[0, 0] = -1
    reversal[0, 2] = batch["images"].shape[-1] - 1
    view_to_wide[mirror] = view_to_wide[mirror] @ reversal
    return view_to_wide


def training_reflection_parity(model, batch, mirrored, unmirrored) -> dict:
    forced = copy.deepcopy(model)
    settings = forced.augmentation_settings
    forced.augmentation_settings = settings.model_copy(
        update={
            "reflection_augmentation": settings.reflection_augmentation.model_copy(
                update={"probability": 1.0}
            ),
            **{
                name: getattr(settings, name).model_copy(update={"probability": 0.0})
                for name in AUGMENTATIONS
            },
        }
    )
    forced.feature_statistics_settings = forced.feature_statistics_settings.model_copy(
        update={"probability": 0.0}
    )
    forced.series_dropout = 0.0
    forced.train()
    reference = forced.encode_batch(batch)
    result = {}
    for name, candidate in (
        ("tta_mirrored_detail", mirrored),
        ("tta_unmirrored_detail", unmirrored),
    ):
        encoded = model.encode_batch(candidate)
        result[name] = {
            **{
                key: float((encoded[key].float() - reference[key].float()).abs().max())
                for key in GEOMETRY
            },
            "visual_valid_mismatches": int(
                (encoded["visual_valid"] != reference["visual_valid"]).sum()
            ),
        }
    return result


def score_fold(settings: ScoreCLI, run: Path, fold: int, check: bool):
    config = load_config(
        run / "effective_config.json",
        data_root=settings.data_root,
        label_manifest=settings.label_manifest,
    )
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
            settings.multiplier,
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
    study_ids = []
    parts = {variant: [[] for _ in settings.tau] for variant in VARIANTS}
    parity = None
    with torch.inference_mode():
        for batch in loader:
            study_ids.extend(batch["study_uid"])
            batch = with_images(move_batch(batch, device))
            unmirrored = reflect_batch(batch, sagittal, model.repeated_middle_slice)
            mirrored = {
                **unmirrored,
                "view_to_wide": mirrored_view_to_wide(batch, sagittal),
            }
            if check and parity is None and bool(batch["reflection_eligible"].all()):
                parity = training_reflection_parity(model, batch, mirrored, unmirrored)
            eligible = batch["reflection_eligible"][:, None]
            direct = pooled(model, batch, settings.tau, dtype)
            old = pooled(model, unmirrored, settings.tau, dtype)
            new = pooled(model, mirrored, settings.tau, dtype)
            for index in range(len(settings.tau)):
                a = direct[index]
                parts["plain"][index].append(a.cpu())
                parts["tta_unmirrored_detail"][index].append(
                    torch.where(eligible, (a + old[index]) / 2, a).cpu()
                )
                parts["tta_mirrored_detail"][index].append(
                    torch.where(eligible, (a + new[index]) / 2, a).cpu()
                )
            del batch, mirrored, unmirrored, direct, old, new
            torch.cuda.empty_cache()
    frames = []
    for variant in VARIANTS:
        for index, tau in enumerate(settings.tau):
            values = torch.cat(parts[variant][index]).numpy()
            frames.append(
                pl.DataFrame(
                    {
                        STUDY: study_ids,
                        "fold": fold,
                        "variant": variant,
                        "tau": tau,
                        **{t: values[:, i] for i, t in enumerate(targets)},
                    }
                )
            )
    return pl.concat(frames), parity, selected["path"]


def main() -> None:
    settings = ScoreCLI()
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(0.85)
    started = time.perf_counter()
    frames = []
    checkpoints = []
    parity = None
    for run in settings.run:
        folds = json.loads((run / "effective_config.json").read_text())["validation"][
            "folds"
        ]
        for fold in folds:
            frame, checked, checkpoint = score_fold(settings, run, fold, parity is None)
            parity = parity or checked
            frames.append(frame)
            checkpoints.append(checkpoint)
            print(f"fold {fold} done {time.perf_counter() - started:.0f}s", flush=True)
    predictions = pl.concat(frames)
    predictions.write_parquet(settings.output_dir / "predictions.parquet")
    labels = pl.read_csv(settings.split_dir / "targets.csv")
    targets = labels.columns[1:]
    covered = predictions.filter(
        (pl.col("variant") == "plain") & (pl.col("tau") == settings.tau[0])
    ).select(STUDY, "fold")
    truth_frame = covered.join(labels, on=STUDY, validate="1:1").sort(STUDY)
    hard = (truth_frame.select(targets).to_numpy() >= TARGET_THRESHOLD).astype(
        np.float64
    )
    folds = truth_frame["fold"].to_numpy()
    counts = np.random.default_rng(settings.seed).multinomial(
        len(hard), np.full(len(hard), 1 / len(hard)), size=settings.draws
    )
    results = []
    for tau in settings.tau:
        scores = {}
        for variant in VARIANTS:
            part = predictions.filter(
                (pl.col("variant") == variant) & (pl.col("tau") == tau)
            )
            scores[variant] = (
                truth_frame.select(STUDY)
                .join(part, on=STUDY, validate="1:1", maintain_order="left")
                .select(targets)
                .to_numpy()
                .astype(np.float64)
            )
        metrics = {
            variant: aucs(hard, values, targets) for variant, values in scores.items()
        }
        boot = {
            variant: bootstrap_auc(hard, values, counts)
            for variant, values in scores.items()
        }
        delta = boot["tta_mirrored_detail"] - boot["tta_unmirrored_detail"]
        results.append(
            {
                "tau": tau,
                "pooled_macro_auc": {
                    variant: metrics[variant]["macro_auc"] for variant in VARIANTS
                },
                "per_fold_macro_auc": {
                    variant: {
                        int(fold): aucs(
                            hard[folds == fold], scores[variant][folds == fold], targets
                        )["macro_auc"]
                        for fold in np.unique(folds)
                    }
                    for variant in VARIANTS
                },
                "per_label_auc": {
                    variant: metrics[variant]["per_target_auc"] for variant in VARIANTS
                },
                "mirrored_minus_unmirrored": {
                    "pooled": metrics["tta_mirrored_detail"]["macro_auc"]
                    - metrics["tta_unmirrored_detail"]["macro_auc"],
                    "bootstrap_interval_80": np.quantile(delta, [0.1, 0.9]).tolist(),
                    "bootstrap_interval_95": np.quantile(
                        delta, [0.025, 0.975]
                    ).tolist(),
                    "share_positive": float((delta > 0).mean()),
                },
            }
        )
    report = {
        "multiplier": settings.multiplier,
        "studies": len(hard),
        "checkpoints": checkpoints,
        "training_reflection_parity_max_abs": parity,
        "bootstrap": {"draws": settings.draws, "seed": settings.seed, "unit": "study"},
        "results": results,
    }
    (settings.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
