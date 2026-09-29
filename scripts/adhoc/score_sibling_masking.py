import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from pydantic import Field
from pydantic_settings import SettingsError
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import INFERENCE_BATCH_SIZE, load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD, load_split
from knee_abnormality.evaluation.metrics import bootstrap_auc
from knee_abnormality.inference.checkpoint import checkpoint_model, read_checkpoint
from knee_abnormality.inference.engine import (
    move_batch,
    prediction_data,
    resolve_device,
    resolve_precision,
)
from knee_abnormality.provenance import copy_sources, sha256_file

PAIRS = (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))
ARMS = ("intact", "token_mask", "pixel_blank")
REMOVED_SERIES_FIELDS = ("oblique_tilt_deg",)


class Settings(CLISettings):
    run_dir: Path = Path("outputs/training/k05-clean")
    output_dir: Path = Field(description="Fresh output directory")
    fold: int = 0
    image_multiplier: int = 4
    midline_band_mm: float = 10.0
    draws: int = 2000
    seed: int = 20260923


def current_config(effective: dict, output_dir: Path):
    for series_type in effective["data"]["series_types"]:
        for name in REMOVED_SERIES_FIELDS:
            series_type.pop(name, None)
    path = output_dir / "effective_config.json"
    path.write_text(json.dumps(effective, indent=2) + "\n")
    return load_config(path)


def side_keep(model, medial_axis: torch.Tensor, band: float) -> torch.Tensor:
    signs = torch.sign(model.compartment_signs)
    return (signs[:, None] == 0) | (signs[:, None] * medial_axis[None] > -band)


def pool(model, encoded: dict, keep: torch.Tensor | None):
    tokens, values = model._pooling_tokens(encoded)
    study = encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten()
    valid = encoded["visual_valid"].flatten()
    axis = encoded["medial_axis"].flatten().float()
    count = encoded["batch_size"]
    scores = model._attention_scores(tokens).transpose(0, 1).float()
    strengths = model.anatomical_log_strength.float().exp()
    signs = model.compartment_signs * strengths[model.compartment_families]
    scores = scores + signs[:, None] * torch.tanh(axis[None] / 20.0)
    member = (torch.arange(count, device=study.device)[:, None] == study[None]) & valid[
        None
    ]
    member = member[:, None, :].expand(-1, scores.shape[0], -1)
    fallback = torch.zeros(member.shape[:2], dtype=torch.bool, device=study.device)
    if keep is not None:
        restricted = member & keep[None]
        fallback = ~restricted.any(-1) & member.any(-1)
        member = torch.where(fallback[..., None], member, restricted)
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        attention = (
            scores[None]
            .expand(count, -1, -1)
            .masked_fill(~member, -torch.inf)
            .softmax(-1)
        )
        pooled = torch.einsum("btn,nf->btf", attention, values.float())
        logits = (pooled * model.head.weight.float()[None]).sum(
            -1
        ) + model.head.bias.float()
    kept = (member & valid[None, None]).sum(-1)
    return logits, fallback, kept


def blank_sibling(batch: dict, keep_sign: float, band: float) -> dict:
    images = batch["images"].clone()
    size = images.shape[-1]
    lines = torch.arange(size, device=images.device, dtype=torch.float32)
    rows, columns = torch.meshgrid(lines, lines, indexing="ij")
    study = batch["series_study_index"]
    side = batch["knee_side_sign"][study].float()
    center = batch["knee_center_x_mm"][study].float()
    maps = batch["patient_maps"].float()
    usable = (
        batch["geometry_valid"]
        & batch["slice_mask"]
        & torch.isfinite(maps).all((-1, -2, -3))
        & (side != 0)[:, None]
    )
    for series in range(images.shape[0]):
        slots = usable[series].nonzero().flatten()
        if not len(slots):
            continue
        coefficients = maps[series, slots, :, 0]
        patient_x = (
            coefficients[..., 0, None, None] * columns
            + coefficients[..., 1, None, None] * rows
            + coefficients[..., 2, None, None]
        )
        medial_axis = side[series] * (patient_x - center[series])
        sibling = keep_sign * medial_axis < -band
        images[series, slots] = images[series, slots] * ~sibling
    return {**batch, "images": images}


def predict_arms(model, loader, device, band: float) -> dict:
    _, autocast_dtype = resolve_precision("auto", device)
    medial_rows = torch.sign(model.compartment_signs) > 0
    lateral_rows = torch.sign(model.compartment_signs) < 0
    outputs = {arm: [] for arm in ARMS}
    ids, diagnostics = [], []
    with torch.inference_mode():
        for batch in loader:
            ids.extend(batch["study_uid"])
            batch = with_images(move_batch(batch, device))
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                encoded = model.encode_batch(batch)
                intact, _, intact_kept = pool(model, encoded, None)
                keep = side_keep(model, encoded["medial_axis"].flatten(), band)
                masked, masked_fallback, masked_kept = pool(model, encoded, keep)
                views = {}
                for name, keep_sign in (("medial", 1.0), ("lateral", -1.0)):
                    view = model.encode_batch(blank_sibling(batch, keep_sign, band))
                    views[name] = pool(
                        model,
                        view,
                        side_keep(model, view["medial_axis"].flatten(), band),
                    )
                blanked = intact.clone()
                blanked[:, medial_rows] = views["medial"][0][:, medial_rows]
                blanked[:, lateral_rows] = views["lateral"][0][:, lateral_rows]
                blank_fallback = torch.where(
                    medial_rows[None], views["medial"][1], views["lateral"][1]
                )
            for arm, logits in zip(ARMS, (intact, masked, blanked)):
                outputs[arm].append(torch.sigmoid(logits.float()).cpu())
            side = batch["knee_side_sign"].cpu()
            for row, uid in enumerate(batch["study_uid"]):
                diagnostics.append(
                    {
                        STUDY: uid,
                        "knee_side_sign": int(side[row]),
                        "intact_tokens": int(intact_kept[row, 0]),
                        **{
                            f"mask_tokens_{t}": int(masked_kept[row, t])
                            for t in range(masked_kept.shape[1])
                        },
                        "mask_fallback": int(masked_fallback[row].sum()),
                        "blank_fallback": int(blank_fallback[row].sum()),
                    }
                )
    return (
        {arm: torch.cat(values).numpy() for arm, values in outputs.items()},
        ids,
        diagnostics,
    )


def bootstrap_counts(groups, truth, draws, seed):
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=draws
    )[:, inverse]
    positive = counts @ truth
    return counts[
        ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(axis=1)
    ]


def stratum_auc(truth, scores, rows, counts):
    y = truth[rows]
    point = float(roc_auc_score(y, scores[rows]))
    sample = counts[:, rows]
    usable = ((sample @ y) > 0) & ((sample @ (1 - y)) > 0)
    return point, sample[usable]


def score(arms: dict, truth: np.ndarray, targets, counts) -> dict:
    index = {name: j for j, name in enumerate(targets)}
    result = {}
    for arm, scores in arms.items():
        result[arm] = {
            "macro_auc": float(
                np.mean(
                    [
                        roc_auc_score(truth[:, j], scores[:, j])
                        for j in range(len(targets))
                    ]
                )
            )
        }
    base = bootstrap_auc(truth, arms["intact"], counts)
    for arm in ARMS[1:]:
        delta = bootstrap_auc(truth, arms[arm], counts) - base
        result[arm]["macro_delta"] = (
            result[arm]["macro_auc"] - result["intact"]["macro_auc"]
        )
        result[arm]["macro_ci80"] = np.percentile(delta, [10, 90]).tolist()
    for medial, lateral in PAIRS:
        for own, sibling in ((medial, lateral), (lateral, medial)):
            y = truth[:, index[own]]
            other = truth[:, index[sibling]]
            strata = {
                "overall": np.ones(len(y), dtype=bool),
                "sibling_negative": (y == 1) | (other == 0),
                "sibling_positive": (y == 1) | (other == 1),
            }
            for stratum, rows in strata.items():
                rows = np.flatnonzero(rows)
                key = f"{own} | {stratum}"
                entry = {
                    "positives": int(y[rows].sum()),
                    "negatives": int((1 - y[rows]).sum()),
                }
                intact, sample = stratum_auc(
                    y, arms["intact"][:, index[own]], rows, counts
                )
                entry["intact"] = intact
                reference = bootstrap_auc(
                    y[rows, None], arms["intact"][rows, index[own], None], sample
                )
                for arm in ARMS[1:]:
                    value = float(roc_auc_score(y[rows], arms[arm][rows, index[own]]))
                    delta = (
                        bootstrap_auc(
                            y[rows, None], arms[arm][rows, index[own], None], sample
                        )
                        - reference
                    )
                    entry[arm] = value
                    entry[f"{arm}_delta"] = value - intact
                    entry[f"{arm}_ci80"] = np.percentile(delta, [10, 90]).tolist()
                result.setdefault("strata", {})[key] = entry
    return result


def run(settings: Settings) -> dict:
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    fold_dir = settings.run_dir / f"fold_{settings.fold}"
    selected = json.loads((fold_dir / "selected_checkpoint.json").read_text())
    checkpoint_path = fold_dir / Path(selected["path"]).name
    config = current_config(
        json.loads((settings.run_dir / "effective_config.json").read_text()),
        settings.output_dir,
    )
    checkpoint = read_checkpoint(checkpoint_path)
    saved_hparams = checkpoint["hyper_parameters"]
    if saved_hparams["fold"] != settings.fold:
        raise ValueError("Checkpoint fold differs from the requested fold")
    dataset, targets, _ = prediction_data(
        config, "validation", settings.fold, image_multiplier=settings.image_multiplier
    )
    if list(targets) != saved_hparams["targets"]:
        raise ValueError("Checkpoint targets differ from prediction targets")
    device = resolve_device("auto")
    model = checkpoint_model(checkpoint, config, targets, device, 1.0)
    if model.residual_readout or config.model.compartment_evidence != "whole_knee":
        raise ValueError(
            "The masked pooling supports whole-knee independent heads only"
        )
    loader = build_data_loader(dataset, config.trainer, INFERENCE_BATCH_SIZE)
    started = time.perf_counter()
    arms, ids, diagnostics = predict_arms(
        model, loader, device, settings.midline_band_mm
    )
    seconds = time.perf_counter() - started
    order = np.argsort(ids)
    ids = [ids[i] for i in order]
    arms = {arm: values[order] for arm, values in arms.items()}
    for arm, values in arms.items():
        pl.DataFrame(
            {STUDY: ids, **{t: values[:, j] for j, t in enumerate(targets)}}
        ).write_csv(settings.output_dir / f"{arm}.csv")
    pl.DataFrame(diagnostics).write_csv(settings.output_dir / "diagnostics.csv")
    saved = pl.read_csv(fold_dir / "validation_predictions.csv").sort(STUDY)
    if saved[STUDY].to_list() != ids:
        raise ValueError("Saved predictions do not cover the predicted studies")
    parity = float(np.abs(saved.select(targets).to_numpy() - arms["intact"]).max())
    split = load_split(config.raw_dir, config.split_dir)
    evaluation = (
        split.membership.select(STUDY, "report_group")
        .join(split.evaluation, on=STUDY)
        .filter(pl.col(STUDY).is_in(ids))
        .sort(STUDY)
    )
    if evaluation[STUDY].to_list() != ids:
        raise ValueError("Evaluation targets do not cover the predicted studies")
    truth = (evaluation.select(targets).to_numpy() >= TARGET_THRESHOLD).astype(
        np.float64
    )
    counts = bootstrap_counts(
        evaluation["report_group"].to_numpy(), truth, settings.draws, settings.seed
    )
    sides = pl.DataFrame(diagnostics)["knee_side_sign"]
    summary = {
        "checkpoint": str(checkpoint_path),
        "image_multiplier": settings.image_multiplier,
        "attention_temperature": 1.0,
        "midline_band_mm": settings.midline_band_mm,
        "studies": len(ids),
        "unknown_side_studies": int((sides == 0).sum()),
        "intact_vs_saved_max_abs_diff": parity,
        "prediction_seconds": seconds,
        "mask_fallback_study_targets": int(
            pl.DataFrame(diagnostics)["mask_fallback"].sum()
        ),
        "blank_fallback_study_targets": int(
            pl.DataFrame(diagnostics)["blank_fallback"].sum()
        ),
        **score(arms, truth, targets, counts),
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    source = Path(__file__).resolve()
    manifest = {
        "inputs": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (
                checkpoint_path,
                fold_dir / "validation_predictions.csv",
                settings.run_dir / "effective_config.json",
                source,
            )
        },
        "checkpoint_identity_check": "skipped; parity is intact_vs_saved_max_abs_diff",
        "source_snapshot": copy_sources([source], settings.output_dir / "source"),
    }
    (settings.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return summary


def main() -> None:
    try:
        summary = run(Settings())
    except (OSError, ValueError, SettingsError) as error:
        print(f"Sibling masking scoring error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
