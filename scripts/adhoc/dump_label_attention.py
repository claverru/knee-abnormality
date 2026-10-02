import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import INFERENCE_BATCH_SIZE, Device, load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.data.splits import STUDY
from knee_abnormality.inference.checkpoint import checkpoint_model, read_checkpoint
from knee_abnormality.inference.engine import (
    move_batch,
    prediction_data,
    resolve_device,
)
from knee_abnormality.modeling.model import COMPARTMENT_MIDLINE_BAND_MM
from knee_abnormality.provenance import copy_sources, sha256_file


class Settings(CLISettings):
    model_config = describe(
        "Dump each validation token's per-label attention and readout evidence from a "
        "plane-readout checkpoint, with its slice, grid cell and knee-centred position"
    )

    run_dir: Path = Field(description="Training run directory")
    output_dir: Path = Field(description="Fresh output directory")
    fold: int = Field(default=0, description="Validation fold of the run")
    image_multiplier: int = Field(default=1, description="Validation image multiplier")
    device: Device = Field(default="cpu", description="Inference device")
    num_workers: int = Field(default=8, description="Loader workers")
    torch_threads: int = Field(default=12, description="CPU threads for the model")
    max_batches: int | None = Field(
        default=None, description="Stop after this many batches"
    )
    manifest_dir: Path = Field(
        default=Path("configs/labels"),
        description="Directory holding the label manifests when the run's recorded "
        "paths no longer exist",
    )


def relocated_config(settings: Settings):
    effective = json.loads((settings.run_dir / "effective_config.json").read_text())
    for name in ("label_manifest", "report_attribute_manifest"):
        path = Path(effective["paths"][name])
        if not path.exists():
            effective["paths"][name] = str(
                (settings.manifest_dir / path.name).resolve(strict=True)
            )
    path = settings.output_dir / "effective_config.json"
    path.write_text(json.dumps(effective, indent=2) + "\n")
    return load_config(path)


def token_attention(model, encoded: dict):
    tokens, values = model._pooling_tokens(encoded)
    grid = encoded["visual_valid"].shape[1]
    study = encoded["study"][:, None].expand(-1, grid).flatten()
    valid = encoded["visual_valid"].flatten()
    axis = encoded["medial_axis"].flatten().float()
    count = encoded["batch_size"]
    if model.anatomical_log_strength is not None or not model.plane_readout:
        raise ValueError("The probe supports plane readouts without anatomical bias")
    scores = model.attention(tokens).transpose(0, 1).float()
    membership = (
        torch.arange(count, device=tokens.device)[:, None] == study[None]
    ) & valid[None]
    signs = model.meniscus_mask_signs[:, None]
    own_side = (signs == 0) | (signs * axis[None] > -COMPARTMENT_MIDLINE_BAND_MM)
    restricted = membership[:, None] & own_side[None]
    allowed = torch.where(
        restricted.any(-1, keepdim=True), restricted, membership[:, None]
    )
    attention = scores[None].expand(count, -1, -1)
    attention = attention.masked_fill(~allowed, -torch.inf).softmax(-1)
    evidence = model.head(values, encoded["image_type"]).float()
    partial = torch.einsum("btn,nt->bt", attention, evidence) + model.head.bias.float()
    index = torch.arange(len(study), device=tokens.device)
    own = attention.permute(2, 0, 1)[index, study]
    return own, evidence, partial, study, valid


def run(settings: Settings) -> dict:
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(settings.torch_threads)
    fold_dir = settings.run_dir / f"fold_{settings.fold}"
    selected = json.loads((fold_dir / "selected_checkpoint.json").read_text())
    checkpoint_path = fold_dir / Path(selected["path"]).name
    config = relocated_config(settings)
    checkpoint = read_checkpoint(checkpoint_path)
    dataset, targets, _ = prediction_data(
        config, "validation", settings.fold, image_multiplier=settings.image_multiplier
    )
    if list(targets) != checkpoint["hyper_parameters"]["targets"]:
        raise ValueError("Checkpoint targets differ from prediction targets")
    device = resolve_device(settings.device)
    model = checkpoint_model(checkpoint, config, targets, device, 1.0)
    loader = build_data_loader(
        dataset,
        config.trainer,
        INFERENCE_BATCH_SIZE,
        num_workers=settings.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    columns = {
        name: []
        for name in (
            "slice",
            "cell",
            "image_type",
            "plane",
            "position_mm",
            "medial_mm",
            "anterior_mm",
            "trusted",
            "valid",
            "attention",
            "evidence",
        )
    }
    token_uids, study_uids, logits, partials = [], [], [], []
    slices = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if settings.max_batches is not None and step >= settings.max_batches:
                break
            batch = with_images(move_batch(batch, device))
            encoded = model.encode_batch(batch)
            attention, evidence, partial, study, valid = token_attention(model, encoded)
            logits.append(model.pool_encoded(encoded).float().cpu().numpy())
            partials.append(partial.cpu().numpy())
            study_uids.extend(batch["study_uid"])
            images, grid = encoded["visual_valid"].shape
            presence = batch["series_presence"]
            keep = presence[batch["series_study_index"], batch["series_type_index"]]
            flat = (keep[:, None] & batch["slice_mask"]).flatten().nonzero().flatten()
            positions = batch["slice_positions"].flatten()[flat].float()
            if len(positions) != images:
                raise ValueError("Slice positions do not match the encoded images")
            for name, value in (
                ("slice", torch.arange(images, device=device) + slices),
                ("image_type", encoded["image_type"]),
                ("plane", encoded["image_plane"]),
                ("position_mm", positions),
            ):
                columns[name].append(value.repeat_interleave(grid).cpu().numpy())
            slices += images
            columns["cell"].append(
                torch.arange(grid, device=device).repeat(images).cpu().numpy()
            )
            columns["medial_mm"].append(
                encoded["medial_axis"].flatten().float().cpu().numpy()
            )
            columns["anterior_mm"].append(
                encoded["anterior_axis"].flatten().float().cpu().numpy()
            )
            columns["trusted"].append(
                encoded["trusted_geometry"].flatten().cpu().numpy()
            )
            columns["valid"].append(valid.cpu().numpy())
            columns["attention"].append(attention.cpu().numpy())
            columns["evidence"].append(evidence.cpu().numpy())
            token_uids.extend(batch["study_uid"][i] for i in study.cpu().tolist())
            if step % 20 == 0:
                print(
                    f"batch {step} of {len(loader)}, {time.perf_counter() - started:.0f} s",
                    file=sys.stderr,
                    flush=True,
                )
    seconds = time.perf_counter() - started
    stacked = {name: np.concatenate(values) for name, values in columns.items()}
    frame = {STUDY: token_uids}
    for name in columns:
        if name in ("attention", "evidence"):
            for j, target in enumerate(targets):
                frame[f"{name}|{target}"] = stacked[name][:, j].astype(np.float32)
        else:
            frame[name] = stacked[name]
    pl.DataFrame(frame).write_parquet(settings.output_dir / "tokens.parquet")
    logits = np.concatenate(logits)
    partials = np.concatenate(partials)
    pl.DataFrame(
        {STUDY: study_uids, **{t: logits[:, j] for j, t in enumerate(targets)}}
    ).write_parquet(settings.output_dir / "logits.parquet")
    oa = list(model.oa_indices)
    other = [j for j in range(len(targets)) if j not in oa]
    source = Path(__file__).resolve()
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "image_multiplier": settings.image_multiplier,
        "attention_temperature": 1.0,
        "device": str(device),
        "studies": len(study_uids),
        "slices": slices,
        "tokens": len(token_uids),
        "targets": list(targets),
        "seconds": seconds,
        "reconstruction_max_abs_diff_non_oa": float(
            np.abs(logits[:, other] - partials[:, other]).max()
        ),
        "source_snapshot": copy_sources([source], settings.output_dir / "source"),
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    print(json.dumps(run(Settings()), indent=2))


if __name__ == "__main__":
    main()
