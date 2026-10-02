import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import INFERENCE_BATCH_SIZE, load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.inference.checkpoint import checkpoint_model, read_checkpoint
from knee_abnormality.inference.engine import (
    move_batch,
    prediction_data,
    resolve_device,
)
from knee_abnormality.provenance import copy_sources, sha256_file

GROUPS = {
    "detail": (
        "ACL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
    ),
    "wide": ("MCL", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"),
}
CANVAS = 448


class Settings(CLISettings):
    model_config = describe(
        "Accumulate per series type the input-gradient magnitude of the detail and "
        "wide label groups on validation slices, knee-centred in coronal and sagittal"
    )

    run_dir: Path = Field(description="Training run directory")
    output_dir: Path = Field(description="Fresh output directory")
    manifest_dir: Path = Field(
        default=Path("configs/labels"),
        description="Directory holding the label manifests when the run's recorded "
        "paths no longer exist",
    )
    fold: int = Field(default=0, description="Validation fold of the run")
    studies: int = Field(default=300, description="Validation studies, in UID order")
    num_workers: int = Field(default=6, description="Loader workers")
    torch_threads: int = Field(default=8, description="CPU threads for the model")


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


def centre_column(axis: torch.Tensor, size: int) -> torch.Tensor:
    grid = axis.shape[-1]
    middle = axis[:, grid // 2]
    slope = (middle[:, -1] - middle[:, 0]) / (grid - 1)
    safe = torch.where(slope.abs() > 1e-6, slope, torch.ones_like(slope))
    cell = -middle[:, 0] / safe
    return cell * 32.0, slope


def run(settings: Settings) -> dict:
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(settings.torch_threads)
    fold_dir = settings.run_dir / f"fold_{settings.fold}"
    selected = json.loads((fold_dir / "selected_checkpoint.json").read_text())
    checkpoint_path = fold_dir / Path(selected["path"]).name
    config = relocated_config(settings)
    checkpoint = read_checkpoint(checkpoint_path)
    dataset, targets, studies = prediction_data(config, "validation", settings.fold)
    chosen = sorted(studies["StudyInstanceUID"].to_list())[: settings.studies]
    dataset, targets, _ = prediction_data(
        config, "validation", settings.fold, image_multiplier=1, study_ids=chosen
    )
    device = resolve_device("cpu")
    model = checkpoint_model(checkpoint, config, targets, device, 1.0)
    loader = build_data_loader(
        dataset,
        config.trainer,
        INFERENCE_BATCH_SIZE,
        num_workers=settings.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    captured = {}

    def capture(module, arguments):
        arguments[0].requires_grad_(True)
        captured["input"] = arguments[0]

    model.backbone.conv1.register_forward_pre_hook(capture)
    indices = {
        name: [targets.index(target) for target in members]
        for name, members in GROUPS.items()
    }
    type_count = len(config.data.series_types)
    maps = {
        name: np.zeros((type_count, CANVAS, CANVAS), dtype=np.float64)
        for name in GROUPS
    }
    slices = np.zeros(type_count, dtype=np.int64)
    anchored = np.zeros(type_count, dtype=np.int64)
    size = config.data.image_size
    pad = (CANVAS - size) // 2
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        batch = with_images(move_batch(batch, device))
        with torch.enable_grad():
            encoded = model.encode_batch(batch)
            logits = model.pool_encoded(encoded).float()
            gradients = {}
            for name, members in indices.items():
                (gradient,) = torch.autograd.grad(
                    logits[:, members].sum(), captured["input"], retain_graph=True
                )
                gradients[name] = gradient.abs().sum(1)
        grid = int(encoded["visual_valid"].shape[1] ** 0.5)
        plane = encoded["image_plane"]
        kind = encoded["image_type"]
        study = encoded["study"]
        trusted = encoded["trusted_geometry"].all(1)
        medial = encoded["medial_axis"].reshape(-1, grid, grid)
        anterior = encoded["anterior_axis"].reshape(-1, grid, grid)
        axis = torch.where((plane == 2)[:, None, None], anterior, medial)
        column, slope = centre_column(axis, size)
        usable = trusted & (slope.abs() > 1e-6) & (plane != 0)
        shift = torch.where(usable, (size / 2 - column).round(), 0.0).long()
        shift = shift.clamp(-pad, pad)
        flip = usable & (slope < 0)
        for name, gradient in gradients.items():
            total = torch.zeros(encoded["batch_size"]).index_add_(
                0, study, gradient.sum((1, 2))
            )
            gradient = gradient / total[study][:, None, None]
            for image in range(len(gradient)):
                value = gradient[image]
                left = pad + int(shift[image])
                if flip[image]:
                    value = value.flip(-1)
                    left = pad - int(shift[image])
                maps[name][int(kind[image]), pad : pad + size, left : left + size] += (
                    value.numpy()
                )
        for image in range(len(kind)):
            slices[int(kind[image])] += 1
            anchored[int(kind[image])] += int(usable[image])
        if step % 10 == 0:
            print(
                f"batch {step} of {len(loader)}, {time.perf_counter() - started:.0f} s",
                file=sys.stderr,
                flush=True,
            )
    np.savez_compressed(
        settings.output_dir / "saliency.npz",
        slices=slices,
        anchored=anchored,
        **maps,
    )
    source = Path(__file__).resolve()
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "studies": len(chosen),
        "slices": slices.tolist(),
        "anchored": anchored.tolist(),
        "groups": GROUPS,
        "canvas": CANVAS,
        "spacing_mm": config.data.spacing_mm,
        "seconds": time.perf_counter() - started,
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
