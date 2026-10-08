import json
from pathlib import Path

import pytorch_lightning as L
import torch
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.dataset import build_data_loader
from knee_abnormality.data.preprocessing.device import with_images
from knee_abnormality.inference.engine import move_batch
from knee_abnormality.inference.reflection import reflect_batch, sagittal_types
from knee_abnormality.modeling.model import SHARED_CHANNELS, TARGET_ROUTES
from knee_abnormality.training.lightning import (
    KneeDataModule,
    KneeModule,
    fixed_class_balance,
)

ALLOWED_LOSSES = {
    "anatomy_loss",
    "train_loss",
    "side_head_loss",
    "report_attribute_loss",
    "shift_locator_loss",
    "anchor_penalty",
}
TOLERANCE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Settings(CLISettings):
    model_config = describe(
        "Check the side-routed design's guardrails on a freshly built model and one "
        "training batch: shared classifiers, their blindness to the compartment, "
        "geometry as the only assignment, routing weights that partition every token "
        "among the targets of a shared classifier, zero "
        "start of the learned geometry, readers that leave the backbone alone, the "
        "allowed loss terms and mirror consistency. Runs on the GPU when one is visible "
        "and on the CPU otherwise. Exits non-zero when one fails"
    )
    config: Path = Field(description="Experiment configuration")
    fold: int = Field(default=1, ge=0, description="Fold whose training batch is used")
    studies: int = Field(default=4, gt=0, description="Studies in the checked batch")
    output: Path | None = Field(default=None, description="Optional JSON report path")


def autocast():
    return torch.autocast(
        DEVICE.type,
        dtype=torch.bfloat16,
        enabled=DEVICE.type == "cuda",
        cache_enabled=False,
    )


def frame(model, encoded, sign: float) -> torch.Tensor:
    with torch.autocast(device_type=DEVICE.type, enabled=False):
        return model.frame(encoded, sign)


def pair_logits(model, encoded, left: float):
    sign = -1.0 if left else 1.0
    flip = torch.where(
        (encoded["left_probability"] > 0.5) == bool(left), 1.0, -1.0
    ).float()
    mirror = torch.stack((flip, torch.ones_like(flip), torch.ones_like(flip)), -1)
    moved = {
        **encoded,
        "left_probability": torch.full_like(encoded["left_probability"], left),
        "frame_shift": encoded["frame_shift"].float() * mirror,
        "frame_rotation": mirror[:, :, None]
        * encoded["frame_rotation"].float()
        * mirror[:, None, :],
    }
    return (
        model.pool_encoded({**moved, "position": frame(model, moved, sign)}).float()
        - model.target_bias.float()
    )


def main() -> None:
    settings = Settings()
    config = load_config(settings.config)
    L.seed_everything(config.seed, workers=True)
    data = KneeDataModule(config, settings.fold, limit_train=settings.studies)
    data.setup("fit")
    targets = data.labels.targets
    module = KneeModule(
        config,
        targets,
        settings.fold,
        settings.studies,
        data.labels.validation.height,
        Path("/tmp/unused"),
        fixed_class_balance(
            data.labels.training, targets, config.trainer.class_balance_exponent
        ),
        load_pretrained=False,
    ).to(DEVICE)
    model = module.model
    logged = set()
    module.log = lambda name, value, **kwargs: logged.add(name)
    raw = next(
        iter(
            build_data_loader(
                data.train_dataset, config.trainer, settings.studies, num_workers=0
            )
        )
    )
    batch = with_images(
        {
            key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value
            for key, value in move_batch(raw, DEVICE).items()
        }
    )
    index = {name: number for number, name in enumerate(targets)}
    results = {}

    channels = {}
    for target, (channel, _) in TARGET_ROUTES.items():
        channels.setdefault(channel, set()).add(
            int(model.target_channel[index[target]])
        )
    own = {
        int(model.target_channel[index[name]])
        for name in targets
        if name not in TARGET_ROUTES
    }
    results["shared_classifiers"] = (
        set(channels) == set(SHARED_CHANNELS)
        and all(len(rows) == 1 for rows in channels.values())
        and not own & set().union(*channels.values())
        and len(own) == len(targets) - len(TARGET_ROUTES)
    )

    module.train()
    with autocast():
        encoded = model.encode_batch(batch)
        model.pool_encoded(encoded)
        side_loss = module._side_step(model, batch, encoded, "")
    trunk = [model.projection.weight, *model.backbone.parameters()]
    gradients = torch.autograd.grad(side_loss, trunk, allow_unused=True)
    results["readers_leave_the_backbone_alone"] = all(
        gradient is None or float(gradient.abs().max()) == 0.0 for gradient in gradients
    )
    module.zero_grad()
    logged.clear()
    with autocast():
        module.training_step(batch, 0)
    module._flush_epoch_metrics()
    losses = {name for name in logged if name.endswith(("loss", "penalty"))}
    results["allowed_loss_terms"] = losses <= ALLOWED_LOSSES
    module.zero_grad()

    module.eval()
    mirrored_batch = reflect_batch(
        batch, sagittal_types(config.data.series_types), model.repeated_middle_slice
    )
    with torch.no_grad(), autocast():
        encoded = model.encode_batch(batch)
        valid = encoded["visual_valid"] & encoded["routable"][:, None]
        identity = torch.eye(3, device=DEVICE)
        prior = model.position_priors(encoded["position"], encoded["routable"])
        results["learned_geometry_starts_at_zero"] = (
            torch.allclose(model.frame_rotation(), identity, atol=1e-6)
            and float(prior.abs().max()) == 0.0
        )

        shared = model.study_tokens(encoded, shared=True)
        moved = model.study_tokens(
            {**encoded, "position": encoded["position"] + 37.0}, shared=True
        )
        results["shared_tokens_are_position_free"] = torch.equal(shared, moved)

        torch.manual_seed(0)
        model.position_prior.weight.normal_(std=0.3)
        as_left = pair_logits(model, encoded, 1.0)
        as_right = pair_logits(model, encoded, 0.0)
        mismatch = 0.0
        gap = float("inf")
        for medial, lateral in (
            ("Medial Meniscus", "Lateral Meniscus"),
            ("Medial OA", "Lateral OA"),
        ):
            mismatch = max(
                mismatch,
                float(
                    (as_left[:, index[medial]] - as_right[:, index[lateral]])
                    .abs()
                    .max()
                ),
                float(
                    (as_left[:, index[lateral]] - as_right[:, index[medial]])
                    .abs()
                    .max()
                ),
            )
            gap = min(
                gap,
                float(
                    (as_left[:, index[medial]] - as_left[:, index[lateral]]).abs().max()
                ),
            )
        results["shared_classifiers_blind_to_the_compartment"] = (
            mismatch < TOLERANCE < gap
        )

        model.position_prior.weight.zero_()
        unrouted = (
            model.pool_encoded(
                {**encoded, "routable": torch.zeros_like(encoded["routable"])}
            ).float()
            - model.target_bias.float()
        )
        routing = model.routing_weights(encoded).float()
        sides = routing[
            [
                int(model.target_route[index[target]])
                for target in ("Medial Meniscus", "Lateral Meniscus")
            ]
        ].sum(0)[valid]
        partition = max(
            float((sides - 1).clamp_min(0).max()),
            float((routing[:, valid] - 0.5).abs().max()) - 0.5,
        )
        results["routing_partitions_every_token"] = partition < TOLERANCE
        results["geometry_is_the_only_assignment"] = all(
            float((unrouted[:, index[a]] - unrouted[:, index[b]]).abs().max())
            < TOLERANCE
            for a, b in (
                ("Medial Meniscus", "Lateral Meniscus"),
                ("Medial OA", "Lateral OA"),
                ("Medial OA", "PF OA"),
            )
        )

        mirrored = model.encode_batch(mirrored_batch)
        grid = int(encoded["fitted_position"].shape[1] ** 0.5)
        certain = batch["knee_side_certain"][encoded["study"]] & encoded["routable"]
        sagittal = encoded["image_plane"] == 2
        plain = encoded["fitted_position"].unflatten(1, (grid, grid))
        other = mirrored["fitted_position"].unflatten(1, (grid, grid))
        pitch = float(config.data.reference_field_mm) / grid
        same = plain[certain & sagittal] - other[certain & sagittal]
        flipped = plain[certain & ~sagittal] - other.flip(2)[certain & ~sagittal]
        results["mirrored_pass_keeps_anatomical_position"] = (
            (same.numel() == 0 or float(same.abs().max()) < 1e-2)
            and (flipped.numel() == 0 or float(flipped.abs().max()) < pitch)
            and bool(valid.any())
        )

    report = {
        "passed": all(results.values()),
        "guardrails": results,
        "details": {
            "loss_terms": sorted(losses),
            "routing_partition_error": partition,
            "side_flip_mismatch": mismatch,
            "medial_lateral_gap": gap,
        },
    }
    text = json.dumps(report, indent=2)
    if settings.output is not None:
        settings.output.write_text(text)
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
