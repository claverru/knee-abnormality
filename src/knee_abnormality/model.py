import math

import timm
import torch
from torch import nn

from knee_abnormality.augmentation import (
    ImageAugmentation,
    affine_matrices,
    apply_reflection,
    augmented_to_prepared,
    reflection_channels,
    reflection_inverse,
    reflection_positions,
    sample_augmentation_parameters,
    sample_reflection_parameters,
)
from knee_abnormality.config import DataConfig, ModelConfig
from knee_abnormality.spatial_tokens import (
    SpatialTokens,
    backbone_mapping,
    feature_geometry,
)


def sinusoidal_positions(
    positions: torch.Tensor, width: int, dtype: torch.dtype
) -> torch.Tensor:
    frequency = torch.exp(
        torch.arange(0, width, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    angles = positions.to(torch.float32).unsqueeze(-1) * frequency
    encoding = torch.zeros(
        (*positions.shape, width), device=positions.device, dtype=torch.float32
    )
    encoding[..., 0::2] = torch.sin(angles)
    encoding[..., 1::2] = torch.cos(angles[..., : width // 2])
    return encoding.to(dtype)


class KneeAbnormalityModel(nn.Module):
    def __init__(
        self,
        model: ModelConfig,
        data: DataConfig,
        target_count: int,
        load_pretrained: bool | None = None,
    ) -> None:
        super().__init__()
        if data.slices_per_input != 3 or data.position_encoding != "physical_mm":
            raise ValueError(
                "Spatial tokens require triplets with physical 1D positions"
            )
        self.series_dropout = model.series_dropout
        self.augmentation_settings = data
        self.image_augmentation = ImageAugmentation()
        self.prepared_channels = data.slices_per_input
        self.register_buffer(
            "mean", torch.tensor(data.mean).reshape(1, -1, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(data.std).reshape(1, -1, 1, 1), persistent=False
        )
        self.projection_width = model.projection_width
        self.channels_last = model.channels_last
        self.checkpoint_backbone = model.checkpoint_backbone
        self.backbone_eval = model.backbone_eval
        pretrained = model.pretrained if load_pretrained is None else load_pretrained
        self.backbone = timm.create_model(
            model.backbone,
            pretrained=pretrained,
            in_chans=data.input_channels,
            num_classes=0,
            global_pool="",
        )
        self.feature_mapping = backbone_mapping(model.backbone, self.backbone)
        backbone_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.inference_mode():
                spatial_output = self.backbone.forward_features(
                    torch.zeros(
                        1,
                        data.input_channels,
                        data.image_size,
                        data.image_size,
                    )
                )
        finally:
            self.backbone.train(backbone_training)
        if not isinstance(spatial_output, torch.Tensor) or spatial_output.ndim != 4:
            raise ValueError("Backbone must return NCHW spatial features")
        if self.checkpoint_backbone:
            self.backbone.set_grad_checkpointing(True)
        self.projection = nn.Linear(spatial_output.shape[1], model.projection_width)
        if self.channels_last:
            self.backbone.to(memory_format=torch.channels_last)
        self.mil_norm = nn.LayerNorm(model.projection_width)
        self.attention = nn.Sequential(
            nn.Linear(model.projection_width, model.mil_attention_width),
            nn.Tanh(),
            nn.Dropout(model.dropout),
            nn.Linear(model.mil_attention_width, target_count),
        )
        self.head = nn.Linear(model.projection_width, target_count)
        self.spatial_tokens = SpatialTokens(model.projection_width, data.series_types)
        self.train(self.training)

    def train(self, mode: bool = True) -> "KneeAbnormalityModel":
        super().train(mode)
        if self.backbone_eval:
            self.backbone.eval()
        return self

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_backbone:
            enabled = self.training and torch.is_grad_enabled()
            if enabled and not self.backbone_eval:
                raise ValueError(
                    "Activation checkpointing requires model.backbone_eval=true"
                )
            self.backbone.set_grad_checkpointing(enabled)
        return self.backbone.forward_features(images)

    def _effective_presence(self, presence: torch.Tensor) -> torch.Tensor:
        if not presence.any(dim=1).all():
            raise ValueError("Every study must contain at least one present series")
        if not self.training or self.series_dropout == 0:
            return presence
        keep = presence & (
            torch.rand(presence.shape, device=presence.device) >= self.series_dropout
        )
        empty = ~keep.any(dim=1)
        if empty.any():
            scores = torch.rand(presence.shape, device=presence.device).masked_fill(
                ~presence, -1
            )
            selected = scores.argmax(dim=1)
            rows = empty.nonzero(as_tuple=False).flatten()
            keep[rows, selected[rows]] = True
        return keep

    def _augment_images(
        self, images: torch.Tensor, slice_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        settings = self.augmentation_settings
        if not self.training or not any(
            augmentation.probability > 0
            for augmentation in (
                settings.intensity_augmentation,
                settings.affine_augmentation,
                settings.noise_augmentation,
            )
        ):
            return images, torch.eye(3, device=images.device)[None].expand(
                len(images), -1, -1
            )
        parameters = sample_augmentation_parameters(
            settings, slice_mask.shape[0], images.device
        )
        series_index = slice_mask.nonzero(as_tuple=True)[0]
        matrices = affine_matrices(parameters.affine, *images.shape[-2:])
        inverse = augmented_to_prepared(matrices, *images.shape[-2:])
        return self.image_augmentation(
            images, series_index, parameters, matrices=matrices
        ), inverse[series_index]

    def _attention_pool(
        self,
        tokens: torch.Tensor,
        token_study: torch.Tensor,
        token_valid: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        scores = self.attention(tokens).transpose(0, 1)
        membership = (
            torch.arange(batch_size, device=tokens.device)[:, None] == token_study[None]
        ) & token_valid[None]
        if not membership.any(-1).all():
            raise ValueError("A study has no visually valid spatial tokens")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            attention = scores.float().unsqueeze(0).expand(batch_size, -1, -1)
            attention = attention.masked_fill(~membership[:, None], -torch.inf).softmax(
                -1
            )
            pooled = torch.einsum("btn,nf->btf", attention, tokens.float())
            logits = (pooled * self.head.weight.float()[None]).sum(
                -1
            ) + self.head.bias.float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Study attention pooling produced non-finite logits")
        return logits

    def encode_batch(self, batch: dict) -> dict:
        if batch["images"].shape[2] != self.prepared_channels:
            raise ValueError(
                "Prepared image channels differ from the configured input mode"
            )
        presence = batch["series_presence"]
        reflected_study = torch.zeros_like(batch["reflection_eligible"])
        if self.training:
            reflected_study = sample_reflection_parameters(
                self.augmentation_settings.reflection_augmentation.probability,
                batch["reflection_eligible"],
            ).applied
        effective = self._effective_presence(presence)
        all_study = batch["series_study_index"]
        all_type = batch["series_type_index"]
        keep_series = effective[all_study, all_type]
        study_index = all_study[keep_series]
        type_index = all_type[keep_series]
        slice_mask = batch["slice_mask"][keep_series]
        flat_indices = (
            (keep_series[:, None] & batch["slice_mask"]).flatten().nonzero().flatten()
        )
        positions = batch["slice_positions"].flatten()[flat_indices]

        valid_images = (
            batch["images"].flatten(0, 1)[flat_indices].to(torch.float32).div_(255.0)
        )
        acquisition_index = slice_mask.nonzero(as_tuple=True)[0]
        token_study = study_index[acquisition_index]
        token_type = type_index[acquisition_index]
        reflected = reflected_study[token_study]
        sagittal = self.spatial_tokens.plane_lookup[token_type] == 2
        if (
            self.training
            and self.augmentation_settings.reflection_augmentation.probability > 0
        ):
            valid_images = apply_reflection(valid_images, reflected, sagittal)
        positions = reflection_positions(positions, reflected, sagittal)
        channels = reflection_channels(reflected, sagittal)
        valid_images, inverse_affine = self._augment_images(valid_images, slice_mask)
        inverse_affine = reflection_inverse(
            inverse_affine, reflected, sagittal, valid_images.shape[-1]
        )
        valid_images = valid_images.sub(self.mean).div_(self.std)
        if self.channels_last:
            valid_images = valid_images.contiguous(memory_format=torch.channels_last)
        maps = self._encode_images(valid_images)
        features = self.projection(maps.flatten(2).transpose(1, 2))
        xy, xyz, visual_valid, available_3d = feature_geometry(
            batch["patient_maps"].flatten(0, 1)[flat_indices],
            batch["valid_rects"].flatten(0, 1)[flat_indices],
            batch["geometry_valid"].flatten(0, 1)[flat_indices],
            inverse_affine,
            maps.shape[-2:],
            valid_images.shape[-2:],
            self.feature_mapping,
            batch["study_centers"][token_study],
            batch["center_valid"][token_study],
            reflected,
            channels,
        )
        terms = self.spatial_tokens(
            xy,
            xyz,
            available_3d,
            token_type,
            token_study,
            len(presence),
            features.dtype,
        )
        return {
            "features": features,
            "position_1d": sinusoidal_positions(
                positions, self.projection_width, features.dtype
            ),
            **terms,
            "study": token_study,
            "visual_valid": visual_valid,
            "available_3d": available_3d,
            "batch_size": len(presence),
        }

    def pool_encoded(self, encoded: dict) -> torch.Tensor:
        tokens = self.mil_norm(
            encoded["features"]
            + encoded["position_1d"][:, None]
            + encoded["position_2d"][None]
            + encoded["position_3d"]
            + encoded["plane"][:, None]
            + encoded["contrast"][:, None]
        )
        return self._attention_pool(
            tokens.flatten(0, 1),
            encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            encoded["visual_valid"].flatten(),
            encoded["batch_size"],
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.pool_encoded(self.encode_batch(batch))
