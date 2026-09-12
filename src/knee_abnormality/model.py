import math

import timm
import torch
from torch import nn

from knee_abnormality.augmentation import (
    ImageAugmentation,
    sample_augmentation_parameters,
)
from knee_abnormality.config import DataConfig, ModelConfig


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
        self.pooling = model.pooling
        self.channels_last = model.channels_last
        self.checkpoint_backbone = model.checkpoint_backbone
        self.backbone_eval = model.backbone_eval
        pretrained = model.pretrained if load_pretrained is None else load_pretrained
        self.backbone = timm.create_model(
            model.backbone,
            pretrained=pretrained,
            in_chans=data.input_channels,
            num_classes=0,
        )
        backbone_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.inference_mode():
                backbone_output = self.backbone(
                    torch.zeros(
                        1,
                        data.input_channels,
                        data.image_size,
                        data.image_size,
                    )
                )
        finally:
            self.backbone.train(backbone_training)
        if not isinstance(backbone_output, torch.Tensor) or backbone_output.ndim != 2:
            raise ValueError(
                "Backbone must return a two-dimensional pooled feature tensor"
            )
        if self.checkpoint_backbone:
            self.backbone.set_grad_checkpointing(True)
        self.projection = nn.Linear(backbone_output.shape[1], model.projection_width)
        if self.channels_last:
            self.backbone.to(memory_format=torch.channels_last)
        if model.pooling == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=model.projection_width,
                nhead=model.attention_heads,
                dim_feedforward=model.feedforward_width,
                dropout=model.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=model.transformer_layers,
                norm=nn.LayerNorm(model.projection_width),
                enable_nested_tensor=False,
            )
            self.cls_token = nn.Parameter(torch.empty(1, 1, model.projection_width))
        else:
            self.mil_norm = nn.LayerNorm(model.projection_width)
            self.attention = nn.Sequential(
                nn.Linear(model.projection_width, model.mil_attention_width),
                nn.Tanh(),
                nn.Dropout(model.dropout),
                nn.Linear(model.mil_attention_width, target_count),
            )
        self.type_embedding = nn.Embedding(
            len(data.series_types), model.projection_width
        )
        self.head = nn.Linear(model.projection_width, target_count)
        if model.pooling == "transformer":
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
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
        return self.backbone(images)

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
    ) -> torch.Tensor:
        settings = self.augmentation_settings
        if not self.training or not any(
            augmentation.probability > 0
            for augmentation in (
                settings.intensity_augmentation,
                settings.affine_augmentation,
                settings.noise_augmentation,
            )
        ):
            return images
        parameters = sample_augmentation_parameters(
            settings, slice_mask.shape[0], images.device
        )
        series_index = slice_mask.nonzero(as_tuple=True)[0]
        return self.image_augmentation(images, series_index, parameters)

    def _attention_pool(
        self,
        features: torch.Tensor,
        slice_mask: torch.Tensor,
        positions: torch.Tensor,
        study_index: torch.Tensor,
        type_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        acquisition_index = slice_mask.nonzero(as_tuple=True)[0]
        tokens = features + self.type_embedding(type_index[acquisition_index])
        tokens = self.mil_norm(
            tokens
            + sinusoidal_positions(
                positions[slice_mask], self.projection_width, tokens.dtype
            )
        )
        scores = self.attention(tokens).transpose(0, 1)
        membership = (
            torch.arange(batch_size, device=features.device)[:, None]
            == study_index[acquisition_index][None]
        )
        with torch.autocast(device_type=features.device.type, enabled=False):
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

    def forward(self, batch: dict) -> torch.Tensor:
        if batch["images"].shape[2] != self.prepared_channels:
            raise ValueError(
                "Prepared image channels differ from the configured input mode"
            )
        presence = batch["series_presence"]
        effective = self._effective_presence(presence)
        all_study = batch["series_study_index"]
        all_type = batch["series_type_index"]
        keep_series = effective[all_study, all_type]
        study_index = all_study[keep_series]
        type_index = all_type[keep_series]
        images = batch["images"][keep_series]
        slice_mask = batch["slice_mask"][keep_series]
        positions = batch["slice_positions"][keep_series]

        valid_images = images[slice_mask].to(torch.float32).div_(255.0)
        valid_images = self._augment_images(valid_images, slice_mask)
        valid_images = valid_images.sub(self.mean).div_(self.std)
        if self.channels_last:
            valid_images = valid_images.contiguous(memory_format=torch.channels_last)
        features = self.projection(self._encode_images(valid_images))
        if self.pooling == "attention_mil":
            return self._attention_pool(
                features,
                slice_mask,
                positions,
                study_index,
                type_index,
                presence.shape[0],
            )
        series_count, slice_count = slice_mask.shape
        tokens = features.new_zeros((series_count, slice_count, self.projection_width))
        tokens[slice_mask] = features
        type_tokens = self.type_embedding(type_index)
        tokens = tokens + sinusoidal_positions(
            positions, self.projection_width, tokens.dtype
        )
        tokens = tokens + type_tokens.unsqueeze(1)
        cls = self.cls_token.to(tokens.dtype).expand(
            series_count, -1, -1
        ) + type_tokens.unsqueeze(1)
        sequence = torch.cat([cls, tokens], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(
                    (series_count, 1), device=slice_mask.device, dtype=torch.bool
                ),
                ~slice_mask,
            ],
            dim=1,
        )
        series_vectors = self.transformer(sequence, src_key_padding_mask=padding_mask)[
            :, 0
        ]

        batch_size = presence.shape[0]
        pooled = series_vectors.new_full(
            (batch_size, self.projection_width), -torch.inf
        )
        scatter_index = study_index.unsqueeze(1).expand_as(series_vectors)
        pooled = pooled.scatter_reduce(
            0, scatter_index, series_vectors, reduce="amax", include_self=True
        )
        if not torch.isfinite(pooled).all():
            raise RuntimeError("Study pooling produced a non-finite representation")
        return self.head(pooled)
