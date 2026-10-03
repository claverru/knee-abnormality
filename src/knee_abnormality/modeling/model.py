import math

import timm
import torch
from torch import nn

from knee_abnormality.config import SLICES_PER_INPUT, DataConfig, ModelConfig
from knee_abnormality.evaluation.metrics import COMPARTMENT_PAIRS
from knee_abnormality.modeling.augmentation import (
    ImageAugmentation,
    affine_matrices,
    apply_contrast_mixing,
    apply_feature_statistics_augmentation,
    apply_reflection,
    augmented_to_prepared,
    reflection_channels,
    reflection_inverse,
    reflection_positions,
    reflection_valid_rects,
    sample_augmentation_parameters,
    sample_contrast_mixing_parameters,
    sample_reflection_parameters,
)
from knee_abnormality.modeling.plane_readout import PlaneContrastReadout
from knee_abnormality.modeling.radimagenet import (
    RADIMAGENET_WEIGHTS,
    radimagenet_state_dict,
)
from knee_abnormality.modeling.spatial_tokens import (
    COATNET_BACKBONE,
    SpatialTokens,
    backbone_mapping,
    backbone_mapping_identity,
    feature_geometry,
)

PROJECTION_WIDTH = 256
MIL_ATTENTION_WIDTH = 128
COMPARTMENT_MIDLINE_BAND_MM = 10.0
REXNET_STRIDE_8_BLOCK_COUNT = 5


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
        targets: tuple[str, ...],
        load_pretrained: bool | None = None,
    ) -> None:
        super().__init__()
        target_count = len(targets)
        self.series_dropout = model.series_dropout
        self.plane_readout = model.plane_readout
        self.augmentation_settings = data
        self.feature_statistics_settings = model.feature_statistics_augmentation
        self.image_augmentation = ImageAugmentation(data)
        self.register_buffer(
            "mean", torch.tensor(data.mean).reshape(1, -1, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(data.std).reshape(1, -1, 1, 1), persistent=False
        )
        self.channels_last = model.channels_last
        self.repeated_middle_slice = model.slice_channels == "repeated_middle"
        self.checkpoint_backbone = model.checkpoint_backbone
        self.backbone_eval = model.backbone_eval
        pretrained = model.pretrained if load_pretrained is None else load_pretrained
        self.feature_mapping_identity = backbone_mapping_identity(model.backbone)
        if (
            model.pretrained_weights == "radimagenet"
            and model.backbone not in RADIMAGENET_WEIGHTS
        ):
            raise ValueError(
                f"RadImageNet weights require one of {sorted(RADIMAGENET_WEIGHTS)}"
            )
        backbone_kwargs = {}
        if model.backbone == COATNET_BACKBONE:
            if data.image_size % 32:
                raise ValueError(
                    "CoAtNet spatial tokens require image_size divisible by 32"
                )
            backbone_kwargs["img_size"] = data.image_size
        self.backbone = timm.create_model(
            model.backbone,
            pretrained=pretrained and model.pretrained_weights == "timm",
            in_chans=SLICES_PER_INPUT,
            num_classes=0,
            global_pool="",
            **backbone_kwargs,
        )
        if pretrained and model.pretrained_weights == "radimagenet":
            self.backbone.load_state_dict(
                radimagenet_state_dict(model.backbone), strict=True
            )
        self.feature_mapping = backbone_mapping(model.backbone, self.backbone)
        self.resnet_stages = hasattr(self.backbone, "layer2")
        if self.feature_statistics_settings.probability > 0 and not (
            self.resnet_stages
            or (
                hasattr(self.backbone, "stem")
                and hasattr(self.backbone, "features")
                and len(self.backbone.features) > REXNET_STRIDE_8_BLOCK_COUNT
            )
        ):
            raise ValueError(
                "Feature statistics augmentation requires a ReXNet or ResNet backbone "
                "with a stride-8 stage to resample"
            )
        backbone_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.inference_mode():
                spatial_output = self.backbone.forward_features(
                    torch.zeros(
                        1,
                        SLICES_PER_INPUT,
                        data.image_size,
                        data.image_size,
                    )
                )
        finally:
            self.backbone.train(backbone_training)
        if not isinstance(spatial_output, torch.Tensor) or spatial_output.ndim != 4:
            raise ValueError("Backbone must return NCHW spatial features")
        if spatial_output.shape[1] != self.backbone.num_features:
            raise ValueError("Backbone feature channels differ from the NCHW contract")
        if self.checkpoint_backbone:
            self.backbone.set_grad_checkpointing(True)
        self.projection = nn.Linear(spatial_output.shape[1], PROJECTION_WIDTH)
        if self.channels_last:
            self.backbone.to(memory_format=torch.channels_last)
        self.mil_norm = nn.LayerNorm(PROJECTION_WIDTH)
        compartment_signs = torch.zeros(target_count)
        compartment_families = torch.zeros(target_count, dtype=torch.long)
        for family, (medial, lateral) in enumerate(COMPARTMENT_PAIRS.values()):
            if medial not in targets or lateral not in targets:
                raise ValueError("Anatomical attention requires both compartment pairs")
            compartment_signs[targets.index(medial)] = 1
            compartment_signs[targets.index(lateral)] = -1
            compartment_families[targets.index(medial)] = family
            compartment_families[targets.index(lateral)] = family
        self.register_buffer("compartment_signs", compartment_signs, persistent=False)
        self.register_buffer(
            "compartment_families", compartment_families, persistent=False
        )
        meniscus_mask_signs = torch.zeros(target_count)
        for target in COMPARTMENT_PAIRS["meniscus"]:
            meniscus_mask_signs[targets.index(target)] = compartment_signs[
                targets.index(target)
            ]
        self.register_buffer(
            "meniscus_mask_signs", meniscus_mask_signs, persistent=False
        )
        self.anatomical_log_strength = (
            nn.Parameter(torch.zeros(len(COMPARTMENT_PAIRS)))
            if model.anatomical_bias
            else None
        )
        self.attention = nn.Sequential(
            nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH),
            nn.Tanh(),
            nn.Dropout(model.dropout),
            nn.Linear(MIL_ATTENTION_WIDTH, target_count),
        )
        self.head = nn.Linear(PROJECTION_WIDTH, target_count)
        if self.plane_readout:
            with torch.random.fork_rng(devices=[]):
                self.head = PlaneContrastReadout(self.head, len(data.series_types))
        self.spatial_tokens = SpatialTokens(PROJECTION_WIDTH, data.series_types)
        if data.contrast_mixing_augmentation.probability > 0:
            plane_lookup = self.spatial_tokens.plane_lookup.tolist()
            self.contrast_mixing_pairs = tuple(
                tuple(
                    index for index, plane in enumerate(plane_lookup) if plane == target
                )
                for target in range(3)
            )
            if any(len(pair) != 2 for pair in self.contrast_mixing_pairs):
                raise ValueError(
                    "Contrast mixing requires exactly two series types per plane"
                )
        with torch.random.fork_rng(devices=[]):
            self.half_knee_detector = nn.Linear(PROJECTION_WIDTH, 1)
        self.oa_indices = tuple(
            targets.index(name) for name in (*COMPARTMENT_PAIRS["oa"], "PF OA")
        )
        with torch.random.fork_rng(devices=[]):
            self.oa_detector = nn.Linear(PROJECTION_WIDTH, 1, bias=False)
            nn.init.xavier_uniform_(self.oa_detector.weight)
        self.oa_detector_scale = nn.Parameter(torch.zeros(3))
        with torch.random.fork_rng(devices=[]):
            self.detail_view_embedding = nn.Parameter(torch.zeros(PROJECTION_WIDTH))
        self.register_buffer(
            "detail_view",
            torch.tensor(
                [view.field_mm != data.reference_field_mm for _, view in data.views]
            ),
            persistent=False,
        )
        self.view_label_bias = (
            nn.Parameter(torch.zeros(len(data.views), target_count))
            if model.view_label_bias
            else None
        )
        self.train(self.training)

    def train(self, mode: bool = True) -> "KneeAbnormalityModel":
        super().train(mode)
        if self.backbone_eval:
            self.backbone.eval()
        return self

    def _encode_images(
        self,
        images: torch.Tensor,
        acquisition_index: torch.Tensor,
        acquisition_type: torch.Tensor,
    ) -> torch.Tensor:
        if self.checkpoint_backbone:
            self.backbone.set_grad_checkpointing(
                self.training and torch.is_grad_enabled()
            )
        if not self.training or self.feature_statistics_settings.probability == 0:
            return self.backbone.forward_features(images)
        features = self._backbone_to_stride_8(images)
        features, self.feature_statistics_activity = (
            apply_feature_statistics_augmentation(
                features,
                acquisition_index,
                acquisition_type,
                self.feature_statistics_settings,
            )
        )
        return self._backbone_from_stride_8(features)

    def _backbone_to_stride_8(self, images: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        if self.resnet_stages:
            stem = backbone.maxpool(backbone.act1(backbone.bn1(backbone.conv1(images))))
            return backbone.layer2(backbone.layer1(stem))
        return backbone.features[:REXNET_STRIDE_8_BLOCK_COUNT](backbone.stem(images))

    def _backbone_from_stride_8(self, features: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        if self.resnet_stages:
            return backbone.layer4(backbone.layer3(features))
        return backbone.features[REXNET_STRIDE_8_BLOCK_COUNT:](features)

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

    def _mix_contrast(
        self,
        images: torch.Tensor,
        effective: torch.Tensor,
        acquisition_index: torch.Tensor,
        token_study: torch.Tensor,
        token_type: torch.Tensor,
        token_plane: torch.Tensor,
        patient_maps: torch.Tensor,
        geometry_valid: torch.Tensor,
    ) -> torch.Tensor:
        settings = self.augmentation_settings.contrast_mixing_augmentation
        if not self.training or settings.probability == 0:
            return images
        pairs = torch.tensor(
            self.contrast_mixing_pairs, device=images.device, dtype=torch.long
        )
        parameters = sample_contrast_mixing_parameters(
            settings.probability, effective[:, pairs[:, 0]] & effective[:, pairs[:, 1]]
        )
        images, self.contrast_mixing_activity = apply_contrast_mixing(
            images,
            parameters,
            self.contrast_mixing_pairs,
            settings.target_weight_range,
            acquisition_index,
            token_study,
            token_type,
            token_plane,
            patient_maps,
            geometry_valid,
        )
        return images

    def _augment_images(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        valid_rects: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        settings = self.augmentation_settings
        if not self.training or not any(
            augmentation.probability > 0
            for augmentation in (
                settings.intensity_augmentation,
                settings.affine_augmentation,
                settings.noise_augmentation,
                settings.smoothing_augmentation,
                settings.resolution_augmentation,
                settings.gin_augmentation,
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
            images,
            series_index,
            parameters,
            matrices=matrices,
            valid_rects=valid_rects,
            inverse_affine=inverse[series_index],
        ), inverse[series_index]

    def _attention_pool(
        self,
        tokens: torch.Tensor,
        values: torch.Tensor,
        token_study: torch.Tensor,
        image_type: torch.Tensor,
        token_valid: torch.Tensor,
        batch_size: int,
        medial_axis: torch.Tensor,
        anterior_axis: torch.Tensor,
        trusted_geometry: torch.Tensor,
        token_view: torch.Tensor,
        diagnostic: bool = False,
    ) -> torch.Tensor:
        scores = self.attention(tokens).transpose(0, 1)
        if self.view_label_bias is not None:
            scores = scores.float() + self.view_label_bias.float()[
                token_view
            ].transpose(0, 1)
        if self.anatomical_log_strength is not None:
            strengths = self.anatomical_log_strength.float().exp()
            signs = self.compartment_signs * strengths[self.compartment_families]
            scaled_axis = medial_axis.float()[None] / 20.0
            scores = scores.float() + signs[:, None] * torch.tanh(scaled_axis)
        membership = (
            torch.arange(batch_size, device=tokens.device)[:, None] == token_study[None]
        ) & token_valid[None]
        if not diagnostic and not membership.any(-1).all():
            raise ValueError("A study has no visually valid spatial tokens")
        mask_signs = self.meniscus_mask_signs[:, None]
        own_side = (mask_signs == 0) | (
            mask_signs * medial_axis.float()[None] > -COMPARTMENT_MIDLINE_BAND_MM
        )
        restricted = membership[:, None] & own_side[None]
        allowed = torch.where(
            restricted.any(-1, keepdim=True), restricted, membership[:, None]
        )
        empty_study = ~membership.any(-1)
        if diagnostic:
            allowed = allowed.clone()
            allowed[..., 0] |= empty_study[:, None]
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            attention = scores.float().unsqueeze(0).expand(batch_size, -1, -1)
            attention = attention.masked_fill(~allowed, -torch.inf).softmax(-1)
            if diagnostic:
                attention = attention.masked_fill(empty_study[:, None, None], 0.0)
            else:
                self.attention_mass_view = torch.stack(
                    [
                        (attention * (token_view == view)[None, None]).sum(-1)
                        for view in range(len(self.detail_view))
                    ]
                ).detach()
            if self.plane_readout:
                evidence = self.head(values, image_type)
                logits = torch.einsum("btn,nt->bt", attention, evidence)
                logits = logits + self.head.bias.float()
            else:
                pooled = torch.einsum("btn,nf->btf", attention, values.float())
                logits = (pooled * self.head.weight.float()[None]).sum(
                    -1
                ) + self.head.bias.float()
            regions = (
                torch.stack(
                    (
                        medial_axis.float() > COMPARTMENT_MIDLINE_BAND_MM,
                        medial_axis.float() < -COMPARTMENT_MIDLINE_BAND_MM,
                        anterior_axis.float() > 0,
                    )
                )
                & trusted_geometry[None]
            )
            region_membership = membership[:, None] & regions[None]
            counts = region_membership.sum(-1)
            empty = counts == 0
            safe_membership = region_membership.clone()
            safe_membership[..., 0] |= empty
            evidence = self.oa_detector(tokens.float()).squeeze(-1)
            scores = evidence[None, None].masked_fill(~safe_membership, -torch.inf)
            pooled_evidence = torch.logsumexp(scores, -1) - counts.clamp_min(1).log()
            pooled_evidence = torch.where(empty, 0.0, pooled_evidence)
            correction = pooled_evidence * self.oa_detector_scale.float()[None]
            logits[:, list(self.oa_indices)] += correction
            if self.training:
                self.oa_detector_activity = (
                    counts.detach(),
                    (token_valid & ~trusted_geometry).sum().detach(),
                    correction.detach().abs().mean(0),
                )
        if not torch.isfinite(logits).all():
            raise RuntimeError("Study attention pooling produced non-finite logits")
        return logits

    def encode_batch(self, batch: dict) -> dict:
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
        valid_rects = batch["valid_rects"].flatten(0, 1)[flat_indices]
        patient_maps = batch["patient_maps"].flatten(0, 1)[flat_indices].float()
        geometry_valid = batch["geometry_valid"].flatten()[
            flat_indices
        ] & torch.isfinite(patient_maps[:, 1]).all((-1, -2))

        images = batch["images"].flatten(0, 1)
        source_valid_rects = valid_rects
        if self.repeated_middle_slice:
            images = images[:, 1:2]
            valid_rects = valid_rects[:, 1:2]
        valid_images = images[flat_indices].to(torch.float32).div_(255.0)
        acquisition_index = slice_mask.nonzero(as_tuple=True)[0]
        token_study = study_index[acquisition_index]
        token_type = type_index[acquisition_index]
        reflected = reflected_study[token_study]
        token_plane = self.spatial_tokens.plane_lookup[token_type]
        sagittal = token_plane == 2
        channels = reflection_channels(reflected, sagittal)
        if self.repeated_middle_slice:
            channels = channels[:, 1:2]
        if (
            self.training
            and self.augmentation_settings.reflection_augmentation.probability > 0
        ):
            valid_images = apply_reflection(valid_images, reflected, sagittal)
        positions = reflection_positions(positions, reflected, sagittal)
        valid_images = self._mix_contrast(
            valid_images,
            effective,
            acquisition_index,
            token_study,
            token_type,
            token_plane,
            patient_maps,
            geometry_valid,
        )
        valid_images, inverse_affine = self._augment_images(
            valid_images,
            slice_mask,
            reflection_valid_rects(
                valid_rects, reflected, sagittal, valid_images.shape[-1]
            ),
        )
        inverse_affine = reflection_inverse(
            inverse_affine, reflected, sagittal, valid_images.shape[-1]
        )
        if self.repeated_middle_slice:
            valid_images = valid_images.expand(-1, SLICES_PER_INPUT, -1, -1)
        valid_images = valid_images.sub(self.mean).div_(self.std)
        if self.channels_last:
            valid_images = valid_images.contiguous(memory_format=torch.channels_last)
        maps = self._encode_images(valid_images, acquisition_index, token_type)
        features = self.projection(maps.flatten(2).transpose(1, 2))
        xy, visual_valid, prepared_xy = feature_geometry(
            source_valid_rects,
            inverse_affine,
            maps.shape[-2:],
            valid_images.shape[-2:],
            self.feature_mapping,
            channels,
        )
        view_index = batch["view_index"][keep_series][acquisition_index]
        detail_view = self.detail_view[view_index]
        with torch.autocast(device_type=features.device.type, enabled=False):
            view_to_wide = batch["view_to_wide"].flatten(0, 1)[flat_indices].float()
            prepared_homogeneous = torch.cat(
                (prepared_xy.float(), torch.ones_like(prepared_xy[..., :1])), -1
            )
            wide_xy = torch.einsum("nij,nkj->nki", view_to_wide, prepared_homogeneous)[
                ..., :2
            ]
            wide_xy = 2 * wide_xy / (valid_images.shape[-1] - 1) - 1
            spatial_xy = torch.where(detail_view[:, None, None], wide_xy, xy[None])
            middle_map = patient_maps[:, 1]
            homogeneous = torch.cat(
                (prepared_xy.float(), torch.ones_like(prepared_xy[..., :1])), -1
            )
            patient_x = torch.einsum("nj,nkj->nk", middle_map[:, 0], homogeneous)
            patient_y = torch.einsum("nj,nkj->nk", middle_map[:, 1], homogeneous)
            side = batch["knee_side_sign"][keep_series][acquisition_index].float()
            center = batch["knee_center_x_mm"][keep_series][acquisition_index].float()
            medial_axis = torch.where(
                geometry_valid[:, None] & (side[:, None] != 0),
                side[:, None] * (patient_x - center[:, None]),
                0.0,
            )
            center_y = batch["knee_center_y_mm"][keep_series][acquisition_index].float()
            anterior_axis = torch.where(
                geometry_valid[:, None] & (side[:, None] != 0),
                center_y[:, None] - patient_y,
                0.0,
            )
            trusted_geometry = (
                geometry_valid[:, None] & (side[:, None] != 0) & visual_valid
            )
        return {
            "features": features,
            "position_1d": sinusoidal_positions(
                positions, PROJECTION_WIDTH, features.dtype
            ),
            **self.spatial_tokens(spatial_xy, token_type, features.dtype),
            "view_index": view_index,
            "detail_view": detail_view,
            "study": token_study,
            "image_plane": token_plane,
            "image_type": token_type,
            "visual_valid": visual_valid,
            "medial_axis": medial_axis,
            "anterior_axis": anterior_axis,
            "trusted_geometry": trusted_geometry,
            "batch_size": len(presence),
        }

    def _pooling_tokens(self, encoded: dict) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = (
            encoded["features"]
            + encoded["position_1d"][:, None]
            + encoded["position_2d"]
            + encoded["plane"][:, None]
            + encoded["contrast"][:, None]
            + encoded["detail_view"][:, None, None]
            * self.detail_view_embedding.to(encoded["features"].dtype)[None, None]
        ).flatten(0, 1)
        tokens = self.mil_norm(tokens)
        return tokens, tokens

    def half_knee_tokens(self, encoded: dict) -> torch.Tensor:
        return self._pooling_tokens(encoded)[0]

    def pool_encoded(self, encoded: dict, view: int | None = None) -> torch.Tensor:
        tokens, values = self._pooling_tokens(encoded)
        return self._attention_pool(
            tokens,
            values,
            encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            encoded["image_type"],
            (
                encoded["visual_valid"]
                if view is None
                else encoded["visual_valid"] & (encoded["view_index"] == view)[:, None]
            ).flatten(),
            encoded["batch_size"],
            encoded["medial_axis"].flatten(),
            encoded["anterior_axis"].flatten(),
            encoded["trusted_geometry"].flatten(),
            encoded["view_index"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            diagnostic=view is not None,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.pool_encoded(self.encode_batch(batch))
