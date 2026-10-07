import math

import timm
import torch
from torch import nn

from knee_abnormality.config import SLICES_PER_INPUT, DataConfig, ModelConfig
from knee_abnormality.modeling.augmentation import (
    ImageAugmentation,
    affine_matrices,
    apply_contrast_mixing,
    apply_feature_statistics_augmentation,
    apply_reflection,
    augmented_to_prepared,
    reflection_channels,
    reflection_inverse,
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
REXNET_STRIDE_8_BLOCK_COUNT = 5
SHARED_CHANNELS = ("meniscus", "oa")
ROUTES = (
    "meniscus_medial",
    "meniscus_lateral",
    "oa_medial",
    "oa_lateral",
    "kneecap",
    "core",
    "collateral_medial",
    "baker",
    "front_core",
    "tear_medial_anterior",
    "tear_medial_body",
    "tear_medial_posterior",
    "tear_lateral_anterior",
    "tear_lateral_body",
    "tear_lateral_posterior",
    "facet_medial",
    "facet_lateral",
)
TARGET_ROUTES = {
    "Medial Meniscus": ("meniscus", "meniscus_medial"),
    "Lateral Meniscus": ("meniscus", "meniscus_lateral"),
    "Medial OA": ("oa", "oa_medial"),
    "Lateral OA": ("oa", "oa_lateral"),
    "PF OA": ("oa", "kneecap"),
}
OWN_ROUTES = {
    "ACL": "core",
    "Synovitis": "front_core",
    "MCL": "collateral_medial",
    "Effusion": "kneecap",
    "Baker's": "baker",
}
FRAME_SIZES = ("band_top", "band_bottom", "core_radius", "ring_radius")
FRAME_SIZE_START_PITCHES = (1.3, -0.9, 1.1, 2.2)
SECTORS = (
    "anteromedial",
    "anterior",
    "anterolateral",
    "lateral",
    "posterolateral",
    "posterior",
    "posteromedial",
    "medial",
)
ROUTING_WIDTHS = ("side",)
ROUTING_WIDTH_RANGE_MM = (1.0, 40.0)
ROUTING_INITIAL_WIDTH_MM = 5.0
ROUTING_MINIMUM_WEIGHT = 1e-6
POSITION_AXES = ("medial", "front", "up")
LANDMARKS = ("centre", "front", "up")
POSITION_WAVELENGTHS_MM = (8.0, 512.0, 7)
SHARED_PRIORS = ("meniscus", "oa", "front")
PRIOR_VARIANTS = {
    "meniscus_medial": 0,
    "meniscus_lateral": 1,
    "kneecap": 2,
    "oa_medial": 0,
    "oa_lateral": 1,
}
FRAME_ROTATIONS = ("nod", "roll", "turn")
FRAME_ROTATION_RANGE_RAD = math.pi / 2
SIDE_COORDINATE_LOG_ODDS = 4.33


def channel_linear(layer: nn.Linear, rows: list[int], bias: bool) -> nn.Linear:
    selected = nn.utils.skip_init(nn.Linear, layer.in_features, len(rows), bias=bias)
    with torch.no_grad():
        selected.weight.copy_(layer.weight[rows])
        if bias:
            selected.bias.copy_(layer.bias[rows])
    return selected


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
        missing = sorted((set(TARGET_ROUTES) | set(OWN_ROUTES)) - set(targets))
        if missing:
            raise ValueError(f"Side-routed heads require the targets {missing}")
        target_channels = [
            TARGET_ROUTES.get(target, (target,))[0] for target in targets
        ]
        self.channels = (
            *(name for name in target_channels if name not in SHARED_CHANNELS),
            *SHARED_CHANNELS,
        )
        target_channel = [self.channels.index(name) for name in target_channels]
        target_route = [
            ROUTES.index(
                TARGET_ROUTES[target][1]
                if target in TARGET_ROUTES
                else OWN_ROUTES[target]
            )
            if target in TARGET_ROUTES or target in OWN_ROUTES
            else len(ROUTES)
            for target in targets
        ]
        channel_source = [
            target_channel.index(channel) for channel in range(len(self.channels))
        ]
        self.register_buffer(
            "target_channel", torch.tensor(target_channel), persistent=False
        )
        self.register_buffer(
            "target_route", torch.tensor(target_route), persistent=False
        )
        self.register_buffer(
            "routed_channel",
            torch.tensor(
                [
                    any(
                        route < len(ROUTES)
                        for member, route in zip(
                            target_channel, target_route, strict=True
                        )
                        if member == channel
                    )
                    for channel in range(len(self.channels))
                ]
            ),
            persistent=False,
        )
        own_channels = len(self.channels) - len(SHARED_CHANNELS)
        self.register_buffer(
            "shared_channel",
            torch.tensor([name in SHARED_CHANNELS for name in self.channels]),
            persistent=False,
        )
        self.register_buffer(
            "target_prior_column",
            torch.tensor(
                [
                    own_channels
                    + SHARED_PRIORS.index(
                        "front"
                        if TARGET_ROUTES[target][1] == "kneecap"
                        else TARGET_ROUTES[target][0]
                    )
                    if target in TARGET_ROUTES
                    else target_channel[index]
                    for index, target in enumerate(targets)
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "target_prior_variant",
            torch.tensor(
                [
                    PRIOR_VARIANTS[TARGET_ROUTES[target][1]]
                    if target in TARGET_ROUTES
                    else 0
                    for target in targets
                ]
            ),
            persistent=False,
        )
        self.register_buffer("routing_temperature", torch.ones(()))
        self.attention = nn.Sequential(
            nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH),
            nn.Tanh(),
            nn.Dropout(model.dropout),
            nn.Linear(MIL_ATTENTION_WIDTH, target_count),
        )
        self.head = nn.Linear(PROJECTION_WIDTH, target_count)
        self.target_bias = nn.Parameter(self.head.bias.detach().clone())
        self.attention[-1] = channel_linear(self.attention[-1], channel_source, True)
        self.head = channel_linear(self.head, channel_source, False)
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
            self.side_head = nn.Sequential(
                nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH),
                nn.Tanh(),
                nn.Linear(MIL_ATTENTION_WIDTH, 1),
            )
            self.position_prior = nn.Linear(
                2 * len(POSITION_AXES) * POSITION_WAVELENGTHS_MM[2],
                own_channels + len(SHARED_PRIORS),
            )
            nn.init.zeros_(self.position_prior.weight)
            nn.init.zeros_(self.position_prior.bias)
        self.frame_rotation_logit = nn.Parameter(torch.zeros(len(FRAME_ROTATIONS)))
        self.frame_shift = nn.Parameter(torch.zeros(len(POSITION_AXES)))
        self.frame_size = nn.Parameter(torch.zeros(len(FRAME_SIZES)))
        self.register_buffer(
            "frame_size_start",
            torch.tensor(FRAME_SIZE_START_PITCHES),
            persistent=False,
        )
        with torch.random.fork_rng(devices=[]):
            self.frame_head = nn.Sequential(
                nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH), nn.Tanh()
            )
            self.frame_attention = nn.Linear(MIL_ATTENTION_WIDTH, 1)
            self.frame_output = nn.Linear(
                MIL_ATTENTION_WIDTH,
                len(POSITION_AXES) + len(FRAME_ROTATIONS) + len(FRAME_SIZES),
                bias=False,
            )
            nn.init.zeros_(self.frame_output.weight)
            self.landmark_attention = nn.Linear(MIL_ATTENTION_WIDTH, len(LANDMARKS))
            nn.init.zeros_(self.landmark_attention.weight)
            nn.init.zeros_(self.landmark_attention.bias)
        shortest, longest, count = POSITION_WAVELENGTHS_MM
        self.register_buffer(
            "code_frequencies",
            2
            * math.pi
            / torch.logspace(
                math.log10(shortest),
                math.log10(longest),
                PROJECTION_WIDTH // (2 * len(POSITION_AXES)),
            ),
            persistent=False,
        )
        self.register_buffer(
            "position_frequencies",
            2
            * math.pi
            / torch.logspace(math.log10(shortest), math.log10(longest), count),
            persistent=False,
        )
        with torch.random.fork_rng(devices=[]):
            low, high = ROUTING_WIDTH_RANGE_MM
            initial_logit = math.log(
                (ROUTING_INITIAL_WIDTH_MM - low) / (high - ROUTING_INITIAL_WIDTH_MM)
            )
            self.routing_width_logit = nn.Parameter(
                torch.full((len(ROUTING_WIDTHS),), initial_logit)
            )
            self.knee_level_scale = nn.Parameter(torch.zeros(len(self.channels)))
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
            nn.Parameter(torch.zeros(len(data.views), len(self.channels)))
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

    def routing_widths(self) -> torch.Tensor:
        low, high = ROUTING_WIDTH_RANGE_MM
        return low + (high - low) * torch.sigmoid(self.routing_width_logit.float())

    def position_priors(
        self, position: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type=position.device.type, enabled=False):
            position = position.float()
            medial = position[..., :1]
            variants = torch.stack(
                [
                    torch.cat((axis, position[..., 1:]), -1)
                    for axis in (medial, -medial, medial.abs())
                ]
            )
            angles = variants.unsqueeze(-1) * self.position_frequencies
            features = torch.stack((angles.sin(), angles.cos()), -1).flatten(-3)
            priors = self.position_prior(features)[
                self.target_prior_variant, ..., self.target_prior_column
            ]
            return torch.where(valid[None, :, None], priors, 0.0)

    def side_probability(
        self,
        side_logit: torch.Tensor,
        side_present: torch.Tensor,
        mirrored: torch.Tensor,
        coordinates: torch.Tensor,
        left: torch.Tensor,
    ) -> torch.Tensor:
        source_logit = torch.where(
            side_present,
            torch.where(mirrored, -side_logit, side_logit).detach().float(),
            0.0,
        )
        prior = SIDE_COORDINATE_LOG_ODDS * torch.where(
            coordinates, torch.where(left, 1.0, -1.0), 0.0
        )
        return torch.sigmoid(source_logit + prior)

    def frame_rotation(self, offset: torch.Tensor | None = None) -> torch.Tensor:
        logit = self.frame_rotation_logit.float()
        if offset is not None:
            logit = logit + offset.float()
        nod, roll, turn = (FRAME_ROTATION_RANGE_RAD * torch.tanh(logit)).unbind(-1)
        zero = torch.zeros_like(nod)
        return torch.linalg.matrix_exp(
            torch.stack(
                (
                    torch.stack((zero, -turn, roll), -1),
                    torch.stack((turn, zero, -nod), -1),
                    torch.stack((-roll, nod, zero), -1),
                ),
                -2,
            )
        )

    @staticmethod
    def rotation_angles(rotation: torch.Tensor) -> torch.Tensor:
        rotation = rotation.float()
        vector = torch.stack(
            (
                rotation[..., 2, 1] - rotation[..., 1, 2],
                rotation[..., 0, 2] - rotation[..., 2, 0],
                rotation[..., 1, 0] - rotation[..., 0, 1],
            ),
            -1,
        )
        sine = vector.norm(dim=-1, keepdim=True) / 2
        cosine = (rotation.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) - 1) / 2
        angle = torch.atan2(sine, cosine)
        return torch.rad2deg(
            torch.where(sine > 1e-6, angle / (2 * sine.clamp_min(1e-6)), 0.5) * vector
        )

    def study_frames(
        self,
        tokens: torch.Tensor,
        position: torch.Tensor,
        valid: torch.Tensor,
        token_study: torch.Tensor,
        pitch: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, ...]:
        hidden = self.frame_head(
            tokens.detach().float()
            + self.position_code(position, valid.any(-1)).float()
        ).flatten(0, 1)
        points = position.detach().float().flatten(0, 1)
        member = (
            torch.arange(batch_size, device=tokens.device)[:, None]
            == token_study[:, None].expand_as(valid).flatten()[None]
        ) & valid.flatten()[None]
        present = member.any(-1)
        allowed = member.clone()
        allowed[:, 0] |= ~present
        attention = (
            self.frame_attention(hidden)
            .squeeze(-1)[None]
            .masked_fill(~allowed, -torch.inf)
            .softmax(-1)
        )
        output = self.frame_output(attention @ hidden) * present[:, None]
        slices = valid.any(-1).float()
        study_pitch = torch.zeros(batch_size, device=tokens.device).index_add(
            0, token_study, pitch * slices
        ) / torch.zeros(batch_size, device=tokens.device).index_add(
            0, token_study, slices
        ).clamp_min(1)
        count = len(POSITION_AXES)
        uniform = allowed.float() / allowed.sum(-1, keepdim=True)
        mean = uniform @ points
        spread = torch.where(
            present[:, None],
            (uniform @ points.square() - mean.square()).clamp_min(1e-6).sqrt(),
            1.0,
        )
        landmarks = (
            self.landmark_attention(hidden)
            .transpose(0, 1)[None]
            .masked_fill(~allowed[:, None], -torch.inf)
            .softmax(-1)
        ) @ points
        centre, front, up = (
            (landmarks - mean[:, None]) * present[:, None, None]
        ).unbind(1)
        forward = front - centre
        upward = up - centre
        shared = len(self.channels) - len(SHARED_CHANNELS)
        with torch.no_grad():
            shared_looks = self.attention(tokens.float()).float().flatten(0, 1)

            def spot(
                looks: torch.Tensor, region: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                chosen = member & region[None]
                found = chosen.any(-1)
                open_ = chosen.clone()
                open_[:, 0] |= ~found
                weight = looks[None].masked_fill(~open_, -torch.inf).softmax(-1).square()
                weight = weight / weight.sum(-1, keepdim=True)
                centre = weight @ points
                spread = (
                    (weight @ points.square() - centre.square())
                    .sum(-1)
                    .clamp_min(1e-6)
                    .sqrt()
                )
                return centre, spread, found

            meniscus_looks = shared_looks[:, shared + SHARED_CHANNELS.index("meniscus")]
            medial_spot, medial_spread, medial_found = spot(
                meniscus_looks, points[:, 0] > 0
            )
            lateral_spot, lateral_spread, lateral_found = spot(
                meniscus_looks, points[:, 0] <= 0
            )
            kneecap_spot, kneecap_spread, kneecap_found = spot(
                shared_looks[:, shared + SHARED_CHANNELS.index("oa")],
                (points[:, 1] > 0) & (points[:, 2] > 0),
            )
        both = (medial_found & lateral_found)[:, None]
        unit = torch.eye(count, device=tokens.device)
        side_axis = nn.functional.normalize(
            torch.where(both, medial_spot - lateral_spot, unit[0]), dim=-1
        )
        up_axis = nn.functional.normalize(
            unit[2] - side_axis[:, 2:3] * side_axis, dim=-1
        )
        basis = torch.stack(
            (side_axis, torch.linalg.cross(up_axis, side_axis), up_axis), 1
        )
        tilt = torch.atan2(
            upward[:, 1] - forward[:, 2],
            spread[:, 1] + forward[:, 1] + spread[:, 2] + upward[:, 2],
        )
        cosine, sine, zero, one = (
            tilt.cos(),
            tilt.sin(),
            torch.zeros_like(tilt),
            torch.ones_like(tilt),
        )
        nod_only = torch.tensor([1.0, 0.0, 0.0], device=tokens.device)
        rotation = self.frame_rotation(
            output[:, count : 2 * count] * nod_only
        ) @ torch.stack(
            (
                torch.stack((one, zero, zero), -1),
                torch.stack((zero, cosine, -sine), -1),
                torch.stack((zero, sine, cosine), -1),
            ),
            1,
        )
        rotation = rotation @ basis
        shift = (
            torch.where(both, (medial_spot + lateral_spot) / 2, 0.0)
            + self.frame_shift.float() * study_pitch[:, None]
        )
        sizes = (
            self.frame_size_start + self.frame_size.float() + output[:, 2 * count :]
        ) * study_pitch[:, None]
        with torch.no_grad():
            medial_place, lateral_place, kneecap_place = (
                torch.einsum("bc,bac->ba", place - shift, rotation)
                for place in (medial_spot, lateral_spot, kneecap_spot)
            )
            meniscus_height = (medial_place[:, 2] + lateral_place[:, 2]) / 2
            meniscus_spread = (medial_spread + lateral_spread) / 2
            top = torch.maximum(
                meniscus_height
                + (kneecap_place[:, 2] - meniscus_height)
                * meniscus_spread
                / (meniscus_spread + kneecap_spread),
                meniscus_height,
            )
            found = both[:, 0] & kneecap_found
            start = self.frame_size_start[:2] * study_pitch[:, None]
            heights = torch.where(
                found[:, None],
                torch.stack((top, 2 * meniscus_height - top), -1),
                start,
            )
            height_cuts_applied = found & (heights != start).any(-1)
        sizes = torch.cat((heights, sizes[:, 2:]), -1)
        return (
            shift,
            rotation,
            torch.stack((medial_spot, lateral_spot), 1),
            sizes,
            height_cuts_applied,
        )

    def frame(self, encoded: dict, medial_sign: torch.Tensor | float) -> torch.Tensor:
        return (
            torch.stack(
                (
                    medial_sign * encoded["knee_x"].float(),
                    encoded["anterior_axis"].float(),
                    encoded["superior_axis"].float(),
                ),
                -1,
            )
            - encoded["frame_shift"][:, None]
        ) @ encoded["frame_rotation"].transpose(-1, -2)

    def position_code(
        self, position: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type=position.device.type, enabled=False):
            angles = position.detach().float().unsqueeze(-1) * self.code_frequencies
            code = torch.stack((angles.sin(), angles.cos()), -1).flatten(-3)
            code = nn.functional.pad(code, (0, PROJECTION_WIDTH - code.shape[-1]))
            return torch.where(valid[:, None, None], code, 0.0)

    def _regions(self, encoded: dict, medial_sign: float) -> torch.Tensor:
        medial, front, height = self.frame(encoded, medial_sign).unbind(-1)
        mirror = torch.tensor([medial_sign, 1.0, 1.0], device=medial.device)
        half = (
            0.5
            * torch.einsum(
                "nac,nvc->nav",
                encoded["frame_rotation"].detach().float(),
                encoded["token_box"].float() * mirror,
            )
            .abs()
            .sum(-1)
        ).clamp_min(1e-3)[:, None]
        across = (half[..., 0] + half[..., 1]) / 2
        top, bottom, core_radius, ring_radius = (
            encoded["frame_sizes"].float()[:, None].unbind(-1)
        )
        step = torch.tensor(0.5, device=medial.device)

        def inside(distance: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
            soft = torch.sigmoid(distance / (2 * extent))
            return torch.heaviside(distance.detach(), step) + soft - soft.detach()

        beyond = [
            inside(
                front * math.cos(math.radians(22.5 + 45.0 * number))
                - medial * math.sin(math.radians(22.5 + 45.0 * number)),
                across,
            )
            for number in range(len(SECTORS))
        ]
        sector = dict(
            zip(
                SECTORS,
                (
                    beyond[number] * (1 - beyond[(number + 1) % len(SECTORS)])
                    for number in range(len(SECTORS))
                ),
                strict=True,
            )
        )
        medial_side = inside(medial, half[..., 0])
        lateral_side = 1 - medial_side
        upper = inside(height - top, half[..., 2])
        tibia = inside(bottom - height, half[..., 2])
        joint = (1 - upper - tibia).clamp_min(0)
        distance = torch.hypot(medial, front).clamp_min(1e-6)
        core = inside(core_radius - distance, across)
        ringed = 1 - core
        inner = ringed * (1 - inside(distance - ring_radius, across))
        medial_trio = (
            sector["anteromedial"] + sector["medial"] + sector["posteromedial"]
        )
        lateral_trio = (
            sector["anterolateral"] + sector["lateral"] + sector["posterolateral"]
        )
        front_trio = (
            sector["anteromedial"] + sector["anterior"] + sector["anterolateral"]
        )
        medial_compartment = (ringed * medial_trio + core * medial_side) * (1 - upper)
        lateral_compartment = (ringed * lateral_trio + core * lateral_side) * (
            1 - upper
        )
        kneecap = ringed * front_trio * (1 - tibia)
        return torch.stack(
            (
                medial_compartment,
                lateral_compartment,
                medial_compartment,
                lateral_compartment,
                kneecap,
                (
                    core
                    + inner * (sector["anterior"] + sector["posterior"] + lateral_trio)
                )
                * (1 - tibia),
                ringed * (sector["anteromedial"] + sector["medial"]),
                ringed
                * (sector["posterior"] + sector["posteromedial"] + sector["medial"]),
                (ringed * front_trio + core) * (1 - tibia),
                ringed * sector["anteromedial"] * joint,
                ringed * sector["medial"] * joint,
                ringed * sector["posteromedial"] * joint,
                ringed * sector["anterolateral"] * joint,
                ringed * sector["lateral"] * joint,
                ringed * sector["posterolateral"] * joint,
                kneecap * medial_side,
                kneecap * lateral_side,
            )
        )

    def routing_weights(self, encoded: dict) -> torch.Tensor:
        knee_x = encoded["knee_x"]
        with torch.autocast(device_type=knee_x.device.type, enabled=False):
            left = encoded["left_probability"][:, None].float()
            weights = left * self._regions(encoded, -1.0) + (1 - left) * self._regions(
                encoded, 1.0
            )
            return torch.where(encoded["routable"][:, None], weights, 0.5)

    def _attention_pool(
        self,
        tokens: torch.Tensor,
        shared_tokens: torch.Tensor,
        token_study: torch.Tensor,
        image_type: torch.Tensor,
        token_valid: torch.Tensor,
        batch_size: int,
        routing: torch.Tensor,
        prior: torch.Tensor,
        token_view: torch.Tensor,
        diagnostic: bool = False,
    ) -> torch.Tensor:
        scores = torch.where(
            self.shared_channel[:, None],
            self.attention(shared_tokens).transpose(0, 1),
            self.attention(tokens).transpose(0, 1),
        )
        if self.view_label_bias is not None:
            scores = scores.float() + self.view_label_bias.float()[
                token_view
            ].transpose(0, 1)
        membership = (
            torch.arange(batch_size, device=tokens.device)[:, None] == token_study[None]
        ) & token_valid[None]
        if not diagnostic and not membership.any(-1).all():
            raise ValueError("A study has no visually valid spatial tokens")
        empty_study = ~membership.any(-1)
        allowed = membership
        if diagnostic:
            allowed = membership.clone()
            allowed[:, 0] |= empty_study
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            routing = routing.float()
            weights = torch.cat((routing, torch.ones_like(routing[:1])))
            routed = (membership.float() @ weights.T >= 1)[:, self.target_route]
            log_weights = (
                weights.clamp_min(ROUTING_MINIMUM_WEIGHT).log()
                / self.routing_temperature
            )
            attention = (
                scores.float()[self.target_channel][None]
                + prior.float()[None]
                + torch.where(
                    routed[:, :, None], log_weights[self.target_route][None], 0.0
                )
            )
            attention = attention.masked_fill(~allowed[:, None], -torch.inf).softmax(-1)
            if diagnostic:
                attention = attention.masked_fill(empty_study[:, None, None], 0.0)
            else:
                self.attention_mass_view = torch.stack(
                    [
                        (attention * (token_view == view)[None, None]).sum(-1)
                        for view in range(len(self.detail_view))
                    ]
                ).detach()
                self.routing_mean = (
                    (routing * token_valid).sum(-1) / token_valid.sum().clamp_min(1)
                ).detach()
            if self.plane_readout:
                evidence = torch.where(
                    self.shared_channel,
                    self.head(shared_tokens, image_type),
                    self.head(tokens, image_type),
                )
                logits = torch.einsum(
                    "btn,nt->bt", attention, evidence[:, self.target_channel]
                )
            else:
                pooled = torch.where(
                    self.shared_channel[self.target_channel][None, :, None],
                    torch.einsum("btn,nf->btf", attention, shared_tokens.float()),
                    torch.einsum("btn,nf->btf", attention, tokens.float()),
                )
                logits = (
                    pooled * self.head.weight.float()[self.target_channel][None]
                ).sum(-1)
                evidence = torch.where(
                    self.shared_channel,
                    nn.functional.linear(
                        shared_tokens.float(), self.head.weight.float()
                    ),
                    nn.functional.linear(tokens.float(), self.head.weight.float()),
                )
            knee_attention = scores.float()[None]
            knee_attention = knee_attention.masked_fill(
                ~allowed[:, None], -torch.inf
            ).softmax(-1)
            if diagnostic:
                knee_attention = knee_attention.masked_fill(
                    empty_study[:, None, None], 0.0
                )
            knee_evidence = torch.einsum("bcn,nc->bc", knee_attention, evidence)
            channel_contribution = (
                knee_evidence
                * self.knee_level_scale.float()[None]
                * self.routed_channel
            )
            if not diagnostic:
                self.knee_level_logit_abs_mean = (
                    channel_contribution.detach().abs().mean()
                )
            logits = (
                logits
                + channel_contribution[:, self.target_channel]
                + self.target_bias.float()
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
        visual_valid, prepared_xy = feature_geometry(
            source_valid_rects,
            inverse_affine,
            maps.shape[-2:],
            self.feature_mapping,
            channels,
        )
        view_index = batch["view_index"][keep_series][acquisition_index]
        detail_view = self.detail_view[view_index]
        batch_size = len(presence)
        mirrored_study = batch.get("mirrored_study", reflected_study)
        encoded = {
            "features": features,
            **self.spatial_tokens(token_type, features.dtype),
            "detail_view": detail_view,
        }
        tokens = self.mil_norm(self._base_tokens(encoded))
        with torch.autocast(device_type=features.device.type, enabled=False):
            homogeneous = torch.cat(
                (prepared_xy.float(), torch.ones_like(prepared_xy[..., :1])), -1
            )
            middle = patient_maps[:, 1]
            patient_position = torch.einsum("nij,nkj->nki", middle, homogeneous)
            center = torch.stack(
                [
                    batch[key][keep_series][acquisition_index].float()
                    for key in (
                        "knee_center_x_mm",
                        "knee_center_y_mm",
                        "knee_center_z_mm",
                    )
                ],
                -1,
            )
            valid_position = (
                geometry_valid
                & torch.isfinite(center).all(-1)
                & torch.isfinite(patient_position).all((-1, -2))
            )
            offset = torch.where(
                valid_position[:, None, None], patient_position - center[:, None], 0.0
            ).detach()
            row = middle[:, :2, 0]
            row = torch.where(row[:, :1] < 0, -row, row)
            length = row.norm(dim=-1)
            aligned = valid_position & ~sagittal & (length > 1e-6)
            axis = torch.zeros((batch_size, 2), device=features.device).index_add(
                0, token_study[aligned], row[aligned] / length[aligned, None]
            )
            axis_length = axis.norm(dim=-1, keepdim=True)
            axis = torch.where(
                axis_length > 1e-6,
                axis / axis_length.clamp_min(1e-6),
                torch.tensor([1.0, 0.0], device=features.device),
            ).detach()
            cosine, sine = axis[token_study][:, None].unbind(-1)
            knee_x = offset[..., 0] * cosine + offset[..., 1] * sine
            anterior_axis = offset[..., 0] * sine - offset[..., 1] * cosine
            side_tokens = visual_valid & (valid_position & ~sagittal)[:, None]
            displayed_x = torch.where(
                mirrored_study[token_study][:, None], -knee_x, knee_x
            )
            side_weight = (
                torch.tanh(
                    displayed_x / self.routing_widths()[ROUTING_WIDTHS.index("side")]
                )
                * side_tokens
            )
            side_count = torch.zeros(batch_size, device=features.device).index_add(
                0, token_study, side_tokens.float().sum(-1)
            )
            side_scores = self.side_head(tokens.detach().float()).squeeze(-1)
            side_logit = torch.zeros(batch_size, device=features.device).index_add(
                0, token_study, (side_scores * side_weight).sum(-1)
            ) / side_count.clamp_min(1)
            side_present = side_count > 0
            head_left = self.side_probability(
                side_logit,
                side_present,
                mirrored_study,
                batch["knee_side_coordinates"],
                batch["knee_side_left"],
            )
            sign = batch["knee_side_sign"][keep_series][acquisition_index]
            certain = batch["knee_side_certain"][token_study]
            metadata_side = certain | (
                batch["knee_side_split"][token_study] & (sign != 0)
            )
            left_probability = torch.where(
                metadata_side,
                torch.where(
                    certain, batch["knee_side_left"][token_study], sign < 0
                ).float(),
                head_left[token_study],
            )
            superior_axis = offset[..., 2]
            sign = (1 - 2 * left_probability)[:, None]
            token_pitch = torch.where(
                valid_position,
                (patient_position[:, 1] - patient_position[:, 0]).norm(dim=-1),
                1.0,
            ).detach()
            steps = torch.stack(
                (
                    patient_position[:, 1] - patient_position[:, 0],
                    patient_position[:, maps.shape[-1]] - patient_position[:, 0],
                    torch.einsum(
                        "nij,nj->ni", patient_maps[:, 2] - middle, homogeneous[:, 0]
                    ),
                ),
                1,
            )
            steps = torch.where(
                valid_position[:, None, None] & torch.isfinite(steps), steps, 0.0
            )
            token_box = torch.stack(
                (
                    steps[..., 0] * cosine + steps[..., 1] * sine,
                    steps[..., 0] * sine - steps[..., 1] * cosine,
                    steps[..., 2],
                ),
                -1,
            ).detach()
            fitted_position = torch.stack(
                (sign * knee_x, anterior_axis, superior_axis), -1
            )
            (
                study_shift,
                study_rotation,
                landmarks,
                study_sizes,
                height_cuts_applied,
            ) = self.study_frames(
                tokens,
                fitted_position,
                visual_valid & valid_position[:, None],
                token_study,
                token_pitch,
                batch_size,
            )
            geometry = {
                "knee_x": knee_x,
                "anterior_axis": anterior_axis,
                "superior_axis": superior_axis,
                "frame_shift": study_shift[token_study],
                "frame_rotation": study_rotation[token_study],
                "frame_sizes": study_sizes[token_study],
                "token_box": token_box,
            }
            position = self.frame(geometry, sign)
        return {
            **encoded,
            "view_index": view_index,
            "study": token_study,
            "image_plane": token_plane,
            "image_type": token_type,
            "visual_valid": visual_valid,
            **geometry,
            "study_frame_shift": study_shift,
            "study_frame_rotation": study_rotation,
            "study_landmarks": landmarks,
            "study_frame_sizes": study_sizes,
            "height_cuts_from_spots_applied": height_cuts_applied,
            "position": position,
            "fitted_position": fitted_position,
            "left_probability": left_probability,
            "routable": valid_position,
            "token_pitch": token_pitch,
            "study_axis": axis,
            "study_left_probability": head_left,
            "side_logit": side_logit,
            "side_present": side_present,
            "mirrored_study": mirrored_study,
            "batch_size": batch_size,
        }

    def _base_tokens(self, encoded: dict) -> torch.Tensor:
        return (
            encoded["features"]
            + encoded["plane"][:, None]
            + encoded["contrast"][:, None]
            + encoded["detail_view"][:, None, None]
            * self.detail_view_embedding.to(encoded["features"].dtype)[None, None]
        )

    def study_tokens(self, encoded: dict, shared: bool = False) -> torch.Tensor:
        tokens = self._base_tokens(encoded)
        if not shared:
            tokens = tokens + self.position_code(
                encoded["fitted_position"], encoded["routable"]
            ).to(tokens.dtype)
        return self.mil_norm(tokens.flatten(0, 1))

    def pool_encoded(self, encoded: dict, view: int | None = None) -> torch.Tensor:
        return self._attention_pool(
            self.study_tokens(encoded),
            self.study_tokens(encoded, shared=True),
            encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            encoded["image_type"],
            (
                encoded["visual_valid"]
                if view is None
                else encoded["visual_valid"] & (encoded["view_index"] == view)[:, None]
            ).flatten(),
            encoded["batch_size"],
            self.routing_weights(encoded).flatten(1),
            torch.where(
                self.shared_channel[self.target_channel][:, None, None],
                self.position_priors(encoded["position"], encoded["routable"]),
                self.position_priors(encoded["fitted_position"], encoded["routable"]),
            ).flatten(1),
            encoded["view_index"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            diagnostic=view is not None,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.pool_encoded(self.encode_batch(batch))
