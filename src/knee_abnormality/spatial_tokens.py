import torch
from timm.layers import to_2tuple
from timm.models.maxxvit import Downsample2d, MbConvBlock, TransformerBlock2d
from torch import nn

COATNET_BACKBONE = "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k"

BACKBONE_MAPPINGS = {
    "rexnet_200.nav_in1k": {
        "layout": "NCHW",
        "origin": [0.0, 0.0],
        "stride": [32.0, 32.0],
        "anchors": "convolution_centers",
    },
    COATNET_BACKBONE: {
        "layout": "NCHW",
        "origin": [12.0, 12.0],
        "stride": [32.0, 32.0],
        "anchors": "mbconv_main_branch_and_attention_pool_centers",
    },
}

SPATIAL_TOKEN_IDENTITY = {
    "algorithm": "all_valid_spatial_tokens_five_embeddings_v1",
    "projection": "shared_linear_C_to_256",
    "position_1d": "sinusoidal_middle_slice_acquisition_midpoint_mm",
    "position_2d": "augmented_input_anchor_align_corners_linear2_64_gelu_linear64_256",
    "position_3d": "middle_slice_LPS_minus_accepted_center_div100_linear3_64_gelu_linear64_256",
    "geometry_dtype": "float32_outside_autocast",
    "plane_order": ["Axial", "Coronal", "Sagittal"],
    "contrast_order": [[0, 0], [1, 1]],
    "metadata_lookup": "configured_series_types_plane_and_metadata_pair",
    "unknown_3d": "shared_learned_256_vector",
    "dropout_3d": {
        "probability": 0.10,
        "unit": "eligible_study_per_training_forward",
        "replacement": "unknown_3d_no_rescaling",
        "rng": "torch_seeded_training_rng",
    },
    "visual_validity": "all_channel_inverse_augmented_anchors_inside_prepared_rectangles",
    "availability_3d": "native_poses_accepted_series_accepted_center_finite_xyz_independent_visual_validity",
    "preparation": "rounded_resampling_half_pixel_crop_v1",
    "augmentation": "kornia_normalized_affine_align_corners_false_v1",
    "aggregation": "layernorm_target_attention_256_128_12",
}


def backbone_mapping_identity(name: str) -> dict:
    if name not in BACKBONE_MAPPINGS:
        raise ValueError(f"Unsupported spatial feature mapping for backbone: {name}")
    return {"backbone": name, **BACKBONE_MAPPINGS[name]}


def _spatial_mapping(
    layers: list[nn.Module],
    mapping: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> tuple[float, float, float, float]:
    origin = list(mapping[:2])
    stride = list(mapping[2:])
    for layer in layers:
        if isinstance(layer, nn.Identity):
            continue
        if not isinstance(layer, (nn.Conv2d, nn.AvgPool2d)):
            raise TypeError(
                f"Unsupported spatial mapping layer: {type(layer).__name__}"
            )
        if isinstance(layer.padding, str) or (
            isinstance(layer, nn.AvgPool2d) and layer.ceil_mode
        ):
            raise ValueError("Spatial anchors require fixed padding and floor pooling")
        kernel = to_2tuple(layer.kernel_size)
        padding = to_2tuple(layer.padding)
        step = to_2tuple(
            layer.stride if layer.stride is not None else layer.kernel_size
        )
        dilation = to_2tuple(layer.dilation) if isinstance(layer, nn.Conv2d) else (1, 1)
        for axis in range(2):
            origin[axis] += (
                (kernel[axis] - 1) * dilation[axis] / 2 - padding[axis]
            ) * stride[axis]
            stride[axis] *= step[axis]
    return origin[0], origin[1], stride[0], stride[1]


def _rexnet_mapping(backbone: nn.Module) -> tuple[float, float, float, float]:
    convolutions = [backbone.stem.conv]
    for block in backbone.features:
        if hasattr(block, "conv_dw"):
            if block.conv_exp is not None:
                convolutions.append(block.conv_exp.conv)
            convolutions.extend((block.conv_dw.conv, block.conv_pwl.conv))
        elif hasattr(block, "conv"):
            convolutions.append(block.conv)
        else:
            raise ValueError("Unsupported ReXNet spatial feature block")
    return _spatial_mapping(convolutions)


def _coatnet_mapping(backbone: nn.Module) -> tuple[float, float, float, float]:
    mapping = _spatial_mapping([backbone.stem.conv1, backbone.stem.conv2])
    for stage in backbone.stages:
        for block in stage.blocks:
            previous = mapping
            if isinstance(block, MbConvBlock):
                if not isinstance(block.down, nn.Identity):
                    raise TypeError(
                        "CoAtNet spatial anchors require depthwise downsampling"
                    )
                layers = [block.conv1_1x1, block.conv2_kxk, block.conv3_1x1]
            elif isinstance(block, TransformerBlock2d):
                layers = (
                    [block.norm1.down.pool, block.norm1.down.expand]
                    if isinstance(block.norm1, nn.Sequential)
                    else []
                )
            else:
                raise TypeError(
                    f"Unsupported CoAtNet spatial block: {type(block).__name__}"
                )
            mapping = _spatial_mapping(layers, previous)
            shortcut = block.shortcut
            shortcut_layers = (
                [shortcut.pool, shortcut.expand]
                if isinstance(shortcut, Downsample2d)
                else [shortcut]
            )
            shortcut_mapping = _spatial_mapping(shortcut_layers, previous)
            expected_shortcut = mapping
            if isinstance(block, MbConvBlock) and isinstance(shortcut, Downsample2d):
                expected_shortcut = (
                    mapping[0] + previous[2] / 2,
                    mapping[1] + previous[3] / 2,
                    mapping[2],
                    mapping[3],
                )
            if shortcut_mapping != expected_shortcut:
                raise ValueError(
                    "CoAtNet residual geometry differs from verified anchors"
                )
    return mapping


def backbone_mapping(
    name: str, backbone: nn.Module
) -> tuple[float, float, float, float]:
    identity = backbone_mapping_identity(name)
    mapping = (
        _coatnet_mapping(backbone)
        if name == COATNET_BACKBONE
        else _rexnet_mapping(backbone)
    )
    if mapping != (*identity["origin"], *identity["stride"]):
        raise ValueError(
            f"Backbone geometry differs from verified spatial anchors: {name} {mapping}"
        )
    return mapping


def feature_geometry(
    patient_maps: torch.Tensor,
    valid_rects: torch.Tensor,
    geometry_valid: torch.Tensor,
    inverse_affine: torch.Tensor,
    grid_shape: tuple[int, int],
    input_shape: tuple[int, int],
    mapping: tuple[float, float, float, float],
    centers: torch.Tensor,
    center_valid: torch.Tensor,
    reflected: torch.Tensor,
    source_channels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=patient_maps.device.type, enabled=False):
        height, width = grid_shape
        input_height, input_width = input_shape
        if min(input_height, input_width) <= 1:
            raise ValueError(
                "Spatial coordinates require input dimensions greater than one"
            )
        origin_y, origin_x, stride_y, stride_x = mapping
        y, x = torch.meshgrid(
            torch.arange(height, device=patient_maps.device, dtype=torch.float32)
            * stride_y
            + origin_y,
            torch.arange(width, device=patient_maps.device, dtype=torch.float32)
            * stride_x
            + origin_x,
            indexing="ij",
        )
        anchors = torch.stack(
            (x.flatten(), y.flatten(), torch.ones_like(x.flatten())), -1
        )
        xy = torch.stack(
            (
                2 * anchors[:, 0] / (input_width - 1) - 1,
                2 * anchors[:, 1] / (input_height - 1) - 1,
            ),
            -1,
        )
        prepared = torch.einsum("nij,kj->nki", inverse_affine.float(), anchors)
        valid_rects = valid_rects.gather(
            1, source_channels[..., None].expand(-1, -1, 4)
        )
        top, left, rows, columns = valid_rects.float().unbind(-1)
        px = prepared[:, None, :, 0]
        py = prepared[:, None, :, 1]
        visual_valid = (
            (px >= left[..., None])
            & (px <= (left + columns - 1)[..., None])
            & (py >= top[..., None])
            & (py <= (top + rows - 1)[..., None])
        ).all(1)
        middle = patient_maps.shape[1] // 2
        points = torch.einsum("nij,nkj->nki", patient_maps[:, middle].float(), prepared)
        xyz = (points - centers[:, None].float()) / 100.0
        xyz = xyz * torch.where(
            reflected[:, None, None], xyz.new_tensor([-1, 1, 1]), xyz.new_ones(3)
        )
        available = (
            geometry_valid[:, None]
            & center_valid[:, None]
            & torch.isfinite(xyz).all(-1)
        )
    return xy, xyz, visual_valid, available


class SpatialTokens(nn.Module):
    def __init__(self, width, series_types):
        super().__init__()
        self.position_2d = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, width)
        )
        self.position_3d = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(), nn.Linear(64, width)
        )
        self.plane = nn.Embedding(3, width)
        self.contrast = nn.Embedding(2, width)
        self.unknown_3d = nn.Parameter(torch.zeros(width))
        planes = SPATIAL_TOKEN_IDENTITY["plane_order"]
        contrasts = [tuple(pair) for pair in SPATIAL_TOKEN_IDENTITY["contrast_order"]]
        self.register_buffer(
            "plane_lookup",
            torch.tensor([planes.index(kind.plane) for kind in series_types]),
        )
        self.register_buffer(
            "contrast_lookup",
            torch.tensor(
                [
                    contrasts.index((kind.fluid_sensitive, kind.fat_suppression))
                    for kind in series_types
                ]
            ),
        )
        nn.init.normal_(self.plane.weight, std=0.02)
        nn.init.normal_(self.contrast.weight, std=0.02)
        nn.init.normal_(self.unknown_3d, std=0.02)

    def forward(self, xy, xyz, available, types, study, batch_size, dtype):
        retained = available
        if self.training:
            eligible = torch.zeros(batch_size, device=study.device, dtype=torch.bool)
            eligible[study[available.any(-1)]] = True
            dropped = torch.zeros_like(eligible)
            dropped[eligible] = (
                torch.rand(int(eligible.sum()), device=study.device) < 0.10
            )
            retained = available & ~dropped[study, None]
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            position_2d = self.position_2d(xy.float())
            position_3d = self.unknown_3d.float().expand(*available.shape, -1).clone()
            position_3d[retained] = self.position_3d(xyz[retained].float())
        return {
            "position_2d": position_2d.to(dtype),
            "position_3d": position_3d.to(dtype),
            "plane": self.plane(self.plane_lookup[types]).to(dtype),
            "contrast": self.contrast(self.contrast_lookup[types]).to(dtype),
        }
