import torch
from timm.layers import to_2tuple
from timm.models._efficientnet_blocks import (
    ConvBnAct,
    DepthwiseSeparableConv,
    EdgeResidual,
    InvertedResidual,
)
from timm.models.maxxvit import Downsample2d, MbConvBlock, TransformerBlock2d
from timm.models.resnet import BasicBlock, Bottleneck
from torch import nn

COATNET_BACKBONE = "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k"

BACKBONE_MAPPINGS = {
    "rexnet_200.nav_in1k": {
        "layout": "NCHW",
        "origin": [0.0, 0.0],
        "stride": [32.0, 32.0],
        "anchors": "convolution_centers",
    },
    "efficientnet_b3.ra2_in1k": {
        "layout": "NCHW",
        "origin": [0.0, 0.0],
        "stride": [32.0, 32.0],
        "anchors": "convolution_centers",
    },
    "efficientnet_b0.ra4_e3600_r224_in1k": {
        "layout": "NCHW",
        "origin": [0.0, 0.0],
        "stride": [32.0, 32.0],
        "anchors": "convolution_centers",
    },
    "efficientnet_b0.ra_in1k": {
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
    **{
        name: {
            "layout": "NCHW",
            "origin": [0.0, 0.0],
            "stride": [32.0, 32.0],
            "anchors": "convolution_centers",
        }
        for name in (
            "resnet50.tv_in1k",
            "resnet50.a1_in1k",
            "resnet18.a1_in1k",
            "resnet10t.c3_in1k",
        )
    },
}

SPATIAL_TOKEN_IDENTITY = {
    "algorithm": "all_valid_spatial_tokens_four_embeddings_v6",
    "projection": "shared_linear_C_to_256",
    "position_1d": "sinusoidal_middle_slice_acquisition_midpoint_mm",
    "position_2d": "augmented_input_anchor_align_corners_linear2_64_gelu_linear64_256",
    "geometry_dtype": "float32_outside_autocast",
    "plane_order": ["Axial", "Coronal", "Sagittal"],
    "contrast_order": [[0, 0], [1, 1]],
    "metadata_lookup": "configured_series_types_plane_and_metadata_pair",
    "visual_validity": "all_channel_inverse_augmented_anchors_inside_prepared_rectangles",
    "preparation": "rounded_resampling_half_pixel_crop_v1",
    "augmentation": "kornia_normalized_affine_align_corners_false_v1",
    "aggregation": "layernorm_target_attention_256_128_12",
    "meniscus_half_knee": "own_side_attention_excludes_opposite_tokens_beyond_10mm_training_shared_float32_smooth_max",
    "oa_shared_detector_regions": "trusted_middle_slice_patient_LPS_medial_axis_beyond_10mm_and_anterior_center_y_minus_patient_y",
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
        if not isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d)):
            raise TypeError(
                f"Unsupported spatial mapping layer: {type(layer).__name__}"
            )
        if isinstance(layer.padding, str) or (
            isinstance(layer, (nn.AvgPool2d, nn.MaxPool2d)) and layer.ceil_mode
        ):
            raise ValueError("Spatial anchors require fixed padding and floor pooling")
        kernel = to_2tuple(layer.kernel_size)
        padding = to_2tuple(layer.padding)
        step = to_2tuple(
            layer.stride if layer.stride is not None else layer.kernel_size
        )
        dilation = (
            to_2tuple(layer.dilation)
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d))
            else (1, 1)
        )
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


def _efficientnet_mapping(backbone: nn.Module) -> tuple[float, float, float, float]:
    convolutions = [backbone.conv_stem]
    for stage in backbone.blocks:
        for block in stage:
            if isinstance(block, DepthwiseSeparableConv):
                convolutions.extend((block.conv_dw, block.conv_pw))
            elif isinstance(block, InvertedResidual):
                convolutions.extend((block.conv_pw, block.conv_dw, block.conv_pwl))
            elif isinstance(block, ConvBnAct):
                convolutions.append(block.conv)
            elif isinstance(block, EdgeResidual):
                convolutions.extend((block.conv_exp, block.conv_pwl))
            else:
                raise TypeError(
                    f"Unsupported EfficientNet spatial block: {type(block).__name__}"
                )
    convolutions.append(backbone.conv_head)
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


def _resnet_mapping(backbone: nn.Module) -> tuple[float, float, float, float]:
    stem = (
        backbone.conv1
        if isinstance(backbone.conv1, nn.Sequential)
        else [backbone.conv1]
    )
    mapping = _spatial_mapping(
        [layer for layer in stem if isinstance(layer, nn.Conv2d)] + [backbone.maxpool]
    )
    for stage in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4):
        for block in stage:
            if isinstance(block, Bottleneck):
                layers = [block.conv1, block.aa, block.conv2, block.conv3]
            elif isinstance(block, BasicBlock):
                layers = [block.conv1, block.aa, block.conv2]
            else:
                raise TypeError(
                    f"Unsupported ResNet spatial block: {type(block).__name__}"
                )
            previous = mapping
            mapping = _spatial_mapping(layers, previous)
            if block.downsample is None:
                shortcut, expected = previous, mapping
            else:
                layers = list(block.downsample)
                expected = mapping
                if isinstance(layers[0], nn.AvgPool2d):
                    pool = layers.pop(0)
                    if (
                        to_2tuple(pool.kernel_size) != (2, 2)
                        or to_2tuple(pool.stride) != (2, 2)
                        or to_2tuple(pool.padding) != (0, 0)
                    ):
                        raise ValueError("Unsupported ResNet average-pool shortcut")
                    previous = (
                        previous[0] + previous[2] / 2,
                        previous[1] + previous[3] / 2,
                        previous[2] * 2,
                        previous[3] * 2,
                    )
                    expected = (
                        mapping[0] + previous[2] / 4,
                        mapping[1] + previous[3] / 4,
                        mapping[2],
                        mapping[3],
                    )
                shortcut = _spatial_mapping(
                    [layer for layer in layers if isinstance(layer, nn.Conv2d)],
                    previous,
                )
            if shortcut != expected:
                raise ValueError(
                    "ResNet residual geometry differs from verified anchors"
                )
    return mapping


_BACKBONE_MAPPING_FUNCTIONS = {
    "rexnet_200.nav_in1k": _rexnet_mapping,
    "efficientnet_b3.ra2_in1k": _efficientnet_mapping,
    "efficientnet_b0.ra4_e3600_r224_in1k": _efficientnet_mapping,
    "efficientnet_b0.ra_in1k": _efficientnet_mapping,
    COATNET_BACKBONE: _coatnet_mapping,
    "resnet50.tv_in1k": _resnet_mapping,
    "resnet50.a1_in1k": _resnet_mapping,
    "resnet18.a1_in1k": _resnet_mapping,
    "resnet10t.c3_in1k": _resnet_mapping,
}


def backbone_mapping(
    name: str, backbone: nn.Module
) -> tuple[float, float, float, float]:
    identity = backbone_mapping_identity(name)
    mapping = _BACKBONE_MAPPING_FUNCTIONS[name](backbone)
    if mapping != (*identity["origin"], *identity["stride"]):
        raise ValueError(
            f"Backbone geometry differs from verified spatial anchors: {name} {mapping}"
        )
    return mapping


def feature_geometry(
    valid_rects: torch.Tensor,
    inverse_affine: torch.Tensor,
    grid_shape: tuple[int, int],
    input_shape: tuple[int, int],
    mapping: tuple[float, float, float, float],
    source_channels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=valid_rects.device.type, enabled=False):
        height, width = grid_shape
        input_height, input_width = input_shape
        if min(input_height, input_width) <= 1:
            raise ValueError(
                "Spatial coordinates require input dimensions greater than one"
            )
        origin_y, origin_x, stride_y, stride_x = mapping
        y, x = torch.meshgrid(
            torch.arange(height, device=valid_rects.device, dtype=torch.float32)
            * stride_y
            + origin_y,
            torch.arange(width, device=valid_rects.device, dtype=torch.float32)
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
    return xy, visual_valid, prepared[..., :2]


class SpatialTokens(nn.Module):
    def __init__(self, width, series_types):
        super().__init__()
        self.position_2d = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, width)
        )
        self.plane = nn.Embedding(3, width)
        self.contrast = nn.Embedding(2, width)
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

    def forward(self, xy, types, dtype):
        with torch.autocast(device_type=xy.device.type, enabled=False):
            position_2d = self.position_2d(xy.float())
        return {
            "position_2d": position_2d.to(dtype),
            "plane": self.plane(self.plane_lookup[types]).to(dtype),
            "contrast": self.contrast(self.contrast_lookup[types]).to(dtype),
        }
