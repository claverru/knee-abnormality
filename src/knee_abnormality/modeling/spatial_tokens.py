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
    **{
        f"{name}.layer3": {
            "layout": "NCHW",
            "origin": [0.0, 0.0],
            "stride": [16.0, 16.0],
            "anchors": "convolution_centers",
        }
        for name in ("resnet18.a1_in1k", "resnet50.a1_in1k")
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
    "oa_shared_detector_regions": "trusted_middle_slice_patient_LPS_medial_axis_beyond_10mm_and_anterior_center_y_minus_patient_y",
    "bone_routing": "contusion_and_fracture_routed_by_each_tokens_detached_bone_map_share_of_femur_tibia_kneecap_and_fibula_unrouted_below_one_token",
    "soft_tissue_routing": "meniscus_and_oa_targets_keep_their_stride32_read_and_add_a_logit_read_from_stride16_layer3_cells_own_layernorm_attention_256_128_2_and_zero_initialized_plane_contrast_readout_no_bias_no_position_prior_no_knee_level_term_seven_class_map_head_routing_weights_detached_menisci_core_at_one_over_32_of_knee_peak_grown_one_cell_weight_max_of_share_and_3x3_mean_kept_nearer_own_meniscus_centre_oa_cells_assigned_to_nearest_of_three_compartment_centres_own_cartilage_plus_assigned_femoral_cartilage_over_self_weighted_mean_share_capped_at_one_3x3_mean_max_with_bone_map_joint_or_kneecap_joint_share_of_parent_token_bone_zone_alone_without_anchors_unrouted_below_one_cell",
    "algorithm": "all_valid_spatial_tokens_rotatable_knee_frame_attention_prior_side_routed_shared_channels_knee_level_term_v14_hard_octants_soft_gradient_own_routes_study_frames_and_landmarks_eighteen_regions_overlap_routed_reports_fixed_frame_own_position_hard_cuts_soft_gradient_knee_level_term_on_every_routed_channel_side_labels_keep_their_half_of_the_core_region_table_from_all_token_attention_synovitis_unrouted_per_study_centre_nod_and_sizes_roll_and_turn_shared_kneecap_region_front_above_the_lower_cut_frame_on_detached_bone_share_centroid_10mm_gaussian_near_anatomy_plane_spot_orientation_and_no_mass_fallback_eight_sections_three_tear_places_per_side_ring_sector_division",
    "projection": "shared_linear_C_to_256",
    "meniscus_map": "independent_zero_initialized_linear_256_to_three_on_stride16_layer3_projection_plus_nearest_repeated_stride32_normalized_tokens_16px_target_footprint_four_by_four_target_samples_preserve_4px_quadrature_pitch_same_visual_validity_detached_probabilities_mean_of_four_per_stride32_token",
    "token": "projection_plus_plane_contrast_and_detail_view_embeddings_layernorm_two_versions_own_channels_plus_fixed_sinusoid_of_detached_knee_frame_position_42_wavelengths_8_to_512_per_axis_zero_without_fitted_center_shared_channels_position_free",
    "initialization_rng": "side_head_position_prior_up_width_and_knee_level_scale_parameters_forked_removed_position_draw_count_and_target_sized_scorer_and_readout_draws_preserve_training_stream_meniscus_stride16_projection_and_head_forked",
    "view_policy": "sagittal_fluid_second_64mm_view_reuses_first_view_exact_slice_selection_and_budget_excluded_from_first_look_shared_patient_shift_jitter_and_turn_configured_fixed_fields_and_slice_counts_all_six_views_shift_turn_scale_detached_single_eval_central224_first_look_multiplier_one_scale_sqrt_medial_bone_sd_times_front_femur_sd_over17.087_clamp0.87_to1.13_zero_mass_one_sagittal_coronal_incumbent_centres_and12deg_turn_axial_center_midpoint_femur_share5th_front_minus30mm_kneecap_share95th_front_frame_medial_zero_projected_medial_axis20deg_turn_shift18mm_sagittal_coronal8mm_axial_rounded_pixels_training_jitter3deg3mm_all_views_side_per_series_in_split_studies",
    "view_embedding": "zero_initialized_additive_detail_only",
    "view_label_bias": "optional_zero_initialized_additive_attention_score_per_view_and_channel",
    "geometry_dtype": "float32_outside_autocast",
    "plane_order": ["Axial", "Coronal", "Sagittal"],
    "contrast_order": [[0, 0], [1, 1]],
    "metadata_lookup": "configured_series_types_plane_and_metadata_pair",
    "visual_validity": "all_channel_inverse_augmented_anchors_inside_rotated_crop_bounds_and_original_canvas_rectangles_through_the_same_crop_transform",
    "preparation": "same_spacing_fixed72px_margin368px_uint8_canvas_all_six_views_normalized_with_incumbent_central_window_quantiles_exact_unscaled_central224_first_look_single_scaled_rotated_bicubic_resampling_all_slices_clamped0_to255_patient_maps_and_valid_rectangles_transformed_together_affine_and_reflection_after_cut",
    "augmentation": "kornia_normalized_affine_align_corners_false_v1_contrast_mixing_nearest_middle_slice_patient_plane_least_squares_bilinear_donor_on_posed_target_grid_reflection_conjugated_target_preserved_outside_donor_coverage_and_on_zero_signal_v1_sagittal_fluid_second_view_shares_primary_augmentation_draws_and_noise_delta_excluded_as_contrast_mixing_target_and_donor_and_from_feature_statistics_population_and_application",
    "aggregation": "sagittal_fluid_second_view_open_only_to_medial_and_lateral_meniscus_medial_and_lateral_oa_pf_oa_and_acl_in_target_and_knee_level_attention_layernorm_channel_attention_256_128_9_seven_own_channels_shared_meniscus_and_oa_twelve_target_biases_plus_zero_started_channel_scale_times_position_free_shared_evidence_pooled_by_unrouted_attention_with_same_scores_and_view_bias_no_position_prior",
    "compartment_routing": "log_routing_weight_over_routing_temperature_added_to_attention_scores_weights_are_hard_steps_with_the_gradient_of_a_sigmoid_of_signed_distance_over_the_projected_token_box_extent_eight_45_degree_sectors_centred_on_the_frame_axes_a_core_cylinder_an_outer_ring_and_three_heights_around_the_study_frame_per_target_region_table_from_the_attention_of_run_11_medial_and_lateral_mirrored_floor_1e-6_unrouted_below_one_token_contusion_and_fracture_unrouted",
    "routing_geometry": "middle_slice_patient_LPS_minus_fitted_center_xyz_rotated_about_z_to_study_mean_folded_axial_coronal_row_direction_identity_without_one_then_medial_positive_frame_rotated_by_learned_rotation_vector_90deg_tanh_zero_initialized_series_without_fitted_center_half",
    "learned_centre": "fitted_center_with_three_learned_frame_rotations_nod_roll_turn_about_it_zero_initialized_common_to_all_studies_no_offsets_no_per_study_part_plus_per_study_frame_head_on_detached_tokens_with_position_code_attention_pooled_256_128_to_three_shifts_in_token_pitches_and_three_rotation_logits_zero_initialized_shared_shift_bias_composed_with_a_frame_from_three_landmarks_centre_front_up_zero_initialized_attention_maps_256_128_3_landmark_minus_uniform_mean_shift_axes_from_per_axis_token_spread_plus_landmark_differences_gram_schmidt_plus_per_study_band_top_band_bottom_core_radius_in_token_pitches_from_0_minus_2_and_1_centre_in_plane_at_detached_femur_plus_tibia_share_weighted_token_centroid_times_gaussian_plane_height_10mm_normal_coordinate_from_fitted_plane_spot_midpoint_projected_to_plane_only_without_bone_mass_side_axis_along_the_two_meniscus_spots_spots_are_squared_attention_centroids_of_the_shared_meniscus_channel_over_the_fitted_medial_and_lateral_halves_no_gradient_nod_and_sizes_regressed_four_sizes_band_top_band_bottom_core_radius_ring_radius_from_1.3_minus_0.9_1.1_and_2.2_pitches",
    "position": "attention_score_prior_per_target_linear42_10_zero_initialized_on_sincos_7_wavelengths_8_to_512_of_rotated_frame_medial_front_up_mm_from_the_fitted_center_own_channels_signed_medial_meniscus_and_oa_sides_one_map_each_on_depth_into_own_compartment_pf_on_absolute_medial_zero_without_fitted_center",
    "knee_side": "explicit_coordinates_agree_metadata_else_sigmoid_detached_source_frame_side_head_logit_plus_4.33_coordinate_side_prior_for_coordinate_source_studies_mixture_bilateral_split_series_own_sign",
    "side_head": "detached_normalized_visually_valid_axial_coronal_fitted_center_tokens_linear_256_128_tanh_linear_128_1_per_token_mean_weighted_by_tanh_displayed_knee_x_over_learned_width_left_logit_flipped_for_mirrored_studies",
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


def _resnet_mapping(
    backbone: nn.Module, stages: int = 4
) -> tuple[float, float, float, float]:
    stem = (
        backbone.conv1
        if isinstance(backbone.conv1, nn.Sequential)
        else [backbone.conv1]
    )
    mapping = _spatial_mapping(
        [layer for layer in stem if isinstance(layer, nn.Conv2d)] + [backbone.maxpool]
    )
    for stage in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)[
        :stages
    ]:
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
    "resnet18.a1_in1k.layer3": lambda backbone: _resnet_mapping(backbone, 3),
    "resnet50.a1_in1k.layer3": lambda backbone: _resnet_mapping(backbone, 3),
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
    mapping: tuple[float, float, float, float],
    source_channels: torch.Tensor,
    canvas_rects: torch.Tensor | None = None,
    crop_affine: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=valid_rects.device.type, enabled=False):
        height, width = grid_shape
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
        if canvas_rects is not None:
            canvas_rects = canvas_rects.gather(
                1, source_channels[..., None].expand(-1, -1, 4)
            )
            canvas_xy = torch.einsum("nij,nkj->nki", crop_affine.float(), prepared)
            top, left, rows, columns = canvas_rects.float().unbind(-1)
            px, py = canvas_xy[:, None, :, 0], canvas_xy[:, None, :, 1]
            visual_valid = visual_valid & (
                (px >= left[..., None])
                & (px <= (left + columns - 1)[..., None])
                & (py >= top[..., None])
                & (py <= (top + rows - 1)[..., None])
            ).all(1)
    return visual_valid, prepared[..., :2]


class SpatialTokens(nn.Module):
    def __init__(self, width, series_types):
        super().__init__()
        removed_position_draws = 3 * 64 + 65 * width
        torch.empty(removed_position_draws).uniform_()
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

    def forward(self, types, dtype):
        return {
            "plane": self.plane(self.plane_lookup[types]).to(dtype),
            "contrast": self.contrast(self.contrast_lookup[types]).to(dtype),
        }
