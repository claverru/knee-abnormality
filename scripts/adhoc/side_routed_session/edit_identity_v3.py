import re
from pathlib import Path

p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
values = {
    "algorithm": "all_valid_spatial_tokens_position_free_knee_frame_attention_prior_side_routed_shared_channels_v10",
    "token": "projection_plus_plane_contrast_and_detail_view_embeddings_layernorm_no_position_code",
    "initialization_rng": "side_head_front_scorer_position_prior_and_half_detector_parameters_forked_removed_position_draw_count_and_target_sized_scorer_and_readout_draws_preserve_training_stream",
    "compartment_routing": "log_routing_weight_over_routing_temperature_added_to_shared_channel_attention_scores_medial_lateral_sigmoid_through_learned_lateral_offset_front_sigmoid_through_study_front_landmark_learned_widths_1_to_40mm_floor_1e-6_unrouted_below_one_token_own_channel_targets_unrouted",
    "learned_centre": "two_learned_offsets_lateral_front_40mm_tanh_zero_initialized_one_zero_initialized_linear_256_1_front_scorer_on_detached_normalized_tokens_softmax_score_minus_anterior_minus_front_offset_squared_over_2_times_15mm_squared_weighted_mean_axial_sagittal_fitted_center_visually_valid_tokens_front_offset_without_any",
    "position": "attention_score_prior_per_target_linear42_10_zero_initialized_on_sincos_7_wavelengths_8_to_512_of_medial_positive_by_side_probability_plus_lateral_offset_anterior_minus_front_landmark_superior_mm_own_channels_signed_medial_meniscus_and_oa_sides_one_map_each_on_depth_into_own_compartment_pf_on_absolute_medial_zero_without_fitted_center",
    "side_head": "detached_projected_visually_valid_axial_coronal_fitted_center_tokens_weighted_by_tanh_displayed_knee_x_over_learned_width_mean_linear_256_1_left_logit_flipped_for_mirrored_studies",
}
for key, value in values.items():
    pattern = re.compile(rf'^    "{key}": ".*",$', re.M)
    assert len(pattern.findall(s)) == 1, key
    s = pattern.sub(f'    "{key}": "{value}",', s)
anchor = '    "knee_side": '
assert s.count(anchor) == 1
s = s.replace(
    anchor,
    '    "routed_half_loss": "linear_256_1_detector_on_tokens_log_mean_exp_weighted_by_medial_and_lateral_routing_weights_bce_on_the_two_meniscus_targets_weight_0.075_training_only",\n'
    + anchor,
)
p.write_text(s)
