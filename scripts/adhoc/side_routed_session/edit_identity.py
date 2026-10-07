import re
from pathlib import Path

p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
values = {
    "algorithm": "all_valid_spatial_tokens_knee_frame_position_side_routed_shared_channels_v9",
    "token": "projection_plus_plane_contrast_detail_view_embeddings_and_knee_frame_position_code_layernorm_two_views_per_token_shared_channels_unsigned_medial_axis_own_channels_signed",
    "initialization_rng": "side_head_landmark_scorer_and_position_parameters_forked_removed_position_draw_count_and_target_sized_scorer_and_readout_draws_preserve_training_stream",
    "compartment_routing": "log_routing_weight_added_to_shared_channel_attention_scores_medial_lateral_sigmoid_through_learned_lateral_offset_front_sigmoid_through_study_front_landmark_learned_widths_1_to_40mm_floor_1e-6_unrouted_below_one_token_own_channel_targets_unrouted",
    "routing_geometry": "middle_slice_patient_LPS_minus_fitted_center_xyz_rotated_about_z_to_study_mean_folded_axial_coronal_row_direction_identity_without_one_series_without_fitted_center_half",
    "learned_centre": "three_learned_offsets_lateral_front_up_40mm_tanh_zero_initialized_one_zero_initialized_linear_256_1_front_scorer_on_position_free_normalized_tokens_softmax_score_minus_anterior_minus_front_offset_squared_over_2_times_15mm_squared_weighted_mean_axial_sagittal_fitted_center_visually_valid_tokens_front_offset_without_any",
}
for key, value in values.items():
    pattern = re.compile(rf'^    "{key}": ".*",$', re.M)
    assert len(pattern.findall(s)) == 1, key
    s = pattern.sub(f'    "{key}": "{value}",', s)
anchor = '    "knee_side": '
assert s.count(anchor) == 1
s = s.replace(
    anchor,
    '    "position": "medial_positive_by_side_probability_plus_lateral_offset_anterior_minus_front_landmark_superior_minus_up_offset_mm_sincos_7_wavelengths_8_to_512_linear42_64_gelu_linear64_256_zero_initialized_output_learned_vector_without_fitted_center",\n'
    + anchor,
)
p.write_text(s)
