import re
from pathlib import Path


def edit(path, changes=(), ranges=()):
    p = Path(path)
    s = p.read_text()
    for start, end, new in ranges:
        a = s.index(start)
        b = s.index(end, a)
        s = s[:a] + new + s[b:]
    for old, new in changes:
        assert s.count(old) == 1, (path, old[:60], s.count(old))
        s = s.replace(old, new)
    p.write_text(s)


edit(
    "src/knee_abnormality/modeling/model.py",
    [
        ('LANDMARKS = ("front",)\nLANDMARK_PRIOR_MM = 15.0\n', ""),
        (
            "            self.landmark_scorers = nn.Linear(PROJECTION_WIDTH, len(LANDMARKS))\n"
            "            nn.init.zeros_(self.landmark_scorers.weight)\n"
            "            nn.init.zeros_(self.landmark_scorers.bias)\n",
            "",
        ),
        ("        routable: torch.Tensor,\n        centre_shift: torch.Tensor,\n", "        routable: torch.Tensor,\n"),
        (
            "            lateral = self.centre_offsets()[0]\n",
            "            lateral, front_offset = self.centre_offsets().unbind()\n",
        ),
        (
            "                (anterior_axis.float() - centre_shift.float()[..., 0]) / width_front\n",
            "                (anterior_axis.float() - front_offset) / width_front\n",
        ),
        (
            "                    anterior_axis - front_centre[token_study][:, None],\n",
            "                    anterior_axis - front_offset,\n",
        ),
        (
            '            "centre_shift": centre_shift[token_study],\n            "study_centre_shift": centre_shift,\n',
            "",
        ),
        (
            '                encoded["routable"][:, None],\n                encoded["centre_shift"][:, None],\n            ).flatten(1),\n',
            '                encoded["routable"][:, None],\n            ).flatten(1),\n',
        ),
    ],
    [
        ("def landmark(\n", "class KneeAbnormalityModel(nn.Module):", ""),
        (
            "            front_centre, front_weights = landmark(\n",
            "            sign = batch[\"knee_side_sign\"]",
            "",
        ),
        (
            '            "landmark_type_weight": torch.zeros(\n',
            '            "study_left_probability": head_left,\n',
            "",
        ),
    ],
)
edit(
    "src/knee_abnormality/training/lightning.py",
    [
        ("    LANDMARKS,\n", ""),
        ('        self.centre_shift_rows = {"": [], "val_": []}\n', ""),
        ('        self.centre_shift_rows[""].clear()\n', ""),
        ('        self._log_centre_shift("")\n', ""),
        ('        self.centre_shift_rows["val_"].clear()\n', ""),
        ('        self._log_centre_shift("val_")\n', ""),
    ],
    [
        ("    def _log_centre_shift(self, prefix: str) -> None:\n", "    def _side_step(\n", ""),
        (
            "        for name, weights in zip(\n            LANDMARKS,",
            '        axis = encoded["study_axis"]\n',
            "",
        ),
    ],
)
edit(
    "scripts/tools/diagnose_encoding_usage.py",
    [('    "centre_shift",\n', "")],
)
p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
values = {
    "algorithm": "all_valid_spatial_tokens_position_free_knee_frame_attention_prior_side_routed_shared_channels_v12",
    "initialization_rng": "side_head_and_position_prior_parameters_forked_removed_position_draw_count_and_target_sized_scorer_and_readout_draws_preserve_training_stream",
    "compartment_routing": "log_routing_weight_over_routing_temperature_added_to_shared_channel_attention_scores_medial_lateral_sigmoid_through_learned_lateral_offset_front_sigmoid_through_learned_front_offset_learned_widths_1_to_40mm_floor_1e-6_unrouted_below_one_token_own_channel_targets_unrouted",
    "learned_centre": "two_learned_offsets_lateral_front_40mm_tanh_zero_initialized_common_to_all_studies_no_per_study_part",
    "position": "attention_score_prior_per_target_linear42_10_zero_initialized_on_sincos_7_wavelengths_8_to_512_of_medial_positive_by_side_probability_plus_lateral_offset_anterior_minus_front_offset_superior_mm_own_channels_signed_medial_meniscus_and_oa_sides_one_map_each_on_depth_into_own_compartment_pf_on_absolute_medial_zero_without_fitted_center",
}
for key, value in values.items():
    pattern = re.compile(rf'^    "{key}": ".*",$', re.M)
    assert len(pattern.findall(s)) == 1, key
    s = pattern.sub(f'    "{key}": "{value}",', s)
p.write_text(s)
