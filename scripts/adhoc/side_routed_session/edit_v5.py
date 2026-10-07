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
        (
            '''ROUTES = ("medial", "lateral", "front")
TARGET_ROUTES = {
    "Medial Meniscus": ("meniscus", "medial"),
    "Lateral Meniscus": ("meniscus", "lateral"),
    "Medial OA": ("oa", "medial"),
    "Lateral OA": ("oa", "lateral"),
    "PF OA": ("oa", "front"),
}
''',
            '''ROUTES = ("medial", "lateral", "front", "medial_back", "lateral_back")
TARGET_ROUTES = {
    "Medial Meniscus": ("meniscus", "medial"),
    "Lateral Meniscus": ("meniscus", "lateral"),
    "Medial OA": ("oa", "medial_back"),
    "Lateral OA": ("oa", "lateral_back"),
    "PF OA": ("oa", "front"),
}
''',
        ),
        (
            '''PRIOR_VARIANTS = ("medial", "lateral", "front")
''',
            '''PRIOR_VARIANTS = {
    "medial": 0,
    "lateral": 1,
    "front": 2,
    "medial_back": 0,
    "lateral_back": 1,
}
FRAME_ROTATIONS = ("nod", "roll", "turn")
FRAME_ROTATION_RANGE_RAD = math.pi / 2
''',
        ),
        (
            "                    PRIOR_VARIANTS.index(TARGET_ROUTES[target][1])\n",
            "                    PRIOR_VARIANTS[TARGET_ROUTES[target][1]]\n",
        ),
        (
            "        own_channels = len(self.channels) - len(SHARED_CHANNELS)\n",
            "        own_channels = len(self.channels) - len(SHARED_CHANNELS)\n"
            "        self.register_buffer(\n"
            '            "shared_channel",\n'
            "            torch.tensor([name in SHARED_CHANNELS for name in self.channels]),\n"
            "            persistent=False,\n"
            "        )\n",
        ),
        (
            "        self.centre_offset_logit = nn.Parameter(torch.zeros(len(CENTRE_OFFSETS)))\n"
            "        shortest, longest, count = POSITION_WAVELENGTHS_MM\n",
            "        self.centre_offset_logit = nn.Parameter(torch.zeros(len(CENTRE_OFFSETS)))\n"
            "        self.frame_rotation_logit = nn.Parameter(torch.zeros(len(FRAME_ROTATIONS)))\n"
            "        shortest, longest, count = POSITION_WAVELENGTHS_MM\n"
            "        self.register_buffer(\n"
            '            "code_frequencies",\n'
            "            2\n"
            "            * math.pi\n"
            "            / torch.logspace(\n"
            "                math.log10(shortest),\n"
            "                math.log10(longest),\n"
            "                PROJECTION_WIDTH // (2 * len(POSITION_AXES)),\n"
            "            ),\n"
            "            persistent=False,\n"
            "        )\n",
        ),
        (
            "    def _attention_pool(\n        self,\n        tokens: torch.Tensor,\n        token_study",
            "    def _attention_pool(\n        self,\n        tokens: torch.Tensor,\n        shared_tokens: torch.Tensor,\n        token_study",
        ),
        (
            "        scores = self.attention(tokens).transpose(0, 1)\n",
            "        scores = torch.where(\n"
            "            self.shared_channel[:, None],\n"
            "            self.attention(shared_tokens).transpose(0, 1),\n"
            "            self.attention(tokens).transpose(0, 1),\n"
            "        )\n",
        ),
        (
            "                evidence = self.head(tokens, image_type)[:, self.target_channel]\n",
            "                evidence = torch.where(\n"
            "                    self.shared_channel,\n"
            "                    self.head(shared_tokens, image_type),\n"
            "                    self.head(tokens, image_type),\n"
            "                )[:, self.target_channel]\n",
        ),
        (
            '                pooled = torch.einsum("btn,nf->btf", attention, tokens.float())\n',
            "                pooled = torch.where(\n"
            "                    self.shared_channel[self.target_channel][None, :, None],\n"
            '                    torch.einsum("btn,nf->btf", attention, shared_tokens.float()),\n'
            '                    torch.einsum("btn,nf->btf", attention, tokens.float()),\n'
            "                )\n",
        ),
        (
            "            lateral_offset, front_offset = self.centre_offsets().unbind()\n            sign = batch",
            "            sign = batch",
        ),
        (
            '''            position = torch.stack(
                (
                    (1 - 2 * left_probability)[:, None] * knee_x + lateral_offset,
                    anterior_axis - front_offset,
                    offset[..., 2],
                ),
                -1,
            )
''',
            '''            superior_axis = offset[..., 2]
            position = self.frame(
                knee_x,
                anterior_axis,
                superior_axis,
                (1 - 2 * left_probability)[:, None],
            )
''',
        ),
        (
            '            "anterior_axis": anterior_axis,\n',
            '            "anterior_axis": anterior_axis,\n            "superior_axis": superior_axis,\n',
        ),
        (
            '''    def study_tokens(self, encoded: dict) -> torch.Tensor:
        return self.mil_norm(self._base_tokens(encoded).flatten(0, 1))
''',
            '''    def study_tokens(self, encoded: dict, shared: bool = False) -> torch.Tensor:
        tokens = self._base_tokens(encoded)
        if not shared:
            tokens = tokens + self.position_code(
                encoded["position"], encoded["routable"]
            ).to(tokens.dtype)
        return self.mil_norm(tokens.flatten(0, 1))
''',
        ),
        (
            '''            self.study_tokens(encoded),
            encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            encoded["image_type"],''',
            '''            self.study_tokens(encoded),
            self.study_tokens(encoded, shared=True),
            encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            encoded["image_type"],''',
        ),
        (
            '''                encoded["anterior_axis"],
                encoded["left_probability"][:, None],
                encoded["routable"][:, None],
            ).flatten(1),
            self.position_priors''',
            '''                encoded["anterior_axis"],
                encoded["superior_axis"],
                encoded["left_probability"][:, None],
                encoded["routable"][:, None],
            ).flatten(1),
            self.position_priors''',
        ),
    ],
    [
        (
            "    def routing_weights(\n",
            "    def _attention_pool(\n",
            '''    def frame_rotation(self) -> torch.Tensor:
        angles = FRAME_ROTATION_RANGE_RAD * torch.tanh(self.frame_rotation_logit.float())
        zero = angles.new_zeros(())
        return torch.linalg.matrix_exp(
            torch.stack(
                (
                    torch.stack((zero, -angles[2], angles[1])),
                    torch.stack((angles[2], zero, -angles[0])),
                    torch.stack((-angles[1], angles[0], zero)),
                )
            )
        )

    def frame(
        self,
        knee_x: torch.Tensor,
        anterior_axis: torch.Tensor,
        superior_axis: torch.Tensor,
        medial_sign: torch.Tensor | float,
    ) -> torch.Tensor:
        lateral, front = self.centre_offsets().unbind()
        coordinates = torch.stack(
            (medial_sign * knee_x.float(), anterior_axis.float(), superior_axis.float()),
            -1,
        ) @ self.frame_rotation().transpose(0, 1)
        return coordinates + torch.stack((lateral, -front, torch.zeros_like(front)))

    def position_code(self, position: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=position.device.type, enabled=False):
            angles = position.detach().float().unsqueeze(-1) * self.code_frequencies
            code = torch.stack((angles.sin(), angles.cos()), -1).flatten(-3)
            code = nn.functional.pad(code, (0, PROJECTION_WIDTH - code.shape[-1]))
            return torch.where(valid[:, None, None], code, 0.0)

    def routing_weights(
        self,
        knee_x: torch.Tensor,
        anterior_axis: torch.Tensor,
        superior_axis: torch.Tensor,
        left_probability: torch.Tensor,
        routable: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=knee_x.device.type, enabled=False):
            width_medial, width_front, _ = self.routing_widths().unbind()
            left = left_probability.float()
            medial, front = (
                left * torch.sigmoid(as_left / width)
                + (1 - left) * torch.sigmoid(as_right / width)
                for as_left, as_right, width in zip(
                    self.frame(knee_x, anterior_axis, superior_axis, -1.0).unbind(-1),
                    self.frame(knee_x, anterior_axis, superior_axis, 1.0).unbind(-1),
                    (width_medial, width_front),
                    strict=False,
                )
            )
            weights = torch.stack(
                (
                    medial,
                    1 - medial,
                    front,
                    medial * (1 - front),
                    (1 - medial) * (1 - front),
                )
            )
            return torch.where(routable, weights, 0.5)

''',
        ),
    ],
)
edit(
    "src/knee_abnormality/training/lightning.py",
    [
        ("    CENTRE_OFFSETS,\n", "    CENTRE_OFFSETS,\n    FRAME_ROTATION_RANGE_RAD,\n    FRAME_ROTATIONS,\n"),
        (
            '            self._log_epoch_metric(f"centre_offset_{name}_mm", value, 1)\n',
            '            self._log_epoch_metric(f"centre_offset_{name}_mm", value, 1)\n'
            "        for name, value in zip(\n"
            "            FRAME_ROTATIONS,\n"
            "            torch.rad2deg(\n"
            "                FRAME_ROTATION_RANGE_RAD\n"
            "                * torch.tanh(self.model.frame_rotation_logit.detach().float())\n"
            "            ),\n"
            "            strict=True,\n"
            "        ):\n"
            '            self._log_epoch_metric(f"frame_rotation_{name}_deg", value, 1)\n',
        ),
    ],
)
p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
values = {
    "algorithm": "all_valid_spatial_tokens_rotatable_knee_frame_attention_prior_side_routed_shared_channels_v13",
    "token": "projection_plus_plane_contrast_and_detail_view_embeddings_layernorm_two_versions_own_channels_plus_fixed_sinusoid_of_detached_knee_frame_position_42_wavelengths_8_to_512_per_axis_zero_without_fitted_center_shared_channels_position_free",
    "compartment_routing": "log_routing_weight_over_routing_temperature_added_to_shared_channel_attention_scores_meniscus_medial_or_lateral_oa_sides_times_one_minus_front_pf_front_sigmoids_of_rotated_frame_medial_and_front_coordinates_learned_widths_1_to_40mm_floor_1e-6_unrouted_below_one_token_own_channel_targets_unrouted",
    "routing_geometry": "middle_slice_patient_LPS_minus_fitted_center_xyz_rotated_about_z_to_study_mean_folded_axial_coronal_row_direction_identity_without_one_then_medial_positive_frame_rotated_by_learned_rotation_vector_90deg_tanh_zero_initialized_series_without_fitted_center_half",
    "learned_centre": "two_learned_offsets_lateral_front_40mm_tanh_and_three_learned_frame_rotations_nod_roll_turn_zero_initialized_common_to_all_studies_no_per_study_part",
    "position": "attention_score_prior_per_target_linear42_10_zero_initialized_on_sincos_7_wavelengths_8_to_512_of_rotated_frame_medial_front_up_mm_own_channels_signed_medial_meniscus_and_oa_sides_one_map_each_on_depth_into_own_compartment_pf_on_absolute_medial_zero_without_fitted_center",
}
for key, value in values.items():
    pattern = re.compile(rf'^    "{key}": ".*",$', re.M)
    assert len(pattern.findall(s)) == 1, key
    s = pattern.sub(f'    "{key}": "{value}",', s)
p.write_text(s)
