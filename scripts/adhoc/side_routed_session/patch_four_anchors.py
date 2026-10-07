import re
from pathlib import Path

p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


sub(
    'LANDMARKS = ("centre", "front", "up")\n',
    'ANCHORS = ("medial_meniscus", "lateral_meniscus", "notch", "kneecap")\nKNEECAP_ELEVATION_START_DEG = 26.0\n',
)
sub(
    "                len(POSITION_AXES) + len(FRAME_ROTATIONS) + len(FRAME_SIZES),\n",
    "                len(FRAME_SIZES),\n",
)
sub(
    """            self.landmark_attention = nn.Linear(MIL_ATTENTION_WIDTH, len(LANDMARKS))
            nn.init.zeros_(self.landmark_attention.weight)
            nn.init.zeros_(self.landmark_attention.bias)
""",
    """        self.kneecap_elevation = nn.Parameter(
            torch.tensor(math.radians(KNEECAP_ELEVATION_START_DEG))
        )
""",
)
a = s.index("    def study_frames(\n")
b = s.index("    def frame(self, encoded: dict, medial_sign: torch.Tensor | float) -> torch.Tensor:\n")
s = (
    s[:a]
    + '''    def study_frames(
        self,
        tokens: torch.Tensor,
        own_tokens: torch.Tensor,
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
        shared = len(self.channels) - len(SHARED_CHANNELS)
        with torch.no_grad():
            shared_looks = self.attention(tokens.float()).float().flatten(0, 1)
            own_looks = self.attention(own_tokens.float()).float().flatten(0, 1)
        medial, front, height = points.unbind(-1)

        def spot(
            looks: torch.Tensor, region: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            chosen = member & region[None]
            found = chosen.any(-1)
            open_ = chosen.clone()
            open_[:, 0] |= ~found
            weight = looks[None].masked_fill(~open_, -torch.inf).softmax(-1).square()
            return (weight / weight.sum(-1, keepdim=True)) @ points, found

        meniscus = shared_looks[:, shared + SHARED_CHANNELS.index("meniscus")]
        medial_spot, medial_found = spot(meniscus, medial > 0)
        lateral_spot, lateral_found = spot(meniscus, medial <= 0)
        notch_spot, notch_found = spot(
            own_looks[:, self.channels.index("ACL")], torch.ones_like(medial > 0)
        )
        kneecap_spot, kneecap_found = spot(
            shared_looks[:, shared + SHARED_CHANNELS.index("oa")],
            (front > 0) & (height > 0),
        )
        both = (medial_found & lateral_found)[:, None]
        unit = torch.eye(len(POSITION_AXES), device=tokens.device)
        side_axis = nn.functional.normalize(
            torch.where(both, medial_spot - lateral_spot, unit[0]), dim=-1
        )
        level = nn.functional.normalize(
            unit[1] - side_axis[:, 1:2] * side_axis, dim=-1
        )
        middle = torch.where(both, (medial_spot + lateral_spot) / 2, 0.0)
        depth = torch.where(
            (notch_found[:, None] & both),
            ((notch_spot - middle) * level).sum(-1, keepdim=True) / 3,
            0.0,
        )
        centre = middle + depth * level
        towards = kneecap_spot - centre
        towards = towards - (towards * side_axis).sum(-1, keepdim=True) * side_axis
        towards = nn.functional.normalize(
            torch.where(kneecap_found[:, None] & both, towards, level), dim=-1
        )
        elevation = torch.where(
            kneecap_found & both[:, 0], self.kneecap_elevation.float(), 0.0
        )[:, None]
        front_axis = elevation.cos() * towards - elevation.sin() * torch.linalg.cross(
            side_axis, towards
        )
        rotation = self.frame_rotation() @ torch.stack(
            (side_axis, front_axis, torch.linalg.cross(side_axis, front_axis)), 1
        )
        shift = centre + self.frame_shift.float() * study_pitch[:, None]
        sizes = (
            self.frame_size_start + self.frame_size.float() + output
        ) * study_pitch[:, None]
        return (
            shift,
            rotation,
            torch.stack((medial_spot, lateral_spot, notch_spot, kneecap_spot), 1),
            sizes,
        )

'''
    + s[b:]
)
sub(
    """            study_shift, study_rotation, landmarks, study_sizes = self.study_frames(
                tokens,
""",
    """            study_shift, study_rotation, landmarks, study_sizes = self.study_frames(
                tokens,
                self.mil_norm(
                    self._base_tokens(encoded)
                    + self.position_code(fitted_position, valid_position).to(
                        features.dtype
                    )
                ),
""",
)
p.write_text(s)

t = Path("src/knee_abnormality/modeling/spatial_tokens.py")
u = t.read_text()
for key, suffix in (
    (
        "learned_centre",
        "_four_anchor_frame_notch_spot_of_the_acl_channel_shares_the_centre_depth_with_the_two_meniscus_spots_front_axis_points_at_the_kneecap_spot_of_the_oa_channel_over_the_fitted_front_up_region_lowered_by_one_shared_learned_elevation_from_26_degrees_no_per_study_nod_regression_no_landmarks",
    ),
    ("algorithm", "_four_anchor_frame"),
):
    pattern = re.compile(rf'^    "{key}": "(.*)",$', re.M)
    old = pattern.search(u).group(1)
    u = pattern.sub(f'    "{key}": "{old}{suffix}",', u)
t.write_text(u)
