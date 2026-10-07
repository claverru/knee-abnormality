import re
from pathlib import Path

p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


def cut(start, end, new):
    global s
    assert s.count(start) == 1, start
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]


sub(
    'POSITION_AXES = ("medial", "front", "up")\n',
    'POSITION_AXES = ("medial", "front", "up")\nLANDMARKS = ("centre", "front", "up")\n',
)
sub(
    "            nn.init.zeros_(self.frame_output.weight)\n",
    "            nn.init.zeros_(self.frame_output.weight)\n"
    "            self.landmark_attention = nn.Linear(MIL_ATTENTION_WIDTH, len(LANDMARKS))\n"
    "            nn.init.zeros_(self.landmark_attention.weight)\n"
    "            nn.init.zeros_(self.landmark_attention.bias)\n",
)
cut(
    "    def study_frames(\n",
    "    def position_code(\n",
    '''    @staticmethod
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        axes = torch.eye(count, device=tokens.device)
        forward = nn.functional.normalize(
            spread[:, 1:2] * axes[1] + front - centre, dim=-1
        )
        upward = spread[:, 2:3] * axes[2] + up - centre
        upward = nn.functional.normalize(
            upward - (upward * forward).sum(-1, keepdim=True) * forward, dim=-1
        )
        rotation = self.frame_rotation(output[:, count:]) @ torch.stack(
            (torch.linalg.cross(forward, upward), forward, upward), 1
        )
        shift = (
            centre
            + (self.frame_shift.float() + output[:, :count]) * study_pitch[:, None]
        )
        return shift, rotation, landmarks

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

''',
)
sub("            study_shift, study_angle = self.study_frames(\n", "            study_shift, study_rotation, landmarks = self.study_frames(\n")
sub('                "frame_angle": study_angle[token_study],\n', '                "frame_rotation": study_rotation[token_study],\n')
sub('            "study_frame_angle": study_angle,\n', '            "study_frame_rotation": study_rotation,\n            "study_landmarks": landmarks,\n')
p.write_text(s)

p = Path("src/knee_abnormality/training/lightning.py")
s = p.read_text()
a = s.index("        study_angles = torch.rad2deg(\n")
b = s.index("        for number, name in enumerate(FRAME_ROTATIONS):\n", a)
s = s[:a] + '        study_angles = model.rotation_angles(encoded["study_frame_rotation"].detach())\n' + s[b:]
p.write_text(s)

p = Path("scripts/tools/diagnose_encoding_usage.py")
s = p.read_text()
sub('    "position",\n    "left_probability",\n', '    "superior_axis",\n    "token_pitch",\n    "position",\n    "frame_shift",\n    "frame_rotation",\n    "left_probability",\n')
p.write_text(s)

p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
for key, suffix in (
    ("learned_centre", "_composed_with_a_frame_from_three_landmarks_centre_front_up_zero_initialized_attention_maps_256_128_3_landmark_minus_uniform_mean_shift_axes_from_per_axis_token_spread_plus_landmark_differences_gram_schmidt"),
    ("algorithm", "_and_landmarks"),
):
    pattern = re.compile(rf'^    "{key}": "(.*)",$', re.M)
    old = pattern.search(s).group(1)
    s = pattern.sub(f'    "{key}": "{old}{suffix}",', s)
p.write_text(s)
