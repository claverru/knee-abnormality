import re
from pathlib import Path

p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


sub(
    '    "tear_medial_front",\n    "tear_medial_back",\n    "tear_lateral_front",\n    "tear_lateral_back",\n',
    '    "tear_medial_anterior",\n    "tear_medial_body",\n    "tear_medial_posterior",\n'
    '    "tear_lateral_anterior",\n    "tear_lateral_body",\n    "tear_lateral_posterior",\n',
)
sub(
    "        forward = front - centre\n        upward = up - centre\n",
    """        forward = front - centre
        upward = up - centre
        meniscus = (
            len(self.channels)
            - len(SHARED_CHANNELS)
            + SHARED_CHANNELS.index("meniscus")
        )
        with torch.no_grad():
            looks = self.attention(tokens.float())[..., meniscus].float().flatten()
        spots = []
        for side in (points[:, 0] > 0, points[:, 0] <= 0):
            chosen = member & side[None]
            found = chosen.any(-1)
            open_ = chosen.clone()
            open_[:, 0] |= ~found
            weight = looks[None].masked_fill(~open_, -torch.inf).softmax(-1).square()
            spots.append(((weight / weight.sum(-1, keepdim=True)) @ points, found))
        (medial_spot, medial_found), (lateral_spot, lateral_found) = spots
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
""",
)
a = s.index("        shift = (\n            centre\n            + (self.frame_shift.float() + output[:, :count]) * study_pitch[:, None]\n        )\n")
b = a + len("        shift = (\n            centre\n            + (self.frame_shift.float() + output[:, :count]) * study_pitch[:, None]\n        )\n")
s = (
    s[:a]
    + """        rotation = rotation @ basis
        shift = (
            torch.where(both, (medial_spot + lateral_spot) / 2, 0.0)
            + self.frame_shift.float() * study_pitch[:, None]
        )
"""
    + s[b:]
)
sub(
    "        return shift, rotation, landmarks, sizes\n",
    "        return shift, rotation, torch.stack((medial_spot, lateral_spot), 1), sizes\n",
)
sub(
    "        core = inside(radius - distance, radial.clamp_min(1e-3))\n",
    "        core = inside(radius - distance, radial.clamp_min(1e-3))\n"
    "        polar = inside(\n"
    "            (front.abs() - medial.abs()) / math.sqrt(2),\n"
    "            (half[..., 0] + half[..., 1]) / 2,\n"
    "        )\n",
)
sub(
    """                medial_side * joint * ahead,
                medial_side * joint * (1 - ahead),
                lateral_side * joint * ahead,
                lateral_side * joint * (1 - ahead),
""",
    """                medial_side * joint * ahead * polar,
                medial_side * joint * (1 - polar),
                medial_side * joint * (1 - ahead) * polar,
                lateral_side * joint * ahead * polar,
                lateral_side * joint * (1 - polar),
                lateral_side * joint * (1 - ahead) * polar,
""",
)
p.write_text(s)

p = Path("src/knee_abnormality/training/report_attributes.py")
s = p.read_text()
sub(
    """            (*options, "body", "multiple"),
            ("not_stated",),
        )
        for side in ("medial", "lateral")
        for part, options in (
            ("front", ("anterior", "anterior_body")),
            ("back", ("posterior", "posterior_body")),
        )
""",
    """            (*options, "multiple"),
            ("not_stated",),
        )
        for side in ("medial", "lateral")
        for part, options in (
            ("anterior", ("anterior", "anterior_body")),
            ("body", ("body", "posterior_body", "anterior_body")),
            ("posterior", ("posterior", "posterior_body")),
        )
""",
)
p.write_text(s)

p = Path("scripts/tools/check_side_routing.py")
s = p.read_text()
sub(
    """    sign = -1.0 if left else 1.0
    return (
        model.pool_encoded(
            {
                **encoded,
                "left_probability": torch.full_like(encoded["left_probability"], left),
                "position": frame(model, encoded, sign),
            }
        ).float()
        - model.target_bias.float()
    )
""",
    """    sign = -1.0 if left else 1.0
    flip = torch.where(
        (encoded["left_probability"] > 0.5) == bool(left), 1.0, -1.0
    ).float()
    mirror = torch.stack((flip, torch.ones_like(flip), torch.ones_like(flip)), -1)
    moved = {
        **encoded,
        "left_probability": torch.full_like(encoded["left_probability"], left),
        "frame_shift": encoded["frame_shift"].float() * mirror,
        "frame_rotation": mirror[:, :, None]
        * encoded["frame_rotation"].float()
        * mirror[:, None, :],
    }
    return (
        model.pool_encoded({**moved, "position": frame(model, moved, sign)}).float()
        - model.target_bias.float()
    )
""",
)
sub(
    """            and float(prior.abs().max()) == 0.0
            and torch.allclose(
                frame(model, encoded, 1.0),
                raw_position,
                atol=1e-4,
            )
        )
""",
    """            and float(prior.abs().max()) == 0.0
        )
""",
)
p.write_text(s)

p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
for key, suffix in (
    (
        "learned_centre",
        "_centre_at_the_midpoint_of_the_two_meniscus_spots_side_axis_along_them_spots_are_squared_attention_centroids_of_the_shared_meniscus_channel_over_the_fitted_medial_and_lateral_halves_no_gradient_nod_and_sizes_regressed",
    ),
    ("algorithm", "_frame_on_meniscus_spots_eight_sections_three_tear_places_per_side"),
):
    pattern = re.compile(rf'^    "{key}": "(.*)",$', re.M)
    old = pattern.search(s).group(1)
    s = pattern.sub(f'    "{key}": "{old}{suffix}",', s)
p.write_text(s)
