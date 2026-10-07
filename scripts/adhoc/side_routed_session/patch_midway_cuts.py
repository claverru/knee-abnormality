import re
from pathlib import Path


def sub(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90])
    return text.replace(old, new)


p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()
s = sub(s, "        self.frame_size = nn.Parameter(torch.zeros(len(FRAME_SIZES)))\n", "")
s = sub(
    s,
    "                len(POSITION_AXES) + len(FRAME_ROTATIONS) + len(FRAME_SIZES),\n",
    "                len(POSITION_AXES) + len(FRAME_ROTATIONS),\n",
)
s = sub(
    s,
    """        tokens: torch.Tensor,
        position: torch.Tensor,
        valid: torch.Tensor,
        token_study: torch.Tensor,
        pitch: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, ...]:
        hidden = self.frame_head(
""",
    """        tokens: torch.Tensor,
        own_tokens: torch.Tensor,
        position: torch.Tensor,
        valid: torch.Tensor,
        token_study: torch.Tensor,
        pitch: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, ...]:
        hidden = self.frame_head(
""",
)
s = sub(
    s,
    """        meniscus = (
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
""",
    """        shared = len(self.channels) - len(SHARED_CHANNELS)
        with torch.no_grad():
            shared_looks = self.attention(tokens.float()).float().flatten(0, 1)
            own_looks = self.attention(own_tokens.float()).float().flatten(0, 1)

        def spot(
            looks: torch.Tensor, region: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            chosen = member & region[None]
            found = chosen.any(-1)
            open_ = chosen.clone()
            open_[:, 0] |= ~found
            weight = looks[None].masked_fill(~open_, -torch.inf).softmax(-1).square()
            weight = weight / weight.sum(-1, keepdim=True)
            centre = weight @ points
            spread = (
                (weight @ points.square() - centre.square())
                .sum(-1)
                .clamp_min(1e-6)
                .sqrt()
            )
            return centre, spread, found

        meniscus_looks = shared_looks[:, shared + SHARED_CHANNELS.index("meniscus")]
        medial_spot, medial_spread, medial_found = spot(
            meniscus_looks, points[:, 0] > 0
        )
        lateral_spot, lateral_spread, lateral_found = spot(
            meniscus_looks, points[:, 0] <= 0
        )
        notch_spot, notch_spread, notch_found = spot(
            own_looks[:, self.channels.index("ACL")], torch.ones_like(points[:, 0] > 0)
        )
        kneecap_spot, kneecap_spread, kneecap_found = spot(
            shared_looks[:, shared + SHARED_CHANNELS.index("oa")],
            (points[:, 1] > 0) & (points[:, 2] > 0),
        )
        both = (medial_found & lateral_found)[:, None]
""",
)
s = sub(
    s,
    """        sizes = (
            self.frame_size_start + self.frame_size.float() + output[:, 2 * count :]
        ) * study_pitch[:, None]
        return shift, rotation, torch.stack((medial_spot, lateral_spot), 1), sizes
""",
    """        with torch.no_grad():
            medial_place, lateral_place, notch_place, kneecap_place = (
                torch.einsum("bc,bac->ba", place - shift, rotation)
                for place in (medial_spot, lateral_spot, notch_spot, kneecap_spot)
            )

            def radius(place: torch.Tensor) -> torch.Tensor:
                return torch.hypot(place[:, 0], place[:, 1])

            def midway(
                near: torch.Tensor,
                far: torch.Tensor,
                near_spread: torch.Tensor,
                far_spread: torch.Tensor,
            ) -> torch.Tensor:
                return near + (far - near) * near_spread / (near_spread + far_spread)

            meniscus_radius = (radius(medial_place) + radius(lateral_place)) / 2
            meniscus_height = (medial_place[:, 2] + lateral_place[:, 2]) / 2
            meniscus_spread = (medial_spread + lateral_spread) / 2
            core_radius = midway(
                radius(notch_place), meniscus_radius, notch_spread, meniscus_spread
            )
            ring_radius = torch.maximum(
                midway(
                    meniscus_radius,
                    radius(kneecap_place),
                    meniscus_spread,
                    kneecap_spread,
                ),
                core_radius,
            )
            top = torch.maximum(
                midway(
                    meniscus_height,
                    kneecap_place[:, 2],
                    meniscus_spread,
                    kneecap_spread,
                ),
                meniscus_height,
            )
            sizes = torch.where(
                (both[:, 0] & notch_found & kneecap_found)[:, None],
                torch.stack(
                    (top, 2 * meniscus_height - top, core_radius, ring_radius), -1
                ),
                self.frame_size_start * study_pitch[:, None],
            )
        return (
            shift,
            rotation,
            torch.stack((medial_spot, lateral_spot, notch_spot, kneecap_spot), 1),
            sizes,
        )
""",
)
s = sub(
    s,
    """            study_shift, study_rotation, landmarks, study_sizes = self.study_frames(
                tokens,
""",
    """            study_shift, study_rotation, landmarks, study_sizes = self.study_frames(
                tokens,
                self.mil_norm(
                    self._base_tokens(encoded).float()
                    + self.position_code(fitted_position, valid_position).float()
                ),
""",
)
p.write_text(s)

t = Path("src/knee_abnormality/modeling/spatial_tokens.py")
u = t.read_text()
pattern = re.compile(r'^    "algorithm": "(.*)",$', re.M)
old = pattern.search(u).group(1)
assert "midway_cuts" not in old
u = pattern.sub(
    f'    "algorithm": "{old}_midway_cuts_the_four_region_sizes_are_placed_per_knee_without_gradient_core_radius_between_the_notch_spot_and_the_meniscus_spots_ring_radius_between_the_meniscus_spots_and_the_kneecap_spot_upper_cut_between_the_meniscus_and_kneecap_heights_lower_cut_mirrored_about_the_meniscus_height_each_weighted_by_the_spots_spreads",',
    u,
)
t.write_text(u)
print("patched")
