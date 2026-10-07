import re
from pathlib import Path


def sub(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90])
    return text.replace(old, new)


p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()
s = sub(
    s,
    'FRAME_SIZES = ("band_top", "band_bottom", "core_radius", "ring_radius")\n'
    "FRAME_SIZE_START_PITCHES = (1.3, -0.9, 1.1, 2.2)\n",
    'FRAME_SIZES = ("cut_height", "core_radius")\n'
    "FRAME_SIZE_START_PITCHES = (0.45, 1.1)\n",
)
s = sub(
    s,
    "class KneeAbnormalityModel(nn.Module):\n",
    '''class BalancedSides(torch.autograd.Function):
    @staticmethod
    def forward(ctx, near: torch.Tensor, far: torch.Tensor):
        return near.view_as(near), far.view_as(far)

    @staticmethod
    def backward(ctx, near: torch.Tensor, far: torch.Tensor):
        near_norm, far_norm = near.norm(), far.norm()
        if near_norm == 0 or far_norm == 0:
            return near, far
        mean = (near_norm + far_norm) / 2
        return near * (mean / near_norm), far * (mean / far_norm)


class KneeAbnormalityModel(nn.Module):
''',
)
start = s.index("        top, bottom, core_radius, ring_radius = (\n")
stop = s.index("    def routing_weights(self, encoded: dict) -> torch.Tensor:\n")
s = (
    s[:start]
    + '''        cut_height, core_radius = encoded["frame_sizes"].float()[:, None].unbind(-1)
        step = torch.tensor(0.5, device=medial.device)

        def inside(distance: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
            soft = torch.sigmoid(distance / (2 * extent))
            return torch.heaviside(distance.detach(), step) + soft - soft.detach()

        def sides(
            distance: torch.Tensor, extent: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            soft = torch.sigmoid(distance / (2 * extent))
            near, far = BalancedSides.apply(soft, soft)
            hard = torch.heaviside(distance.detach(), step)
            return hard + near - near.detach(), 1 - hard - far + far.detach()

        beyond = [
            inside(
                front * math.cos(math.radians(22.5 + 45.0 * number))
                - medial * math.sin(math.radians(22.5 + 45.0 * number)),
                across,
            )
            for number in range(len(SECTORS))
        ]
        sector = dict(
            zip(
                SECTORS,
                (
                    beyond[number] * (1 - beyond[(number + 1) % len(SECTORS)])
                    for number in range(len(SECTORS))
                ),
                strict=True,
            )
        )
        medial_side, lateral_side = sides(medial, half[..., 0])
        upper, lower = sides(height - cut_height, half[..., 2])
        core, ringed = sides(
            core_radius - torch.hypot(medial, front).clamp_min(1e-6), across
        )
        medial_compartment = ringed * medial_side * lower
        lateral_compartment = ringed * lateral_side * lower
        kneecap = ringed * upper
        below = ringed.detach() * lower.detach()
        return torch.stack(
            (
                medial_compartment,
                lateral_compartment,
                medial_compartment,
                lateral_compartment,
                kneecap,
                core,
                medial_compartment,
                medial_compartment,
                kneecap,
                below * sector["anteromedial"],
                below * sector["medial"],
                below * sector["posteromedial"],
                below * sector["anterolateral"],
                below * sector["lateral"],
                below * sector["posterolateral"],
                kneecap * medial_side,
                kneecap * lateral_side,
            )
        )

'''
    + s[stop:]
)
assert "        step = torch.tensor(0.5, device=medial.device)\n\n        def inside" in s
p.write_text(s)

t = Path("src/knee_abnormality/modeling/spatial_tokens.py")
u = t.read_text()
pattern = re.compile(r'^    "algorithm": "(.*)",$', re.M)
old = pattern.search(u).group(1)
assert "held_cuts" not in old
u = pattern.sub(
    f'    "algorithm": "{old}_held_cuts_only_three_label_cuts_midline_one_height_cut_from_0p45_pitch_and_core_radius_medial_and_lateral_compartments_below_the_cut_outside_the_core_for_menisci_oa_mcl_and_bakers_kneecap_compartment_above_the_cut_for_pf_oa_effusion_and_synovitis_core_for_acl_alone_sectors_only_for_the_report_tear_places_without_gradient_to_the_three_cuts_gradient_of_the_two_sides_of_each_cut_rescaled_to_their_mean_norm",',
    u,
)
t.write_text(u)
print("patched")
