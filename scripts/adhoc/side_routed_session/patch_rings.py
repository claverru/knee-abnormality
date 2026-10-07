import re
from pathlib import Path

p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


sub('    "femur",\n', '    "front_core",\n')
sub('    "ACL": "core",\n', '    "ACL": "core",\n    "Synovitis": "front_core",\n')
sub(
    'FRAME_SIZES = ("band_top", "band_bottom", "core_radius")\nFRAME_SIZE_START_PITCHES = (0.0, -2.0, 1.0)\n',
    'FRAME_SIZES = ("band_top", "band_bottom", "core_radius", "ring_radius")\n'
    "FRAME_SIZE_START_PITCHES = (1.3, -0.9, 1.1, 2.2)\n"
    'SECTORS = (\n    "anteromedial",\n    "anterior",\n    "anterolateral",\n    "lateral",\n'
    '    "posterolateral",\n    "posterior",\n    "posteromedial",\n    "medial",\n)\n',
)
a = s.index("    def _regions(self, encoded: dict, medial_sign: float) -> torch.Tensor:\n")
b = s.index("    def routing_weights(self, encoded: dict) -> torch.Tensor:\n")
s = (
    s[:a]
    + '''    def _regions(self, encoded: dict, medial_sign: float) -> torch.Tensor:
        medial, front, height = self.frame(encoded, medial_sign).unbind(-1)
        mirror = torch.tensor([medial_sign, 1.0, 1.0], device=medial.device)
        half = (
            0.5
            * torch.einsum(
                "nac,nvc->nav",
                encoded["frame_rotation"].detach().float(),
                encoded["token_box"].float() * mirror,
            )
            .abs()
            .sum(-1)
        ).clamp_min(1e-3)[:, None]
        across = (half[..., 0] + half[..., 1]) / 2
        top, bottom, core_radius, ring_radius = (
            encoded["frame_sizes"].float()[:, None].unbind(-1)
        )
        step = torch.tensor(0.5, device=medial.device)

        def inside(distance: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
            soft = torch.sigmoid(distance / (2 * extent))
            return torch.heaviside(distance.detach(), step) + soft - soft.detach()

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
        medial_side = inside(medial, half[..., 0])
        lateral_side = 1 - medial_side
        upper = inside(height - top, half[..., 2])
        tibia = inside(bottom - height, half[..., 2])
        joint = (1 - upper - tibia).clamp_min(0)
        distance = torch.hypot(medial, front).clamp_min(1e-6)
        core = inside(core_radius - distance, across)
        ringed = 1 - core
        inner = ringed * (1 - inside(distance - ring_radius, across))
        medial_trio = (
            sector["anteromedial"] + sector["medial"] + sector["posteromedial"]
        )
        lateral_trio = (
            sector["anterolateral"] + sector["lateral"] + sector["posterolateral"]
        )
        front_trio = (
            sector["anteromedial"] + sector["anterior"] + sector["anterolateral"]
        )
        medial_compartment = (ringed * medial_trio + core * medial_side) * (1 - upper)
        lateral_compartment = (ringed * lateral_trio + core * lateral_side) * (
            1 - upper
        )
        kneecap = ringed * front_trio * (1 - tibia)
        return torch.stack(
            (
                medial_compartment,
                lateral_compartment,
                medial_compartment,
                lateral_compartment,
                kneecap,
                (
                    core
                    + inner
                    * (
                        sector["anterior"]
                        + sector["posterior"]
                        + lateral_trio
                    )
                )
                * (1 - tibia),
                ringed * (sector["anteromedial"] + sector["medial"]),
                ringed
                * (sector["posterior"] + sector["posteromedial"] + sector["medial"]),
                (ringed * front_trio + core) * (1 - tibia),
                ringed * sector["anteromedial"] * joint,
                ringed * sector["medial"] * joint,
                ringed * sector["posteromedial"] * joint,
                ringed * sector["anterolateral"] * joint,
                ringed * sector["lateral"] * joint,
                ringed * sector["posterolateral"] * joint,
                kneecap * medial_side,
                kneecap * lateral_side,
            )
        )

'''
    + s[b:]
)
p.write_text(s)

t = Path("src/knee_abnormality/modeling/spatial_tokens.py")
u = t.read_text()
for key, value, append in (
    (
        "compartment_routing",
        "log_routing_weight_over_routing_temperature_added_to_attention_scores_weights_are_hard_steps_with_the_gradient_of_a_sigmoid_of_signed_distance_over_the_projected_token_box_extent_eight_45_degree_sectors_centred_on_the_frame_axes_a_core_cylinder_an_outer_ring_and_three_heights_around_the_study_frame_per_target_region_table_from_the_attention_of_run_11_medial_and_lateral_mirrored_floor_1e-6_unrouted_below_one_token_contusion_and_fracture_unrouted",
        False,
    ),
    ("learned_centre", "_four_sizes_band_top_band_bottom_core_radius_ring_radius_from_1.3_minus_0.9_1.1_and_2.2_pitches", True),
    ("algorithm", "_ring_sector_division", True),
):
    pattern = re.compile(rf'^    "{key}": "(.*)",$', re.M)
    old = pattern.search(u).group(1)
    u = pattern.sub(f'    "{key}": "{old + value if append else value}",', u)
t.write_text(u)
