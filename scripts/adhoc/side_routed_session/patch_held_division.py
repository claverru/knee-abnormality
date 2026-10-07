import re
from pathlib import Path


def sub(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90])
    return text.replace(old, new)


p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()
s = sub(
    s,
    '''    "kneecap",
    "core",
    "collateral_medial",
    "baker",
    "front_core",
''',
    '''    "kneecap",
    "core",
    "collateral_medial",
    "baker",
    "front_core",
    "pouch",
''',
)
s = sub(s, '    "Effusion": "kneecap",\n', '    "Effusion": "pouch",\n')
s = sub(
    s,
    'FRAME_SIZES = ("band_top", "band_bottom", "core_radius", "ring_radius")\n'
    "FRAME_SIZE_START_PITCHES = (1.3, -0.9, 1.1, 2.2)\n",
    'FRAME_SIZES = ("band_top", "band_bottom", "core_radius")\n'
    "FRAME_SIZE_START_PITCHES = (2.1, 0.7, 1.65)\n",
)
s = sub(
    s,
    '''SECTORS = (
    "anteromedial",
    "anterior",
    "anterolateral",
    "lateral",
    "posterolateral",
    "posterior",
    "posteromedial",
    "medial",
)
''',
    '''SECTORS = ("front_medial", "front_lateral", "back_lateral", "back_medial")
SECTOR_START_DEG = (30.0, 90.0, 150.0, 270.0)
''',
)
start = s.index("        top, bottom, core_radius, ring_radius = (\n")
stop = s.index("    def routing_weights(self, encoded: dict) -> torch.Tensor:\n")
s = (
    s[:start]
    + '''        top, bottom, core_radius = encoded["frame_sizes"].float()[:, None].unbind(-1)
        step = torch.tensor(0.5, device=medial.device)

        def inside(distance: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
            soft = torch.sigmoid(distance / (2 * extent))
            return torch.heaviside(distance.detach(), step) + soft - soft.detach()

        beyond = [
            inside(
                front * math.cos(math.radians(angle))
                - medial * math.sin(math.radians(angle)),
                across,
            )
            for angle in SECTOR_START_DEG
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
        pouch = inside(height - top, half[..., 2])
        joint = inside(bottom - height, half[..., 2])
        middle = (1 - pouch - joint).clamp_min(0)
        core = inside(
            core_radius - torch.hypot(medial, front).clamp_min(1e-6), across
        )
        ringed = 1 - core
        medial_core = core * medial_side
        lateral_core = core * lateral_side
        lower = joint + middle
        upper = middle + pouch
        front = sector["front_medial"] + sector["front_lateral"]
        medial_ring = ringed * (sector["front_medial"] + sector["back_medial"])
        lateral_ring = ringed * (sector["front_lateral"] + sector["back_lateral"])
        return torch.stack(
            (
                (medial_core + ringed * sector["back_medial"]) * joint,
                (lateral_core + ringed * sector["back_lateral"]) * joint,
                (medial_core + medial_ring) * lower,
                (lateral_core + lateral_ring) * lower,
                ringed * front * upper,
                core * lower,
                ringed * sector["back_medial"] * lower,
                ringed * sector["back_medial"] * lower,
                (core + ringed * front) * upper,
                ringed * front * pouch,
                ringed * sector["front_medial"] * joint,
                medial_ring * joint,
                ringed * sector["back_medial"] * joint,
                ringed * sector["front_lateral"] * joint,
                lateral_ring * joint,
                ringed * sector["back_lateral"] * joint,
                ringed * sector["front_medial"] * upper,
                ringed * sector["front_lateral"] * upper,
            )
        )

'''
    + s[stop:]
)
p.write_text(s)

t = Path("src/knee_abnormality/modeling/spatial_tokens.py")
u = t.read_text()
pattern = re.compile(r'^    "algorithm": "(.*)",$', re.M)
old = pattern.search(u).group(1)
assert "held_division" not in old
u = pattern.sub(
    f'    "algorithm": "{old}_held_division_three_heights_cut_from_0p7_and_2p1_pitch_core_from_1p65_pitch_no_ring_four_sectors_from_30_90_150_270_degrees_mirrored_region_table_from_the_free_attention_at_20_percent_side_labels_keep_their_half_of_the_core_effusion_on_its_own_pouch_route",',
    u,
)
t.write_text(u)
print("patched")
