import re
from pathlib import Path


def edit(path, changes):
    p = Path(path)
    s = p.read_text()
    for old, new in changes:
        assert s.count(old) == 1, (path, old[:60], s.count(old))
        s = s.replace(old, new)
    p.write_text(s)


edit(
    "src/knee_abnormality/modeling/model.py",
    [
        ('CENTRE_OFFSETS = ("lateral", "front")\n', 'CENTRE_OFFSETS = ("lateral",)\n'),
        (
            "        lateral, front = self.centre_offsets().unbind()\n        coordinates = torch.stack(",
            "        coordinates = torch.stack(",
        ),
        (
            "        return coordinates + torch.stack((lateral, -front, torch.zeros_like(front)))\n",
            "        return coordinates + nn.functional.pad(self.centre_offsets(), (0, 2))\n",
        ),
    ],
)
edit(
    "scripts/adhoc/render_routing_views.py",
    [
        (
            "    lateral, front = model.centre_offsets().unbind()\n",
            "    lateral = model.centre_offsets()[0]\n",
        ),
        (
            '        encoded["anterior_axis"],\n        encoded["left_probability"][:, None],\n',
            '        encoded["anterior_axis"],\n        encoded["superior_axis"],\n        encoded["left_probability"][:, None],\n',
        ),
        (
            "((2 * left.mean() - 1) * lateral, front)",
            "((2 * left.mean() - 1) * lateral, torch.zeros_like(lateral))",
        ),
    ],
)
p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
key = "learned_centre"
value = "one_learned_lateral_offset_40mm_tanh_and_three_learned_frame_rotations_nod_roll_turn_about_the_fitted_center_zero_initialized_common_to_all_studies_no_per_study_part"
pattern = re.compile(rf'^    "{key}": ".*",$', re.M)
assert len(pattern.findall(s)) == 1
p.write_text(pattern.sub(f'    "{key}": "{value}",', s))

p = Path(
    "/tmp/claude-1000/-home-claudio-src-knee-abnormality/d2f4bec7-d221-4005-8675-56c8c41acf6d/scratchpad/side-routed/smoke_v5.py"
)
s = p.read_text()
old = '''    model.frame_rotation_logit.copy_(torch.tensor([0.3, -0.1, 0.15]))
    model.centre_offset_logit.copy_(torch.tensor([0.05, 0.2]))
    encoded = model.encode_batch(batch)
'''
assert s.count(old) == 1
s = s.replace(old, "")
old = "    routing = model.routing_weights("
assert s.count(old) == 1
s = s.replace(
    old,
    "    model.frame_rotation_logit.copy_(torch.tensor([0.3, -0.1, 0.15]))\n"
    "    model.centre_offset_logit.copy_(torch.tensor([0.05]))\n"
    "    encoded = model.encode_batch(batch)\n"
    "    assert torch.isfinite(model.pool_encoded(encoded)).all()\n"
    "    print('frame rotation set', [[round(v, 3) for v in row] for row in model.frame_rotation().tolist()])\n"
    + old,
)
p.write_text(s)
