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
        ('CENTRE_OFFSETS = ("lateral",)\nCENTRE_OFFSET_RANGE_MM = 40.0\n', ""),
        (
            "        self.centre_offset_logit = nn.Parameter(torch.zeros(len(CENTRE_OFFSETS)))\n",
            "",
        ),
        (
            "    def centre_offsets(self) -> torch.Tensor:\n"
            "        return CENTRE_OFFSET_RANGE_MM * torch.tanh(self.centre_offset_logit.float())\n\n",
            "",
        ),
        (
            "        coordinates = torch.stack(\n            (\n                medial_sign * knee_x.float(),",
            "        return torch.stack(\n            (\n                medial_sign * knee_x.float(),",
        ),
        (
            "        ) @ self.frame_rotation().transpose(0, 1)\n"
            "        return coordinates + nn.functional.pad(self.centre_offsets(), (0, 2))\n",
            "        ) @ self.frame_rotation().transpose(0, 1)\n",
        ),
    ],
)
edit(
    "src/knee_abnormality/training/lightning.py",
    [
        ("    CENTRE_OFFSETS,\n", ""),
        (
            "        for name, value in zip(\n"
            "            CENTRE_OFFSETS, self.model.centre_offsets().detach(), strict=True\n"
            "        ):\n"
            '            self._log_epoch_metric(f"centre_offset_{name}_mm", value, 1)\n',
            "",
        ),
    ],
)
edit(
    "scripts/adhoc/render_routing_views.py",
    [
        ("    lateral = model.centre_offsets()[0]\n", ""),
        (
            "((2 * left.mean() - 1) * lateral, torch.zeros_like(lateral))",
            "(torch.zeros(()), torch.zeros(()))",
        ),
    ],
)
p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
values = {
    "learned_centre": "fitted_center_with_three_learned_frame_rotations_nod_roll_turn_about_it_zero_initialized_common_to_all_studies_no_offsets_no_per_study_part",
    "position": "attention_score_prior_per_target_linear42_10_zero_initialized_on_sincos_7_wavelengths_8_to_512_of_rotated_frame_medial_front_up_mm_from_the_fitted_center_own_channels_signed_medial_meniscus_and_oa_sides_one_map_each_on_depth_into_own_compartment_pf_on_absolute_medial_zero_without_fitted_center",
}
for key, value in values.items():
    pattern = re.compile(rf'^    "{key}": ".*",$', re.M)
    assert len(pattern.findall(s)) == 1, key
    s = pattern.sub(f'    "{key}": "{value}",', s)
p.write_text(s)

p = Path(
    "/tmp/claude-1000/-home-claudio-src-knee-abnormality/d2f4bec7-d221-4005-8675-56c8c41acf6d/scratchpad/side-routed/smoke_v5.py"
)
s = p.read_text()
for old, new in (
    ('print("offsets", model.centre_offsets().tolist(), "widths", model.routing_widths().tolist())', 'print("widths", model.routing_widths().tolist())\nassert not hasattr(model, "centre_offset_logit")'),
    ('print("offset logit grad", model.centre_offset_logit.grad.tolist())\n', ""),
    ("    model.centre_offset_logit.copy_(torch.tensor([0.05]))\n", ""),
    ('print(sorted(key for key in logged if "offset" in key or "width" in key or "routed_half" in key))', 'print(sorted(key for key in logged if "rotation" in key or "width" in key))'),
):
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
p.write_text(s)
