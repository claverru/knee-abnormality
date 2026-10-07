import re
from pathlib import Path


def edit(path, changes, ranges=()):
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
            "            self.side_head = nn.Linear(PROJECTION_WIDTH, 1)\n",
            "            self.side_head = nn.Sequential(\n"
            "                nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH),\n"
            "                nn.Tanh(),\n"
            "                nn.Linear(MIL_ATTENTION_WIDTH, 1),\n"
            "            )\n",
        ),
        ("            self.half_detector = nn.Linear(PROJECTION_WIDTH, 1)\n", ""),
    ],
    [
        (
            "            side_features = (\n",
            "            side_present = side_count > 0\n",
            "            side_scores = self.side_head(tokens.detach().float()).squeeze(-1)\n"
            "            side_logit = torch.zeros(batch_size, device=features.device).index_add(\n"
            "                0, token_study, (side_scores * side_weight).sum(-1)\n"
            "            ) / side_count.clamp_min(1)\n",
        ),
    ],
)
edit(
    "src/knee_abnormality/training/lightning.py",
    [
        (
            'ROUTED_HALF_LOSS_WEIGHT = 0.075\nROUTED_HALF_TARGETS = ("Medial Meniscus", "Lateral Meniscus")\n',
            "",
        ),
    ],
    [
        ("def routed_half_loss(\n", "class KneeDataModule(L.LightningDataModule):", ""),
        (
            "        auxiliary, used_studies = routed_half_loss(\n",
            "        loss = loss + self._report_attribute_step(encoded, batch)\n",
            "",
        ),
    ],
)
p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
values = {
    "algorithm": "all_valid_spatial_tokens_position_free_knee_frame_attention_prior_side_routed_shared_channels_v11",
    "initialization_rng": "side_head_front_scorer_and_position_prior_parameters_forked_removed_position_draw_count_and_target_sized_scorer_and_readout_draws_preserve_training_stream",
    "side_head": "detached_normalized_visually_valid_axial_coronal_fitted_center_tokens_linear_256_128_tanh_linear_128_1_per_token_mean_weighted_by_tanh_displayed_knee_x_over_learned_width_left_logit_flipped_for_mirrored_studies",
}
for key, value in values.items():
    pattern = re.compile(rf'^    "{key}": ".*",$', re.M)
    assert len(pattern.findall(s)) == 1, key
    s = pattern.sub(f'    "{key}": "{value}",', s)
pattern = re.compile(r'^    "routed_half_loss": ".*",\n', re.M)
assert len(pattern.findall(s)) == 1
s = pattern.sub("", s)
p.write_text(s)
