import re
from pathlib import Path


def sub(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90])
    return text.replace(old, new)


p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()
s = sub(
    s,
    'FRAME_SIZES = ("band_top", "band_bottom", "core_radius", "ring_radius")\n',
    'FRAME_SIZES = ("band_top", "band_bottom", "core_radius", "ring_radius")\n'
    'MENISCUS_TARGETS = ("Medial Meniscus", "Lateral Meniscus")\n'
    'OA_TARGETS = ("Medial OA", "Lateral OA", "PF OA")\n'
    "TEAR_DETECTOR_TEMPERATURE = 1.0\n",
)
s = sub(
    s,
    """        with torch.random.fork_rng(devices=[]):
            self.detail_view_embedding = nn.Parameter(torch.zeros(PROJECTION_WIDTH))
""",
    """        with torch.random.fork_rng(devices=[]):
            self.detail_view_embedding = nn.Parameter(torch.zeros(PROJECTION_WIDTH))
        with torch.random.fork_rng(devices=[]):
            self.tear_detector = nn.Linear(PROJECTION_WIDTH, 1)
            self.oa_detector = nn.Linear(PROJECTION_WIDTH, 1, bias=False)
            nn.init.xavier_uniform_(self.oa_detector.weight)
        self.oa_detector_scale = nn.Parameter(torch.zeros(len(OA_TARGETS)))
        self.register_buffer(
            "meniscus_targets",
            torch.tensor([targets.index(name) for name in MENISCUS_TARGETS]),
            persistent=False,
        )
        self.register_buffer(
            "oa_targets",
            torch.tensor([targets.index(name) for name in OA_TARGETS]),
            persistent=False,
        )
""",
)
s = sub(
    s,
    """            logits = (
                logits
                + channel_contribution[:, self.target_channel]
                + self.target_bias.float()
            )
""",
    """            logits = (
                logits
                + channel_contribution[:, self.target_channel]
                + self.target_bias.float()
            )
            compartments = (
                membership[:, None]
                & (weights[self.target_route[self.oa_targets]] > 0.5)[None]
            )
            counts = compartments.sum(-1)
            absent = counts == 0
            compartments = compartments.clone()
            compartments[..., 0] |= absent
            detected = self.oa_detector(shared_tokens.float()).squeeze(-1)
            pooled_detection = torch.logsumexp(
                detected[None, None].masked_fill(~compartments, -torch.inf), -1
            ) - counts.clamp_min(1).log()
            correction = (
                torch.where(absent, 0.0, pooled_detection)
                * self.oa_detector_scale.float()[None]
            )
            logits = logits.index_add(1, self.oa_targets, correction)
""",
)
s = sub(
    s,
    "    def pool_encoded(self, encoded: dict, view: int | None = None) -> torch.Tensor:\n",
    """    def tear_detector_scores(self, encoded: dict) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.study_tokens(encoded, shared=True)
        valid = encoded["visual_valid"].flatten()
        study = encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten()
        member = (
            torch.arange(encoded["batch_size"], device=tokens.device)[:, None]
            == study[None]
        ) & valid[None]
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            routing = self.routing_weights(encoded).flatten(1).float()
            chosen = (
                member[:, None]
                & (routing[self.target_route[self.meniscus_targets]] > 0.5)[None]
            )
            counts = chosen.sum(-1)
            present = counts > 0
            chosen = chosen.clone()
            chosen[..., 0] |= ~present
            values = self.tear_detector(tokens.float()).squeeze(-1)
            scores = TEAR_DETECTOR_TEMPERATURE * (
                torch.logsumexp(
                    (values / TEAR_DETECTOR_TEMPERATURE)[None, None].masked_fill(
                        ~chosen, -torch.inf
                    ),
                    -1,
                )
                - counts.clamp_min(1).log()
            )
        return scores, present

    def pool_encoded(self, encoded: dict, view: int | None = None) -> torch.Tensor:
""",
)
p.write_text(s)

p = Path("src/knee_abnormality/training/lightning.py")
s = p.read_text()
s = sub(
    s,
    '        loss = loss + self._report_attribute_step(self.model, encoded, batch, "")\n',
    '        loss = loss + self._report_attribute_step(self.model, encoded, batch, "")\n'
    "        loss = loss + TEAR_DETECTOR_LOSS_WEIGHT * self._tear_detector_step(\n"
    "            encoded, batch\n"
    "        )\n",
)
s = sub(
    s,
    "    def _log_epoch_metric(\n",
    '''    def _tear_detector_step(self, encoded: dict, batch: dict) -> torch.Tensor:
        scores, present = self.model.tear_detector_scores(encoded)
        sides = self.model.meniscus_targets
        used = present.all(-1) & batch["target_observed"][:, sides].all(-1)
        if not used.any():
            return scores.sum() * 0
        targets = batch["target"][:, sides][used].float()
        weights = self.class_balance_weights[:, sides]
        loss = (
            F.binary_cross_entropy_with_logits(scores[used], targets, reduction="none")
            * torch.where(targets >= 0.5, weights[0], weights[1])
        ).mean()
        self._log_epoch_metric("tear_detector_loss", loss, int(used.sum()))
        return loss

    def _log_epoch_metric(
''',
)
s = sub(
    s,
    """        for channel, scale, routed in zip(
            self.model.channels,
            self.model.knee_level_scale.detach(),
""",
    """        for name, scale in zip(
            ("medial", "lateral", "pf"),
            self.model.oa_detector_scale.detach().float(),
            strict=True,
        ):
            self._log_epoch_metric(f"oa_detector_scale_{name}", scale, 1)
        for channel, scale, routed in zip(
            self.model.channels,
            self.model.knee_level_scale.detach(),
""",
)
s = sub(s, "SIDE_LOSS_WEIGHT = ", "TEAR_DETECTOR_LOSS_WEIGHT = 0.075\nSIDE_LOSS_WEIGHT = ")
p.write_text(s)

t = Path("src/knee_abnormality/modeling/spatial_tokens.py")
u = t.read_text()
pattern = re.compile(r'^    "algorithm": "(.*)",$', re.M)
old = pattern.search(u).group(1)
assert "routed_detectors" not in old
u = pattern.sub(
    f'    "algorithm": "{old}_routed_detectors_shared_tear_detector_smooth_maximum_over_each_meniscus_region_loss_weight_0p075_and_oa_detector_log_mean_exp_over_the_three_oa_regions_added_through_a_learned_scale_from_zero",',
    u,
)
t.write_text(u)
print("patched")
