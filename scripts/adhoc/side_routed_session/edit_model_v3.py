from pathlib import Path

p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)


def between(start, end, new):
    global s
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]


sub(
    '''ROUTING_WIDTHS = ("medial", "front")
''',
    '''ROUTING_WIDTHS = ("medial", "front", "side")
''',
)
sub(
    '''CENTRE_OFFSETS = ("lateral", "front", "up")
CENTRE_OFFSET_RANGE_MM = 40.0
POSITION_WAVELENGTHS_MM = (8.0, 512.0, 7)
POSITION_WIDTH = 64
''',
    '''CENTRE_OFFSETS = ("lateral", "front")
CENTRE_OFFSET_RANGE_MM = 40.0
POSITION_AXES = ("medial", "front", "up")
POSITION_WAVELENGTHS_MM = (8.0, 512.0, 7)
SHARED_PRIORS = ("meniscus", "oa", "front")
PRIOR_VARIANTS = ("medial", "lateral", "front")
''',
)
between(
    "def study_mean(",
    "def landmark(",
    "",
)
sub(
    '''        self.register_buffer(
            "shared_channel",
            torch.tensor([name in SHARED_CHANNELS for name in self.channels]),
            persistent=False,
        )
''',
    '''        own_channels = len(self.channels) - len(SHARED_CHANNELS)
        self.register_buffer(
            "target_prior_column",
            torch.tensor(
                [
                    own_channels
                    + SHARED_PRIORS.index(
                        "front"
                        if TARGET_ROUTES[target][1] == "front"
                        else TARGET_ROUTES[target][0]
                    )
                    if target in TARGET_ROUTES
                    else target_channel[index]
                    for index, target in enumerate(targets)
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "target_prior_variant",
            torch.tensor(
                [
                    PRIOR_VARIANTS.index(TARGET_ROUTES[target][1])
                    if target in TARGET_ROUTES
                    else 0
                    for target in targets
                ]
            ),
            persistent=False,
        )
        self.register_buffer("routing_temperature", torch.ones(()))
''',
)
between(
    "            self.position = nn.Sequential(\n",
    "        self.centre_offset_logit = nn.Parameter(",
    '''            self.position_prior = nn.Linear(
                2 * len(POSITION_AXES) * POSITION_WAVELENGTHS_MM[2],
                own_channels + len(SHARED_PRIORS),
            )
            nn.init.zeros_(self.position_prior.weight)
            nn.init.zeros_(self.position_prior.bias)
            self.half_detector = nn.Linear(PROJECTION_WIDTH, 1)
''',
)
between(
    "    def position_code(\n",
    "    def side_probability(\n",
    '''    def position_priors(
        self, position: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type=position.device.type, enabled=False):
            position = position.float()
            medial = position[..., :1]
            variants = torch.stack(
                [
                    torch.cat((axis, position[..., 1:]), -1)
                    for axis in (medial, -medial, medial.abs())
                ]
            )
            angles = variants.unsqueeze(-1) * self.position_frequencies
            features = torch.stack((angles.sin(), angles.cos()), -1).flatten(-3)
            priors = self.position_prior(features)[
                self.target_prior_variant, ..., self.target_prior_column
            ]
            return torch.where(valid[None, :, None], priors, 0.0)

''',
)
sub(
    "            width_medial, width_front = self.routing_widths().unbind()\n",
    "            width_medial, width_front, _ = self.routing_widths().unbind()\n",
)
between(
    "    def _attention_pool(\n",
    "    def encode_batch(",
    '''    def _attention_pool(
        self,
        tokens: torch.Tensor,
        token_study: torch.Tensor,
        image_type: torch.Tensor,
        token_valid: torch.Tensor,
        batch_size: int,
        routing: torch.Tensor,
        prior: torch.Tensor,
        token_view: torch.Tensor,
        diagnostic: bool = False,
    ) -> torch.Tensor:
        scores = self.attention(tokens).transpose(0, 1)
        if self.view_label_bias is not None:
            scores = scores.float() + self.view_label_bias.float()[
                token_view
            ].transpose(0, 1)
        membership = (
            torch.arange(batch_size, device=tokens.device)[:, None] == token_study[None]
        ) & token_valid[None]
        if not diagnostic and not membership.any(-1).all():
            raise ValueError("A study has no visually valid spatial tokens")
        empty_study = ~membership.any(-1)
        allowed = membership
        if diagnostic:
            allowed = membership.clone()
            allowed[:, 0] |= empty_study
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            routing = routing.float()
            weights = torch.cat((routing, torch.ones_like(routing[:1])))
            routed = (membership.float() @ weights.T >= 1)[:, self.target_route]
            log_weights = (
                weights.clamp_min(ROUTING_MINIMUM_WEIGHT).log()
                / self.routing_temperature
            )
            attention = (
                scores.float()[self.target_channel][None]
                + prior.float()[None]
                + torch.where(
                    routed[:, :, None], log_weights[self.target_route][None], 0.0
                )
            )
            attention = attention.masked_fill(~allowed[:, None], -torch.inf).softmax(-1)
            if diagnostic:
                attention = attention.masked_fill(empty_study[:, None, None], 0.0)
            else:
                self.attention_mass_view = torch.stack(
                    [
                        (attention * (token_view == view)[None, None]).sum(-1)
                        for view in range(len(self.detail_view))
                    ]
                ).detach()
                self.routing_mean = (
                    (routing * token_valid).sum(-1) / token_valid.sum().clamp_min(1)
                ).detach()
            if self.plane_readout:
                evidence = self.head(tokens, image_type)[:, self.target_channel]
                logits = torch.einsum("btn,nt->bt", attention, evidence)
            else:
                pooled = torch.einsum("btn,nf->btf", attention, tokens.float())
                logits = (
                    pooled * self.head.weight.float()[self.target_channel][None]
                ).sum(-1)
            logits = logits + self.target_bias.float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Study attention pooling produced non-finite logits")
        return logits

''',
)
between(
    "            side_features, side_count = study_mean(\n",
    "            head_left = self.side_probability(\n",
    '''            side_tokens = visual_valid & (valid_position & ~sagittal)[:, None]
            displayed_x = torch.where(
                mirrored_study[token_study][:, None], -knee_x, knee_x
            )
            side_weight = (
                torch.tanh(displayed_x / self.routing_widths()[2]) * side_tokens
            )
            side_count = torch.zeros(batch_size, device=features.device).index_add(
                0, token_study, side_tokens.float().sum(-1)
            )
            side_features = torch.zeros(
                (batch_size, features.shape[-1]), device=features.device
            ).index_add(
                0,
                token_study,
                (features.detach().float() * side_weight[..., None]).sum(1),
            ) / side_count.clamp_min(1)[:, None]
            side_logit = self.side_head(side_features).squeeze(-1)
            side_present = side_count > 0
''',
)
sub(
    "            lateral_offset, front_offset, up_offset = self.centre_offsets().unbind()\n",
    "            lateral_offset, front_offset = self.centre_offsets().unbind()\n",
)
sub(
    "                self.landmark_scorers(tokens.float())[..., 0],\n",
    "                self.landmark_scorers(tokens.detach().float())[..., 0],\n",
)
sub(
    "                    offset[..., 2] - up_offset,\n",
    "                    offset[..., 2],\n",
)
between(
    "    def study_tokens(self, encoded: dict, shared: bool = False)",
    "    def pool_encoded(",
    '''    def study_tokens(self, encoded: dict) -> torch.Tensor:
        return self.mil_norm(self._base_tokens(encoded).flatten(0, 1))

''',
)
sub(
    '''            self.study_tokens(encoded),
            self.study_tokens(encoded, shared=True),
            encoded["study"]''',
    '''            self.study_tokens(encoded),
            encoded["study"]''',
)
sub(
    '''                encoded["centre_shift"][:, None],
            ).flatten(1),
            encoded["view_index"]''',
    '''                encoded["centre_shift"][:, None],
            ).flatten(1),
            self.position_priors(encoded["position"], encoded["routable"]).flatten(1),
            encoded["view_index"]''',
)
p.write_text(s)
