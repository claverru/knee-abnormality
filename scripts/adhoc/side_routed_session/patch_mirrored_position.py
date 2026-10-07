import re
from pathlib import Path

p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


sub(
    """        scores = torch.where(
            self.shared_channel[:, None],
            self.attention(shared_tokens).transpose(0, 1),
            self.attention(tokens).transpose(0, 1),
        )
        if self.view_label_bias is not None:
            scores = scores.float() + self.view_label_bias.float()[
                token_view
            ].transpose(0, 1)
""",
    """        shared_target = self.shared_channel[self.target_channel]
        variant = self.target_prior_variant
        own_scores = self.attention(tokens).transpose(0, 1).float()
        variant_scores = self.attention(shared_tokens).transpose(1, 2).float()
        if self.view_label_bias is not None:
            view_bias = self.view_label_bias.float()[token_view].transpose(0, 1)
            own_scores = own_scores + view_bias
            variant_scores = variant_scores + view_bias[None]
        scores = torch.where(
            shared_target[:, None],
            variant_scores[variant, self.target_channel],
            own_scores[self.target_channel],
        )
        knee_scores = variant_scores[-1, -len(SHARED_CHANNELS) :]
""",
)
sub(
    "                scores.float()[self.target_channel][None]\n                + prior.float()[None]\n",
    "                scores[None]\n                + prior.float()[None]\n",
)
a = s.index("            if self.plane_readout:\n                evidence = torch.where(")
b = s.index("            knee_attention = scores.float()[-len(SHARED_CHANNELS) :][None]\n")
s = (
    s[:a]
    + """            own_evidence = self._evidence(tokens, image_type)
            variant_evidence = torch.stack(
                [self._evidence(values, image_type) for values in shared_tokens]
            )
            shared_evidence = variant_evidence[-1][:, -len(SHARED_CHANNELS) :]
            logits = torch.einsum(
                "btn,nt->bt",
                attention,
                torch.where(
                    shared_target,
                    variant_evidence[variant, :, self.target_channel].transpose(0, 1),
                    own_evidence[:, self.target_channel],
                ),
            )
            knee_attention = knee_scores[None]
"""
    + s[b + len("            knee_attention = scores.float()[-len(SHARED_CHANNELS) :][None]\n") :]
)
sub(
    "    def _attention_pool(\n",
    """    def _evidence(self, tokens: torch.Tensor, image_type: torch.Tensor) -> torch.Tensor:
        if self.plane_readout:
            return self.head(tokens, image_type)
        return nn.functional.linear(tokens.float(), self.head.weight.float())

    def _attention_pool(
""",
)
a = s.index("    def study_tokens(self, encoded: dict, shared: bool = False) -> torch.Tensor:\n")
b = s.index("    def pool_encoded(")
s = (
    s[:a]
    + """    def study_tokens(self, encoded: dict, shared: bool = False) -> torch.Tensor:
        tokens = self._base_tokens(encoded)
        if not shared:
            return self.mil_norm(
                (
                    tokens
                    + self.position_code(
                        encoded["fitted_position"], encoded["routable"]
                    ).to(tokens.dtype)
                ).flatten(0, 1)
            )
        position = encoded["position"]
        medial = position[..., :1]
        return torch.stack(
            [
                self.mil_norm(
                    (
                        tokens
                        + self.position_code(
                            torch.cat((axis, position[..., 1:]), -1), encoded["routable"]
                        ).to(tokens.dtype)
                    ).flatten(0, 1)
                )
                for axis in (medial, -medial, medial.abs())
            ]
        )

"""
    + s[b:]
)
p.write_text(s)

p = Path("scripts/tools/check_side_routing.py")
s = p.read_text()
a = s.index("        shared = model.study_tokens(encoded, shared=True)\n")
b = s.index('        results["shared_tokens_are_position_free"] = torch.equal(shared, moved)\n') + len(
    '        results["shared_tokens_are_position_free"] = torch.equal(shared, moved)\n'
)
s = (
    s[:a]
    + """        shared = model.study_tokens(encoded, shared=True).float()
        mirrored = model.study_tokens(
            {
                **encoded,
                "position": encoded["position"]
                * torch.tensor([-1.0, 1.0, 1.0], device=DEVICE),
            },
            shared=True,
        ).float()
        results["shared_tokens_mirror_between_sides"] = (
            torch.allclose(shared[0], mirrored[1], atol=1e-4)
            and torch.allclose(shared[1], mirrored[0], atol=1e-4)
            and torch.allclose(shared[2], mirrored[2], atol=1e-4)
            and not torch.allclose(shared[0], shared[1], atol=1e-4)
        )
"""
    + s[b:]
)
p.write_text(s)

p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
pattern = re.compile(r'^    "algorithm": "(.*)",$', re.M)
old = pattern.search(s).group(1)
s = pattern.sub(
    f'    "algorithm": "{old}_shared_heads_read_fixed_sinusoid_of_compartment_relative_frame_position_medial_axis_as_is_mirrored_or_absolute",',
    s,
)
p.write_text(s)
