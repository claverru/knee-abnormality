from pathlib import Path

p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:60], s.count(old))
    s = s.replace(old, new)


sub(
    '''LANDMARKS = ("midline", "front")
LANDMARK_PRIOR_MM = 15.0
''',
    '''LANDMARKS = ("front",)
LANDMARK_PRIOR_MM = 15.0
CENTRE_OFFSETS = ("lateral", "front", "up")
CENTRE_OFFSET_RANGE_MM = 40.0
POSITION_WAVELENGTHS_MM = (8.0, 512.0, 7)
POSITION_WIDTH = 64
''',
)

sub(
    '''def landmark(
    scores: torch.Tensor,
    coordinate: torch.Tensor,
    eligible: torch.Tensor,
    token_study: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    study = token_study[:, None].expand_as(eligible)[eligible]
    position = coordinate.float()[eligible]
    values = scores.float()[eligible] - position.square() / (2 * LANDMARK_PRIOR_MM**2)
''',
    '''def landmark(
    scores: torch.Tensor,
    coordinate: torch.Tensor,
    prior_centre: torch.Tensor,
    eligible: torch.Tensor,
    token_study: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    study = token_study[:, None].expand_as(eligible)[eligible]
    position = coordinate.float()[eligible]
    values = scores.float()[eligible] - (position - prior_centre).square() / (
        2 * LANDMARK_PRIOR_MM**2
    )
''',
)
sub(
    '''    centre = torch.zeros(batch_size, device=scores.device).index_add(
        0, study, weight * position
    )
    weights = torch.zeros(eligible.shape, device=scores.device)
''',
    '''    centre = torch.zeros(batch_size, device=scores.device).index_add(
        0, study, weight * position
    )
    centre = torch.where(total > 0, centre, prior_centre)
    weights = torch.zeros(eligible.shape, device=scores.device)
''',
)

sub(
    '''        self.register_buffer(
            "target_route", torch.tensor(target_route), persistent=False
        )
''',
    '''        self.register_buffer(
            "target_route", torch.tensor(target_route), persistent=False
        )
        self.register_buffer(
            "shared_channel",
            torch.tensor([name in SHARED_CHANNELS for name in self.channels]),
            persistent=False,
        )
''',
)
sub(
    '''            self.landmark_scorers = nn.Linear(PROJECTION_WIDTH, len(LANDMARKS))
            nn.init.zeros_(self.landmark_scorers.weight)
            nn.init.zeros_(self.landmark_scorers.bias)
''',
    '''            self.landmark_scorers = nn.Linear(PROJECTION_WIDTH, len(LANDMARKS))
            nn.init.zeros_(self.landmark_scorers.weight)
            nn.init.zeros_(self.landmark_scorers.bias)
            self.position = nn.Sequential(
                nn.Linear(
                    2 * len(CENTRE_OFFSETS) * POSITION_WAVELENGTHS_MM[2], POSITION_WIDTH
                ),
                nn.GELU(),
                nn.Linear(POSITION_WIDTH, PROJECTION_WIDTH),
            )
            nn.init.zeros_(self.position[-1].weight)
            nn.init.zeros_(self.position[-1].bias)
        self.unknown_position = nn.Parameter(torch.zeros(PROJECTION_WIDTH))
        self.centre_offset_logit = nn.Parameter(torch.zeros(len(CENTRE_OFFSETS)))
        shortest, longest, count = POSITION_WAVELENGTHS_MM
        self.register_buffer(
            "position_frequencies",
            2
            * math.pi
            / torch.logspace(math.log10(shortest), math.log10(longest), count),
            persistent=False,
        )
''',
)

sub(
    '''    def side_probability(
''',
    '''    def centre_offsets(self) -> torch.Tensor:
        return CENTRE_OFFSET_RANGE_MM * torch.tanh(self.centre_offset_logit.float())

    def position_code(
        self, position: torch.Tensor, valid: torch.Tensor, shared: bool
    ) -> torch.Tensor:
        with torch.autocast(device_type=position.device.type, enabled=False):
            position = position.float()
            if shared:
                position = torch.cat((position[..., :1].abs(), position[..., 1:]), -1)
            angles = position.unsqueeze(-1) * self.position_frequencies
            features = torch.stack((angles.sin(), angles.cos()), -1).flatten(-3)
            return torch.where(
                valid[:, None, None],
                self.position(features),
                self.unknown_position.float(),
            )

    def side_probability(
''',
)

sub(
    '''            shift_midline, shift_front = centre_shift.float().unbind(-1)
            width_medial, width_front = self.routing_widths().unbind()
            as_right = torch.sigmoid((knee_x.float() - shift_midline) / width_medial)
            as_left = 1 - as_right
            left = left_probability.float()
            medial = left * as_left + (1 - left) * as_right
            front = torch.sigmoid((anterior_axis.float() - shift_front) / width_front)
''',
    '''            lateral = self.centre_offsets()[0]
            width_medial, width_front = self.routing_widths().unbind()
            as_right = torch.sigmoid((knee_x.float() + lateral) / width_medial)
            as_left = torch.sigmoid((lateral - knee_x.float()) / width_medial)
            left = left_probability.float()
            medial = left * as_left + (1 - left) * as_right
            front = torch.sigmoid(
                (anterior_axis.float() - centre_shift.float()[..., 0]) / width_front
            )
''',
)

sub(
    '''    def _attention_pool(
        self,
        tokens: torch.Tensor,
        token_study''',
    '''    def _attention_pool(
        self,
        tokens: torch.Tensor,
        shared_tokens: torch.Tensor,
        token_study''',
)
sub(
    '''        scores = self.attention(tokens).transpose(0, 1)
''',
    '''        scores = torch.where(
            self.shared_channel[:, None],
            self.attention(shared_tokens).transpose(0, 1),
            self.attention(tokens).transpose(0, 1),
        )
''',
)
sub(
    '''                evidence = self.head(tokens, image_type)[:, self.target_channel]
                logits = torch.einsum("btn,nt->bt", attention, evidence)
            else:
                pooled = torch.einsum("btn,nf->btf", attention, tokens.float())
''',
    '''                evidence = torch.where(
                    self.shared_channel,
                    self.head(shared_tokens, image_type),
                    self.head(tokens, image_type),
                )[:, self.target_channel]
                logits = torch.einsum("btn,nt->bt", attention, evidence)
            else:
                pooled = torch.where(
                    self.shared_channel[self.target_channel][None, :, None],
                    torch.einsum("btn,nf->btf", attention, shared_tokens.float()),
                    torch.einsum("btn,nf->btf", attention, tokens.float()),
                )
''',
)

start = s.index(
    "        tokens = self.study_tokens(encoded).unflatten(0, visual_valid.shape)\n"
)
end = s.index("    def pool_encoded(")
new_tail = '''        tokens = self.mil_norm(self._base_tokens(encoded))
        with torch.autocast(device_type=features.device.type, enabled=False):
            homogeneous = torch.cat(
                (prepared_xy.float(), torch.ones_like(prepared_xy[..., :1])), -1
            )
            middle = patient_maps[:, 1]
            patient_position = torch.einsum("nij,nkj->nki", middle, homogeneous)
            center = torch.stack(
                [
                    batch[key][keep_series][acquisition_index].float()
                    for key in (
                        "knee_center_x_mm",
                        "knee_center_y_mm",
                        "knee_center_z_mm",
                    )
                ],
                -1,
            )
            valid_position = (
                geometry_valid
                & torch.isfinite(center).all(-1)
                & torch.isfinite(patient_position).all((-1, -2))
            )
            offset = torch.where(
                valid_position[:, None, None], patient_position - center[:, None], 0.0
            ).detach()
            row = middle[:, :2, 0]
            row = torch.where(row[:, :1] < 0, -row, row)
            length = row.norm(dim=-1)
            aligned = valid_position & ~sagittal & (length > 1e-6)
            axis = torch.zeros((batch_size, 2), device=features.device).index_add(
                0, token_study[aligned], row[aligned] / length[aligned, None]
            )
            axis_length = axis.norm(dim=-1, keepdim=True)
            axis = torch.where(
                axis_length > 1e-6,
                axis / axis_length.clamp_min(1e-6),
                torch.tensor([1.0, 0.0], device=features.device),
            ).detach()
            cosine, sine = axis[token_study][:, None].unbind(-1)
            knee_x = offset[..., 0] * cosine + offset[..., 1] * sine
            anterior_axis = offset[..., 0] * sine - offset[..., 1] * cosine
            side_features, side_count = study_mean(
                features, visual_valid & ~sagittal[:, None], token_study, batch_size
            )
            side_logit = self.side_head(side_features).squeeze(-1)
            side_present = side_count > 0
            head_left = self.side_probability(
                side_logit,
                side_present,
                mirrored_study,
                batch["knee_side_coordinates"],
                batch["knee_side_left"],
            )
            lateral_offset, front_offset, up_offset = self.centre_offsets().unbind()
            front_centre, front_weights = landmark(
                self.landmark_scorers(tokens.float())[..., 0],
                anterior_axis,
                front_offset,
                visual_valid & (valid_position & (token_plane != 1))[:, None],
                token_study,
                batch_size,
            )
            centre_shift = front_centre[:, None]
            sign = batch["knee_side_sign"][keep_series][acquisition_index]
            certain = batch["knee_side_certain"][token_study]
            metadata_side = certain | (
                batch["knee_side_split"][token_study] & (sign != 0)
            )
            left_probability = torch.where(
                metadata_side,
                torch.where(
                    certain, batch["knee_side_left"][token_study], sign < 0
                ).float(),
                head_left[token_study],
            )
            position = torch.stack(
                (
                    (1 - 2 * left_probability)[:, None] * knee_x + lateral_offset,
                    anterior_axis - front_centre[token_study][:, None],
                    offset[..., 2] - up_offset,
                ),
                -1,
            )
        return {
            **encoded,
            "view_index": view_index,
            "study": token_study,
            "image_plane": token_plane,
            "image_type": token_type,
            "visual_valid": visual_valid,
            "knee_x": knee_x,
            "anterior_axis": anterior_axis,
            "position": position,
            "left_probability": left_probability,
            "routable": valid_position,
            "centre_shift": centre_shift[token_study],
            "study_centre_shift": centre_shift,
            "study_axis": axis,
            "landmark_type_weight": torch.zeros(
                len(self.spatial_tokens.plane_lookup), device=features.device
            ).index_add(0, token_type, front_weights.sum(-1))[None],
            "study_left_probability": head_left,
            "side_logit": side_logit,
            "side_present": side_present,
            "mirrored_study": mirrored_study,
            "batch_size": batch_size,
        }

    def _base_tokens(self, encoded: dict) -> torch.Tensor:
        return (
            encoded["features"]
            + encoded["plane"][:, None]
            + encoded["contrast"][:, None]
            + encoded["detail_view"][:, None, None]
            * self.detail_view_embedding.to(encoded["features"].dtype)[None, None]
        )

    def study_tokens(self, encoded: dict, shared: bool = False) -> torch.Tensor:
        tokens = self._base_tokens(encoded)
        code = self.position_code(encoded["position"], encoded["routable"], shared)
        return self.mil_norm((tokens + code.to(tokens.dtype)).flatten(0, 1))

'''
s = s[:start] + new_tail + s[end:]
sub(
    '''        return self._attention_pool(
            self.study_tokens(encoded),
            encoded["study"]''',
    '''        return self._attention_pool(
            self.study_tokens(encoded),
            self.study_tokens(encoded, shared=True),
            encoded["study"]''',
)
p.write_text(s)
