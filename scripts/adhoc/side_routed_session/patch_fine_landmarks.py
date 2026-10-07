from pathlib import Path

p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


def cut(start, end, new):
    global s
    assert s.count(start) == 1, start
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]


sub("    backbone_mapping_identity,\n", "    backbone_mapping_identity,\n    fine_backbone_mapping,\n")
sub(
    'POSITION_AXES = ("medial", "front", "up")\n',
    'POSITION_AXES = ("medial", "front", "up")\nLANDMARKS = ("centre", "front", "up")\n'
    'FINE_KEYS = (\n    "features",\n    "visual_valid",\n    "knee_x",\n    "anterior_axis",\n'
    '    "superior_axis",\n    "token_pitch",\n    "position",\n)\n',
)
sub(
    "        self.projection = nn.Linear(spatial_output.shape[1], PROJECTION_WIDTH)\n",
    "        self.projection = nn.Linear(spatial_output.shape[1], PROJECTION_WIDTH)\n"
    "        if not self.resnet_stages:\n"
    '            raise ValueError("Side-routed heads require a ResNet stride-16 stage")\n'
    "        self.fine_mapping = fine_backbone_mapping(self.backbone)\n"
    "        with torch.random.fork_rng(devices=[]):\n"
    "            self.fine_projection = nn.Linear(\n"
    "                self.backbone.layer3[-1].bn2.num_features\n"
    '                if not hasattr(self.backbone.layer3[-1], "bn3")\n'
    "                else self.backbone.layer3[-1].bn3.num_features,\n"
    "                PROJECTION_WIDTH,\n"
    "            )\n"
    "            self.fine_norm = nn.LayerNorm(PROJECTION_WIDTH)\n",
)
cut(
    "        self.frame_shift = nn.Parameter(torch.zeros(len(POSITION_AXES)))\n",
    "        shortest, longest, count = POSITION_WAVELENGTHS_MM\n",
    """        with torch.random.fork_rng(devices=[]):
            self.frame_head = nn.Sequential(
                nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH), nn.Tanh()
            )
            self.frame_attention = nn.Linear(MIL_ATTENTION_WIDTH, len(LANDMARKS))
            nn.init.zeros_(self.frame_attention.weight)
            nn.init.zeros_(self.frame_attention.bias)
""",
)
cut(
    "    def _encode_images(\n",
    "    def _backbone_to_stride_8(",
    """    def _encode_images(
        self,
        images: torch.Tensor,
        acquisition_index: torch.Tensor,
        acquisition_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._backbone_to_stride_8(images)
        if self.training and self.feature_statistics_settings.probability > 0:
            features, self.feature_statistics_activity = (
                apply_feature_statistics_augmentation(
                    features,
                    acquisition_index,
                    acquisition_type,
                    self.feature_statistics_settings,
                )
            )
        fine = self.backbone.layer3(features)
        return fine, self.backbone.layer4(fine)

""",
)
cut(
    "    def _backbone_from_stride_8(",
    "    def _effective_presence(",
    "",
)
cut(
    "    def frame_rotation(self, offset: torch.Tensor | None = None) -> torch.Tensor:\n",
    "    def position_code(\n",
    '''    def frame_rotation(self) -> torch.Tensor:
        nod, roll, turn = (
            FRAME_ROTATION_RANGE_RAD * torch.tanh(self.frame_rotation_logit.float())
        ).unbind(-1)
        zero = torch.zeros_like(nod)
        return torch.linalg.matrix_exp(
            torch.stack(
                (
                    torch.stack((zero, -turn, roll), -1),
                    torch.stack((turn, zero, -nod), -1),
                    torch.stack((-roll, nod, zero), -1),
                ),
                -2,
            )
        )

    @staticmethod
    def rotation_angles(rotation: torch.Tensor) -> torch.Tensor:
        rotation = rotation.float()
        vector = torch.stack(
            (
                rotation[..., 2, 1] - rotation[..., 1, 2],
                rotation[..., 0, 2] - rotation[..., 2, 0],
                rotation[..., 1, 0] - rotation[..., 0, 1],
            ),
            -1,
        )
        sine = vector.norm(dim=-1, keepdim=True) / 2
        cosine = (rotation.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) - 1) / 2
        angle = torch.atan2(sine, cosine)
        return torch.rad2deg(
            torch.where(sine > 1e-6, angle / (2 * sine.clamp_min(1e-6)), 0.5) * vector
        )

    def study_frames(
        self,
        tokens: torch.Tensor,
        position: torch.Tensor,
        valid: torch.Tensor,
        token_study: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.frame_head(
            tokens.detach().float()
            + self.position_code(position, valid.any(-1)).float()
        ).flatten(0, 1)
        points = position.detach().float().flatten(0, 1)
        member = (
            torch.arange(batch_size, device=tokens.device)[:, None]
            == token_study[:, None].expand_as(valid).flatten()[None]
        ) & valid.flatten()[None]
        present = member.any(-1)
        allowed = member.clone()
        allowed[:, 0] |= ~present
        uniform = allowed.float() / allowed.sum(-1, keepdim=True)
        attention = (
            self.frame_attention(hidden)
            .transpose(0, 1)[None]
            .masked_fill(~allowed[:, None], -torch.inf)
            .softmax(-1)
        )
        mean = uniform @ points
        spread = torch.where(
            present[:, None],
            (uniform @ points.square() - mean.square()).clamp_min(1e-6).sqrt(),
            1.0,
        )
        landmarks = attention @ points
        centre, front, up = (
            (landmarks - mean[:, None]) * present[:, None, None]
        ).unbind(1)
        axes = torch.eye(len(POSITION_AXES), device=tokens.device)
        forward = nn.functional.normalize(
            spread[:, 1:2] * axes[1] + front - centre, dim=-1
        )
        upward = spread[:, 2:3] * axes[2] + up - centre
        upward = nn.functional.normalize(
            upward - (upward * forward).sum(-1, keepdim=True) * forward, dim=-1
        )
        rotation = self.frame_rotation() @ torch.stack(
            (torch.linalg.cross(forward, upward), forward, upward), 1
        )
        return centre, rotation, landmarks

    def frame(self, encoded: dict, medial_sign: torch.Tensor | float) -> torch.Tensor:
        return (
            torch.stack(
                (
                    medial_sign * encoded["knee_x"].float(),
                    encoded["anterior_axis"].float(),
                    encoded["superior_axis"].float(),
                ),
                -1,
            )
            - encoded["frame_shift"][:, None]
        ) @ encoded["frame_rotation"].transpose(-1, -2)

    def fine(self, encoded: dict) -> dict:
        return {**encoded, **{key: encoded[f"fine_{key}"] for key in FINE_KEYS}}

''',
)
cut(
    "    def _attention_pool(\n",
    "        if self.view_label_bias is not None:\n",
    """    def _attention_pool(
        self, tokens: torch.Tensor, encoded: dict, view: int | None
    ) -> torch.Tensor:
        diagnostic = view is not None
        batch_size = encoded["batch_size"]
        image_type = encoded["image_type"]
        visual_valid = encoded["visual_valid"]
        token_study = encoded["study"][:, None].expand_as(visual_valid).flatten()
        token_view = encoded["view_index"][:, None].expand_as(visual_valid).flatten()
        token_valid = (
            visual_valid
            if view is None
            else visual_valid & (encoded["view_index"] == view)[:, None]
        ).flatten()
        routing = self.routing_weights(encoded).flatten(1)
        prior = self.position_priors(encoded["position"], encoded["routable"]).flatten(
            1
        )
        scores = self.attention(tokens).transpose(0, 1)
""",
)
sub(
    """                evidence = torch.where(
                    self.shared_channel,
                    self.head(shared_tokens, image_type),
                    self.head(tokens, image_type),
                )
""",
    """                evidence = self.head(tokens, image_type)
""",
)
sub(
    """                pooled = torch.where(
                    self.shared_channel[self.target_channel][None, :, None],
                    torch.einsum("btn,nf->btf", attention, shared_tokens.float()),
                    torch.einsum("btn,nf->btf", attention, tokens.float()),
                )
""",
    """                pooled = torch.einsum("btn,nf->btf", attention, tokens.float())
""",
)
sub(
    """                shared_evidence = nn.functional.linear(
                    shared_tokens.float(),
""",
    """                shared_evidence = nn.functional.linear(
                    tokens.float(),
""",
)
sub(
    """        maps = self._encode_images(valid_images, acquisition_index, token_type)
        features = self.projection(maps.flatten(2).transpose(1, 2))
""",
    """        fine_maps, maps = self._encode_images(
            valid_images, acquisition_index, token_type
        )
        features = self.projection(maps.flatten(2).transpose(1, 2))
        fine_features = self.fine_projection(fine_maps.flatten(2).transpose(1, 2))
        fine_valid, fine_xy = feature_geometry(
            source_valid_rects,
            inverse_affine,
            fine_maps.shape[-2:],
            self.fine_mapping,
            channels,
        )
""",
)
sub(
    """        tokens = self.mil_norm(self._base_tokens(encoded))
        with torch.autocast(device_type=features.device.type, enabled=False):
            homogeneous = torch.cat(
                (prepared_xy.float(), torch.ones_like(prepared_xy[..., :1])), -1
            )
            middle = patient_maps[:, 1]
            patient_position = torch.einsum("nij,nkj->nki", middle, homogeneous)
""",
    """        tokens = self.mil_norm(self._base_tokens(encoded))
        fine_tokens = self.fine_norm(
            self._base_tokens({**encoded, "features": fine_features})
        )
        with torch.autocast(device_type=features.device.type, enabled=False):
            middle = patient_maps[:, 1]

            def patient(xy: torch.Tensor) -> torch.Tensor:
                return torch.einsum(
                    "nij,nkj->nki",
                    middle,
                    torch.cat((xy.float(), torch.ones_like(xy[..., :1])), -1),
                )

            patient_position = patient(prepared_xy)
            fine_patient = patient(fine_xy)
""",
)
sub(
    """                & torch.isfinite(patient_position).all((-1, -2))
            )
            offset = torch.where(
                valid_position[:, None, None], patient_position - center[:, None], 0.0
            ).detach()
""",
    """                & torch.isfinite(patient_position).all((-1, -2))
                & torch.isfinite(fine_patient).all((-1, -2))
            )
""",
)
sub(
    """            knee_x = offset[..., 0] * cosine + offset[..., 1] * sine
            anterior_axis = offset[..., 0] * sine - offset[..., 1] * cosine
""",
    """
            def axes(points: torch.Tensor) -> tuple[torch.Tensor, ...]:
                offset = torch.where(
                    valid_position[:, None, None], points - center[:, None], 0.0
                ).detach()
                return (
                    offset[..., 0] * cosine + offset[..., 1] * sine,
                    offset[..., 0] * sine - offset[..., 1] * cosine,
                    offset[..., 2],
                    torch.where(
                        valid_position, (points[:, 1] - points[:, 0]).norm(dim=-1), 1.0
                    ).detach(),
                )

            knee_x, anterior_axis, superior_axis, token_pitch = axes(patient_position)
            fine_x, fine_anterior, fine_superior, fine_pitch = axes(fine_patient)
""",
)
cut(
    "            superior_axis = offset[..., 2]\n",
    "            position = self.frame(geometry, sign)\n",
    """            sign = (1 - 2 * left_probability)[:, None]
            study_shift, study_rotation, landmarks = self.study_frames(
                fine_tokens,
                torch.stack((sign * fine_x, fine_anterior, fine_superior), -1),
                fine_valid & valid_position[:, None],
                token_study,
                batch_size,
            )
            study_frame = {
                "frame_shift": study_shift[token_study],
                "frame_rotation": study_rotation[token_study],
            }
            geometry = {
                "knee_x": knee_x,
                "anterior_axis": anterior_axis,
                "superior_axis": superior_axis,
                **study_frame,
            }
            fine_position = self.frame(
                {
                    "knee_x": fine_x,
                    "anterior_axis": fine_anterior,
                    "superior_axis": fine_superior,
                    **study_frame,
                },
                sign,
            )
""",
)
sub(
    """            "study_frame_shift": study_shift,
            "study_frame_angle": study_angle,
            "position": position,
""",
    """            "study_frame_shift": study_shift,
            "study_frame_rotation": study_rotation,
            "study_landmarks": landmarks,
            "position": position,
            "fine_features": fine_features,
            "fine_visual_valid": fine_valid,
            "fine_knee_x": fine_x,
            "fine_anterior_axis": fine_anterior,
            "fine_superior_axis": fine_superior,
            "fine_token_pitch": fine_pitch,
            "fine_position": fine_position,
""",
)
cut(
    "    def study_tokens(self, encoded: dict, shared: bool = False) -> torch.Tensor:\n",
    "    def forward(self, batch: dict) -> torch.Tensor:\n",
    """    def study_tokens(self, encoded: dict, shared: bool = False) -> torch.Tensor:
        if shared:
            return self.fine_norm(
                self._base_tokens(self.fine(encoded)).flatten(0, 1)
            )
        tokens = self._base_tokens(encoded) + self.position_code(
            encoded["position"], encoded["routable"]
        ).to(encoded["features"].dtype)
        return self.mil_norm(tokens.flatten(0, 1))

    def pool_encoded(self, encoded: dict, view: int | None = None) -> torch.Tensor:
        shared_target = self.shared_channel[self.target_channel]
        own = self._attention_pool(self.study_tokens(encoded), encoded, view)
        if view is None:
            own_mass = self.attention_mass_view
        shared = self._attention_pool(
            self.study_tokens(encoded, shared=True), self.fine(encoded), view
        )
        if view is None:
            self.attention_mass_view = torch.where(
                shared_target, self.attention_mass_view, own_mass
            )
        return torch.where(shared_target, shared, own)

""",
)
p.write_text(s)

t = Path("src/knee_abnormality/modeling/spatial_tokens.py")
u = t.read_text()
old = "def _resnet_mapping(backbone: nn.Module) -> tuple[float, float, float, float]:\n"
assert u.count(old) == 1
u = u.replace(
    old,
    "def _resnet_mapping(\n    backbone: nn.Module, stage_count: int = 4\n) -> tuple[float, float, float, float]:\n",
)
old = "    for stage in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4):\n"
assert u.count(old) == 1
u = u.replace(
    old,
    "    for stage in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)[\n        :stage_count\n    ]:\n",
)
old = "def feature_geometry(\n"
assert u.count(old) == 1
u = u.replace(
    old,
    "def fine_backbone_mapping(backbone: nn.Module) -> tuple[float, float, float, float]:\n    return _resnet_mapping(backbone, 3)\n\n\ndef feature_geometry(\n",
)
t.write_text(u)
