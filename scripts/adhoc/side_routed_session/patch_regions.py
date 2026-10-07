import re
import shutil
from pathlib import Path

p = Path("src/knee_abnormality/modeling/model.py")
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


def cut(start, end, new):
    global s
    assert s.count(start) == 1, start
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]


cut(
    "ROUTES = (\n",
    "ROUTING_WIDTHS = (",
    '''ROUTES = (
    "meniscus_medial",
    "meniscus_lateral",
    "oa_medial",
    "oa_lateral",
    "kneecap",
    "core",
    "collateral_medial",
    "baker",
    "femur",
    "tear_medial_front",
    "tear_medial_back",
    "tear_lateral_front",
    "tear_lateral_back",
    "facet_medial",
    "facet_lateral",
)
TARGET_ROUTES = {
    "Medial Meniscus": ("meniscus", "meniscus_medial"),
    "Lateral Meniscus": ("meniscus", "meniscus_lateral"),
    "Medial OA": ("oa", "oa_medial"),
    "Lateral OA": ("oa", "oa_lateral"),
    "PF OA": ("oa", "kneecap"),
}
OWN_ROUTES = {
    "ACL": "core",
    "MCL": "collateral_medial",
    "Effusion": "kneecap",
    "Synovitis": "femur",
    "Baker's": "baker",
}
FRAME_SIZES = ("band_top", "band_bottom", "core_radius")
FRAME_SIZE_START_PITCHES = (0.0, -2.0, 1.0)
''',
)
cut(
    "PRIOR_VARIANTS = {\n",
    "FRAME_ROTATIONS = (",
    '''PRIOR_VARIANTS = {
    "meniscus_medial": 0,
    "meniscus_lateral": 1,
    "kneecap": 2,
    "oa_medial": 0,
    "oa_lateral": 1,
}
''',
)
sub('                        if TARGET_ROUTES[target][1] == "front_up"\n', '                        if TARGET_ROUTES[target][1] == "kneecap"\n')
sub(
    "        self.frame_shift = nn.Parameter(torch.zeros(len(POSITION_AXES)))\n",
    "        self.frame_shift = nn.Parameter(torch.zeros(len(POSITION_AXES)))\n"
    "        self.frame_size = nn.Parameter(torch.zeros(len(FRAME_SIZES)))\n"
    "        self.register_buffer(\n"
    '            "frame_size_start",\n'
    "            torch.tensor(FRAME_SIZE_START_PITCHES),\n"
    "            persistent=False,\n"
    "        )\n",
)
sub(
    "                len(POSITION_AXES) + len(FRAME_ROTATIONS),\n",
    "                len(POSITION_AXES) + len(FRAME_ROTATIONS) + len(FRAME_SIZES),\n",
)
sub(
    "    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n        hidden = self.frame_head(\n",
    "    ) -> tuple[torch.Tensor, ...]:\n        hidden = self.frame_head(\n",
)
sub(
    "        rotation = self.frame_rotation(output[:, count:]) @ torch.stack(\n",
    "        rotation = self.frame_rotation(output[:, count : 2 * count]) @ torch.stack(\n",
)
sub(
    "        return shift, rotation, landmarks\n",
    "        sizes = (\n"
    "            self.frame_size_start + self.frame_size.float() + output[:, 2 * count :]\n"
    "        ) * study_pitch[:, None]\n"
    "        return shift, rotation, landmarks, sizes\n",
)
cut(
    "    def routing_weights(self, encoded: dict) -> torch.Tensor:\n",
    "    def _attention_pool(\n",
    '''    def _regions(self, encoded: dict, medial_sign: float) -> torch.Tensor:
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
        top, bottom, radius = encoded["frame_sizes"].float()[:, None].unbind(-1)

        def inside(distance: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
            return (0.5 + distance / (2 * extent)).clamp(0, 1)

        medial_side = inside(medial, half[..., 0])
        lateral_side = 1 - medial_side
        ahead = inside(front, half[..., 1])
        femur = inside(height - top, half[..., 2])
        tibia = inside(bottom - height, half[..., 2])
        joint = (1 - femur - tibia).clamp_min(0)
        distance = torch.hypot(medial, front).clamp_min(1e-6)
        radial = (
            (medial.abs() * half[..., 0] + front.abs() * half[..., 1]) / distance
        ).detach()
        core = inside(radius - distance, radial.clamp_min(1e-3))
        outer = 1 - core
        kneecap = outer * ahead * femur
        above_tibia = 1 - tibia
        return torch.stack(
            (
                medial_side * outer * joint,
                lateral_side * outer * joint,
                medial_side * (1 - kneecap),
                lateral_side * (1 - kneecap),
                femur * (ahead * outer + core),
                core * above_tibia,
                medial_side * outer * above_tibia,
                medial_side * (1 - ahead) * outer * above_tibia,
                femur,
                medial_side * outer * joint * ahead,
                medial_side * outer * joint * (1 - ahead),
                lateral_side * outer * joint * ahead,
                lateral_side * outer * joint * (1 - ahead),
                kneecap * medial_side,
                kneecap * lateral_side,
            )
        )

    def routing_weights(self, encoded: dict) -> torch.Tensor:
        knee_x = encoded["knee_x"]
        with torch.autocast(device_type=knee_x.device.type, enabled=False):
            left = encoded["left_probability"][:, None].float()
            weights = left * self._regions(encoded, -1.0) + (1 - left) * self._regions(
                encoded, 1.0
            )
            return torch.where(encoded["routable"][:, None], weights, 0.5)

''',
)
sub(
    "            study_shift, study_rotation, landmarks = self.study_frames(\n",
    """            steps = torch.stack(
                (
                    patient_position[:, 1] - patient_position[:, 0],
                    patient_position[:, maps.shape[-1]] - patient_position[:, 0],
                    torch.einsum(
                        "nij,nj->ni", patient_maps[:, 2] - middle, homogeneous[:, 0]
                    ),
                ),
                1,
            )
            steps = torch.where(
                valid_position[:, None, None] & torch.isfinite(steps), steps, 0.0
            )
            token_box = torch.stack(
                (
                    steps[..., 0] * cosine + steps[..., 1] * sine,
                    steps[..., 0] * sine - steps[..., 1] * cosine,
                    steps[..., 2],
                ),
                -1,
            ).detach()
            fitted_position = torch.stack(
                (sign * knee_x, anterior_axis, superior_axis), -1
            )
            study_shift, study_rotation, landmarks, study_sizes = self.study_frames(
""",
)
sub(
    "            sign = (1 - 2 * left_probability)[:, None]\n            steps = torch.stack(\n",
    "            steps = torch.stack(\n",
)
sub(
    "            superior_axis = offset[..., 2]\n",
    "            superior_axis = offset[..., 2]\n            sign = (1 - 2 * left_probability)[:, None]\n",
)
sub(
    "                torch.stack((sign * knee_x, anterior_axis, superior_axis), -1),\n                visual_valid & valid_position[:, None],\n",
    "                fitted_position,\n                visual_valid & valid_position[:, None],\n",
)
sub(
    '                "frame_rotation": study_rotation[token_study],\n            }\n',
    '                "frame_rotation": study_rotation[token_study],\n'
    '                "frame_sizes": study_sizes[token_study],\n'
    '                "token_box": token_box,\n'
    "            }\n",
)
sub(
    '            "study_landmarks": landmarks,\n            "position": position,\n',
    '            "study_landmarks": landmarks,\n'
    '            "study_frame_sizes": study_sizes,\n'
    '            "position": position,\n'
    '            "fitted_position": fitted_position,\n',
)
sub(
    """            tokens = tokens + self.position_code(
                encoded["position"], encoded["routable"]
            ).to(tokens.dtype)
""",
    """            tokens = tokens + self.position_code(
                encoded["fitted_position"], encoded["routable"]
            ).to(tokens.dtype)
""",
)
sub(
    '            self.position_priors(encoded["position"], encoded["routable"]).flatten(1),\n',
    """            torch.where(
                self.shared_channel[self.target_channel][:, None, None],
                self.position_priors(encoded["position"], encoded["routable"]),
                self.position_priors(encoded["fitted_position"], encoded["routable"]),
            ).flatten(1),
""",
)
p.write_text(s)

shutil.copy2(
    "/tmp/claude-1000/-home-claudio-src-knee-abnormality/d2f4bec7-d221-4005-8675-56c8c41acf6d/scratchpad/run10-build/src/knee_abnormality/training/report_attributes.py",
    "src/knee_abnormality/training/report_attributes.py",
)
p = Path("src/knee_abnormality/training/report_attributes.py")
s = p.read_text()
cut(
    "QUESTION_ROUTES = {\n",
    "class Region(NamedTuple):",
    '''QUESTION_ROUTES = {
    "medial_meniscus_status": ("meniscus_status", "meniscus_medial"),
    "lateral_meniscus_status": ("meniscus_status", "meniscus_lateral"),
    "medial_meniscus_tear_type": ("meniscus_tear_type", "meniscus_medial"),
    "lateral_meniscus_tear_type": ("meniscus_tear_type", "meniscus_lateral"),
    "medial_cartilage_depth": ("cartilage_depth", "oa_medial"),
    "lateral_cartilage_depth": ("cartilage_depth", "oa_lateral"),
    "pf_cartilage_depth": ("cartilage_depth", "kneecap"),
    "medial_cartilage_extent": ("cartilage_extent", "oa_medial"),
    "lateral_cartilage_extent": ("cartilage_extent", "oa_lateral"),
    "pf_cartilage_extent": ("cartilage_extent", "kneecap"),
    "acl_status": ("acl_status", "core"),
    "extra_pcl": ("extra_pcl", "core"),
}


''',
)
sub('            f"{side}_{part}",\n', '            f"tear_{side}_{part}",\n')
for old, new in (
    ('        "medial_rest",\n        "marrow_edema_location",', '        "oa_medial",\n        "marrow_edema_location",'),
    ('        "lateral_rest",\n        "marrow_edema_location",', '        "oa_lateral",\n        "marrow_edema_location",'),
    ('        "front_up",\n        "marrow_edema_location",', '        "kneecap",\n        "marrow_edema_location",'),
    ('        "front_up_medial",\n', '        "facet_medial",\n'),
    ('        "front_up_lateral",\n', '        "facet_lateral",\n'),
):
    sub(old, new)
p.write_text(s)

p = Path("src/knee_abnormality/training/lightning.py")
s = p.read_text()
cut(
    "    def _report_attribute_step(self, encoded: dict, batch: dict) -> torch.Tensor:\n",
    "    def _scanner_confusion_step(",
    '''    def _report_attribute_step(
        self, model, encoded: dict, batch: dict, prefix: str
    ) -> torch.Tensor:
        heads = self.report_attribute_heads
        batch_size = len(batch["study_uid"])
        logits, region_logits = heads(
            model.study_tokens(encoded),
            encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            encoded["visual_valid"].flatten(),
            model.study_tokens(encoded, shared=True),
            model.routing_weights(encoded).flatten(1),
            encoded["batch_size"],
        )
        targets, weights, region_targets, region_weights, observed = heads.targets(
            batch["study_uid"]
        )
        loss, divergence, region_loss = report_attribute_loss(
            logits,
            targets,
            weights,
            region_logits,
            region_targets,
            region_weights,
            observed,
            heads.option_counts,
        )
        self.log(
            f"{prefix}report_attribute_{'fit' if prefix else 'loss'}",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        for region, value in zip(REGION_IDS, region_loss, strict=True):
            self.log(
                f"{prefix}report_attribute_bce_{region}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        if prefix:
            return loss
        self.log(
            "report_attribute_studies",
            observed.sum().float(),
            on_step=False,
            on_epoch=True,
            reduce_fx="sum",
            batch_size=1,
        )
        for question, value in zip(heads.question_ids, divergence, strict=True):
            self.log(
                f"report_attribute_kl_{question}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return self.config.trainer.report_attribute_loss_weight * loss

''',
)
sub(
    "        loss = loss + self._report_attribute_step(encoded, batch)\n",
    '        loss = loss + self._report_attribute_step(self.model, encoded, batch, "")\n',
)
sub(
    '        self._side_step(saved_model, batch, encoded, "val_")\n',
    '        self._side_step(saved_model, batch, encoded, "val_")\n'
    '        self._report_attribute_step(saved_model, encoded, batch, "val_")\n',
)
sub(
    '        study_shift = encoded["study_frame_shift"].detach().float()\n',
    '        study_sizes = encoded["study_frame_sizes"].detach().float()\n'
    "        for number, name in enumerate(FRAME_SIZES):\n"
    "            self._log_epoch_metric(\n"
    '                f"{prefix}study_frame_{name}_mm", study_sizes[:, number].mean(), studies\n'
    "            )\n"
    '        study_shift = encoded["study_frame_shift"].detach().float()\n',
)
sub("    FRAME_ROTATION_RANGE_RAD,\n    FRAME_ROTATIONS,\n", "    FRAME_ROTATION_RANGE_RAD,\n    FRAME_ROTATIONS,\n    FRAME_SIZES,\n")
sub("    ReportAttributeHeads,\n", "    REGION_IDS,\n    ReportAttributeHeads,\n")
p.write_text(s)

p = Path("src/knee_abnormality/training/report_attributes.py")
s = p.read_text()
sub(
    """        shared_tokens: torch.Tensor,
        shared_study: torch.Tensor,
        shared_valid: torch.Tensor,
        routing: torch.Tensor,
""",
    """        shared_tokens: torch.Tensor,
        routing: torch.Tensor,
""",
)
sub(
    "            shared_membership = (studies == shared_study[None]) & shared_valid[None]\n",
    "",
)
s = s.replace("                shared_membership,\n", "                membership,\n")
p.write_text(s)

for name in ("configs/training.yaml", "configs/refined-resnet50.yaml"):
    q = Path(name)
    t = q.read_text()
    assert t.count("jevk5-report-attribute-targets-v3.1.json") == 1, name
    q.write_text(t.replace("jevk5-report-attribute-targets-v3.1.json", "jevk5-report-attribute-targets-v4.0.json"))

p = Path("scripts/tools/check_side_routing.py")
s = p.read_text()
cut(
    "        partition = 0.0\n",
    '        results["geometry_is_the_only_assignment"]',
    """        sides = routing[
            [
                int(model.target_route[index[target]])
                for target in ("Medial Meniscus", "Lateral Meniscus")
            ]
        ].sum(0)[valid]
        partition = max(
            float((sides - 1).clamp_min(0).max()),
            float((routing[:, valid] - 0.5).abs().max()) - 0.5,
        )
        results["routing_partitions_every_token"] = partition < TOLERANCE
""",
)
p.write_text(s)

p = Path("scripts/tools/diagnose_encoding_usage.py")
s = p.read_text()
sub('    "frame_rotation",\n', '    "frame_rotation",\n    "frame_sizes",\n    "token_box",\n    "fitted_position",\n')
p.write_text(s)

p = Path("src/knee_abnormality/modeling/spatial_tokens.py")
s = p.read_text()
for key, value, append in (
    (
        "compartment_routing",
        "log_routing_weight_over_routing_temperature_added_to_attention_scores_weights_are_token_box_overlap_fractions_linear_in_signed_distance_over_projected_half_extent_of_four_quadrants_plus_core_cylinder_at_three_heights_femur_joint_tibia_around_the_study_frame_per_target_region_table_floor_1e-6_unrouted_below_one_token_contusion_and_fracture_unrouted",
        False,
    ),
    ("learned_centre", "_plus_per_study_band_top_band_bottom_core_radius_in_token_pitches_from_0_minus_2_and_1", True),
    ("algorithm", "_eighteen_regions_overlap_routed_reports_fixed_frame_own_position", True),
):
    pattern = re.compile(rf'^    "{key}": "(.*)",$', re.M)
    old = pattern.search(s).group(1)
    s = pattern.sub(f'    "{key}": "{old + value if append else value}",', s)
p.write_text(s)
