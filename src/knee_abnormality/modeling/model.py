import math

import timm
import torch
import torch.nn.functional as F
from torch import nn

from knee_abnormality.config import SLICES_PER_INPUT, DataConfig, ModelConfig
from knee_abnormality.evaluation.metrics import COMPARTMENT_PAIRS
from knee_abnormality.inference.reflection import reflect_crops
from knee_abnormality.modeling.augmentation import (
    ImageAugmentation,
    affine_matrices,
    apply_contrast_mixing,
    apply_feature_statistics_augmentation,
    apply_reflection,
    augmented_to_prepared,
    reflection_channels,
    reflection_inverse,
    reflection_valid_rects,
    sample_augmentation_parameters,
    sample_contrast_mixing_parameters,
    sample_reflection_parameters,
)
from knee_abnormality.modeling.plane_readout import PlaneContrastReadout
from knee_abnormality.modeling.radimagenet import (
    RADIMAGENET_WEIGHTS,
    radimagenet_state_dict,
    resnet_avg_down_state_dict,
)
from knee_abnormality.modeling.spatial_tokens import (
    COATNET_BACKBONE,
    SpatialTokens,
    backbone_mapping,
    backbone_mapping_identity,
    feature_geometry,
)
from knee_abnormality.training.anatomy import MENISCUS_POINTS


def sample_windows(
    canvas: torch.Tensor, origin: torch.Tensor, size: int
) -> torch.Tensor:
    height, width = canvas.shape[-2:]
    integral = origin.long()
    lines = torch.arange(size, device=canvas.device)
    rows = integral[:, 0, None] + lines
    columns = integral[:, 1, None] + lines
    indices = (
        rows.clamp(0, height - 1)[:, :, None] * width
        + columns.clamp(0, width - 1)[:, None, :]
    )
    exact = (
        canvas.flatten(2)
        .gather(2, indices.flatten(1)[:, None].expand(-1, canvas.shape[1], -1))
        .view(len(canvas), canvas.shape[1], size, size)
    )
    inside = ((rows >= 0) & (rows < height))[:, :, None] & (
        (columns >= 0) & (columns < width)
    )[:, None, :]
    return exact * inside[:, None]


def crop_rectangles(
    rects: torch.Tensor, origin: torch.Tensor, size: int
) -> torch.Tensor:
    first = rects[..., :2] - origin
    last = first + rects[..., 2:]
    first = first.clamp(0, size)
    last = last.clamp(0, size)
    return torch.cat((first, (last - first).clamp_min(0)), -1)


MAX_VIEW_ROTATION = math.radians(12.0)
VIEW_ROTATION_JITTER_DEGREES = 3.0
VIEW_SHIFT_JITTER_MM = 3.0
CORONAL_JOINT_TARGET_MM = (-2.075445, 2.031255)
SAGITTAL_JOINT_TARGET_MM = (4.894627, 2.727388)


def rotated_rectangles(
    rects: torch.Tensor, affine: torch.Tensor, size: int
) -> torch.Tensor:
    top, left, rows, columns = rects.float().unbind(-1)
    right, bottom = left + columns - 1, top + rows - 1
    x = torch.stack((left, right, right, left), -1)
    y = torch.stack((top, top, bottom, bottom), -1)
    corners = torch.stack((x, y, torch.ones_like(x)), -1)
    inverse = torch.linalg.inv(affine.float())
    placed = torch.einsum("ntij,ntckj->ntcki", inverse, corners)[..., :2]
    first = placed.amin(-2).clamp(0, size - 1)
    last = placed.amax(-2).clamp(0, size - 1)
    extent = (last - first + 1).clamp_min(0)
    extent = torch.where(((rows > 0) & (columns > 0))[..., None], extent, 0.0)
    return torch.cat((first.flip(-1), extent.flip(-1)), -1)


class ViewWindows(nn.Module):
    def __init__(self, data: DataConfig) -> None:
        super().__init__()
        self.image_size = data.image_size
        views = [view for _, view in data.views]
        self.first_look_width = max(view.slices for view in views)
        with torch.random.fork_rng(devices=[]):
            self.register_buffer(
                "margin", torch.full((len(views),), 72.0), persistent=False
            )
            self.register_buffer(
                "pixel_spacing",
                torch.tensor([view.field_mm / data.image_size for view in views]),
                persistent=False,
            )
            self.register_buffer(
                "shift_limit",
                torch.tensor(
                    [
                        round(
                            (8.0 if data.series_types[kind].plane == "Axial" else 18.0)
                            / (view.field_mm / data.image_size)
                        )
                        for kind, view in data.views
                    ],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.register_buffer(
                "rotation_limit",
                torch.tensor(
                    [
                        math.radians(20.0)
                        if kind.plane == "Axial"
                        else MAX_VIEW_ROTATION
                        for kind in data.series_types
                    ]
                ),
                persistent=False,
            )
            self.register_buffer(
                "sagittal",
                torch.tensor([kind.plane == "Sagittal" for kind in data.series_types]),
                persistent=False,
            )
            self.register_buffer(
                "coronal",
                torch.tensor([kind.plane == "Coronal" for kind in data.series_types]),
                persistent=False,
            )
            self.register_buffer(
                "coronal_target",
                torch.tensor(CORONAL_JOINT_TARGET_MM),
                persistent=False,
            )
            self.register_buffer(
                "sagittal_target",
                torch.tensor(SAGITTAL_JOINT_TARGET_MM),
                persistent=False,
            )
            self.register_buffer(
                "first_look_slices",
                torch.tensor([view.slices for view in views]),
                persistent=False,
            )

    def first_look_batch(self, batch: dict) -> dict:
        primary_count = len(batch["primary_series_index"])
        batch = {
            **batch,
            **{
                key: batch[key][:primary_count]
                for key in (
                    "images",
                    "slice_mask",
                    "slice_positions",
                    "valid_rects",
                    "patient_maps",
                    "geometry_valid",
                    "view_index",
                    "series_study_index",
                    "series_type_index",
                    "window_center_mm",
                    "window_rule_distance_mm",
                    "knee_side_sign",
                    "knee_center_x_mm",
                    "knee_center_y_mm",
                    "knee_center_z_mm",
                    "source_series_index",
                )
            },
        }
        mask = batch["slice_mask"]
        count = mask.sum(-1)
        budget = torch.minimum(count, self.first_look_slices[batch["view_index"]])
        width = self.first_look_width
        steps = torch.arange(width, device=mask.device)[None]
        ranks = (
            (
                steps
                * (count - 1).clamp_min(0)[:, None]
                / (budget - 1).clamp_min(1)[:, None]
            )
            .round()
            .long()
        )
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        ordered = torch.where(mask, positions, mask.shape[1]).sort(-1).values
        indices = ordered.gather(1, ranks.clamp_max(mask.shape[1] - 1)).clamp_max(
            mask.shape[1] - 1
        )
        prepared = {
            key: value for key, value in batch.items() if key != "mirrored_study"
        }
        prepared["images"] = (
            batch["images"][:, :, 1:2]
            if batch["images"].shape[2] > 1
            else batch["images"]
        )
        for key in (
            "images",
            "slice_mask",
            "slice_positions",
            "valid_rects",
            "patient_maps",
            "geometry_valid",
        ):
            value = prepared[key]
            selection = indices.reshape(*indices.shape, *([1] * (value.ndim - 2)))
            prepared[key] = value.gather(
                1, selection.expand(*indices.shape, *value.shape[2:])
            )
        prepared["slice_mask"] = steps < budget[:, None]
        return prepared

    def geometry(self, batch: dict) -> tuple[torch.Tensor, ...]:
        maps = batch["patient_maps"].float()[:, :, 1]
        linear = torch.where(torch.isfinite(maps[..., :2]), maps[..., :2], 0.0)
        length = linear.norm(dim=-2)
        usable = (length > 1e-4).all(-1) & batch["geometry_valid"] & batch["slice_mask"]
        usable = usable & torch.isfinite(maps).all((-1, -2))
        series = batch["series_type_index"]
        coronal = self.coronal[series][:, None]
        sagittal = self.sagittal[series][:, None]
        movable = (self.margin[batch["view_index"]] > 0)[:, None]
        usable = usable & movable
        half = (self.image_size - 1) / 2
        offset = torch.where(
            coronal[..., None],
            self.coronal_target,
            torch.where(sagittal[..., None], self.sagittal_target, 0.0),
        ) / length.clamp_min(1e-4)
        return maps, linear, length, usable, half + offset

    def axial_front_center(self, encoded: dict) -> torch.Tensor:
        front = encoded["position"].detach().float()[..., 1].flatten()
        valid = encoded["visual_valid"] & encoded["routable"][:, None]
        shares = encoded["anatomy_shares"].detach().float()
        weights = (
            torch.stack((shares[..., FEMUR_CLASS], shares[..., KNEECAP_CLASS]), -1)
            * valid[..., None]
        )
        member = encoded["study"][:, None].expand_as(valid).flatten()
        order = front.argsort()
        order = order[member[order].argsort(stable=True)]
        ordered_member = member[order]
        weights = weights.flatten(0, 1)[order]
        mass = torch.zeros(encoded["batch_size"], 2, device=front.device).index_add(
            0, ordered_member, weights
        )
        before = mass.cumsum(0) - mass
        cumulative = weights.cumsum(0) - before[ordered_member]
        target = torch.stack((0.05 * mass[:, 0], 0.95 * mass[:, 1]), -1)[ordered_member]
        eligible = (cumulative >= target) & (weights > 0)
        candidates = torch.where(eligible, front[order, None], float("inf"))
        bounds = torch.full_like(mass, float("inf")).scatter_reduce(
            0,
            ordered_member[:, None].expand_as(weights),
            candidates,
            reduce="amin",
            include_self=True,
        )
        center = (bounds[:, 0] - 30.0 + bounds[:, 1]) / 2
        return torch.where((mass > 0).all(-1) & torch.isfinite(center), center, 0.0)

    def frame_pose(self, batch: dict, encoded: dict) -> torch.Tensor:
        with torch.autocast(device_type=batch["images"].device.type, enabled=False):
            study = batch["series_study_index"]
            studies = encoded["batch_size"]
            side = torch.zeros(studies, device=study.device).index_add(
                0, encoded["study"], 1 - 2 * encoded["left_probability"].float()
            )
            side = torch.where(side < 0, -1.0, 1.0)[study]
            sign = batch["knee_side_sign"]
            own = (
                batch["knee_side_split"][study]
                & ~batch["knee_side_certain"][study]
                & (sign != 0)
            )
            side = torch.where(own, torch.where(sign < 0, -1.0, 1.0), side)
            cosine, sine = encoded["study_axis"].float()[study].unbind(-1)
            rotation = encoded["study_frame_rotation"].detach().float()[study]
            up = rotation[:, 2]
            centre = encoded["study_frame_shift"].float()[study]

            def patient(vector: torch.Tensor) -> torch.Tensor:
                return torch.stack(
                    (
                        side * vector[:, 0] * cosine + vector[:, 1] * sine,
                        side * vector[:, 0] * sine - vector[:, 1] * cosine,
                        vector[:, 2],
                    ),
                    -1,
                )

            knee = torch.stack(
                [
                    batch[key].float()
                    for key in (
                        "knee_center_x_mm",
                        "knee_center_y_mm",
                        "knee_center_z_mm",
                    )
                ],
                -1,
            )
            maps, linear, length, usable, pivot = self.geometry(batch)
            across, down = torch.einsum(
                "ntpa,np->nta",
                linear / length.clamp_min(1e-6)[..., None, :],
                patient(up),
            ).unbind(-1)
            angle = torch.atan(-across / torch.where(down.abs() < 1e-6, 1e-6, down))
            axial = ~(
                self.sagittal[batch["series_type_index"]]
                | self.coronal[batch["series_type_index"]]
            )
            medial_across, medial_down = torch.einsum(
                "ntpa,np->nta",
                linear / length.clamp_min(1e-6)[..., None, :],
                patient(rotation[:, 0]),
            ).unbind(-1)
            axial_angle = torch.atan(
                medial_down
                / torch.where(medial_across.abs() < 1e-6, 1e-6, medial_across)
            )
            angle = torch.where(axial[:, None], axial_angle, angle)
            turn_limit = self.rotation_limit[batch["series_type_index"]][:, None]
            angle = angle.clamp(-turn_limit, turn_limit)
            front = self.axial_front_center(encoded)[study]
            center = centre + axial[:, None] * front[:, None] * rotation[:, 1]
            point = (knee + patient(center))[:, None] - maps[..., 2]
            point = torch.where(torch.isfinite(point), point, 0.0)
            gram = torch.einsum("ntpa,ntpb->ntab", linear, linear)
            gram = gram + 1e-6 * torch.eye(2, device=gram.device)
            pixel = torch.linalg.solve(
                gram, torch.einsum("ntpa,ntp->nta", linear, point)
            )
            margin = self.margin[batch["view_index"]][:, None, None]
            limit = self.shift_limit[batch["view_index"]][:, None, None]
            shift = (pixel - margin - pivot).clamp(-limit, limit).round()
            usable = usable & torch.isfinite(angle) & torch.isfinite(shift).all(-1)
            pose = torch.cat((angle[..., None], shift), -1)
            return torch.where(usable[..., None], pose, 0.0)

    def jitter(self, batch: dict, pose: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=pose.device.type, enabled=False):
            _, _, length, usable, _ = self.geometry(batch)
            source = batch["source_series_index"]
            series = len(batch["primary_series_index"])
            turn = (
                math.radians(VIEW_ROTATION_JITTER_DEGREES)
                * (2 * torch.rand((series, 1), device=pose.device) - 1)[source]
            )
            move = (
                VIEW_SHIFT_JITTER_MM
                * (2 * torch.rand((series, 1, 2), device=pose.device) - 1)[source]
                / length.clamp_min(1e-4)
            ).round()
            limit = self.shift_limit[batch["view_index"]][:, None, None]
            turn_limit = self.rotation_limit[batch["series_type_index"]][:, None, None]
            moved = torch.cat(
                (
                    (pose[..., :1] + turn[..., None]).clamp(-turn_limit, turn_limit),
                    (pose[..., 1:3] + move).clamp(
                        -limit,
                        limit,
                    ),
                ),
                -1,
            )
            moved = torch.where(usable[..., None], moved, 0.0)
            return torch.cat((moved, pose[..., 3:]), -1)

    def forward(
        self,
        batch: dict,
        middle_only: bool = False,
        pose: torch.Tensor | None = None,
    ) -> dict:
        with torch.autocast(device_type=batch["images"].device.type, enabled=False):
            canvas = batch["images"]
            maps = batch["patient_maps"].float()
            margin = self.margin[batch["view_index"]][:, None]
            origin = margin[..., None].expand(-1, canvas.shape[1], 2)
            if middle_only and canvas.shape[2] > 1:
                canvas = canvas[:, :, 1:2]
            affine = torch.eye(3, device=canvas.device).repeat(
                len(canvas), canvas.shape[1], 1, 1
            )
            if pose is None:
                crops = (
                    sample_windows(
                        canvas.flatten(0, 1),
                        origin.flatten(0, 1),
                        self.image_size,
                    )
                    .float()
                    .view(*canvas.shape[:3], self.image_size, self.image_size)
                )
                affine[..., :2, 2] = origin
                angle = None
            else:
                angle = pose[..., 0]
                shift = pose[..., 1:3]
                canvas_pivot = self.geometry(batch)[4]
                scale = pose[..., 3]
                half = (self.image_size - 1) / 2
                pivot = half + (canvas_pivot - half) / scale[..., None]
                cosine, sine = scale * angle.cos(), scale * angle.sin()
                affine[..., 0, 0], affine[..., 0, 1] = cosine, -sine
                affine[..., 1, 0], affine[..., 1, 1] = sine, cosine
                affine[..., 0, 2] = (
                    margin
                    + canvas_pivot[..., 0]
                    + shift[..., 0]
                    - (cosine * pivot[..., 0] - sine * pivot[..., 1])
                )
                affine[..., 1, 2] = (
                    margin
                    + canvas_pivot[..., 1]
                    + shift[..., 1]
                    - (sine * pivot[..., 0] + cosine * pivot[..., 1])
                )
                flat_canvas = canvas.flatten(0, 1)
                lines = torch.arange(
                    self.image_size, device=canvas.device, dtype=torch.float32
                )
                rows, columns = torch.meshgrid(lines, lines, indexing="ij")
                points = torch.stack((columns, rows, torch.ones_like(rows)), -1)
                placed = torch.einsum(
                    "nij,hwj->nhwi", affine.flatten(0, 1)[:, :2, :], points
                )
                height, width = canvas.shape[-2:]
                grid = torch.stack(
                    (
                        (2 * placed[..., 0] + 1) / width - 1,
                        (2 * placed[..., 1] + 1) / height - 1,
                    ),
                    -1,
                )
                crops = F.grid_sample(
                    flat_canvas.float(),
                    grid,
                    mode="bicubic",
                    padding_mode="zeros",
                    align_corners=False,
                ).clamp(0, 255)
                crops = crops.view(*canvas.shape[:3], self.image_size, self.image_size)
            prepared = {
                **batch,
                "images": crops.expand(-1, -1, 3, -1, -1),
                "valid_rects": (
                    crop_rectangles(
                        batch["valid_rects"], origin[:, :, None], self.image_size
                    )
                    if angle is None
                    else rotated_rectangles(
                        batch["valid_rects"], affine, self.image_size
                    )
                ),
                "patient_maps": maps @ affine[:, :, None],
                "crop_affine": affine,
                "canvas_valid_rects": batch["valid_rects"],
                "applied_view_angle": torch.zeros_like(batch["slice_positions"])
                if angle is None
                else angle,
                "applied_view_shift": torch.zeros_like(affine[..., :2, 2])
                if pose is None
                else pose[..., 1:3],
            }
            return (
                reflect_crops(prepared, self.sagittal)
                if "mirrored_study" in batch
                else prepared
            )


PROJECTION_WIDTH = 256
COMPARTMENT_MIDLINE_BAND_MM = 10.0
MIL_ATTENTION_WIDTH = 128
REXNET_STRIDE_8_BLOCK_COUNT = 5
SHARED_CHANNELS = ("meniscus", "oa")
ROUTES = (
    "meniscus_medial",
    "meniscus_lateral",
    "oa_medial",
    "oa_lateral",
    "kneecap",
    "core",
    "collateral_medial",
    "baker",
    "front_core",
    "tear_medial_anterior",
    "tear_medial_body",
    "tear_medial_posterior",
    "tear_lateral_anterior",
    "tear_lateral_body",
    "tear_lateral_posterior",
    "facet_medial",
    "facet_lateral",
    "bones",
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
    "Synovitis": "front_core",
    "MCL": "collateral_medial",
    "Effusion": "kneecap",
    "Baker's": "baker",
    "Contusion": "bones",
    "Fracture": "bones",
}
FRAME_SIZES = ("band_top", "band_bottom", "core_radius", "ring_radius")
FRAME_SIZE_START_PITCHES = (1.3, -0.9, 1.1, 2.2)
SECTORS = (
    "anteromedial",
    "anterior",
    "anterolateral",
    "lateral",
    "posterolateral",
    "posterior",
    "posteromedial",
    "medial",
)
ROUTING_WIDTHS = ("side",)
ROUTING_WIDTH_RANGE_MM = (1.0, 40.0)
ROUTING_INITIAL_WIDTH_MM = 5.0
ROUTING_MINIMUM_WEIGHT = 1e-6
POSITION_AXES = ("medial", "front", "up")
LANDMARKS = ("centre", "front", "up")
POSITION_WAVELENGTHS_MM = (8.0, 512.0, 7)
SHARED_PRIORS = ("meniscus", "oa", "front")
PRIOR_VARIANTS = {
    "meniscus_medial": 0,
    "meniscus_lateral": 1,
    "kneecap": 2,
    "oa_medial": 0,
    "oa_lateral": 1,
}
FRAME_ROTATIONS = ("nod", "roll", "turn")
FRAME_ROTATION_RANGE_RAD = math.pi / 2
PLANE_SCALE_MM = 50.0
PLANE_RIDGE = 0.5
PLANE_START_SLOPE = 0.05
PLANE_REFITS = 3
PLANE_REFIT_WIDTH_MM = 12.0
ANATOMY_CENTRE_WIDTH_MM = 10.0
ANATOMY_CLASS_COUNT = 7
MENISCUS_CLASS_COUNT = 3
SOFT_TISSUE_CLASS_COUNT = 7
(
    MEDIAL_MENISCUS_CLASS,
    LATERAL_MENISCUS_CLASS,
    MEDIAL_TIBIAL_CARTILAGE_CLASS,
    LATERAL_TIBIAL_CARTILAGE_CLASS,
    FEMORAL_CARTILAGE_CLASS,
    PATELLAR_CARTILAGE_CLASS,
) = range(1, SOFT_TISSUE_CLASS_COUNT)
MENISCUS_CORE_SHARE = 1 / (2 * MENISCUS_POINTS**2)
(
    FEMUR_CLASS,
    TIBIA_CLASS,
    KNEECAP_CLASS,
    FIBULA_CLASS,
    JOINT_CLASS,
    KNEECAP_JOINT_CLASS,
) = range(1, ANATOMY_CLASS_COUNT)
SECTION_LOGIT = 2.0
SECTION_RATE = 5.0
KNEECAP_FULL_SHARE = 0.3
FRONT_FLOOR = 0.25
ROUTE_SECTIONS = {
    "meniscus_medial": (0, 1, 1),
    "meniscus_lateral": (0, 1, 1),
    "oa_medial": (0, 1, 1),
    "oa_lateral": (0, 1, 1),
    "kneecap": (1, 0, 0),
    "core": (1, 1, 0),
    "collateral_medial": (1, 1, 1),
    "baker": (1, 1, 1),
    "front_core": (1, 0, 0),
    "tear_medial_anterior": (0, 1, 0),
    "tear_medial_body": (0, 1, 0),
    "tear_medial_posterior": (0, 1, 0),
    "tear_lateral_anterior": (0, 1, 0),
    "tear_lateral_body": (0, 1, 0),
    "tear_lateral_posterior": (0, 1, 0),
    "facet_medial": (1, 0, 0),
    "facet_lateral": (1, 0, 0),
    "bones": (1, 1, 1),
}
ANATOMICAL_BAND_PITCHES = (0.5, -0.5)
SIDE_COORDINATE_LOG_ODDS = 4.33


def channel_linear(layer: nn.Linear, rows: list[int], bias: bool) -> nn.Linear:
    selected = nn.utils.skip_init(nn.Linear, layer.in_features, len(rows), bias=bias)
    with torch.no_grad():
        selected.weight.copy_(layer.weight[rows])
        if bias:
            selected.bias.copy_(layer.bias[rows])
    return selected


class MaskDecoder(nn.Module):
    def __init__(self, widths: tuple[int, int, int]) -> None:
        super().__init__()
        self.second = nn.Conv2d(widths[0], 64, 1)
        self.third = nn.Conv2d(widths[1], 64, 1)
        self.fourth = nn.Conv2d(widths[2], 64, 1)
        self.third_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.second_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.bone = nn.Conv2d(64, ANATOMY_CLASS_COUNT, 1)
        self.soft_tissue = nn.Conv2d(64, SOFT_TISSUE_CLASS_COUNT, 1)

    def forward(
        self, second: torch.Tensor, third: torch.Tensor, fourth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        merged = self.third_block(
            self.third(third)
            + F.interpolate(
                self.fourth(fourth), scale_factor=2, mode="bilinear", align_corners=False
            )
        )
        merged = self.second_block(
            self.second(second)
            + F.interpolate(merged, scale_factor=2, mode="bilinear", align_corners=False)
        )
        return (
            self.bone(merged).flatten(2).transpose(1, 2),
            self.soft_tissue(merged).flatten(2).transpose(1, 2),
        )


class KneeAbnormalityModel(nn.Module):
    def __init__(
        self,
        model: ModelConfig,
        data: DataConfig,
        targets: tuple[str, ...],
        load_pretrained: bool | None = None,
    ) -> None:
        super().__init__()
        target_count = len(targets)
        self.series_dropout = model.series_dropout
        self.plane_readout = model.plane_readout
        self.augmentation_settings = data
        self.feature_statistics_settings = model.feature_statistics_augmentation
        self.image_augmentation = ImageAugmentation(data)
        self.register_buffer(
            "mean", torch.tensor(data.mean).reshape(1, -1, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(data.std).reshape(1, -1, 1, 1), persistent=False
        )
        self.channels_last = model.channels_last
        self._pooling_cache = None
        self.repeated_middle_slice = model.slice_channels == "repeated_middle"
        self.checkpoint_backbone = model.checkpoint_backbone
        self.backbone_eval = model.backbone_eval
        pretrained = model.pretrained if load_pretrained is None else load_pretrained
        self.feature_mapping_identity = backbone_mapping_identity(model.backbone)
        if (
            model.pretrained_weights == "radimagenet"
            and model.backbone not in RADIMAGENET_WEIGHTS
        ):
            raise ValueError(
                f"RadImageNet weights require one of {sorted(RADIMAGENET_WEIGHTS)}"
            )
        backbone_kwargs = {}
        if model.backbone == "resnet50.a1_in1k":
            backbone_kwargs["avg_down"] = True
        if model.backbone == COATNET_BACKBONE:
            if data.image_size % 32:
                raise ValueError(
                    "CoAtNet spatial tokens require image_size divisible by 32"
                )
            backbone_kwargs["img_size"] = data.image_size
        self.backbone = timm.create_model(
            model.backbone,
            pretrained=pretrained and model.pretrained_weights == "timm",
            in_chans=SLICES_PER_INPUT,
            num_classes=0,
            global_pool="",
            **backbone_kwargs,
        )
        if pretrained and model.pretrained_weights == "radimagenet":
            state = radimagenet_state_dict(model.backbone)
            if model.backbone == "resnet50.a1_in1k":
                state = resnet_avg_down_state_dict(state)
            self.backbone.load_state_dict(state, strict=True)
        self.feature_mapping = backbone_mapping(model.backbone, self.backbone)
        self.resnet_stages = hasattr(self.backbone, "layer2")
        if self.feature_statistics_settings.probability > 0 and not (
            self.resnet_stages
            or (
                hasattr(self.backbone, "stem")
                and hasattr(self.backbone, "features")
                and len(self.backbone.features) > REXNET_STRIDE_8_BLOCK_COUNT
            )
        ):
            raise ValueError(
                "Feature statistics augmentation requires a ReXNet or ResNet backbone "
                "with a stride-8 stage to resample"
            )
        backbone_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.inference_mode():
                spatial_output = self.backbone.forward_features(
                    torch.zeros(
                        1,
                        SLICES_PER_INPUT,
                        data.image_size,
                        data.image_size,
                    )
                )
        finally:
            self.backbone.train(backbone_training)
        if not isinstance(spatial_output, torch.Tensor) or spatial_output.ndim != 4:
            raise ValueError("Backbone must return NCHW spatial features")
        if spatial_output.shape[1] != self.backbone.num_features:
            raise ValueError("Backbone feature channels differ from the NCHW contract")
        if self.checkpoint_backbone:
            self.backbone.set_grad_checkpointing(True)
        self.projection = nn.Linear(spatial_output.shape[1], PROJECTION_WIDTH)
        if self.channels_last:
            self.backbone.to(memory_format=torch.channels_last)
        self.mil_norm = nn.LayerNorm(PROJECTION_WIDTH)
        missing = sorted((set(TARGET_ROUTES) | set(OWN_ROUTES)) - set(targets))
        if missing:
            raise ValueError(f"Side-routed heads require the targets {missing}")
        target_channels = [
            TARGET_ROUTES.get(target, (target,))[0] for target in targets
        ]
        self.channels = (
            *(name for name in target_channels if name not in SHARED_CHANNELS),
            *SHARED_CHANNELS,
        )
        target_channel = [self.channels.index(name) for name in target_channels]
        target_route = [
            ROUTES.index(
                TARGET_ROUTES[target][1]
                if target in TARGET_ROUTES
                else OWN_ROUTES[target]
            )
            if target in TARGET_ROUTES or target in OWN_ROUTES
            else len(ROUTES)
            for target in targets
        ]
        channel_source = [
            target_channel.index(channel) for channel in range(len(self.channels))
        ]
        self.register_buffer(
            "target_channel", torch.tensor(target_channel), persistent=False
        )
        self.register_buffer(
            "target_route", torch.tensor(target_route), persistent=False
        )
        self.register_buffer(
            "routed_channel",
            torch.tensor(
                [
                    any(
                        route < len(ROUTES)
                        for member, route in zip(
                            target_channel, target_route, strict=True
                        )
                        if member == channel
                    )
                    for channel in range(len(self.channels))
                ]
            ),
            persistent=False,
        )
        own_channels = len(self.channels) - len(SHARED_CHANNELS)
        self.register_buffer(
            "shared_channel",
            torch.tensor([name in SHARED_CHANNELS for name in self.channels]),
            persistent=False,
        )
        self.register_buffer(
            "target_prior_column",
            torch.tensor(
                [
                    own_channels
                    + SHARED_PRIORS.index(
                        "front"
                        if TARGET_ROUTES[target][1] == "kneecap"
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
                    PRIOR_VARIANTS[TARGET_ROUTES[target][1]]
                    if target in TARGET_ROUTES
                    else 0
                    for target in targets
                ]
            ),
            persistent=False,
        )
        self.register_buffer("routing_temperature", torch.ones(()))
        self.attention = nn.Sequential(
            nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH),
            nn.Tanh(),
            nn.Dropout(model.dropout),
            nn.Linear(MIL_ATTENTION_WIDTH, target_count),
        )
        self.head = nn.Linear(PROJECTION_WIDTH, target_count)
        self.target_bias = nn.Parameter(self.head.bias.detach().clone())
        self.attention[-1] = channel_linear(self.attention[-1], channel_source, True)
        self.head = channel_linear(self.head, channel_source, False)
        if self.plane_readout:
            with torch.random.fork_rng(devices=[]):
                self.head = PlaneContrastReadout(self.head, len(data.series_types))
        self.spatial_tokens = SpatialTokens(PROJECTION_WIDTH, data.series_types)
        if data.contrast_mixing_augmentation.probability > 0:
            plane_lookup = self.spatial_tokens.plane_lookup.tolist()
            self.contrast_mixing_pairs = tuple(
                tuple(
                    index for index, plane in enumerate(plane_lookup) if plane == target
                )
                for target in range(3)
            )
            if any(len(pair) != 2 for pair in self.contrast_mixing_pairs):
                raise ValueError(
                    "Contrast mixing requires exactly two series types per plane"
                )
        with torch.random.fork_rng(devices=[]):
            self.side_head = nn.Sequential(
                nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH),
                nn.Tanh(),
                nn.Linear(MIL_ATTENTION_WIDTH, 1),
            )
            self.position_prior = nn.Linear(
                2 * len(POSITION_AXES) * POSITION_WAVELENGTHS_MM[2],
                own_channels + len(SHARED_PRIORS),
            )
            nn.init.zeros_(self.position_prior.weight)
            nn.init.zeros_(self.position_prior.bias)
        self.frame_rotation_logit = nn.Parameter(torch.zeros(len(FRAME_ROTATIONS)))
        self.frame_shift = nn.Parameter(torch.zeros(len(POSITION_AXES)))
        self.frame_size = nn.Parameter(torch.zeros(len(FRAME_SIZES)))
        self.register_buffer(
            "frame_size_start",
            torch.tensor(FRAME_SIZE_START_PITCHES),
            persistent=False,
        )
        self.meniscus_map = model.meniscus_map
        if self.meniscus_map:
            self.meniscus_mapping = backbone_mapping(
                f"{model.backbone}.layer3", self.backbone
            )
            with torch.random.fork_rng(devices=[]):
                block = self.backbone.layer3[-1]
                self.meniscus_projection = nn.Linear(
                    getattr(block, "conv3", block.conv2).out_channels, PROJECTION_WIDTH
                )
                self.meniscus_head = nn.Linear(
                    PROJECTION_WIDTH,
                    SOFT_TISSUE_CLASS_COUNT
                    if model.soft_tissue_routing
                    else MENISCUS_CLASS_COUNT,
                )
            nn.init.zeros_(self.meniscus_head.weight)
            nn.init.zeros_(self.meniscus_head.bias)
        self.soft_tissue_routing = model.soft_tissue_routing
        if self.soft_tissue_routing:
            soft_target = [targets.index(target) for target in TARGET_ROUTES]
            self.register_buffer(
                "soft_target", torch.tensor(soft_target), persistent=False
            )
            self.register_buffer(
                "soft_target_channel",
                torch.tensor(
                    [SHARED_CHANNELS.index(route[0]) for route in TARGET_ROUTES.values()]
                ),
                persistent=False,
            )
            with torch.random.fork_rng(devices=[]):
                self.soft_norm = nn.LayerNorm(PROJECTION_WIDTH)
                self.soft_attention = nn.Sequential(
                    nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH),
                    nn.Tanh(),
                    nn.Dropout(model.dropout),
                    nn.Linear(MIL_ATTENTION_WIDTH, len(SHARED_CHANNELS)),
                )
                self.soft_head = nn.Linear(
                    PROJECTION_WIDTH, len(SHARED_CHANNELS), bias=False
                )
                nn.init.zeros_(self.soft_head.weight)
                if self.plane_readout:
                    self.soft_head = PlaneContrastReadout(
                        self.soft_head, len(data.series_types)
                    )
        self.anatomical_frame = model.anatomical_frame
        if self.anatomical_frame:
            with torch.random.fork_rng(devices=[]):
                self.anatomy_head = nn.Linear(PROJECTION_WIDTH, ANATOMY_CLASS_COUNT)
                self.route_section_shift = nn.Parameter(torch.zeros((len(ROUTES), 3)))
            self.register_buffer(
                "route_section_start",
                SECTION_LOGIT
                * (2 * torch.tensor([ROUTE_SECTIONS[route] for route in ROUTES]) - 1.0),
                persistent=False,
            )
            self.register_buffer(
                "anatomical_band_start",
                torch.tensor(ANATOMICAL_BAND_PITCHES),
                persistent=False,
            )
        with torch.random.fork_rng(devices=[]):
            self.frame_head = nn.Sequential(
                nn.Linear(PROJECTION_WIDTH, MIL_ATTENTION_WIDTH), nn.Tanh()
            )
            self.frame_attention = nn.Linear(MIL_ATTENTION_WIDTH, 1)
            self.frame_output = nn.Linear(
                MIL_ATTENTION_WIDTH,
                len(POSITION_AXES) + len(FRAME_ROTATIONS) + len(FRAME_SIZES),
                bias=False,
            )
            nn.init.zeros_(self.frame_output.weight)
            self.landmark_attention = nn.Linear(MIL_ATTENTION_WIDTH, len(LANDMARKS))
            nn.init.zeros_(self.landmark_attention.weight)
            nn.init.zeros_(self.landmark_attention.bias)
        shortest, longest, count = POSITION_WAVELENGTHS_MM
        self.register_buffer(
            "code_frequencies",
            2
            * math.pi
            / torch.logspace(
                math.log10(shortest),
                math.log10(longest),
                PROJECTION_WIDTH // (2 * len(POSITION_AXES)),
            ),
            persistent=False,
        )
        self.register_buffer(
            "position_frequencies",
            2
            * math.pi
            / torch.logspace(math.log10(shortest), math.log10(longest), count),
            persistent=False,
        )
        with torch.random.fork_rng(devices=[]):
            low, high = ROUTING_WIDTH_RANGE_MM
            initial_logit = math.log(
                (ROUTING_INITIAL_WIDTH_MM - low) / (high - ROUTING_INITIAL_WIDTH_MM)
            )
            self.routing_width_logit = nn.Parameter(
                torch.full((len(ROUTING_WIDTHS),), initial_logit)
            )
            self.knee_level_scale = nn.Parameter(torch.zeros(len(self.channels)))
        self.oa_indices = tuple(
            targets.index(name) for name in (*COMPARTMENT_PAIRS["oa"], "PF OA")
        )
        with torch.random.fork_rng(devices=[]):
            self.oa_detector = nn.Linear(PROJECTION_WIDTH, 1, bias=False)
            nn.init.xavier_uniform_(self.oa_detector.weight)
            self.oa_detector_scale = nn.Parameter(torch.zeros(3))
        with torch.random.fork_rng(devices=[]):
            self.detail_view_embedding = nn.Parameter(torch.zeros(PROJECTION_WIDTH))
        self.register_buffer(
            "detail_view",
            torch.tensor(
                [view.field_mm != data.reference_field_mm for _, view in data.views]
            ),
            persistent=False,
        )
        with torch.random.fork_rng(devices=[]):
            self.register_buffer(
                "shared_slice_view",
                torch.tensor(
                    [index != source for index, source in enumerate(data.view_sources)]
                ),
                persistent=False,
            )
            self.register_buffer(
                "detail_target_allowed",
                torch.tensor([target in (*TARGET_ROUTES, "ACL") for target in targets]),
                persistent=False,
            )
            self.register_buffer(
                "detail_channel_allowed",
                torch.tensor(
                    [channel in (*SHARED_CHANNELS, "ACL") for channel in self.channels]
                ),
                persistent=False,
            )
        self.primary_view_count = sum(
            index == source for index, source in enumerate(data.view_sources)
        )
        self.view_label_bias = (
            nn.Parameter(torch.zeros(len(data.views), len(self.channels)))
            if model.view_label_bias
            else None
        )
        with torch.random.fork_rng(devices=[]):
            self.view_windows = ViewWindows(data)
            self.mask_decoder = MaskDecoder(
                tuple(
                    getattr(stage[-1], "conv3", stage[-1].conv2).out_channels
                    for stage in (
                        self.backbone.layer2,
                        self.backbone.layer3,
                        self.backbone.layer4,
                    )
                )
            )
        self.decoder_mapping = (
            *self.meniscus_mapping[:2],
            *(step * 0.5 for step in self.meniscus_mapping[2:]),
        )
        self.train(self.training)

    def train(self, mode: bool = True) -> "KneeAbnormalityModel":
        super().train(mode)
        if self.backbone_eval:
            self.backbone.eval()
        return self

    def _encode_images(
        self,
        images: torch.Tensor,
        acquisition_index: torch.Tensor,
        acquisition_type: torch.Tensor,
        primary_image: torch.Tensor,
        primary_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.checkpoint_backbone:
            self.backbone.set_grad_checkpointing(
                self.training and torch.is_grad_enabled()
            )
        if not self.training:
            maps, meniscus_maps = self._backbone_features(images)
            return maps, meniscus_maps, None
        features = self._backbone_to_stride_8(images)
        if self.feature_statistics_settings.probability > 0:
            features, self.feature_statistics_activity = (
                apply_feature_statistics_augmentation(
                    features,
                    acquisition_index,
                    acquisition_type,
                    self.feature_statistics_settings,
                    primary_image,
                    primary_count,
                )
            )
        maps, meniscus_maps = self._backbone_stage_outputs(features)
        return maps, meniscus_maps, features

    def _backbone_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.meniscus_map:
            return self.backbone.forward_features(images), None
        return self._backbone_stage_outputs(self._backbone_to_stride_8(images))

    def _backbone_stage_outputs(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.meniscus_map:
            meniscus_maps = self.backbone.layer3(features)
            return self.backbone.layer4(meniscus_maps), meniscus_maps
        return self._backbone_from_stride_8(features), None

    def _backbone_to_stride_8(self, images: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        if self.resnet_stages:
            stem = backbone.maxpool(backbone.act1(backbone.bn1(backbone.conv1(images))))
            return backbone.layer2(backbone.layer1(stem))
        return backbone.features[:REXNET_STRIDE_8_BLOCK_COUNT](backbone.stem(images))

    def _backbone_from_stride_8(self, features: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        if self.resnet_stages:
            return backbone.layer4(backbone.layer3(features))
        return backbone.features[REXNET_STRIDE_8_BLOCK_COUNT:](features)

    def _effective_presence(self, presence: torch.Tensor) -> torch.Tensor:
        if not presence.any(dim=1).all():
            raise ValueError("Every study must contain at least one present series")
        if not self.training or self.series_dropout == 0:
            return presence
        keep = presence & (
            torch.rand(presence.shape, device=presence.device) >= self.series_dropout
        )
        empty = ~keep.any(dim=1)
        if empty.any():
            scores = torch.rand(presence.shape, device=presence.device).masked_fill(
                ~presence, -1
            )
            selected = scores.argmax(dim=1)
            rows = empty.nonzero(as_tuple=False).flatten()
            keep[rows, selected[rows]] = True
        return keep

    def _mix_contrast(
        self,
        images: torch.Tensor,
        effective: torch.Tensor,
        acquisition_index: torch.Tensor,
        token_study: torch.Tensor,
        token_type: torch.Tensor,
        token_plane: torch.Tensor,
        patient_maps: torch.Tensor,
        geometry_valid: torch.Tensor,
        reflected: torch.Tensor,
    ) -> torch.Tensor:
        settings = self.augmentation_settings.contrast_mixing_augmentation
        if not self.training or settings.probability == 0:
            return images
        pairs = torch.tensor(
            self.contrast_mixing_pairs, device=images.device, dtype=torch.long
        )
        parameters = sample_contrast_mixing_parameters(
            settings.probability, effective[:, pairs[:, 0]] & effective[:, pairs[:, 1]]
        )
        images, self.contrast_mixing_activity = apply_contrast_mixing(
            images,
            parameters,
            self.contrast_mixing_pairs,
            settings.target_weight_range,
            acquisition_index,
            token_study,
            token_type,
            token_plane,
            patient_maps,
            geometry_valid,
            reflected,
        )
        return images

    def _augment_images(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor,
        valid_rects: torch.Tensor,
        source_series_index: torch.Tensor,
        primary_series_index: torch.Tensor,
        source_image_index: torch.Tensor,
        primary_image_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        settings = self.augmentation_settings
        if not self.training or not any(
            augmentation.probability > 0
            for augmentation in (
                settings.intensity_augmentation,
                settings.affine_augmentation,
                settings.noise_augmentation,
                settings.smoothing_augmentation,
                settings.resolution_augmentation,
                settings.gin_augmentation,
            )
        ):
            return images, torch.eye(3, device=images.device)[None].expand(
                len(images), -1, -1
            )
        parameters = sample_augmentation_parameters(
            settings, len(primary_series_index), images.device
        )
        series_index = slice_mask.nonzero(as_tuple=True)[0]
        series_index = source_series_index[series_index]
        matrices = affine_matrices(parameters.affine, *images.shape[-2:])
        inverse = augmented_to_prepared(matrices, *images.shape[-2:])
        return self.image_augmentation(
            images,
            series_index,
            parameters,
            matrices=matrices,
            valid_rects=valid_rects,
            inverse_affine=inverse[series_index],
            source_image_index=source_image_index,
            primary_image_index=primary_image_index,
        ), inverse[series_index]

    def routing_widths(self) -> torch.Tensor:
        low, high = ROUTING_WIDTH_RANGE_MM
        return low + (high - low) * torch.sigmoid(self.routing_width_logit.float())

    def position_priors(
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

    def side_probability(
        self,
        side_logit: torch.Tensor,
        side_present: torch.Tensor,
        mirrored: torch.Tensor,
        coordinates: torch.Tensor,
        left: torch.Tensor,
    ) -> torch.Tensor:
        source_logit = torch.where(
            side_present,
            torch.where(mirrored, -side_logit, side_logit).detach().float(),
            0.0,
        )
        prior = SIDE_COORDINATE_LOG_ODDS * torch.where(
            coordinates, torch.where(left, 1.0, -1.0), 0.0
        )
        return torch.sigmoid(source_logit + prior)

    def frame_rotation(self, offset: torch.Tensor | None = None) -> torch.Tensor:
        logit = self.frame_rotation_logit.float()
        if offset is not None:
            logit = logit + offset.float()
        nod, roll, turn = (FRAME_ROTATION_RANGE_RAD * torch.tanh(logit)).unbind(-1)
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
        pitch: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, ...]:
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
        attention = (
            self.frame_attention(hidden)
            .squeeze(-1)[None]
            .masked_fill(~allowed, -torch.inf)
            .softmax(-1)
        )
        output = self.frame_output(attention @ hidden) * present[:, None]
        slices = valid.any(-1).float()
        study_pitch = torch.zeros(batch_size, device=tokens.device).index_add(
            0, token_study, pitch * slices
        ) / torch.zeros(batch_size, device=tokens.device).index_add(
            0, token_study, slices
        ).clamp_min(1)
        count = len(POSITION_AXES)
        uniform = allowed.float() / allowed.sum(-1, keepdim=True)
        mean = uniform @ points
        spread = torch.where(
            present[:, None],
            (uniform @ points.square() - mean.square()).clamp_min(1e-6).sqrt(),
            1.0,
        )
        landmarks = (
            self.landmark_attention(hidden)
            .transpose(0, 1)[None]
            .masked_fill(~allowed[:, None], -torch.inf)
            .softmax(-1)
        ) @ points
        centre, front, up = (
            (landmarks - mean[:, None]) * present[:, None, None]
        ).unbind(1)
        forward = front - centre
        upward = up - centre
        meniscus = (
            len(self.channels)
            - len(SHARED_CHANNELS)
            + SHARED_CHANNELS.index("meniscus")
        )
        with torch.no_grad():
            looks = self.attention(tokens.float())[..., meniscus].float().flatten()
        spots = []
        for side in (points[:, 0] > 0, points[:, 0] <= 0):
            chosen = member & side[None]
            found = chosen.any(-1)
            open_ = chosen.clone()
            open_[:, 0] |= ~found
            weight = looks[None].masked_fill(~open_, -torch.inf).softmax(-1).square()
            spots.append(((weight / weight.sum(-1, keepdim=True)) @ points, found))
        (medial_spot, medial_found), (lateral_spot, lateral_found) = spots
        both = (medial_found & lateral_found)[:, None]
        unit = torch.eye(count, device=tokens.device)
        side_axis = nn.functional.normalize(
            torch.where(both, medial_spot - lateral_spot, unit[0]), dim=-1
        )
        up_axis = nn.functional.normalize(
            unit[2] - side_axis[:, 2:3] * side_axis, dim=-1
        )
        basis = torch.stack(
            (side_axis, torch.linalg.cross(up_axis, side_axis), up_axis), 1
        )
        tilt = torch.atan2(
            upward[:, 1] - forward[:, 2],
            spread[:, 1] + forward[:, 1] + spread[:, 2] + upward[:, 2],
        )
        cosine, sine, zero, one = (
            tilt.cos(),
            tilt.sin(),
            torch.zeros_like(tilt),
            torch.ones_like(tilt),
        )
        nod_only = torch.tensor([1.0, 0.0, 0.0], device=tokens.device)
        rotation = self.frame_rotation(
            output[:, count : 2 * count] * nod_only
        ) @ torch.stack(
            (
                torch.stack((one, zero, zero), -1),
                torch.stack((zero, cosine, -sine), -1),
                torch.stack((zero, sine, cosine), -1),
            ),
            1,
        )
        rotation = rotation @ basis
        shift = (
            torch.where(both, (medial_spot + lateral_spot) / 2, 0.0)
            + self.frame_shift.float() * study_pitch[:, None]
        )
        sizes = (
            self.frame_size_start + self.frame_size.float() + output[:, 2 * count :]
        ) * study_pitch[:, None]
        return shift, rotation, torch.stack((medial_spot, lateral_spot), 1), sizes

    def anatomical_frames(
        self,
        scores: torch.Tensor,
        position: torch.Tensor,
        valid: torch.Tensor,
        token_study: torch.Tensor,
        pitch: torch.Tensor,
        batch_size: int,
        shift: torch.Tensor,
        rotation: torch.Tensor,
        sizes: torch.Tensor,
        diagnostics: dict | None = None,
    ) -> tuple[torch.Tensor, ...]:
        device = position.device
        shares = scores.detach().float().softmax(-1)
        femur, tibia = shares[..., 1], shares[..., 2]
        study = token_study[:, None].expand_as(valid).flatten()
        points = (
            torch.cat(
                (position / PLANE_SCALE_MM, torch.ones_like(position[..., :1])), -1
            )
            .flatten(0, 1)
            .double()
        )
        bone = (valid & (femur + tibia > 0.5)).flatten().double()
        target = (femur - tibia).flatten().double()
        eye = PLANE_RIDGE * torch.eye(4, dtype=torch.float64, device=device)
        prior = torch.zeros((batch_size, 4), dtype=torch.float64, device=device)
        prior[:, 2] = PLANE_SCALE_MM * PLANE_START_SLOPE
        prior[:, 3] = -PLANE_START_SLOPE * shift[:, 2].detach().double()
        weight = bone
        for _ in range(PLANE_REFITS + 1):
            moments = torch.zeros(
                (batch_size, 4, 4), dtype=torch.float64, device=device
            ).index_add(
                0, study, points[:, :, None] * points[:, None] * weight[:, None, None]
            )
            response = torch.zeros(
                (batch_size, 4), dtype=torch.float64, device=device
            ).index_add(0, study, points * (target * weight)[:, None])
            solution = torch.linalg.solve(moments + eye, response + PLANE_RIDGE * prior)
            slope = solution[:, :3] / PLANE_SCALE_MM
            length = slope.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            distance = (
                (points[:, :3] * PLANE_SCALE_MM * slope[study]).sum(-1)
                + solution[study, 3]
            ) / length[study, 0]
            weight = bone * torch.exp(-0.5 * (distance / PLANE_REFIT_WIDTH_MM) ** 2)
        normal = (slope / length).float()
        normal = torch.where(normal[:, 2:] < 0, -normal, normal)
        level = (
            torch.where(slope[:, 2:] < 0, -solution[:, 3:], solution[:, 3:]).float()
            / length.float()
        )
        side = nn.functional.normalize(
            rotation[:, 0] - (rotation[:, 0] * normal).sum(-1, keepdim=True) * normal,
            dim=-1,
        )
        rotation = torch.stack((side, torch.linalg.cross(normal, side), normal), 1)
        shift = shift - normal * ((shift * normal).sum(-1, keepdim=True) + level)
        height = (position.detach().float() * normal[token_study, None]).sum(
            -1
        ) + level[token_study]
        centre_weight = (
            valid.float()
            * (femur + tibia)
            * torch.exp(-0.5 * (height / ANATOMY_CENTRE_WIDTH_MM).square())
        ).flatten()
        mass = torch.zeros(batch_size, device=device).index_add(0, study, centre_weight)
        centroid = (
            torch.zeros((batch_size, 3), device=device).index_add(
                0,
                study,
                position.detach().float().flatten(0, 1) * centre_weight[:, None],
            )
            / mass.clamp_min(torch.finfo(torch.float32).tiny)[:, None]
        )
        centroid = centroid - normal * (
            (centroid * normal).sum(-1, keepdim=True) + level
        )
        centred = torch.where((mass > 0)[:, None], centroid, shift)
        if diagnostics is not None:
            diagnostics["anatomy_centre_delta_mm"] = (centred - shift).detach()
            diagnostics["anatomy_centre_applied"] = mass > 0
        shift = centred
        slices = valid.any(-1).float()
        study_pitch = torch.zeros(batch_size, device=device).index_add(
            0, token_study, pitch * slices
        ) / torch.zeros(batch_size, device=device).index_add(
            0, token_study, slices
        ).clamp_min(1)
        band = (self.anatomical_band_start + self.frame_size.float()[:2]) * study_pitch[
            :, None
        ]
        radii = self.frame_size_start[2:] * study_pitch[:, None]
        return shift, rotation, torch.cat((band, radii), -1)

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

    def position_code(
        self, position: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type=position.device.type, enabled=False):
            angles = position.detach().float().unsqueeze(-1) * self.code_frequencies
            code = torch.stack((angles.sin(), angles.cos()), -1).flatten(-3)
            code = nn.functional.pad(code, (0, PROJECTION_WIDTH - code.shape[-1]))
            return torch.where(valid[:, None, None], code, 0.0)

    def _regions(self, encoded: dict, medial_sign: float) -> torch.Tensor:
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
        across = (half[..., 0] + half[..., 1]) / 2
        top, bottom, core_radius, ring_radius = (
            encoded["frame_sizes"].float()[:, None].unbind(-1)
        )
        step = torch.tensor(0.5, device=medial.device)

        def inside(distance: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
            soft = torch.sigmoid(distance / (2 * extent))
            return torch.heaviside(distance.detach(), step) + soft - soft.detach()

        beyond = [
            inside(
                front * math.cos(math.radians(22.5 + 45.0 * number))
                - medial * math.sin(math.radians(22.5 + 45.0 * number)),
                across,
            )
            for number in range(len(SECTORS))
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
        upper = inside(height - top, half[..., 2])
        tibia = inside(bottom - height, half[..., 2])
        joint = (1 - upper - tibia).clamp_min(0)
        distance = torch.hypot(medial, front).clamp_min(1e-6)
        core = inside(core_radius - distance, across)
        ringed = 1 - core
        inner = ringed * (1 - inside(distance - ring_radius, across))
        medial_trio = (
            sector["anteromedial"] + sector["medial"] + sector["posteromedial"]
        )
        lateral_trio = (
            sector["anterolateral"] + sector["lateral"] + sector["posterolateral"]
        )
        front_trio = (
            sector["anteromedial"] + sector["anterior"] + sector["anterolateral"]
        )
        medial_compartment = (ringed * medial_trio + core * medial_side) * (1 - upper)
        lateral_compartment = (ringed * lateral_trio + core * lateral_side) * (
            1 - upper
        )
        kneecap = ringed * front_trio * (1 - tibia)
        shares = encoded.get("anatomy_shares")
        if shares is not None:
            femur_share, tibia_share, joint_share = (
                shares[..., FEMUR_CLASS],
                shares[..., TIBIA_CLASS],
                shares[..., JOINT_CLASS],
            )
            free = (1 - femur_share - tibia_share - joint_share).clamp_min(0)
            sections = torch.stack(
                (
                    femur_share + free * upper,
                    joint_share + free * joint,
                    tibia_share + free * tibia,
                )
            )
            heights = torch.einsum("rs,snt->rnt", self.route_sections(), sections)
            front_plane = ringed * front_trio
            behind_kneecap = (
                shares[..., KNEECAP_CLASS] + shares[..., KNEECAP_JOINT_CLASS]
            )
            behind_kneecap = (behind_kneecap / KNEECAP_FULL_SHARE).clamp(max=1)
            kneecap = behind_kneecap + (1 - behind_kneecap) * FRONT_FLOOR * (
                front_plane * heights[4]
            )
            front_core = (front_plane + core) * heights[8]
            front_core = front_core + (1 - front_core) * behind_kneecap
            medial_plane = ringed * medial_trio + core * medial_side
            lateral_plane = ringed * lateral_trio + core * lateral_side
            return torch.stack(
                (
                    medial_plane * heights[0],
                    lateral_plane * heights[1],
                    medial_plane * heights[2],
                    lateral_plane * heights[3],
                    kneecap,
                    (
                        core
                        + inner
                        * (sector["anterior"] + sector["posterior"] + lateral_trio)
                    )
                    * heights[5],
                    ringed * (sector["anteromedial"] + sector["medial"]) * heights[6],
                    ringed
                    * (sector["posterior"] + sector["posteromedial"] + sector["medial"])
                    * heights[7],
                    front_core,
                    ringed * sector["anteromedial"] * heights[9],
                    ringed * sector["medial"] * heights[10],
                    ringed * sector["posteromedial"] * heights[11],
                    ringed * sector["anterolateral"] * heights[12],
                    ringed * sector["lateral"] * heights[13],
                    ringed * sector["posterolateral"] * heights[14],
                    kneecap * medial_side,
                    kneecap * lateral_side,
                    shares[
                        ..., [FEMUR_CLASS, TIBIA_CLASS, KNEECAP_CLASS, FIBULA_CLASS]
                    ].sum(-1),
                )
            )
        return torch.stack(
            (
                medial_compartment,
                lateral_compartment,
                medial_compartment,
                lateral_compartment,
                kneecap,
                (
                    core
                    + inner * (sector["anterior"] + sector["posterior"] + lateral_trio)
                )
                * (1 - tibia),
                ringed * (sector["anteromedial"] + sector["medial"]),
                ringed
                * (sector["posterior"] + sector["posteromedial"] + sector["medial"]),
                (ringed * front_trio + core) * (1 - tibia),
                ringed * sector["anteromedial"] * joint,
                ringed * sector["medial"] * joint,
                ringed * sector["posteromedial"] * joint,
                ringed * sector["anterolateral"] * joint,
                ringed * sector["lateral"] * joint,
                ringed * sector["posterolateral"] * joint,
                kneecap * medial_side,
                kneecap * lateral_side,
                torch.ones_like(core),
            )
        )

    def route_sections(self) -> torch.Tensor:
        return torch.sigmoid(
            self.route_section_start + SECTION_RATE * self.route_section_shift.float()
        )

    @staticmethod
    def _memo(encoded: dict, name: str, compute):
        inputs = tuple(
            value for key, value in encoded.items() if not key.startswith("memo_")
        )
        entry = encoded.get(name)
        if (
            entry is not None
            and len(entry[0]) == len(inputs)
            and all(old is new for old, new in zip(entry[0], inputs, strict=True))
        ):
            return entry[1]
        value = compute()
        encoded[name] = (inputs, value)
        return value

    def routing_weights(self, encoded: dict) -> torch.Tensor:
        return self._memo(
            encoded, "memo_routing_weights", lambda: self._routing_weights(encoded)
        )

    def _routing_weights(self, encoded: dict) -> torch.Tensor:
        knee_x = encoded["knee_x"]
        with torch.autocast(device_type=knee_x.device.type, enabled=False):
            left = encoded["left_probability"][:, None].float()
            weights = left * self._regions(encoded, -1.0) + (1 - left) * self._regions(
                encoded, 1.0
            )
            return torch.where(encoded["routable"][:, None], weights, 0.5)

    def _attention_pool(
        self,
        tokens: torch.Tensor,
        shared_tokens: torch.Tensor,
        token_study: torch.Tensor,
        image_type: torch.Tensor,
        token_valid: torch.Tensor,
        batch_size: int,
        routing: torch.Tensor,
        prior: torch.Tensor,
        medial_axis: torch.Tensor,
        anterior_axis: torch.Tensor,
        trusted_geometry: torch.Tensor,
        token_view: torch.Tensor,
        diagnostic: bool = False,
    ) -> torch.Tensor:
        cached = self._pooling_cache
        if (
            diagnostic
            and cached is not None
            and cached[0] is tokens
            and cached[1] is shared_tokens
        ):
            scores, evidence = cached[2], cached[3]
        else:
            scores = torch.where(
                self.shared_channel[:, None],
                self.attention(shared_tokens).transpose(0, 1),
                self.attention(tokens).transpose(0, 1),
            )
            evidence = None
            if self.plane_readout:
                with torch.autocast(device_type=tokens.device.type, enabled=False):
                    evidence = torch.where(
                        self.shared_channel,
                        self.head(shared_tokens, image_type),
                        self.head(tokens, image_type),
                    )
            self._pooling_cache = (
                None if self.training else (tokens, shared_tokens, scores, evidence)
            )
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
            target_gate = (
                ~self.shared_slice_view[token_view][None]
                | self.detail_target_allowed[:, None]
            )
            target_weights = weights[self.target_route] * target_gate
            routed = membership.float() @ target_weights.T >= 1
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
            target_allowed = allowed[:, None] & target_gate[None]
            target_empty = empty_study[:, None] | ~target_allowed.any(-1)
            target_allowed = target_allowed.clone()
            target_allowed[:, :, 0] |= target_empty
            attention = attention.masked_fill(~target_allowed, -torch.inf).softmax(-1)
            attention = attention.masked_fill(target_empty[:, :, None], 0.0)
            if not diagnostic:
                self.attention_mass_view = torch.stack(
                    [
                        (attention * (token_view == view)[None, None]).sum(-1)
                        for view in range(
                            self.primary_view_count
                            if self.training
                            else len(self.detail_view)
                        )
                    ]
                ).detach()
                self.routing_mean = (
                    (routing * token_valid).sum(-1) / token_valid.sum().clamp_min(1)
                ).detach()
            if self.plane_readout:
                logits = torch.einsum(
                    "btn,nt->bt", attention, evidence[:, self.target_channel]
                )
            else:
                pooled = torch.where(
                    self.shared_channel[self.target_channel][None, :, None],
                    torch.einsum("btn,nf->btf", attention, shared_tokens.float()),
                    torch.einsum("btn,nf->btf", attention, tokens.float()),
                )
                logits = (
                    pooled * self.head.weight.float()[self.target_channel][None]
                ).sum(-1)
                evidence = torch.where(
                    self.shared_channel,
                    nn.functional.linear(
                        shared_tokens.float(), self.head.weight.float()
                    ),
                    nn.functional.linear(tokens.float(), self.head.weight.float()),
                )
            knee_allowed = allowed[:, None] & (
                ~self.shared_slice_view[token_view][None, None]
                | self.detail_channel_allowed[None, :, None]
            )
            knee_empty = empty_study[:, None] | ~knee_allowed.any(-1)
            knee_allowed = knee_allowed.clone()
            knee_allowed[:, :, 0] |= knee_empty
            knee_attention = (
                scores.float()[None].masked_fill(~knee_allowed, -torch.inf).softmax(-1)
            )
            knee_attention = knee_attention.masked_fill(knee_empty[:, :, None], 0.0)
            knee_evidence = torch.einsum("bcn,nc->bc", knee_attention, evidence)
            channel_contribution = (
                knee_evidence
                * self.knee_level_scale.float()[None]
                * self.routed_channel
            )
            if not diagnostic:
                self.knee_level_logit_abs_mean = (
                    channel_contribution.detach().abs().mean()
                )
            logits = (
                logits
                + channel_contribution[:, self.target_channel]
                + self.target_bias.float()
            )
            regions = (
                torch.stack(
                    (
                        medial_axis.float() > COMPARTMENT_MIDLINE_BAND_MM,
                        medial_axis.float() < -COMPARTMENT_MIDLINE_BAND_MM,
                        anterior_axis.float() > 0,
                    )
                )
                & trusted_geometry[None]
            )
            region_membership = membership[:, None] & regions[None]
            counts = region_membership.sum(-1)
            empty = counts == 0
            safe_membership = region_membership.clone()
            safe_membership[..., 0] |= empty
            evidence = self.oa_detector(tokens.float()).squeeze(-1)
            scores = evidence[None, None].masked_fill(~safe_membership, -torch.inf)
            pooled_evidence = torch.logsumexp(scores, -1) - counts.clamp_min(1).log()
            pooled_evidence = torch.where(empty, 0.0, pooled_evidence)
            correction = pooled_evidence * self.oa_detector_scale.float()[None]
            logits[:, self.oa_indices] += correction
        if not torch.isfinite(logits).all():
            raise RuntimeError("Study attention pooling produced non-finite logits")
        return logits

    def encode_batch(self, batch: dict, view_pose: torch.Tensor | None = None) -> dict:
        if view_pose is None:
            view_pose = self.first_look_pose(batch)
            if self.training:
                view_pose = self.view_windows.jitter(batch, view_pose)
        return self._encode_crops(batch, view_pose)

    def first_look_pose(self, batch: dict) -> torch.Tensor:
        training = self.training
        self.eval()
        with torch.no_grad():
            first = self._encode_crops(self.view_windows.first_look_batch(batch))
            pose = self.view_windows.frame_pose(batch, first)
            with torch.autocast(device_type=batch["images"].device.type, enabled=False):
                position = first["position"].detach().float()
                medial, front, up = position.unbind(-1)
                shares = first["anatomy_shares"].detach().float()
                valid = first["visual_valid"] & first["routable"][:, None]
                plane = first["image_plane"][:, None]
                width_weight = (
                    valid
                    * ((plane == 0) | (plane == 1))
                    * (
                        shares[..., FEMUR_CLASS] * ((up > -5) & (up < 45))
                        + shares[..., TIBIA_CLASS] * ((up > -30) & (up < 5))
                    )
                )
                depth_weight = (
                    valid
                    * ((plane == 0) | (plane == 2))
                    * shares[..., FEMUR_CLASS]
                    * torch.exp(-0.5 * ((up - 20) / 20).square())
                )
                weights = torch.stack((width_weight, depth_weight), -1)
                coordinates = torch.stack((medial, front), -1)
                moments = torch.cat(
                    (
                        weights.sum(1),
                        (weights * coordinates).sum(1),
                        (weights * coordinates.square()).sum(1),
                    ),
                    -1,
                )
                totals = torch.zeros(
                    first["batch_size"], 6, device=position.device
                ).index_add(0, first["study"], moments)
                mass, total, square = totals.split(2, -1)
                denominator = mass.clamp_min(1e-6)
                spread = (
                    (square / denominator - (total / denominator).square())
                    .clamp_min(0)
                    .sqrt()
                )
                size = spread.prod(-1).sqrt()
                scale = torch.where(
                    (mass > 0).all(-1), (size / 17.087).clamp(0.87, 1.13), 1.0
                )
                pose = torch.cat(
                    (
                        pose,
                        scale[batch["series_study_index"], None, None].expand(
                            -1, pose.shape[1], -1
                        ),
                    ),
                    -1,
                )
        self.train(training)
        return pose

    def _encode_crops(self, batch: dict, view_pose: torch.Tensor | None = None) -> dict:
        batch = self.view_windows(batch, self.repeated_middle_slice, view_pose)
        presence = batch["series_presence"]
        reflected_study = torch.zeros_like(batch["reflection_eligible"])
        if self.training:
            reflected_study = sample_reflection_parameters(
                self.augmentation_settings.reflection_augmentation.probability,
                batch["reflection_eligible"],
            ).applied
        effective = self._effective_presence(presence)
        all_study = batch["series_study_index"]
        all_type = batch["series_type_index"]
        keep_series = effective[all_study, all_type]
        study_index = all_study[keep_series]
        type_index = all_type[keep_series]
        slice_mask = batch["slice_mask"][keep_series]
        flat_indices = (
            (keep_series[:, None] & batch["slice_mask"]).flatten().nonzero().flatten()
        )
        valid_rects = batch["valid_rects"].flatten(0, 1)[flat_indices]
        patient_maps = batch["patient_maps"].flatten(0, 1)[flat_indices].float()
        geometry_valid = batch["geometry_valid"].flatten()[
            flat_indices
        ] & torch.isfinite(patient_maps[:, 1]).all((-1, -2))

        images = batch["images"].flatten(0, 1)
        source_valid_rects = valid_rects
        if self.repeated_middle_slice:
            images = images[:, 1:2]
            valid_rects = valid_rects[:, 1:2]
        valid_images = images[flat_indices].to(torch.float32).div_(255.0)
        acquisition_index = slice_mask.nonzero(as_tuple=True)[0]
        token_study = study_index[acquisition_index]
        token_type = type_index[acquisition_index]
        reflected = reflected_study[token_study]
        token_plane = self.spatial_tokens.plane_lookup[token_type]
        sagittal = token_plane == 2
        channels = reflection_channels(reflected, sagittal)
        if self.repeated_middle_slice:
            channels = channels[:, 1:2]
        if (
            self.training
            and self.augmentation_settings.reflection_augmentation.probability > 0
        ):
            valid_images = apply_reflection(valid_images, reflected, sagittal)
        view_index = batch["view_index"][keep_series][acquisition_index]
        primary_image = ~self.shared_slice_view[view_index]
        mixing_type = torch.where(primary_image, token_type, -1)
        valid_images = self._mix_contrast(
            valid_images,
            effective,
            acquisition_index,
            token_study,
            mixing_type,
            token_plane,
            patient_maps,
            geometry_valid,
            reflected,
        )
        valid_images, inverse_affine = self._augment_images(
            valid_images,
            slice_mask,
            reflection_valid_rects(
                valid_rects, reflected, sagittal, valid_images.shape[-1]
            ),
            batch["source_series_index"],
            batch["primary_series_index"],
            batch["source_image_index"],
            batch["primary_image_index"],
        )
        inverse_affine = reflection_inverse(
            inverse_affine, reflected, sagittal, valid_images.shape[-1]
        )
        if self.repeated_middle_slice:
            valid_images = valid_images.expand(-1, SLICES_PER_INPUT, -1, -1)
        valid_images = valid_images.sub(self.mean).div_(self.std)
        if self.channels_last:
            valid_images = valid_images.contiguous(memory_format=torch.channels_last)
        maps, meniscus_maps, decoder_maps = self._encode_images(
            valid_images,
            acquisition_index,
            token_type,
            primary_image,
            len(batch["primary_series_index"]),
        )
        features = self.projection(maps.flatten(2).transpose(1, 2))
        visual_valid, prepared_xy = feature_geometry(
            source_valid_rects,
            inverse_affine,
            maps.shape[-2:],
            self.feature_mapping,
            channels,
            batch["canvas_valid_rects"].flatten(0, 1)[flat_indices],
            batch["crop_affine"].flatten(0, 1)[flat_indices],
        )
        detail_view = self.detail_view[view_index]
        batch_size = len(presence)
        mirrored_study = batch.get("mirrored_study", reflected_study)
        encoded = {
            "features": features,
            **self.spatial_tokens(token_type, features.dtype),
            "detail_view": detail_view,
        }
        if self.training:
            encoded["decoder_maps"] = (decoder_maps, meniscus_maps, maps)
        tokens = self.mil_norm(self._base_tokens(encoded))
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
            source_middle = middle @ torch.linalg.inv(
                batch["crop_affine"].flatten(0, 1)[flat_indices].float()
            )
            row = source_middle[:, :2, 0]
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
            side_tokens = visual_valid & (valid_position & ~sagittal)[:, None]
            displayed_x = torch.where(
                mirrored_study[token_study][:, None], -knee_x, knee_x
            )
            side_weight = (
                torch.tanh(
                    displayed_x / self.routing_widths()[ROUTING_WIDTHS.index("side")]
                )
                * side_tokens
            )
            side_count = torch.zeros(batch_size, device=features.device).index_add(
                0, token_study, side_tokens.float().sum(-1)
            )
            side_scores = self.side_head(tokens.detach().float()).squeeze(-1)
            side_logit = torch.zeros(batch_size, device=features.device).index_add(
                0, token_study, (side_scores * side_weight).sum(-1)
            ) / side_count.clamp_min(1)
            side_present = side_count > 0
            head_left = self.side_probability(
                side_logit,
                side_present,
                mirrored_study,
                batch["knee_side_coordinates"],
                batch["knee_side_left"],
            )
            sign = batch["knee_side_sign"][keep_series][acquisition_index]
            oa_trusted_geometry = (
                geometry_valid[:, None] & (sign[:, None] != 0) & visual_valid
            )
            oa_medial_axis = torch.where(
                geometry_valid[:, None] & (sign[:, None] != 0),
                sign[:, None] * (patient_position[..., 0] - center[:, :1]),
                0.0,
            )
            oa_anterior_axis = torch.where(
                geometry_valid[:, None] & (sign[:, None] != 0),
                center[:, 1:2] - patient_position[..., 1],
                0.0,
            )
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
            superior_axis = offset[..., 2]
            sign = (1 - 2 * left_probability)[:, None]
            token_pitch = torch.where(
                valid_position,
                (patient_position[:, 1] - patient_position[:, 0]).norm(dim=-1),
                1.0,
            ).detach()
            steps = torch.stack(
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
            meniscus_scores = None
            meniscus_encoded = {}
            if self.meniscus_map:
                height, width = maps.shape[-2:]
                fine_height, fine_width = meniscus_maps.shape[-2:]
                meniscus_features = self.meniscus_projection(
                    meniscus_maps.flatten(2).transpose(1, 2).float()
                )
                deep = (
                    tokens.float()
                    .unflatten(1, (height, width))
                    .repeat_interleave(2, 1)
                    .repeat_interleave(2, 2)
                    .flatten(1, 2)
                )
                meniscus_scores = self.meniscus_head(meniscus_features + deep)
                meniscus_valid, meniscus_xy = feature_geometry(
                    source_valid_rects,
                    inverse_affine,
                    (fine_height, fine_width),
                    self.meniscus_mapping,
                    channels,
                    batch["canvas_valid_rects"].flatten(0, 1)[flat_indices],
                    batch["crop_affine"].flatten(0, 1)[flat_indices],
                )
                fine_homogeneous = torch.cat(
                    (meniscus_xy, torch.ones_like(meniscus_xy[..., :1])), -1
                )
                meniscus_position = torch.einsum(
                    "nij,nkj->nki", middle, fine_homogeneous
                )
                soft_tissue_valid = meniscus_valid
                meniscus_valid = meniscus_valid & (
                    valid_position & torch.isfinite(meniscus_position).all((-1, -2))
                )[:, None]
                meniscus_encoded = {
                    "meniscus_visual_valid": meniscus_valid,
                    "meniscus_patient_position": meniscus_position.detach(),
                    "meniscus_token_steps": torch.cat(
                        (steps[:, :2] * 0.5, steps[:, 2:]), 1
                    ).detach(),
                    "meniscus_shares": (
                        meniscus_scores.detach().softmax(-1)
                        .reshape(
                            -1, height, 2, width, 2, meniscus_scores.shape[-1]
                        )
                        .mean((2, 4))
                        .flatten(1, 2)
                    ),
                }
                if self.training:
                    decoder_valid, decoder_xy = feature_geometry(
                        source_valid_rects,
                        inverse_affine,
                        decoder_maps.shape[-2:],
                        self.decoder_mapping,
                        channels,
                        batch["canvas_valid_rects"].flatten(0, 1)[flat_indices],
                        batch["crop_affine"].flatten(0, 1)[flat_indices],
                    )
                    decoder_homogeneous = torch.cat(
                        (decoder_xy, torch.ones_like(decoder_xy[..., :1])), -1
                    )
                    decoder_position = torch.einsum(
                        "nij,nkj->nki", middle, decoder_homogeneous
                    )
                    meniscus_encoded |= {
                        "decoder_visual_valid": decoder_valid
                        & (
                            valid_position
                            & torch.isfinite(decoder_position).all((-1, -2))
                        )[:, None],
                        "decoder_patient_position": decoder_position.detach(),
                        "decoder_token_steps": (steps[:, :2] * 0.25).detach(),
                    }
                if self.soft_tissue_routing:
                    meniscus_encoded |= {
                        "soft_tissue_features": meniscus_features,
                        "soft_tissue_shares": meniscus_scores.detach().softmax(-1),
                        "soft_tissue_valid": soft_tissue_valid,
                        "soft_tissue_offset": torch.where(
                            meniscus_valid[..., None],
                            meniscus_position - center[:, None],
                            0.0,
                        ).detach(),
                    }
            study_shift, study_rotation, landmarks, study_sizes = self.study_frames(
                tokens,
                fitted_position,
                visual_valid & valid_position[:, None],
                token_study,
                token_pitch,
                batch_size,
            )
            anatomy_scores = None
            anatomy_shares = None
            anatomy_diagnostics = {}
            if self.anatomical_frame:
                anatomy_scores = self.anatomy_head(tokens.float())
                anatomy_shares = anatomy_scores.detach().float().softmax(-1)
                study_shift, study_rotation, study_sizes = self.anatomical_frames(
                    anatomy_scores,
                    fitted_position.detach(),
                    visual_valid & valid_position[:, None],
                    token_study,
                    token_pitch,
                    batch_size,
                    study_shift,
                    study_rotation,
                    study_sizes,
                    anatomy_diagnostics if view_pose is not None else None,
                )
            geometry = {
                "knee_x": knee_x,
                "anterior_axis": anterior_axis,
                "superior_axis": superior_axis,
                "frame_shift": study_shift[token_study],
                "frame_rotation": study_rotation[token_study],
                "frame_sizes": study_sizes[token_study],
                "anatomy_shares": anatomy_shares,
                "token_box": token_box,
            }
            position = self.frame(geometry, sign)
        return {
            **encoded,
            "view_index": view_index,
            "study": token_study,
            "image_plane": token_plane,
            "image_type": token_type,
            "crop_angle": batch["applied_view_angle"].flatten()[flat_indices],
            "crop_shift": batch["applied_view_shift"].flatten(0, 1)[flat_indices],
            **(
                {"crop_affine": batch["crop_affine"].flatten(0, 1)[flat_indices]}
                if not self.training and view_pose is not None
                else {}
            ),
            "visual_valid": visual_valid,
            "oa_medial_axis": oa_medial_axis,
            "oa_anterior_axis": oa_anterior_axis,
            "oa_trusted_geometry": oa_trusted_geometry,
            **geometry,
            "study_frame_shift": study_shift,
            "study_frame_rotation": study_rotation,
            "study_landmarks": landmarks,
            "study_frame_sizes": study_sizes,
            "position": position,
            "fitted_position": fitted_position,
            "anatomy_scores": anatomy_scores,
            "meniscus_scores": meniscus_scores,
            **meniscus_encoded,
            **anatomy_diagnostics,
            "patient_position": patient_position.detach(),
            "token_steps": steps.detach(),
            "left_probability": left_probability,
            "routable": valid_position,
            "token_pitch": token_pitch,
            "study_axis": axis,
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
        return self._memo(
            encoded,
            f"memo_study_tokens_{int(shared)}",
            lambda: self._study_tokens(encoded, shared),
        )

    def _study_tokens(self, encoded: dict, shared: bool) -> torch.Tensor:
        tokens = self._base_tokens(encoded)
        if not shared:
            tokens = tokens + self.position_code(
                encoded["fitted_position"], encoded["routable"]
            ).to(tokens.dtype)
        return self.mil_norm(tokens.flatten(0, 1))

    def soft_tissue_routing_weights(
        self, encoded: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._memo(
            encoded,
            "memo_soft_tissue_routing_weights",
            lambda: self._soft_tissue_routing_weights(encoded),
        )

    @torch.no_grad()
    def _soft_tissue_routing_weights(
        self, encoded: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shares = encoded["soft_tissue_shares"]
        with torch.autocast(device_type=shares.device.type, enabled=False):
            shares = shares.float()
            valid = encoded["meniscus_visual_valid"]
            offset = encoded["soft_tissue_offset"].float()
            study = encoded["study"]
            batch_size = encoded["batch_size"]
            side = math.isqrt(valid.shape[1])
            tiny = torch.finfo(torch.float32).tiny

            def per_study(values: torch.Tensor) -> torch.Tensor:
                return torch.zeros(
                    (batch_size, *values.shape[2:]), device=values.device
                ).index_add(0, study, values.sum(1))

            def grid(values: torch.Tensor) -> torch.Tensor:
                return values.transpose(1, 2).unflatten(2, (side, side))

            def cells(values: torch.Tensor) -> torch.Tensor:
                return values.flatten(2).transpose(1, 2)

            def centres(mass: torch.Tensor) -> torch.Tensor:
                return per_study(
                    mass[..., None] * offset[:, :, None]
                ) / per_study(mass).clamp_min(tiny)[..., None]

            def distances(centre: torch.Tensor) -> torch.Tensor:
                return (offset[:, :, None] - centre[study][:, None]).norm(dim=-1)

            meniscus = (
                shares[..., [MEDIAL_MENISCUS_CLASS, LATERAL_MENISCUS_CLASS]]
                * valid[..., None]
            )
            peak = torch.zeros((batch_size, 2), device=shares.device).index_reduce_(
                0, study, meniscus.amax(1), "amax"
            )
            core = (meniscus > 0) & (
                meniscus >= MENISCUS_CORE_SHARE * peak[study][:, None]
            )
            opened = nn.functional.max_pool2d(grid(core.float()), 3, 1, 1) > 0
            spread = nn.functional.avg_pool2d(grid(meniscus), 3, 1, 1)
            distance = distances(centres(meniscus)).masked_fill(
                ~(peak > 0)[study][:, None], torch.inf
            )
            meniscus_weight = (
                cells(torch.where(opened, torch.maximum(grid(meniscus), spread), 0.0))
                * (distance <= distance.flip(-1))
                * valid[..., None]
            )

            anchors = (
                torch.stack(
                    (
                        shares[..., MEDIAL_TIBIAL_CARTILAGE_CLASS]
                        + shares[..., MEDIAL_MENISCUS_CLASS],
                        shares[..., LATERAL_TIBIAL_CARTILAGE_CLASS]
                        + shares[..., LATERAL_MENISCUS_CLASS],
                        shares[..., PATELLAR_CARTILAGE_CLASS],
                    ),
                    -1,
                )
                * valid[..., None]
            )
            found = per_study(anchors) >= 1
            known = found[study][:, None]
            distance = distances(centres(anchors)).masked_fill(~known, torch.inf)
            own = (
                nn.functional.one_hot(distance.argmin(-1), anchors.shape[-1]).bool()
                & known
            )
            cartilage = (
                shares[
                    ...,
                    [
                        MEDIAL_TIBIAL_CARTILAGE_CLASS,
                        LATERAL_TIBIAL_CARTILAGE_CLASS,
                        PATELLAR_CARTILAGE_CLASS,
                    ],
                ]
                + own * shares[..., FEMORAL_CARTILAGE_CLASS, None]
            ) * valid[..., None]
            typical = per_study(cartilage.square()) / per_study(cartilage).clamp_min(
                tiny
            )
            rim = cells(
                nn.functional.avg_pool2d(
                    grid(
                        (cartilage / typical[study][:, None].clamp_min(tiny)).clamp(
                            max=1
                        )
                    ),
                    3,
                    1,
                    1,
                )
            )
            zone = (
                encoded["anatomy_shares"]
                .float()[..., [JOINT_CLASS, JOINT_CLASS, KNEECAP_JOINT_CLASS]]
                .unflatten(1, (side // 2, side // 2))
                .repeat_interleave(2, 1)
                .repeat_interleave(2, 2)
                .flatten(1, 2)
            )
            cartilage_weight = (
                torch.where(known, torch.maximum(rim, zone) * own, zone)
                * valid[..., None]
            )
            return (
                torch.cat((meniscus_weight, cartilage_weight), -1).permute(2, 0, 1),
                found,
            )

    def _soft_tissue_tokens(self, encoded: dict) -> torch.Tensor:
        tokens = (
            encoded["soft_tissue_features"].float()
            + encoded["plane"][:, None].float()
            + encoded["contrast"][:, None].float()
            + encoded["detail_view"][:, None, None]
            * self.detail_view_embedding.float()[None, None]
        )
        return self.soft_norm(tokens.flatten(0, 1))

    def _soft_tissue_pool(self, encoded: dict, view: int | None) -> torch.Tensor:
        valid = encoded["soft_tissue_valid"]
        if view is not None:
            valid = valid & (encoded["view_index"] == view)[:, None]
        study = encoded["study"][:, None].expand_as(valid).flatten()
        cell_view = encoded["view_index"][:, None].expand_as(valid).flatten()
        valid = valid.flatten()
        batch_size = encoded["batch_size"]
        with torch.autocast(device_type=valid.device.type, enabled=False):
            tokens = self._memo(
                encoded,
                "memo_soft_tissue_tokens",
                lambda: self._soft_tissue_tokens(encoded),
            )
            scores = self.soft_attention(tokens).transpose(0, 1)[
                self.soft_target_channel
            ]
            if self.view_label_bias is not None:
                scores = scores + self.view_label_bias.float()[cell_view][
                    :, self.target_channel[self.soft_target]
                ].transpose(0, 1)
            membership = (
                torch.arange(batch_size, device=valid.device)[:, None] == study[None]
            ) & valid[None]
            empty_study = ~membership.any(-1)
            if view is None and empty_study.any():
                raise ValueError("A study has no visually valid soft-tissue cells")
            allowed = membership.clone()
            allowed[:, 0] |= empty_study
            routing, found = self.soft_tissue_routing_weights(encoded)
            routing = routing.flatten(1)
            total = membership.float() @ routing.T
            routed = total >= 1
            attention = scores[None] + torch.where(
                routed[:, :, None],
                (
                    routing.clamp_min(ROUTING_MINIMUM_WEIGHT).log()
                    / self.routing_temperature
                )[None],
                0.0,
            )
            attention = attention.masked_fill(~allowed[:, None], -torch.inf).softmax(-1)
            if view is not None:
                attention = attention.masked_fill(empty_study[:, None, None], 0.0)
            else:
                self.soft_tissue_routing_stats = {
                    "open_cells": (
                        total.square()
                        / (membership.float() @ routing.square().T).clamp_min(
                            torch.finfo(torch.float32).tiny
                        )
                    ).detach(),
                    "weight": total.detach(),
                    "routed": routed.float(),
                    "compartment_found": found.float(),
                }
            evidence = (
                self.soft_head(tokens, encoded["image_type"])
                if self.plane_readout
                else nn.functional.linear(tokens, self.soft_head.weight.float())
            )[:, self.soft_target_channel]
            logits = torch.einsum("btn,nt->bt", attention, evidence)
            if view is None:
                self.soft_tissue_routing_stats["logit_abs"] = logits.detach().abs()
            return logits

    def pool_encoded(self, encoded: dict, view: int | None = None) -> torch.Tensor:
        logits = self._pool_tokens(encoded, view)
        if not self.soft_tissue_routing:
            return logits
        soft = self._soft_tissue_pool(encoded, view)
        if not torch.isfinite(soft).all():
            raise RuntimeError("Soft-tissue pooling produced non-finite logits")
        logits = logits.clone()
        logits[:, self.soft_target] += soft.to(logits.dtype)
        return logits

    def _pool_tokens(self, encoded: dict, view: int | None = None) -> torch.Tensor:
        return self._attention_pool(
            self.study_tokens(encoded),
            self.study_tokens(encoded, shared=True),
            encoded["study"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            encoded["image_type"],
            (
                encoded["visual_valid"]
                if view is None
                else encoded["visual_valid"] & (encoded["view_index"] == view)[:, None]
            ).flatten(),
            encoded["batch_size"],
            self.routing_weights(encoded).flatten(1),
            self._memo(
                encoded,
                "memo_position_prior",
                lambda: torch.where(
                    self.shared_channel[self.target_channel][:, None, None],
                    self.position_priors(encoded["position"], encoded["routable"]),
                    self.position_priors(
                        encoded["fitted_position"], encoded["routable"]
                    ),
                ).flatten(1),
            ),
            encoded["oa_medial_axis"].flatten(),
            encoded["oa_anterior_axis"].flatten(),
            encoded["oa_trusted_geometry"].flatten(),
            encoded["view_index"][:, None].expand_as(encoded["visual_valid"]).flatten(),
            diagnostic=view is not None,
        )

    def forward(
        self, batch: dict, view_pose: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.pool_encoded(self.encode_batch(batch, view_pose))
