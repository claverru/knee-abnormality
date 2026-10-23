import math

import torch
import torch.nn.functional as F
from torch import nn

from knee_abnormality.config import DataConfig
from knee_abnormality.inference.reflection import reflect_crops
from knee_abnormality.modeling.anatomy_classes import FEMUR_CLASS, KNEECAP_CLASS


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
    inverse = torch.linalg.inv_ex(affine.float())[0]
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
            pixel = torch.linalg.solve_ex(
                gram, torch.einsum("ntpa,ntp->nta", linear, point)
            )[0]
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
