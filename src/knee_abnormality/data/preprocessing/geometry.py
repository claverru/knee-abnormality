from collections import Counter
from dataclasses import dataclass

import numpy as np

RELIABILITY = {
    "orientation_tolerance": 1e-3,
    "center_line_tolerance_mm": 0.1,
    "maximum_condition": 1e3,
    "maximum_rms_mm": 10.0,
    "maximum_residual_mm": 20.0,
    "fitted_midline": "validated_study_reference_x_replaces_series_median",
}


@dataclass(frozen=True, slots=True)
class PoseLayout:
    orientation: tuple[float, ...]
    spacing: tuple[float, float]
    rows: int
    columns: int
    frame_uid: str | None


@dataclass(frozen=True, slots=True)
class NativePose:
    position: tuple[float, float, float]
    layout: PoseLayout


@dataclass(frozen=True, slots=True)
class SeriesGeometry:
    center: tuple[float, float, float] | None
    normal: tuple[float, float, float] | None
    bounds: tuple[float, float] | None
    frame_uids: tuple[str, ...]
    reason: str | None
    orientation_change: float
    line_departure_mm: float
    native_poses_valid: bool
    orientations: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class StudyReference:
    point: tuple[float, float, float] | None
    contributors: tuple[str, ...]
    condition: float | None
    rms_mm: float | None
    residuals_mm: tuple[tuple[str, float], ...]
    frame_uids: tuple[str, ...]
    reason: str | None


def reflection_eligibility(geometry: SeriesGeometry, plane: str) -> tuple[bool, str]:
    if geometry.reason is not None:
        return False, geometry.reason
    if not geometry.native_poses_valid or not geometry.orientations:
        return False, "invalid_native_pose"
    expected = {
        "Axial": (0, 1, 2),
        "Coronal": (0, 2, 1),
        "Sagittal": (1, 2, 0),
    }.get(plane)
    if expected is None:
        return False, "unsupported_plane"
    tolerance = RELIABILITY["orientation_tolerance"]
    orientations = geometry.orientations
    for orientation in orientations:
        axes = np.asarray(orientation, dtype=np.float64).reshape(2, 3)
        vectors = (*axes, np.cross(axes[0], axes[1]))
        for vector, axis in zip(vectors, expected, strict=True):
            if (
                abs(vector[axis])
                <= max(
                    (abs(value) for index, value in enumerate(vector) if index != axis),
                    default=0.0,
                )
                + tolerance
            ):
                return False, "ambiguous_orientation_axes"
    return True, "eligible"


def valid_pose(pose: NativePose) -> bool:
    layout = pose.layout
    values = (*pose.position, *layout.orientation, *layout.spacing)
    if not np.isfinite(values).all() or min(layout.spacing) <= 0:
        return False
    if min(layout.rows, layout.columns) < 1:
        return False
    a, b = np.asarray(layout.orientation).reshape(2, 3)
    tolerance = RELIABILITY["orientation_tolerance"]
    return bool(
        abs(np.linalg.norm(a) - 1) <= tolerance
        and abs(np.linalg.norm(b) - 1) <= tolerance
        and abs(np.dot(a, b)) <= tolerance
    )


def image_center(pose: NativePose) -> np.ndarray:
    layout = pose.layout
    a, b = np.asarray(layout.orientation).reshape(2, 3)
    return (
        np.asarray(pose.position)
        + a * (layout.columns - 1) * layout.spacing[1] / 2
        + b * (layout.rows - 1) * layout.spacing[0] / 2
    )


def summarize_series(records: tuple) -> SeriesGeometry:
    poses = [record.pose for record in sorted(records, key=lambda item: item.path)]
    frames = tuple(
        sorted(
            {pose.layout.frame_uid for pose in poses if pose and pose.layout.frame_uid}
        )
    )
    bounds = (
        (records[0].physical_position, records[-1].physical_position)
        if records
        else None
    )
    reason = None
    native_poses_valid = bool(poses) and all(
        pose is not None and valid_pose(pose) for pose in poses
    )
    if not records:
        reason = "empty_series"
    elif any(record.position_source != "patient" for record in records):
        reason = "non_patient_positions"
    elif not native_poses_valid:
        reason = "invalid_native_pose"
    if reason:
        return SeriesGeometry(
            None,
            None,
            bounds,
            frames,
            reason,
            0.0,
            0.0,
            native_poses_valid,
            (),
        )
    representative = poses[0]
    orientation = np.asarray(representative.layout.orientation)
    normal = np.cross(orientation[:3], orientation[3:])
    unit = normal / np.linalg.norm(normal)
    center = image_center(representative)
    change = max(
        float(np.max(np.abs(np.asarray(pose.layout.orientation) - orientation)))
        for pose in poses
    )
    departures = [image_center(pose) - center for pose in poses]
    departure = max(
        float(np.linalg.norm(delta - np.dot(delta, unit) * unit))
        for delta in departures
    )
    if change > RELIABILITY["orientation_tolerance"]:
        reason = "inconsistent_orientation"
    elif departure > RELIABILITY["center_line_tolerance_mm"]:
        reason = "inconsistent_center_line"
    elif len(frames) > 1:
        reason = "inconsistent_frame"
    return SeriesGeometry(
        tuple(center),
        tuple(normal),
        bounds,
        frames,
        reason,
        change,
        departure,
        native_poses_valid,
        tuple(sorted({pose.layout.orientation for pose in poses})),
    )


def normal_projection(geometry: SeriesGeometry) -> np.ndarray:
    normal = np.asarray(geometry.normal)
    normal = normal / np.linalg.norm(normal)
    return np.eye(3) - np.outer(normal, normal)


def fit_center(
    usable: list[tuple[str, str, SeriesGeometry]],
) -> tuple[str | None, np.ndarray | None, float | None, float | None, tuple]:
    counts = Counter(plane for _, plane, _ in usable)
    if len(counts) < 2:
        return "insufficient_planes", None, None, None, ()
    matrix = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    projections = []
    for uid, plane, geometry in usable:
        projection = normal_projection(geometry)
        weight = 1 / counts[plane]
        matrix += weight * projection
        rhs += weight * projection @ np.asarray(geometry.center)
        projections.append((uid, weight, projection, geometry.center))
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > RELIABILITY["maximum_condition"]:
        return "ill_conditioned", None, condition, None, ()
    point = np.linalg.solve(matrix, rhs)
    residuals = tuple(
        (uid, float(np.linalg.norm(projection @ (point - center))))
        for uid, _, projection, center in projections
    )
    rms = float(
        np.sqrt(
            sum(
                weight * distance**2
                for (_, weight, _, _), (_, distance) in zip(
                    projections, residuals, strict=True
                )
            )
            / len(counts)
        )
    )
    return None, point, condition, rms, residuals


def estimate_reference(
    series: tuple[tuple[str, str, SeriesGeometry], ...],
) -> StudyReference:
    usable = sorted(
        (uid, plane, geometry)
        for uid, plane, geometry in series
        if geometry.reason is None
    )
    contributors = tuple(uid for uid, _, _ in usable)
    frames = tuple(
        sorted({frame for _, _, geometry in usable for frame in geometry.frame_uids})
    )
    reason, point, condition, rms, residuals = fit_center(usable)
    if reason is None and rms > RELIABILITY["maximum_rms_mm"]:
        reason = "excessive_rms"
    elif (
        reason is None
        and max(distance for _, distance in residuals)
        > RELIABILITY["maximum_residual_mm"]
    ):
        reason = "excessive_residual"
    return StudyReference(
        tuple(point) if reason is None else None,
        contributors,
        condition,
        rms,
        residuals,
        frames,
        reason,
    )
