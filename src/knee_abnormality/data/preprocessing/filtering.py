import numpy as np

from knee_abnormality.data.preprocessing.geometry import SeriesGeometry
from knee_abnormality.data.preprocessing.series import IndexedSeries

NON_KNEE_COIL_WORDS = (
    "ankle",
    "foot",
    "wrist",
    "hand",
    "elbow",
    "shoulder",
    "hip",
    "head",
    "neck",
    "breast",
)
SIDE_SPLIT_MM = 150.0
FILTERING_IDENTITY = {
    "instance_guard": "modal_rows_columns_spacing_orientation_within_orientation_tolerance",
    "non_knee_coil": "drop_coil_naming_non_knee_part_when_study_has_other_coil_name",
    "non_knee_coil_words": ",".join(NON_KNEE_COIL_WORDS),
    "other_knee": "split_volume_center_x_spread_at_largest_gap_drop_single_series_side",
    "side_split_mm": SIDE_SPLIT_MM,
    "stage": "before_reference_estimate_never_every_series",
}


def non_knee_coil(name: str) -> bool:
    lowered = name.lower()
    return "knee" not in lowered and any(
        word in lowered for word in NON_KNEE_COIL_WORDS
    )


def volume_center(geometry: SeriesGeometry) -> np.ndarray | None:
    if geometry.center is None or geometry.normal is None or geometry.bounds is None:
        return None
    normal = np.asarray(geometry.normal, dtype=np.float64)
    scale = float(np.linalg.norm(normal))
    center = np.asarray(geometry.center, dtype=np.float64)
    if not np.isfinite(scale) or scale <= 0:
        return None
    unit = normal / scale
    value = center + unit * (sum(geometry.bounds) / 2 / scale - center @ unit)
    return value if np.isfinite(value).all() else None


def excluded_series(series: dict[str, IndexedSeries]) -> frozenset[str]:
    coils = {
        uid: indexed.summary.receive_coil
        for uid, indexed in series.items()
        if indexed.summary.receive_coil
    }
    excluded = set()
    if len(set(coils.values())) > 1 and not all(map(non_knee_coil, coils.values())):
        excluded = {uid for uid, name in coils.items() if non_knee_coil(name)}
    centers = {
        uid: center[0]
        for uid, indexed in series.items()
        if uid not in excluded
        and (center := volume_center(indexed.geometry)) is not None
    }
    values = sorted(centers.values())
    if len(values) > 2 and values[-1] - values[0] > SIDE_SPLIT_MM:
        cut = values[int(np.argmax(np.diff(values)))]
        lower = {uid for uid, x in centers.items() if x <= cut}
        upper = set(centers) - lower
        for side, other in ((lower, upper), (upper, lower)):
            if len(side) == 1 and len(other) >= 2:
                excluded |= side
    return frozenset() if excluded >= set(series) else frozenset(excluded)
