import numpy as np
from scipy import ndimage

from knee_abnormality.data.preprocessing.dicom import (
    PREPARATION_ERRORS,
    DecodedFrame,
    read_frame,
)
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
COPY_POSITION_DECIMALS = 2
CONTRAST_SLICES = (0.25, 0.5, 0.75)
FILTERING_IDENTITY = {
    "order": "non_knee_coil,superseded,contrast_relabel,exact_copy,other_knee",
    "superseded": "same_slot_signature_count_matrix_sorted_positions_keep_non_reformat_then_highest_series_number_then_uid",
    "contrast_relabel": "header_candidate_moves_when_nearest_within_study_reference_by_body_intensity_is_other_contrast_same_plane_only_if_both_contrasts",
    "contrast_slices": ",".join(map(str, CONTRAST_SLICES)),
    "exact_copy": "same_slot_count_matrix_sorted_positions_equal_middle_frame_keep_non_reformat_then_uid",
    "copy_position_decimals": COPY_POSITION_DECIMALS,
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


def _layout_key(indexed: IndexedSeries) -> tuple:
    record = indexed.records[len(indexed.records) // 2]
    if record.descriptor is not None:
        shape = (record.descriptor.rows, record.descriptor.columns)
    elif record.pose is not None:
        shape = (record.pose.layout.rows, record.pose.layout.columns)
    else:
        shape = None
    positions = tuple(
        round(item.physical_position, COPY_POSITION_DECIMALS)
        for item in indexed.records
    )
    return len(indexed.records), shape, positions


def _middle_frame(indexed: IndexedSeries) -> np.ndarray | None:
    record = indexed.records[len(indexed.records) // 2]
    try:
        return read_frame(record.path, record.descriptor).pixels
    except PREPARATION_ERRORS:
        return None


def non_knee_coils(series: dict[str, IndexedSeries]) -> frozenset[str]:
    coils = {
        uid: indexed.summary.receive_coil
        for uid, indexed in series.items()
        if indexed.summary.receive_coil
    }
    if len(set(coils.values())) > 1 and not all(map(non_knee_coil, coils.values())):
        return frozenset(uid for uid, name in coils.items() if non_knee_coil(name))
    return frozenset()


def superseded_series(
    series: dict[str, IndexedSeries], slots: dict[str, int]
) -> frozenset[str]:
    groups: dict[tuple, list[str]] = {}
    for uid, indexed in series.items():
        signature = indexed.summary.signature
        if indexed.records and signature is not None and uid in slots:
            groups.setdefault(
                (slots[uid], signature, *_layout_key(indexed)), []
            ).append(uid)
    superseded = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        kept = max(
            members,
            key=lambda uid: (
                not series[uid].summary.reformat,
                series[uid].summary.series_number
                if series[uid].summary.series_number is not None
                else -np.inf,
                uid,
            ),
        )
        superseded.update(uid for uid in members if uid != kept)
    return frozenset(superseded)


def _slice_intensity(frame: DecodedFrame) -> np.ndarray | None:
    pixels = frame.pixels * frame.slope + frame.intercept
    step = max(1, max(pixels.shape) // 160)
    image = pixels[::step, ::step].astype(np.float64)
    spacing = max(frame.row_spacing, frame.column_spacing) * step
    top = float(np.quantile(image, 0.995))
    if not np.isfinite(top) or top <= 0:
        return None
    mask = ndimage.binary_opening(image > 0.12 * top, iterations=1)
    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    body = ndimage.binary_fill_holes(labels == int(np.argmax(sizes)) + 1)
    values = image[body]
    if values.size < 50:
        return None
    reference = float(np.quantile(values, 0.99))
    inner = ndimage.binary_erosion(body, iterations=max(1, round(8.0 / spacing)))
    ring = body & ~inner
    rows, columns = np.nonzero(body)
    top_row, bottom_row = np.quantile(rows, [0.3, 0.7])
    left, right = np.quantile(columns, [0.3, 0.7])
    central = image[int(top_row) : int(bottom_row) + 1, int(left) : int(right) + 1]
    return np.asarray(
        [
            *(np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]) / reference),
            (values > 0.5 * reference).mean(),
            np.median(image[ring]) / reference if ring.any() else np.nan,
            np.median(image[inner]) / reference if inner.any() else np.nan,
            np.median(central) / reference if central.size else np.nan,
        ]
    )


def _series_intensity(indexed: IndexedSeries) -> np.ndarray | None:
    values = []
    for fraction in CONTRAST_SLICES:
        record = indexed.records[round(fraction * (len(indexed.records) - 1))]
        try:
            value = _slice_intensity(read_frame(record.path, record.descriptor))
        except PREPARATION_ERRORS:
            value = None
        if value is not None:
            values.append(value)
    if not values:
        return None
    mean = np.nanmean(np.stack(values), axis=0)
    return mean if np.isfinite(mean).all() else None


def _contrast_candidate(indexed: IndexedSeries, fluid: int) -> bool:
    contrast = indexed.summary.contrast
    return (fluid == 0 and contrast == "fluid_fat_suppressed") or (
        fluid == 1 and contrast == "not_fat_suppressed"
    )


def relabelled_slots(
    series: dict[str, IndexedSeries],
    slots: dict[str, int],
    kinds: tuple[tuple[str, int, int], ...],
) -> dict[str, int]:
    present = {
        uid for uid, indexed in series.items() if indexed.records and uid in slots
    }
    candidates = {
        uid for uid in present if _contrast_candidate(series[uid], kinds[slots[uid]][1])
    }
    if not candidates:
        return dict(slots)
    intensity = {
        uid: value
        for uid in sorted(present)
        if (value := _series_intensity(series[uid])) is not None
    }
    assigned = dict(slots)
    for uid in sorted(candidates & intensity.keys()):
        plane, fluid, fat = kinds[slots[uid]]
        references = {
            value: [
                other
                for other in intensity
                if other not in candidates and kinds[slots[other]][1] == value
            ]
            for value in (0, 1)
        }
        same_plane = {
            value: [other for other in members if kinds[slots[other]][0] == plane]
            for value, members in references.items()
        }
        used = same_plane if same_plane[0] and same_plane[1] else references
        if not used[0] or not used[1]:
            continue
        distance = {
            value: min(
                float(np.linalg.norm(intensity[other] - intensity[uid]))
                for other in members
            )
            for value, members in used.items()
        }
        target = next(
            (
                slot
                for slot, kind in enumerate(kinds)
                if kind == (plane, 1 - fluid, 1 - fat)
            ),
            None,
        )
        if target is not None and distance[1 - fluid] < distance[fluid]:
            assigned[uid] = target
    return assigned


def exact_copies(
    series: dict[str, IndexedSeries], slots: dict[str, int]
) -> frozenset[str]:
    groups: dict[tuple, list[str]] = {}
    for uid, indexed in series.items():
        if indexed.records and uid in slots:
            groups.setdefault((slots[uid], *_layout_key(indexed)), []).append(uid)
    copies = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        kept: list[np.ndarray] = []
        for uid in sorted(members, key=lambda uid: (series[uid].summary.reformat, uid)):
            frame = _middle_frame(series[uid])
            if frame is None:
                continue
            if any(np.array_equal(frame, other) for other in kept):
                copies.add(uid)
            else:
                kept.append(frame)
    return frozenset(copies)


def other_knee_series(series: dict[str, IndexedSeries]) -> frozenset[str]:
    centers = {
        uid: center[0]
        for uid, indexed in series.items()
        if (center := volume_center(indexed.geometry)) is not None
    }
    values = sorted(centers.values())
    excluded = set()
    if len(values) > 2 and values[-1] - values[0] > SIDE_SPLIT_MM:
        cut = values[int(np.argmax(np.diff(values)))]
        lower = {uid for uid, x in centers.items() if x <= cut}
        upper = set(centers) - lower
        for side, other in ((lower, upper), (upper, lower)):
            if len(side) == 1 and len(other) >= 2:
                excluded |= side
    return frozenset(excluded)


def clean_series(
    series: dict[str, IndexedSeries],
    slots: dict[str, int],
    kinds: tuple[tuple[str, int, int], ...],
) -> tuple[frozenset[str], dict[str, int]]:
    coils = non_knee_coils(series)
    kept = {uid: indexed for uid, indexed in series.items() if uid not in coils}
    superseded = superseded_series(kept, slots)
    kept = {uid: indexed for uid, indexed in kept.items() if uid not in superseded}
    assigned = relabelled_slots(kept, slots, kinds)
    copies = exact_copies(kept, assigned)
    kept = {uid: indexed for uid, indexed in kept.items() if uid not in copies}
    ruled = coils | other_knee_series(kept)
    if ruled >= set(kept) | coils:
        ruled = frozenset()
    return superseded | copies | ruled, assigned
