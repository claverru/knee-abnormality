import hashlib
import json
import math
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import log_ndtr, logsumexp
from scipy.stats import truncnorm

from knee_abnormality.data.preprocessing.geometry import (
    RELIABILITY,
    StudyReference,
    summarize_series,
    valid_pose,
    within_acquired_range,
)
from knee_abnormality.data.preprocessing.series import IndexedSeries, SliceRecord

SAMPLING_IDENTITY = {
    "reference_physical": "within_acquired_range_else_local_all_fit_methods",
    "gap_floor_mm": 0.01,
    "gap_median_multiplier": 1.5,
    "gap_margin_mm": 0.05,
    "plane_group_tolerance_mm": 0.1,
    "grouping": "complete_link_offset_plus_crop_radius_normal_distance",
    "support": "union_of_run_middle_extents_singleton_neighbor_cells",
    "targets": "stratified_truncated_gaussian_patient_axis_sigma_fraction_of_pool_limiting_reach",
    "suppliers": "uniform_random_or_exact_fractional_entitlement_before_assignment",
    "source_ranking": "original_then_newest_same_signature_then_crop_coverage_0.1_then_acquired_resolution_0.01_then_uid",
    "assignment": "maximum_cardinality_then_supplier_changes_then_squared_error",
    "fallback": "union_uniform_acquisition_local_affine_run_support_unknown_xyz",
    "normalization": "unique_retained_complete_triplet_slices_once",
    "detail_view": "sagittal_fluid_second_view_reuses_first_view_selection_and_budget_no_reallocation",
    "retry": "split_original_runs_at_failed_paths_fixed_center_and_spacing",
    "seed": "sha256_training_seed_study_type_operation_epoch_evaluation_unseeded",
}
AXES = {"Axial": 2, "Coronal": 1, "Sagittal": 0}


@dataclass(frozen=True)
class Triplet:
    uid: str
    records: tuple[SliceRecord, ...]
    run: int
    depth: float
    support: tuple[float, float]
    normal: tuple[float, ...] | None
    offset: float | None


@dataclass(frozen=True)
class CandidatePool:
    candidates: tuple[Triplet, ...]
    physical: bool
    diagnostics: dict
    ranking: tuple[str, ...]


@dataclass(frozen=True)
class Selection:
    triplets: tuple[Triplet, ...]
    metadata: dict


def operation_seed(
    seed: int, study: str, kind: tuple, operation: str, epoch: int
) -> int:
    payload = json.dumps((seed, study, kind, operation, epoch), separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def valid_runs(
    records: tuple[SliceRecord, ...], failed: set[Path]
) -> tuple[tuple[SliceRecord, ...], ...]:
    if len({r.position_source for r in records}) != 1:
        return ()
    positions = np.asarray([r.physical_position for r in records])
    gaps = np.diff(positions)
    positive = gaps[np.isfinite(gaps) & (gaps > 0.01)]
    threshold = 1.5 * float(np.median(positive)) + 0.05 if len(positive) else 0.05
    runs, current = [], []
    for i, record in enumerate(records):
        broken = i and (
            not np.isfinite(gaps[i - 1]) or not 0.01 < gaps[i - 1] <= threshold
        )
        if broken or record.path in failed or not np.isfinite(record.physical_position):
            if len(current) >= 3:
                runs.append(tuple(current))
            current = []
        if record.path not in failed and np.isfinite(record.physical_position):
            current.append(record)
    if len(current) >= 3:
        runs.append(tuple(current))
    return tuple(runs)


def usable_local_order(indexed: IndexedSeries) -> bool:
    geometry, records = indexed.geometry, indexed.records
    if geometry.reason is None and geometry.native_poses_valid:
        return True
    if not records or any(r.position_source != "slice_location" for r in records):
        return False
    if len(geometry.frame_uids) > 1:
        return False
    orientations = [
        np.asarray(r.pose.layout.orientation)
        for r in records
        if r.pose is not None and np.isfinite(r.pose.layout.orientation).all()
    ]
    if orientations and any(
        np.max(np.abs(value - orientations[0])) > RELIABILITY["orientation_tolerance"]
        for value in orientations[1:]
    ):
        return False
    native = summarize_series(
        tuple(replace(r, position_source="patient") for r in records)
    )
    return native.reason not in {
        "inconsistent_orientation",
        "inconsistent_center_line",
        "inconsistent_frame",
    }


def crop_coverage(
    indexed: IndexedSeries, point: tuple[float, float, float] | None, radius: float
) -> float | None:
    records = indexed.records
    pose = records[len(records) // 2].pose if records else None
    if point is None or pose is None or not valid_pose(pose):
        return None
    layout = pose.layout
    axes = np.asarray(layout.orientation, dtype=np.float64).reshape(2, 3)
    u, v = axes @ (np.asarray(point) - np.asarray(pose.position, dtype=np.float64))
    width = (layout.columns - 1) * layout.spacing[1]
    height = (layout.rows - 1) * layout.spacing[0]
    overlap = max(0.0, min(u + radius, width) - max(u - radius, 0.0)) * max(
        0.0, min(v + radius, height) - max(v - radius, 0.0)
    )
    value = float(overlap / (2 * radius) ** 2)
    return value if math.isfinite(value) else None


def source_ranking(
    sources: dict[str, IndexedSeries],
    point: tuple[float, float, float] | None,
    spacing_mm: float,
    radius: float,
) -> tuple[str, ...]:
    summaries = {uid: indexed.summary for uid, indexed in sources.items()}
    coverage = {
        uid: crop_coverage(indexed, point, radius) for uid, indexed in sources.items()
    }
    covered = all(value is not None for value in coverage.values())
    resolved = all(
        summary.acquired_resolution_mm is not None for summary in summaries.values()
    )

    def newer(uid: str) -> int:
        own = summaries[uid]
        if own.signature is None or own.series_number is None:
            return 0
        return sum(
            other.signature == own.signature
            and other.series_number is not None
            and other.series_number > own.series_number
            for key, other in summaries.items()
            if key != uid
        )

    return tuple(
        sorted(
            sources,
            key=lambda uid: (
                summaries[uid].reformat,
                newer(uid),
                -round(coverage[uid], 1) if covered else 0,
                round(max(summaries[uid].acquired_resolution_mm, spacing_mm), 2)
                if resolved
                else 0,
                uid,
            ),
        )
    )


def build_pool(
    sources: tuple[tuple[str, IndexedSeries], ...],
    plane: str,
    reference: StudyReference,
    spacing_mm: float,
    radius: float,
    failed_paths: set[Path] | None = None,
    failed_sources: set[str] | None = None,
) -> CandidatePool:
    physical, local = [], []
    diagnostics = {
        "sources": {},
        "reference_method": reference.method,
        "singleton_runs": 0,
        "split_sources": 0,
        "plane_disagreements": 0,
        "no_valid_run_sources": 0,
    }
    axis = AXES[plane]
    slice_normals: dict[tuple, np.ndarray] = {}
    for uid, indexed in sorted(sources, key=lambda item: item[0]):
        if uid in (failed_sources or set()):
            diagnostics["sources"][uid] = "preparation_failure"
            continue
        geometry, records = indexed.geometry, indexed.records
        reliable = geometry.reason is None and geometry.native_poses_valid
        normal = np.asarray(geometry.normal, dtype=float) if reliable else None
        agrees = reliable and int(np.argmax(np.abs(normal))) == axis
        if reliable and not agrees:
            diagnostics["plane_disagreements"] += 1
        inside = reference.point is not None and within_acquired_range(
            geometry, reference.point
        )
        use_physical = agrees and inside and reference.point is not None
        local_ok = usable_local_order(indexed)
        if not local_ok or any(r.relative_position_mm is None for r in records):
            diagnostics["sources"][uid] = geometry.reason or "untrustworthy_local_order"
            continue
        runs = valid_runs(records, failed_paths or set())
        diagnostics["no_valid_run_sources"] += not bool(runs)
        diagnostics["split_sources"] += len(runs) > 1
        diagnostics["singleton_runs"] += sum(len(run) == 3 for run in runs)
        diagnostics["sources"][uid] = (
            "physical"
            if use_physical
            else "centre_outside_slab"
            if agrees and reference.point is not None and not inside
            else geometry.reason or reference.reason or "plane_disagreement"
        )
        if not runs:
            diagnostics["sources"][uid] = "no_valid_runs"
        values = []
        for run in runs:
            normals = []
            offsets = []
            for record in run:
                if use_physical:
                    layout = record.pose.layout.orientation
                    n = slice_normals.get(layout)
                    if n is None:
                        row, column = np.asarray(layout).reshape(2, 3)
                        n = np.array(
                            [
                                row[1] * column[2] - row[2] * column[1],
                                row[2] * column[0] - row[0] * column[2],
                                row[0] * column[1] - row[1] * column[0],
                            ]
                        )
                        n /= np.linalg.norm(n)
                        slice_normals[layout] = n
                    if n[axis] < 0:
                        n = -n
                    normals.append(tuple(n))
                    offsets.append(
                        float(
                            np.dot(
                                n, np.asarray(record.pose.position) - reference.point
                            )
                        )
                    )
            depths = np.asarray(
                [offset / n[axis] for offset, n in zip(offsets, normals, strict=True)]
                if use_physical
                else [r.physical_position for r in run]
            )
            support = (
                tuple(
                    sorted(((depths[0] + depths[1]) / 2, (depths[1] + depths[2]) / 2))
                )
                if len(run) == 3
                else (float(min(depths[1:-1])), float(max(depths[1:-1])))
            )
            for middle in range(1, len(run) - 1):
                values.append(
                    Triplet(
                        uid,
                        run[middle - 1 : middle + 2],
                        run[0].original_index,
                        float(depths[middle]),
                        support,
                        normals[middle] if use_physical else None,
                        offsets[middle] if use_physical else None,
                    )
                )
        if use_physical:
            physical.extend(values)
        elif values:
            lo = min(v.support[0] for v in values)
            hi = max(v.support[1] for v in values)
            local.extend(
                replace(
                    v,
                    depth=(v.depth - lo) / (hi - lo) if len(values) > 1 else 0.5,
                    support=(
                        (v.support[0] - lo) / (hi - lo),
                        (v.support[1] - lo) / (hi - lo),
                    )
                    if len(values) > 1
                    else (0.0, 1.0),
                )
                for v in values
            )
    candidates = tuple(physical or local)
    used = {v.uid for v in candidates}
    ranking = source_ranking(
        {uid: indexed for uid, indexed in sources if uid in used},
        reference.point,
        spacing_mm,
        radius,
    )
    diagnostics["path"] = "physical" if physical else "local" if local else "unusable"
    diagnostics["source_ranking"] = list(ranking)
    return CandidatePool(candidates, bool(physical), diagnostics, ranking)


def union_support(candidates: tuple[Triplet, ...]) -> np.ndarray:
    intervals = []
    for lo, hi in sorted({v.support for v in candidates}):
        if intervals and lo <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], hi)
        else:
            intervals.append([lo, hi])
    return np.asarray(intervals, dtype=float).reshape(-1, 2)


def limiting_reach(pool: CandidatePool) -> float:
    intervals = union_support(pool.candidates)
    lo = float(intervals[:, 0].min())
    hi = float(intervals[:, 1].max())
    return max(min(-lo, hi), 1.0)


def interval_log_mass(lo: float, hi: float) -> float:
    if lo >= 0:
        a, b = log_ndtr(-lo), log_ndtr(-hi)
    else:
        a, b = log_ndtr(hi), log_ndtr(lo)
    return float(a + np.log(-np.expm1(b - a)))


def target_depths(
    pool: CandidatePool, percentiles: np.ndarray, sigma: float | None
) -> np.ndarray:
    intervals = union_support(pool.candidates)
    logs = (
        np.asarray([interval_log_mass(lo / sigma, hi / sigma) for lo, hi in intervals])
        if pool.physical
        else np.log(intervals[:, 1] - intervals[:, 0])
    )
    masses = np.exp(logs - logsumexp(logs))
    cumulative = masses.cumsum()
    cumulative[-1] = 1
    indices = np.minimum(
        np.searchsorted(cumulative, percentiles, side="right"), len(intervals) - 1
    )
    previous = np.concatenate(([0.0], cumulative[:-1]))
    fractions = np.clip(
        (percentiles - previous[indices]) / masses[indices],
        np.finfo(float).eps,
        1 - np.finfo(float).eps,
    )
    lo, hi = intervals[indices].T
    values = (
        sigma * truncnorm.ppf(fractions, lo / sigma, hi / sigma)
        if pool.physical
        else lo + fractions * (hi - lo)
    )
    return np.clip(values, lo, hi)


def geometry_groups(
    pool: CandidatePool, radius: float
) -> tuple[tuple[Triplet, ...], ...]:
    key = lambda v: (
        ((*v.normal, v.offset) if pool.physical else (v.depth,))
        + (v.uid, str(v.records[1].path))
    )
    groups = []
    for value in sorted(pool.candidates, key=key):
        normal = np.asarray(value.normal)

        def linked(other, value=value, normal=normal):
            offset = abs(value.offset - other.offset)
            return offset <= 0.1 and (
                offset + radius * np.linalg.norm(normal - other.normal) <= 0.1
            )

        group = next(
            (g for g in groups if pool.physical and all(linked(other) for other in g)),
            None,
        )
        if group is None:
            groups.append([value])
        else:
            group.append(value)
    return tuple(
        tuple(sorted(g, key=lambda v: (v.uid, str(v.records[1].path)))) for g in groups
    )


def largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return np.zeros(len(weights), dtype=np.int64)
    raw = weights / weights.sum() * total
    counts = np.floor(raw).astype(np.int64)
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[: int(total) - int(counts.sum())]] += 1
    return counts


def study_budgets(
    base: tuple[int, ...],
    sources: tuple[int, ...],
    total: int,
) -> tuple[int, ...]:
    present = np.asarray(sources, dtype=np.int64) > 0
    counts = np.asarray(base, dtype=np.int64) * present
    if not present.any():
        return tuple(int(value) for value in counts)
    if counts.sum() < total:
        counts = largest_remainder(total, counts.astype(float))
    counts = np.where(present, counts, np.asarray(base, dtype=np.int64))
    return tuple(int(value) for value in counts)


def shared_view_budgets(
    base: tuple[int, ...],
    sources: tuple[int, ...],
    total: int,
    view_sources: tuple[int, ...],
) -> tuple[int, ...]:
    primary = tuple(
        index for index, source in enumerate(view_sources) if index == source
    )
    counts = study_budgets(
        tuple(base[index] for index in primary),
        tuple(sources[index] for index in primary),
        total,
    )
    lookup = dict(zip(primary, counts, strict=True))
    return tuple(lookup[source] for source in view_sources)


def select_triplets(
    pool: CandidatePool,
    budget: int,
    sigma_fraction: float,
    radius: float,
    study: str,
    kind: tuple,
    training_seed: int | None,
    epoch: int = 0,
) -> Selection:
    if budget < 1:
        raise ValueError("Sampling requires a positive budget")
    sigma = sigma_fraction * limiting_reach(pool) if pool.physical else None
    if pool.physical and (not np.isfinite(sigma) or sigma <= 0):
        raise ValueError(
            "Sampling requires a finite positive sigma for a physical pool"
        )
    groups = geometry_groups(pool, radius)
    count = min(budget, len(groups))
    metadata = {
        **pool.diagnostics,
        "requested_triplets": budget,
        "capacity": len(groups),
        "sampling_sigma_fraction": sigma_fraction,
        "sampling_sigma_mm": sigma,
        "triplets": 0,
        "channel_appearances": 0,
        "unique_source_slices": 0,
        "distinct_middle_geometry": 0,
        "shortfall": budget,
        "targets": [],
        "assignments": [],
    }
    if not count:
        return Selection((), metadata)

    training = training_seed is not None

    def rng(operation):
        return np.random.default_rng(
            operation_seed(training_seed, study, kind, operation, epoch)
        )

    percentiles = (
        np.arange(count) + (rng("targets").random(count) if training else 0.5)
    ) / count
    targets = target_depths(pool, percentiles, sigma)
    sources = pool.ranking
    supplier_rng = rng("suppliers") if training else None
    entitlement = dict.fromkeys(sources, Fraction(0))
    assigned = dict.fromkeys(sources, 0)
    preferred = [None] * count
    eligible = [
        {v.uid for v in pool.candidates if v.support[0] <= t <= v.support[1]}
        for t in targets
    ]
    order = rng("band_order").permutation(count) if training else np.argsort(targets)
    for row in order:
        choices = [s for s in sources if s in eligible[row]]
        for source in choices:
            entitlement[source] += Fraction(1, len(choices))
        source = (
            choices[int(supplier_rng.integers(len(choices)))]
            if training
            else max(choices, key=lambda s: entitlement[s] - assigned[s])
        )
        preferred[row] = source
        assigned[source] += 1
    ordered = [
        [
            (value, value.uid, value.support[0], value.support[1], value.depth, middle)
            for value in group
            for middle in (str(value.records[1].path),)
        ]
        for group in groups
    ]
    alternatives, errors = {}, []
    for row, target in enumerate(targets):
        wanted = preferred[row]
        for column, group in enumerate(ordered):
            best = None
            best_key = None
            for value, uid, low, high, depth, middle in group:
                if low <= target <= high:
                    key = (uid != wanted, (depth - target) ** 2, uid, middle)
                    if best_key is None or key < best_key:
                        best, best_key = value, key
            if best is not None:
                error = (best.depth - target) ** 2
                alternatives[row, column] = (best, error)
                errors.append(error)
    scale = max(errors, default=0.0) or 1.0
    change_cost = count + 1.0
    missing_cost = count * (change_cost + 1) + 1
    costs = np.full((count, len(groups) + count), missing_cost)
    costs[:, : len(groups)] = np.inf
    for (row, col), (value, error) in alternatives.items():
        costs[row, col] = (value.uid != preferred[row]) * change_cost + error / scale
    rows, cols = linear_sum_assignment(costs)
    selected, placements = [], []
    for row, col in zip(rows, cols, strict=True):
        if col >= len(groups):
            continue
        value, error = alternatives[row, col]
        selected.append(value)
        placements.append(
            {
                "band": int(row),
                "target": float(targets[row]),
                "depth": value.depth,
                "error": float(np.sqrt(error)),
                "preferred_uid": preferred[row],
                "uid": value.uid,
                "supplier_changed": value.uid != preferred[row],
                "middle_path": str(value.records[1].path),
                "run": value.run,
                "support": value.support,
            }
        )
    metadata.update(
        triplets=len(selected),
        channel_appearances=3 * len(selected),
        unique_source_slices=len({r.path for v in selected for r in v.records}),
        distinct_middle_geometry=len(selected),
        shortfall=budget - len(selected),
        targets=targets.tolist(),
        assignments=placements,
        target_entitlements={s: float(value) for s, value in entitlement.items()},
        preferred_counts=assigned,
        realized_counts={s: sum(v.uid == s for v in selected) for s in sources},
    )
    return Selection(tuple(selected), metadata)
