import logging
from dataclasses import replace
from functools import partial
from multiprocessing import Value
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from knee_abnormality.config import SLICES_PER_INPUT, DataConfig, TrainerConfig
from knee_abnormality.data.preprocessing.device import prepare_images
from knee_abnormality.data.preprocessing.dicom import PREPARATION_ERRORS
from knee_abnormality.data.preprocessing.filtering import clean_series
from knee_abnormality.data.preprocessing.geometry import (
    StudyReference,
    estimate_reference,
    reflection_eligibility,
)
from knee_abnormality.data.preprocessing.laterality import (
    LEFT,
    RIGHT,
    KneeSide,
    resolve_knee_side,
    split_knee_geometry,
)
from knee_abnormality.data.preprocessing.preparation import (
    PreparedSeries,
    decode_series,
    prepare_series,
)
from knee_abnormality.data.preprocessing.sampling import (
    build_pool,
    select_triplets,
    study_budgets,
    usable_local_order,
    valid_runs,
)
from knee_abnormality.data.preprocessing.series import IndexedSeries, index_series
from knee_abnormality.data.preprocessing.spatial import (
    fallback_window_center,
    view_position,
)
from knee_abnormality.data.sources import SeriesCandidate, StudySeries
from knee_abnormality.data.splits import STUDY

UNKNOWN_GEOMETRY = (0.0, (0.0, 0.0, 0.0), False)
CERTAIN_SIDE_REASON = "explicit_coordinates_agree"


class KneeStudyDataset(Dataset):
    def __init__(
        self,
        studies: pl.DataFrame,
        series_index: dict[str, StudySeries],
        data: DataConfig,
        targets: tuple[str, ...],
        training_seed: int | None,
        image_limit_multiplier: int = 1,
        io_threads: int = 1,
    ) -> None:
        if image_limit_multiplier < 1:
            raise ValueError("Image limit multiplier must be positive")
        if io_threads < 1:
            raise ValueError("Loader IO threads must be positive")
        self.studies = studies
        self.series_index = series_index
        self.data = data
        self.targets = targets
        self.training_seed = training_seed
        self.image_limit_multiplier = image_limit_multiplier
        self.io_threads = io_threads
        self.epoch = Value("q", 0)
        self._slice_index: dict[Path, IndexedSeries] = {}
        self.slot_views = tuple(
            tuple(
                (index, view)
                for index, (owner, view) in enumerate(data.views)
                if owner == slot
            )
            for slot in range(len(data.series_types))
        )
        self._view_specs: dict[tuple[str, str, int], tuple] = {}
        self._study_references: dict[str, StudyReference] = {}
        self._study_sides: dict[str, KneeSide] = {}
        self._split_studies: set[str] = set()
        self._series_geometry: dict[
            str, dict[str, tuple[float, tuple[float, float, float], bool]]
        ] = {}
        self._excluded_series: dict[str, frozenset[str]] = {}
        self._cleaned_studies: dict[str, StudySeries] = {}
        missing = set(studies[STUDY].to_list()) - set(series_index)
        if missing:
            raise ValueError(
                f"Series metadata is missing {len(missing)} requested studies"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch.value = epoch

    def __len__(self) -> int:
        return self.studies.height

    def _indexed(self, candidate: SeriesCandidate) -> IndexedSeries:
        indexed = self._slice_index.get(candidate.directory)
        if indexed is None:
            indexed = index_series(
                candidate.directory,
                self.io_threads,
                list(candidate.listing) if candidate.listing is not None else None,
            )
            self._slice_index[candidate.directory] = indexed
        return indexed

    def _index_study_geometry(self, study: StudySeries) -> StudySeries:
        if study.uid in self._cleaned_studies:
            return self._cleaned_studies[study.uid]
        candidates = {}
        slots = {}
        for slot, values in enumerate(study.slots):
            for candidate in values:
                candidates.setdefault(
                    candidate.uid, (candidate, self.data.series_types[slot].plane)
                )
                slots.setdefault(candidate.uid, slot)
        indexed = {}
        for uid, (candidate, plane) in sorted(candidates.items()):
            try:
                indexed[uid] = (self._indexed(candidate), plane)
            except PREPARATION_ERRORS:
                continue
        excluded, assigned = clean_series(
            {uid: value for uid, (value, _) in indexed.items()},
            slots,
            tuple(
                (kind.plane, kind.fluid_sensitive, kind.fat_suppression)
                for kind in self.data.series_types
            ),
        )
        regrouped = [[] for _ in study.slots]
        for slot, values in enumerate(study.slots):
            for candidate in values:
                regrouped[assigned.get(candidate.uid, slot)].append(candidate)
        study = StudySeries(study.uid, tuple(tuple(values) for values in regrouped))
        eligible = tuple(
            (uid, plane, value.geometry)
            for uid, (value, plane) in indexed.items()
            if uid not in excluded
            and usable_local_order(value)
            and all(record.relative_position_mm is not None for record in value.records)
            and valid_runs(value.records, set())
        )
        eligible_uids = frozenset(uid for uid, _, _ in eligible)
        self._excluded_series[study.uid] = excluded
        sides = tuple((uid, indexed[uid][0].side) for uid in sorted(eligible_uids))
        groups, bilateral, strays = split_knee_geometry(sides)
        fluid_uids = frozenset(
            candidate.uid
            for slot, values in enumerate(study.slots)
            if self.data.series_types[slot].plane == "Sagittal"
            and self.data.series_types[slot].fluid_sensitive == 1
            for candidate in values
        )
        reference = estimate_reference(
            tuple(item for item in eligible if item[0] not in strays),
            not bilateral,
            fluid_uids,
        )
        self._study_references[study.uid] = reference
        side = resolve_knee_side(tuple(item for item in sides if item[0] not in strays))
        self._study_sides[study.uid] = side
        if bilateral:
            self._split_studies.add(study.uid)
        knee_references = (
            {
                group: estimate_reference(
                    tuple(item for item in eligible if groups.get(item[0]) == group),
                    True,
                    fluid_uids,
                )
                for group in (-1, 1)
            }
            if bilateral
            else {}
        )
        series_geometry = {}
        for uid in indexed:
            if uid not in eligible_uids:
                series_geometry[uid] = UNKNOWN_GEOMETRY
                continue
            group = groups.get(uid, 0)
            own_reference = knee_references.get(group) if bilateral else reference
            if uid in strays or (bilateral and group == 0):
                series_geometry[uid] = UNKNOWN_GEOMETRY
                continue
            own_side = {1: LEFT, -1: RIGHT}.get(group, 0) if bilateral else side.value
            sign = {LEFT: -1.0, RIGHT: 1.0}.get(own_side, 0.0)
            fitted = (
                own_reference.point[0]
                if own_reference is not None and own_reference.point is not None
                else None
            )
            center = fitted if fitted is not None else side.center_x_mm
            valid = bool(
                center is not None
                and own_reference is not None
                and own_reference.point is not None
                and own_reference.method in {"fit", "robust"}
            )
            center_y, center_z = own_reference.point[1:] if valid else (0.0, 0.0)
            series_geometry[uid] = (
                sign if valid else 0.0,
                (center or 0.0, center_y, center_z),
                valid,
            )
        self._series_geometry[study.uid] = series_geometry
        self._cleaned_studies[study.uid] = study
        return study

    def _decode_series(
        self, records, spacing_mm, image_size, offset_y, offset_x, margin
    ):
        return decode_series(
            records, spacing_mm, image_size, self.io_threads, offset_y, offset_x, margin
        )

    def _prepare_candidate(
        self, selected, cache, spacing, offset_y, offset_x, margin
    ) -> PreparedSeries:
        return prepare_series(
            selected,
            spacing,
            self.data.image_size,
            self.data.lower_quantile,
            self.data.upper_quantile,
            decoder=partial(
                self._decode_series, offset_y=offset_y, offset_x=offset_x, margin=margin
            ),
            cache=cache,
        )

    def _try_prepare_candidate(
        self,
        candidate,
        selected,
        cache,
        failures,
        physical,
        kind,
        view_index,
        view,
        study_uid,
    ):
        try:
            plane = kind.plane
            indexed = self._slice_index[candidate.directory]
            geometry = indexed.geometry
            sign, (center, center_y, center_z), valid = self._series_geometry[
                study_uid
            ].get(candidate.uid, UNKNOWN_GEOMETRY)
            detail = view.field_mm != self.data.reference_field_mm
            key = (study_uid, candidate.uid, view_index)
            if detail and key not in self._view_specs:
                self._view_specs[key] = view_position(
                    indexed.records,
                    plane,
                    view,
                    sign,
                    (center, center_y, center_z),
                    valid and geometry.native_poses_valid and geometry.reason is None,
                    self.data.reference_spacing_mm,
                    self.data.image_size,
                )
            offset_y, offset_x, window_center, rule_distance = (
                self._view_specs[key] if detail else (0, 0, (0.0, 0.0), 0.0)
            )
            value = self._prepare_candidate(
                selected,
                cache,
                view.field_mm / self.data.image_size,
                offset_y,
                offset_x,
                72,
            )
            if detail and window_center is None:
                window_center, rule_distance = fallback_window_center(
                    value, view, self.data.image_size
                )
            failures.extend(value.failures)
            eligible, reason = reflection_eligibility(geometry, plane)
            if not bool(value.geometry_valid.all()):
                eligible, reason = False, "invalid_selected_pose"
            return replace(
                value,
                uid=candidate.uid,
                geometry_valid=value.geometry_valid
                & geometry.native_poses_valid
                & (geometry.reason is None)
                & valid,
                sampling_path="physical" if physical else "local",
                reflection_eligible=eligible,
                reflection_reason=reason,
                knee_side_sign=sign,
                knee_center_x_mm=center,
                knee_center_y_mm=center_y,
                knee_center_z_mm=center_z,
                view_index=view_index,
                window_center_mm=window_center,
                window_rule_distance_mm=rule_distance,
            )
        except PREPARATION_ERRORS as error:
            failures.append(f"{candidate.uid}: {type(error).__name__}: {error}")
            return None

    def _slot_sources(self, study, slot, failures):
        excluded = self._excluded_series[study.uid]
        candidates = {c.uid: c for c in study.slots[slot] if c.uid not in excluded}
        sources = []
        for uid, candidate in sorted(candidates.items()):
            try:
                sources.append((uid, self._indexed(candidate)))
            except PREPARATION_ERRORS as error:
                failures.append(f"{uid}: {type(error).__name__}: {error}")
        return candidates, sources

    def _study_budgets(self, slot_sources):
        multiplier = self.image_limit_multiplier
        base = tuple(view.slices * multiplier for _, view in self.data.views)
        return study_budgets(
            base,
            tuple(len(slot_sources[slot][1]) for slot, _ in self.data.views),
            self.data.study_slice_total * multiplier,
        )

    def _prepare_slot(self, study, slot, candidates, sources, budgets, failures):
        excluded = self._excluded_series[study.uid]
        kind = self.data.series_types[slot]
        kind_key = (kind.plane, kind.fluid_sensitive, kind.fat_suppression)
        views = self.slot_views[slot]
        radius = self.data.reference_field_mm / 2
        failed_paths, failed_sources = set(), set()
        caches = [{} for _ in views]
        attempts = 0
        while True:
            pool = build_pool(
                tuple(sources),
                kind.plane,
                self._study_references[study.uid],
                self.data.reference_spacing_mm,
                radius,
                failed_paths,
                failed_sources,
            )
            values, selections, new_paths, new_sources = [], [], set(), set()
            for order, (view_index, view) in enumerate(views):
                selection = select_triplets(
                    pool,
                    budgets[view_index],
                    self.data.sampling_sigma_fraction,
                    radius,
                    study.uid,
                    (*kind_key, order) if order else kind_key,
                    self.training_seed,
                    self.epoch.value,
                )
                selections.append(selection)
                for uid in sorted({t.uid for t in selection.triplets}):
                    selected = tuple(
                        r for t in selection.triplets if t.uid == uid for r in t.records
                    )
                    value = self._try_prepare_candidate(
                        candidates[uid],
                        selected,
                        caches[order],
                        failures,
                        pool.physical,
                        kind,
                        view_index,
                        view,
                        study.uid,
                    )
                    if value is None:
                        new_sources.add(uid)
                    elif value.failed_paths:
                        new_paths.update(value.failed_paths)
                    else:
                        values.append(value)
            attempts += 1
            if not new_paths and not new_sources:
                metadata = tuple(
                    {
                        **selection.metadata,
                        "attempts": attempts,
                        "failed_paths": sorted(map(str, failed_paths)),
                        "failed_sources": sorted(failed_sources),
                        "excluded_sources": sorted(
                            c.uid for c in study.slots[slot] if c.uid in excluded
                        ),
                        "decoded_unique_paths": len(cache),
                    }
                    for selection, cache in zip(selections, caches, strict=True)
                )
                return tuple(values), metadata
            if not ((new_paths - failed_paths) or (new_sources - failed_sources)):
                raise RuntimeError("Preparation retry made no progress")
            failed_paths.update(new_paths)
            failed_sources.update(new_sources)

    def __getitem__(self, index: int) -> dict:
        row = self.studies.row(index, named=True)
        study = self._index_study_geometry(self.series_index[row[STUDY]])
        prepared = []
        failures = []
        sampling = []
        slot_sources = [
            self._slot_sources(study, slot, failures)
            for slot in range(len(study.slots))
        ]
        budgets = self._study_budgets(slot_sources)
        for slot in range(len(study.slots)):
            candidates, sources = slot_sources[slot]
            values, metadata = self._prepare_slot(
                study, slot, candidates, sources, budgets, failures
            )
            prepared.append(values)
            sampling.extend(metadata)
        if not any(prepared):
            logging.getLogger(__name__).error(
                "Study %s has no decodable supported series after %d conditions: %s",
                study.uid,
                len(failures),
                failures,
            )
            raise RuntimeError(f"Study {study.uid} has no decodable supported series")
        if failures:
            logging.getLogger(__name__).warning(
                "Study %s encountered %d preparation conditions",
                study.uid,
                len(failures),
            )
        target = torch.tensor(
            [
                float(row.get(f"target_{name}"))
                if row.get(f"target_{name}") is not None
                else 0.0
                for name in self.targets
            ],
            dtype=torch.float32,
        )
        observed = torch.tensor(
            [bool(row.get(f"observed_{name}", False)) for name in self.targets],
            dtype=torch.bool,
        )
        return {
            "study_uid": study.uid,
            "image_size": self.data.image_size,
            "is_gold": bool(row.get("is_gold", False)),
            "series": tuple(prepared),
            "study_reference": self._study_references.get(study.uid),
            "knee_side": self._study_sides[study.uid],
            "knee_side_split": study.uid in self._split_studies,
            "target": target,
            "target_observed": observed,
            "failures": tuple(failures),
            "sampling": tuple(sampling),
            "reflection_reasons": tuple(
                sorted(
                    {
                        series.reflection_reason
                        for acquisitions in prepared
                        for series in acquisitions
                        if series.reflection_reason != "eligible"
                    }
                )
            ),
        }


class Metadata:
    __slots__ = ("values",)

    def __init__(self, values) -> None:
        self.values = list(values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)


def collate_studies(items: list[dict]) -> dict:
    batch_size = len(items)
    type_count = len(items[0]["series"])
    image_size = items[0]["image_size"]
    entries = []
    presence = torch.zeros((batch_size, type_count), dtype=torch.bool)
    for study_index, item in enumerate(items):
        for type_index, acquisitions in enumerate(item["series"]):
            presence[study_index, type_index] = bool(acquisitions)
            for series in acquisitions:
                entries.append((study_index, type_index, series))
    max_slices = max(series.triplet_count for _, _, series in entries)
    positions = torch.zeros((len(entries), max_slices), dtype=torch.float32)
    slice_mask = torch.zeros((len(entries), max_slices), dtype=torch.bool)
    valid_rects = torch.zeros((len(entries), max_slices, SLICES_PER_INPUT, 4))
    patient_maps = torch.zeros((len(entries), max_slices, SLICES_PER_INPUT, 3, 3))
    geometry_valid = torch.zeros((len(entries), max_slices), dtype=torch.bool)
    view_index = torch.zeros(len(entries), dtype=torch.long)
    window_center = torch.zeros((len(entries), 2))
    window_rule_distance = torch.zeros(len(entries))
    knee_side_sign = torch.zeros(len(entries), dtype=torch.float32)
    knee_center_x_mm = torch.zeros(len(entries), dtype=torch.float32)
    knee_center_y_mm = torch.zeros(len(entries), dtype=torch.float32)
    knee_center_z_mm = torch.zeros(len(entries), dtype=torch.float32)
    reflection_eligible = torch.ones(batch_size, dtype=torch.bool)
    for index, item in enumerate(items):
        supplied = [
            series for acquisitions in item["series"] for series in acquisitions
        ]
        reflection_eligible[index] = bool(supplied) and all(
            series.reflection_eligible for series in supplied
        )
    study_indices = torch.empty(len(entries), dtype=torch.int64)
    type_indices = torch.empty(len(entries), dtype=torch.int64)
    frames = []
    buffers = []
    buffer_indices = {}
    series_frames = []
    image_positions = []
    image_sources = []
    for series_index, (study_index, type_index, series) in enumerate(entries):
        start = len(frames)
        for frame in series.frames:
            buffer = buffer_indices.setdefault(id(frame.pixels), len(buffers))
            if buffer == len(buffers):
                buffers.append(frame.pixels)
            frames.append(
                (frame.plan_key is not None, frame.signed, buffer, frame.offset)
                + (*frame.shape, *frame.columns, frame.slope, frame.intercept)
                + (frame.rect, frame.plan_key)
            )
        lower, upper, fraction = series.quantile_ranks
        series_frames.append(
            (
                start,
                len(series.frames),
                (lower.tolist(), upper.tolist(), fraction.tolist()),
                series.canvas_margin,
            )
        )
        count = series.triplet_count
        view_index[series_index] = series.view_index
        window_center[series_index] = torch.tensor(series.window_center_mm)
        window_rule_distance[series_index] = series.window_rule_distance_mm
        first = series_index * max_slices * SLICES_PER_INPUT
        image_positions.append(
            torch.arange(first, first + count * SLICES_PER_INPUT, dtype=torch.int64)
        )
        image_sources.append(torch.from_numpy(series.gather + start))
        positions[series_index, :count] = series.positions
        valid_rects[series_index, :count] = series.valid_rects
        patient_maps[series_index, :count] = series.patient_maps
        geometry_valid[series_index, :count] = series.geometry_valid
        knee_side_sign[series_index] = series.knee_side_sign
        knee_center_x_mm[series_index] = series.knee_center_x_mm
        knee_center_y_mm[series_index] = series.knee_center_y_mm
        knee_center_z_mm[series_index] = series.knee_center_z_mm
        slice_mask[series_index, :count] = True
        study_indices[series_index] = study_index
        type_indices[series_index] = type_index
    return {
        "study_uid": [item["study_uid"] for item in items],
        "is_gold": torch.tensor([item["is_gold"] for item in items], dtype=torch.bool),
        "image_size": image_size,
        "canvas_size": image_size
        + 2 * max(series.canvas_margin for _, _, series in entries),
        "frames": frames,
        "frame_buffers": buffers,
        "series_frames": series_frames,
        "image_positions": torch.cat(image_positions),
        "image_sources": torch.cat(image_sources),
        "slice_positions": positions,
        "valid_rects": valid_rects,
        "patient_maps": patient_maps,
        "geometry_valid": geometry_valid,
        "view_index": view_index,
        "window_center_mm": window_center,
        "window_rule_distance_mm": window_rule_distance,
        "knee_side_sign": knee_side_sign,
        "knee_center_x_mm": knee_center_x_mm,
        "knee_center_y_mm": knee_center_y_mm,
        "knee_center_z_mm": knee_center_z_mm,
        "knee_side_certain": torch.tensor(
            [item["knee_side"].reason == CERTAIN_SIDE_REASON for item in items]
        ),
        "knee_side_coordinates": torch.tensor(
            [item["knee_side"].source == "coordinates" for item in items]
        ),
        "knee_side_left": torch.tensor(
            [item["knee_side"].value == LEFT for item in items]
        ),
        "knee_side_split": torch.tensor([item["knee_side_split"] for item in items]),
        "knee_side_reasons": [item["knee_side"].reason for item in items],
        "reflection_eligible": reflection_eligible,
        "reflection_reasons": [item["reflection_reasons"] for item in items],
        "selected_paths": Metadata(series.selected_paths for _, _, series in entries),
        "sampling": Metadata(item["sampling"] for item in items),
        "slice_mask": slice_mask,
        "series_presence": presence,
        "series_study_index": study_indices,
        "series_type_index": type_indices,
        "target": torch.stack([item["target"] for item in items]),
        "target_observed": torch.stack([item["target_observed"] for item in items]),
        "failures": [item["failures"] for item in items],
    }


def collate_prepared(items: list[dict]) -> dict:
    return prepare_images(collate_studies(items))


def collate_prepared_middle(items: list[dict]) -> dict:
    batch = collate_prepared(items)
    batch["images"] = batch["images"][:, :, 1:2].contiguous()
    return batch


def build_data_loader(
    dataset: Dataset,
    settings: TrainerConfig,
    batch_size: int,
    *,
    shuffle: bool = False,
    num_workers: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    generator: torch.Generator | None = None,
    prepare_on_loader: bool = True,
    middle_only: bool = False,
) -> DataLoader:
    workers = settings.num_workers if num_workers is None else num_workers
    persistent = (
        settings.persistent_workers
        if persistent_workers is None
        else persistent_workers
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": (
            (collate_prepared_middle if middle_only else collate_prepared)
            if prepare_on_loader
            else collate_studies
        ),
        "pin_memory": pin_memory,
        "persistent_workers": persistent and workers > 0,
        "generator": generator,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = settings.prefetch_factor
    return DataLoader(**kwargs)
