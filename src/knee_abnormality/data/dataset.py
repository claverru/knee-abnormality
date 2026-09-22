import logging
from dataclasses import replace
from multiprocessing import Value
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from knee_abnormality.config import SLICES_PER_INPUT, DataConfig, TrainerConfig
from knee_abnormality.data.labels import STUDY
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
)
from knee_abnormality.data.preprocessing.series import IndexedSeries, index_series
from knee_abnormality.data.sources import SeriesCandidate, StudySeries


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
        self._study_references: dict[str, StudyReference] = {}
        self._study_sides: dict[str, KneeSide] = {}
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
            if uid not in excluded and len(value.records) >= SLICES_PER_INPUT
        )
        self._excluded_series[study.uid] = excluded
        self._study_references[study.uid] = estimate_reference(eligible)
        self._study_sides[study.uid] = resolve_knee_side(
            tuple((uid, value.side) for uid, (value, _) in indexed.items())
        )
        self._cleaned_studies[study.uid] = study
        return study

    def _decode_series(self, records, spacing_mm, image_size):
        return decode_series(records, spacing_mm, image_size, self.io_threads)

    def _prepare_candidate(self, selected, cache) -> PreparedSeries:
        return prepare_series(
            selected,
            self.data.spacing_mm,
            self.data.image_size,
            self.data.lower_quantile,
            self.data.upper_quantile,
            decoder=self._decode_series,
            cache=cache,
        )

    def _try_prepare_candidate(
        self, candidate, selected, cache, failures, physical, plane
    ):
        try:
            value = self._prepare_candidate(selected, cache)
            failures.extend(value.failures)
            geometry = self._slice_index[candidate.directory].geometry
            eligible, reason = reflection_eligibility(geometry, plane)
            if not bool(value.geometry_valid.all()):
                eligible, reason = False, "invalid_selected_pose"
            return replace(
                value,
                uid=candidate.uid,
                geometry_valid=value.geometry_valid
                & geometry.native_poses_valid
                & (geometry.reason is None)
                & physical,
                sampling_path="physical" if physical else "local",
                reflection_eligible=eligible,
                reflection_reason=reason,
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
        base = tuple(
            kind.max_images * multiplier // 3 for kind in self.data.series_types
        )
        total = self.data.study_triplet_total
        return study_budgets(
            base,
            tuple(len(sources) for _, sources in slot_sources),
            total * multiplier if total is not None else None,
            self.data.repeated_series_share,
        )

    def _prepare_slot(self, study, slot, candidates, sources, budget, failures):
        excluded = self._excluded_series[study.uid]
        kind = self.data.series_types[slot]
        radius = self.data.image_size * self.data.spacing_mm / 2
        failed_paths, failed_sources, cache = set(), set(), {}
        attempts = 0
        while True:
            pool = build_pool(
                tuple(sources),
                kind.plane,
                self._study_references[study.uid],
                self.data.spacing_mm,
                radius,
                failed_paths,
                failed_sources,
            )
            selection = select_triplets(
                pool,
                budget,
                self.data.sampling_sigma_fraction,
                radius,
                study.uid,
                (kind.plane, kind.fluid_sensitive, kind.fat_suppression),
                self.training_seed,
                self.epoch.value,
            )
            values, new_paths, new_sources = [], set(), set()
            for uid in sorted({t.uid for t in selection.triplets}):
                selected = tuple(
                    r for t in selection.triplets if t.uid == uid for r in t.records
                )
                value = self._try_prepare_candidate(
                    candidates[uid],
                    selected,
                    cache,
                    failures,
                    pool.physical,
                    kind.plane,
                )
                if value is None:
                    new_sources.add(uid)
                elif value.failed_paths:
                    new_paths.update(value.failed_paths)
                else:
                    values.append(value)
            attempts += 1
            if not new_paths and not new_sources:
                metadata = {
                    **selection.metadata,
                    "attempts": attempts,
                    "failed_paths": sorted(map(str, failed_paths)),
                    "failed_sources": sorted(failed_sources),
                    "excluded_sources": sorted(
                        c.uid for c in study.slots[slot] if c.uid in excluded
                    ),
                    "decoded_unique_paths": len(cache),
                }
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
                study, slot, candidates, sources, budgets[slot], failures
            )
            prepared.append(values)
            sampling.append(metadata)
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
            )
        )
        count = series.triplet_count
        first = series_index * max_slices * SLICES_PER_INPUT
        image_positions.append(
            torch.arange(first, first + count * SLICES_PER_INPUT, dtype=torch.int64)
        )
        image_sources.append(torch.from_numpy(series.gather + start))
        positions[series_index, :count] = series.positions
        valid_rects[series_index, :count] = series.valid_rects
        patient_maps[series_index, :count] = series.patient_maps
        geometry_valid[series_index, :count] = series.geometry_valid
        slice_mask[series_index, :count] = True
        study_indices[series_index] = study_index
        type_indices[series_index] = type_index
    return {
        "study_uid": [item["study_uid"] for item in items],
        "is_gold": torch.tensor([item["is_gold"] for item in items], dtype=torch.bool),
        "image_size": image_size,
        "frames": frames,
        "frame_buffers": buffers,
        "series_frames": series_frames,
        "image_positions": torch.cat(image_positions),
        "image_sources": torch.cat(image_sources),
        "slice_positions": positions,
        "valid_rects": valid_rects,
        "patient_maps": patient_maps,
        "geometry_valid": geometry_valid,
        "knee_side_sign": torch.tensor(
            [
                {LEFT: -1.0, RIGHT: 1.0}.get(item["knee_side"].value, 0.0)
                if item["knee_side"].center_x_mm is not None
                else 0.0
                for item in items
            ],
            dtype=torch.float32,
        ),
        "knee_center_x_mm": torch.tensor(
            [item["knee_side"].center_x_mm or 0.0 for item in items],
            dtype=torch.float32,
        ),
        "knee_side_reasons": [item["knee_side"].reason for item in items],
        "reflection_eligible": reflection_eligible,
        "reflection_reasons": [item["reflection_reasons"] for item in items],
        "selected_paths": [series.selected_paths for _, _, series in entries],
        "sampling": [item["sampling"] for item in items],
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
        "collate_fn": collate_prepared if prepare_on_loader else collate_studies,
        "pin_memory": pin_memory,
        "persistent_workers": persistent and workers > 0,
        "generator": generator,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = settings.prefetch_factor
    return DataLoader(**kwargs)
