import logging
from dataclasses import dataclass, replace
from multiprocessing import Value
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from knee_abnormality.config import DataConfig, TrainerConfig
from knee_abnormality.preprocessing.dicom import PREPARATION_ERRORS
from knee_abnormality.preprocessing.geometry import (
    CenterProjection,
    StudyReference,
    estimate_reference,
    project_reference,
    reflection_eligibility,
)
from knee_abnormality.preprocessing.preparation import (
    PreparedSeries,
    decode_series,
    prepare_series,
)
from knee_abnormality.preprocessing.sampling import build_pool, select_triplets
from knee_abnormality.preprocessing.series import (
    IndexedSeries,
    SliceRecord,
    index_series,
)

STUDY = "StudyInstanceUID"
SERIES = "SeriesInstanceUID"
KNOWN_BAD_SERIES = {
    "1.2.826.0.1.3680043.8.498.31160785238781353848727311763596115703",
    "1.2.826.0.1.3680043.8.498.39396636671446532796538574315802928348",
}


@dataclass(frozen=True)
class SeriesCandidate:
    uid: str
    directory: Path
    image_count: int


@dataclass(frozen=True)
class StudySeries:
    uid: str
    slots: tuple[tuple[SeriesCandidate, ...], ...]


def build_series_index(
    data_root: Path, data: DataConfig, split: str
) -> dict[str, StudySeries]:
    raw_dir = resolve_raw_dir(data_root, split)
    metadata_path = raw_dir / f"{split}_series.csv"
    image_root = raw_dir / f"{split}_series"
    frame = pl.read_csv(metadata_path)
    required = {STUDY, SERIES, "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"}
    if set(frame.columns) != required:
        raise ValueError(f"Unexpected series metadata schema in {metadata_path}")
    type_to_slot = {
        (item.plane, item.fluid_sensitive, item.fat_suppression): slot
        for slot, item in enumerate(data.series_types)
    }
    observed = set(
        frame.select(
            "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"
        ).iter_rows()
    )
    unsupported = observed - set(type_to_slot)
    if unsupported:
        unsupported_rows = frame.filter(
            pl.struct(
                "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"
            ).map_elements(
                lambda value: (
                    (
                        value["Anatomical_Plane"],
                        value["Fluid_Sensitive"],
                        value["Fat_Suppression"],
                    )
                    in unsupported
                ),
                return_dtype=pl.Boolean,
            )
        )
        logging.getLogger(__name__).warning(
            "Ignoring %d unsupported series across %d studies: combinations=%s studies=%s series=%s",
            unsupported_rows.height,
            unsupported_rows[STUDY].n_unique(),
            sorted(unsupported),
            sorted(unsupported_rows[STUDY].unique().to_list()),
            sorted(unsupported_rows[SERIES].to_list()),
        )
        frame = frame.filter(~pl.col(SERIES).is_in(unsupported_rows[SERIES]))

    counts_path = data_root / "metadata" / f"{split}_series_image_counts.parquet"
    if counts_path.exists():
        counts = pl.read_parquet(counts_path).select(SERIES, "n_images")
        frame = frame.join(counts, on=SERIES, how="left", validate="1:1")
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.UInt32).alias("n_images"))

    grouped: dict[str, list[list[SeriesCandidate]]] = {}
    for row in frame.iter_rows(named=True):
        study_uid = row[STUDY]
        series_uid = row[SERIES]
        directory = image_root / study_uid / series_uid
        image_count = row["n_images"]
        if image_count is None:
            image_count = sum(1 for _ in directory.glob("*.dcm"))
        if series_uid in KNOWN_BAD_SERIES:
            image_count -= 1
        slot = type_to_slot[
            (row["Anatomical_Plane"], row["Fluid_Sensitive"], row["Fat_Suppression"])
        ]
        slots = grouped.setdefault(study_uid, [[] for _ in data.series_types])
        slots[slot].append(SeriesCandidate(series_uid, directory, int(image_count)))

    result = {}
    for study_uid, slots in grouped.items():
        ordered = tuple(
            tuple(sorted(values, key=lambda item: (item.image_count, item.uid)))
            for values in slots
        )
        result[study_uid] = StudySeries(study_uid, ordered)
    return result


def resolve_raw_dir(data_root: Path, split: str) -> Path:
    data_root = data_root.expanduser().resolve()
    required = (f"{split}_series.csv", f"{split}_series")
    candidates = (data_root / "raw", data_root)
    matches = [
        candidate
        for candidate in candidates
        if (candidate / required[0]).is_file() and (candidate / required[1]).is_dir()
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {required[0]} and {required[1]}/ under {data_root} or {data_root / 'raw'}"
        )
    return matches[0]


class KneeStudyDataset(Dataset):
    def __init__(
        self,
        studies: pl.DataFrame,
        series_index: dict[str, StudySeries],
        data: DataConfig,
        targets: tuple[str, ...],
        training: bool,
        seed: int,
        image_limit_multiplier: int = 1,
    ) -> None:
        if image_limit_multiplier < 1:
            raise ValueError("Image limit multiplier must be positive")
        self.studies = studies
        self.series_index = series_index
        self.data = data
        self.targets = targets
        self.training = training
        self.seed = seed
        self.image_limit_multiplier = image_limit_multiplier
        self.epoch = Value("q", 0)
        self._slice_index: dict[Path, IndexedSeries] = {}
        self._study_references: dict[str, StudyReference] = {}
        self._center_projections: dict[Path, CenterProjection] = {}
        missing = set(studies[STUDY].to_list()) - set(series_index)
        if missing:
            raise ValueError(
                f"Series metadata is missing {len(missing)} requested studies"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch.value = epoch

    def __len__(self) -> int:
        return self.studies.height

    def _records(self, candidate: SeriesCandidate) -> tuple[SliceRecord, ...]:
        records = self._slice_index.get(candidate.directory)
        if records is None:
            records = index_series(candidate.directory, self.data.position_reference)
            self._slice_index[candidate.directory] = records
        return records.records

    def _index_study_geometry(self, study: StudySeries) -> None:
        if study.uid in self._study_references:
            return
        candidates = {}
        for slot, values in enumerate(study.slots):
            for candidate in values:
                candidates.setdefault(
                    candidate.uid, (candidate, self.data.series_types[slot].plane)
                )
        eligible = []
        indexed = []
        for uid, (candidate, plane) in sorted(candidates.items()):
            try:
                records = self._records(candidate)
            except PREPARATION_ERRORS:
                continue
            geometry = self._slice_index[candidate.directory].geometry
            indexed.append((candidate.directory, geometry, len(records)))
            if len(records) >= self.data.slices_per_input:
                eligible.append((uid, plane, geometry))
        reference = estimate_reference(tuple(eligible))
        self._study_references[study.uid] = reference
        for directory, geometry, count in indexed:
            self._center_projections[directory] = (
                project_reference(reference, geometry)
                if count >= self.data.slices_per_input
                else CenterProjection(None, "no_complete_input_groups")
            )

    def _decode_series(self, records, spacing_mm, image_size):
        return decode_series(records, spacing_mm, image_size)

    def _prepare_candidate(self, candidate, selected, cache) -> PreparedSeries:
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
            value = self._prepare_candidate(candidate, selected, cache)
            failures.extend(value.failures)
            if value.constant:
                failures.append(f"{candidate.uid}: constant prepared series")
                return None
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

    def _prepare_slot(self, study, slot, failures):
        candidates = {c.uid: c for c in study.slots[slot]}
        sources = []
        for uid, candidate in sorted(candidates.items()):
            try:
                self._records(candidate)
                sources.append((uid, self._slice_index[candidate.directory]))
            except PREPARATION_ERRORS as error:
                failures.append(f"{uid}: {type(error).__name__}: {error}")
        kind = self.data.series_types[slot]
        failed_paths, failed_sources, cache = set(), set(), {}
        attempts = 0
        while True:
            pool = build_pool(
                tuple(sources),
                kind.plane,
                self._study_references[study.uid],
                failed_paths,
                failed_sources,
            )
            selection = select_triplets(
                pool,
                kind.max_images * self.image_limit_multiplier // 3,
                self.data.sampling_sigma_mm,
                self.data.image_size * self.data.spacing_mm / 2,
                self.seed,
                study.uid,
                (kind.plane, kind.fluid_sensitive, kind.fat_suppression),
                self.training,
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
                    "decoded_unique_paths": len(cache),
                }
                return tuple(values), metadata
            if not ((new_paths - failed_paths) or (new_sources - failed_sources)):
                raise RuntimeError("Preparation retry made no progress")
            failed_paths.update(new_paths)
            failed_sources.update(new_sources)

    def __getitem__(self, index: int) -> dict:
        row = self.studies.row(index, named=True)
        study = self.series_index[row[STUDY]]
        self._index_study_geometry(study)
        prepared = []
        failures = []
        sampling = []
        for slot in range(len(study.slots)):
            values, metadata = self._prepare_slot(study, slot, failures)
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
            "is_gold": bool(row.get("is_gold", False)),
            "series": tuple(prepared),
            "study_reference": self._study_references.get(study.uid),
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
    entries = []
    presence = torch.zeros((batch_size, type_count), dtype=torch.bool)
    for study_index, item in enumerate(items):
        for type_index, acquisitions in enumerate(item["series"]):
            presence[study_index, type_index] = bool(acquisitions)
            for series in acquisitions:
                entries.append((study_index, type_index, series))
    max_slices = max(series.images.shape[0] for _, _, series in entries)
    image_size = entries[0][2].images.shape[-1]
    channels = entries[0][2].images.shape[1] if entries[0][2].images.ndim == 4 else 1
    images = torch.zeros(
        (len(entries), max_slices, channels, image_size, image_size), dtype=torch.uint8
    )
    positions = torch.zeros((len(entries), max_slices), dtype=torch.float32)
    slice_mask = torch.zeros((len(entries), max_slices), dtype=torch.bool)
    patient_maps = torch.zeros((len(entries), max_slices, channels, 3, 3))
    valid_rects = torch.zeros((len(entries), max_slices, channels, 4))
    geometry_valid = torch.zeros((len(entries), max_slices), dtype=torch.bool)
    study_centers = torch.zeros((batch_size, 3))
    center_valid = torch.zeros(batch_size, dtype=torch.bool)
    reflection_eligible = torch.ones(batch_size, dtype=torch.bool)
    for index, item in enumerate(items):
        supplied = [
            series for acquisitions in item["series"] for series in acquisitions
        ]
        reflection_eligible[index] = bool(supplied) and all(
            series.reflection_eligible for series in supplied
        )
        reference = item["study_reference"]
        if reference is not None and reference.point is not None:
            study_centers[index] = torch.tensor(reference.point)
            center_valid[index] = True
    study_indices = torch.empty(len(entries), dtype=torch.int64)
    type_indices = torch.empty(len(entries), dtype=torch.int64)
    for series_index, (study_index, type_index, series) in enumerate(entries):
        count = series.images.shape[0]
        values = series.images if series.images.ndim == 4 else series.images[:, None]
        if values.shape[1] != channels:
            raise ValueError("A batch cannot mix single-slice and triplet inputs")
        images[series_index, :count] = values
        positions[series_index, :count] = series.positions
        patient_maps[series_index, :count] = series.patient_maps
        valid_rects[series_index, :count] = series.valid_rects
        geometry_valid[series_index, :count] = series.geometry_valid
        slice_mask[series_index, :count] = True
        study_indices[series_index] = study_index
        type_indices[series_index] = type_index
    return {
        "study_uid": [item["study_uid"] for item in items],
        "is_gold": torch.tensor([item["is_gold"] for item in items], dtype=torch.bool),
        "images": images,
        "slice_positions": positions,
        "patient_maps": patient_maps,
        "valid_rects": valid_rects,
        "geometry_valid": geometry_valid,
        "study_centers": study_centers,
        "center_valid": center_valid,
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


def build_data_loader(
    dataset: Dataset,
    settings: TrainerConfig,
    *,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_workers: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    generator: torch.Generator | None = None,
) -> DataLoader:
    workers = settings.num_workers if num_workers is None else num_workers
    persistent = (
        settings.persistent_workers
        if persistent_workers is None
        else persistent_workers
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": settings.batch_size if batch_size is None else batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate_studies,
        "pin_memory": pin_memory,
        "persistent_workers": persistent and workers > 0,
        "generator": generator,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = settings.prefetch_factor
    return DataLoader(**kwargs)
