import logging
from dataclasses import dataclass, replace
from multiprocessing import Value
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from knee_abnormality.config import DataConfig, TrainerConfig
from knee_abnormality.preprocessing import (
    PREPARATION_ERRORS,
    PreparedSeries,
    SliceRecord,
    index_series,
    prepare_series,
    select_slices,
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


def allocate_acquisition_inputs(
    capacities: tuple[int, ...],
    limit: int,
    training: bool,
    generator: torch.Generator,
) -> tuple[int, ...]:
    total = sum(capacities)
    count = min(total, limit)
    if count == total:
        return capacities
    if training:
        cumulative = torch.tensor(capacities).cumsum(0)
        selected = torch.randperm(total, generator=generator)[:count]
        acquisitions = torch.bucketize(selected, cumulative, right=True)
        return tuple(torch.bincount(acquisitions, minlength=len(capacities)).tolist())
    allocated = [capacity * count // total for capacity in capacities]
    remainders = [capacity * count % total for capacity in capacities]
    order = sorted(
        range(len(capacities)), key=lambda index: (-remainders[index], index)
    )
    for index in order[: count - sum(allocated)]:
        allocated[index] += 1
    return tuple(allocated)


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
        self._slice_index: dict[Path, tuple[SliceRecord, ...]] = {}
        missing = set(studies[STUDY].to_list()) - set(series_index)
        if missing:
            raise ValueError(
                f"Series metadata is missing {len(missing)} requested studies"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch.value = epoch

    def __len__(self) -> int:
        return self.studies.height

    def _generator(self, index: int) -> torch.Generator:
        value = (self.seed + 1_000_003 * self.epoch.value + 97_409 * index) % (
            2**63 - 1
        )
        return torch.Generator().manual_seed(value)

    def _records(self, candidate: SeriesCandidate) -> tuple[SliceRecord, ...]:
        records = self._slice_index.get(candidate.directory)
        if records is None:
            records = index_series(candidate.directory, self.data.position_reference)
            self._slice_index[candidate.directory] = records
        return records

    def _prepare_candidate(
        self, candidate: SeriesCandidate, limit: int, generator: torch.Generator
    ) -> PreparedSeries:
        records = self._records(candidate)
        selected = select_slices(
            records,
            limit,
            self.training,
            generator,
            self.data.slices_per_input,
            self.data.slice_sampling,
            self.data.center_sampling_weight,
        )
        return prepare_series(
            selected,
            self.data.spacing_mm,
            self.data.image_size,
            self.data.lower_quantile,
            self.data.upper_quantile,
            self.data.position_encoding,
            self.data.slices_per_input,
        )

    def _try_prepare_candidate(
        self,
        candidate: SeriesCandidate,
        limit: int,
        generator: torch.Generator,
        failures: list[str],
    ) -> PreparedSeries | None:
        try:
            value = self._prepare_candidate(candidate, limit, generator)
            failures.extend(value.failures)
            if value.constant:
                failures.append(f"{candidate.uid}: constant prepared series")
                return None
            return replace(value, uid=candidate.uid)
        except PREPARATION_ERRORS as error:
            failures.append(f"{candidate.uid}: {type(error).__name__}: {error}")
            return None

    def _prepare_slot(
        self,
        candidates: tuple[SeriesCandidate, ...],
        limit: int,
        generator: torch.Generator,
        failures: list[str],
    ) -> tuple[PreparedSeries, ...]:
        if self.data.acquisition_sampling == "shortest":
            for candidate in candidates:
                value = self._try_prepare_candidate(
                    candidate, limit, generator, failures
                )
                if value is not None:
                    return (value,)
            return ()
        available = []
        group_size = self.data.slices_per_input
        for candidate in sorted(candidates, key=lambda candidate: candidate.uid):
            try:
                capacity = len(self._records(candidate)) // group_size
                if capacity:
                    available.append((candidate, capacity))
                else:
                    failures.append(f"{candidate.uid}: no complete input groups")
            except PREPARATION_ERRORS as error:
                failures.append(f"{candidate.uid}: {type(error).__name__}: {error}")
        cache = {}
        while available:
            allocations = allocate_acquisition_inputs(
                tuple(capacity for _, capacity in available),
                limit // group_size,
                self.training,
                generator,
            )
            values = []
            failed = set()
            for (candidate, _), count in zip(available, allocations, strict=True):
                if not count:
                    continue
                key = (candidate.uid, count)
                if key not in cache:
                    cache[key] = self._try_prepare_candidate(
                        candidate, count * group_size, generator, failures
                    )
                value = cache[key]
                if value is None:
                    failed.add(candidate.uid)
                else:
                    values.append(value)
            if not failed:
                return tuple(values)
            available = [
                (candidate, capacity)
                for candidate, capacity in available
                if candidate.uid not in failed
            ]
        return ()

    def __getitem__(self, index: int) -> dict:
        row = self.studies.row(index, named=True)
        study = self.series_index[row[STUDY]]
        generator = self._generator(index)
        prepared = []
        failures = []
        for slot, candidates in enumerate(study.slots):
            prepared.append(
                self._prepare_slot(
                    candidates,
                    self.data.series_types[slot].max_images
                    * self.image_limit_multiplier,
                    generator,
                    failures,
                )
            )
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
            "target": target,
            "target_observed": observed,
            "failures": tuple(failures),
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
    study_indices = torch.empty(len(entries), dtype=torch.int64)
    type_indices = torch.empty(len(entries), dtype=torch.int64)
    for series_index, (study_index, type_index, series) in enumerate(entries):
        count = series.images.shape[0]
        values = series.images if series.images.ndim == 4 else series.images[:, None]
        if values.shape[1] != channels:
            raise ValueError("A batch cannot mix single-slice and triplet inputs")
        images[series_index, :count] = values
        positions[series_index, :count] = series.positions
        slice_mask[series_index, :count] = True
        study_indices[series_index] = study_index
        type_indices[series_index] = type_index
    return {
        "study_uid": [item["study_uid"] for item in items],
        "is_gold": torch.tensor([item["is_gold"] for item in items], dtype=torch.bool),
        "images": images,
        "slice_positions": positions,
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
