import logging
import os
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from knee_abnormality.config import SERIES_LISTING_THREADS, DataConfig, ExperimentConfig
from knee_abnormality.data.labels import STUDY, load_fold_labels
from knee_abnormality.data.preprocessing.exclusions import KNOWN_BAD_SERIES

SERIES = "SeriesInstanceUID"


def list_series(series_dir: Path) -> list[Path]:
    return sorted(
        Path(entry.path)
        for entry in os.scandir(series_dir)
        if entry.name.endswith(".dcm")
    )


@dataclass(frozen=True)
class SeriesCandidate:
    uid: str
    directory: Path
    image_count: int
    listing: tuple[Path, ...] | None = None


@dataclass(frozen=True)
class StudySeries:
    uid: str
    slots: tuple[tuple[SeriesCandidate, ...], ...]


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


def build_series_index(
    data_root: Path,
    data: DataConfig,
    split: str,
    study_ids: Collection[str] | None = None,
) -> dict[str, StudySeries]:
    raw_dir = resolve_raw_dir(data_root, split)
    metadata_path = raw_dir / f"{split}_series.csv"
    image_root = raw_dir / f"{split}_series"
    frame = pl.read_csv(metadata_path)
    required = {STUDY, SERIES, "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"}
    if set(frame.columns) != required:
        raise ValueError(f"Unexpected series metadata schema in {metadata_path}")
    if study_ids is not None:
        frame = frame.filter(pl.col(STUDY).is_in(list(study_ids)))
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

    rows = list(frame.iter_rows(named=True))
    directories = [image_root / row[STUDY] / row[SERIES] for row in rows]
    unlisted = [index for index, row in enumerate(rows) if row["n_images"] is None]
    listings: dict[int, tuple[Path, ...]] = {}
    if unlisted:
        with ThreadPoolExecutor(min(SERIES_LISTING_THREADS, len(unlisted))) as pool:
            listings = {
                index: tuple(listing)
                for index, listing in zip(
                    unlisted,
                    pool.map(list_series, (directories[index] for index in unlisted)),
                    strict=True,
                )
            }

    grouped: dict[str, list[list[SeriesCandidate]]] = {}
    for index, row in enumerate(rows):
        study_uid = row[STUDY]
        series_uid = row[SERIES]
        directory = directories[index]
        listing = listings.get(index)
        image_count = row["n_images"] if listing is None else len(listing)
        if series_uid in KNOWN_BAD_SERIES:
            image_count -= 1
        slot = type_to_slot[
            (row["Anatomical_Plane"], row["Fluid_Sensitive"], row["Fat_Suppression"])
        ]
        slots = grouped.setdefault(study_uid, [[] for _ in data.series_types])
        slots[slot].append(
            SeriesCandidate(series_uid, directory, int(image_count), listing)
        )

    result = {}
    for study_uid, slots in grouped.items():
        ordered = tuple(
            tuple(sorted(values, key=lambda item: (item.image_count, item.uid)))
            for values in slots
        )
        result[study_uid] = StudySeries(study_uid, ordered)
    return result


def prediction_studies(
    config: ExperimentConfig, split: str, fold: int
) -> tuple[tuple[str, ...], pl.DataFrame, str]:
    if split == "validation":
        labels = load_fold_labels(config, fold)
        studies = labels.validation
        targets = labels.targets
        metadata_split = "train"
    elif split == "test":
        raw_dir = resolve_raw_dir(config.paths.data_root, "test")
        template = pl.read_csv(raw_dir / "sample_submission.csv")
        targets = tuple(template.columns[1:])
        if template.columns[0] != STUDY:
            raise ValueError("Submission template must start with StudyInstanceUID")
        studies = template.select(STUDY)
        metadata_studies = (
            pl.read_csv(raw_dir / "test_series.csv").select(STUDY).unique()
        )
        if set(studies[STUDY]) != set(metadata_studies[STUDY]):
            raise ValueError(
                "Mounted test series membership differs from sample_submission.csv"
            )
        metadata_split = "test"
    else:
        raise ValueError(f"Unsupported prediction split: {split}")
    if (
        studies.is_empty()
        or studies[STUDY].null_count()
        or studies[STUDY].n_unique() != studies.height
    ):
        raise ValueError("Prediction studies must be nonempty with unique study IDs")
    return targets, studies, metadata_split


def select_studies(studies: pl.DataFrame, study_ids: list[str]) -> pl.DataFrame:
    if not study_ids or len(set(study_ids)) != len(study_ids):
        raise ValueError("Requested study IDs must be nonempty and unique")
    selected = studies.filter(pl.col(STUDY).is_in(study_ids))
    if selected.height != len(study_ids):
        raise ValueError("Requested study IDs include unknown studies")
    return selected
