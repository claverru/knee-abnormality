import json
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.config import load_config
from knee_abnormality.data.preprocessing.dicom import PREPARATION_ERRORS, read_frame
from knee_abnormality.data.preprocessing.filtering import (
    _layout_key,
    exact_copies,
    non_knee_coils,
    relabelled_slots,
    superseded_series,
    volume_center,
)
from knee_abnormality.data.preprocessing.series import index_series
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.provenance import copy_sources


class Settings(CLISettings):
    model_config = describe(
        "Census of the series the superseded and exact-copy rules delete: how far each "
        "deleted series lies from the one kept, and whether every frame matches"
    )
    config: Path = Field(
        default=Path("configs/training.yaml"), description="Experiment configuration"
    )
    output_dir: Path = Field(description="Fresh output directory for the census")
    workers: int = Field(default=6, gt=0, description="Worker processes")
    studies: int | None = Field(default=None, description="Limit to the first studies")


def _layout(indexed):
    record = indexed.records[len(indexed.records) // 2]
    return record.pose.layout if record.pose is not None else None


def _frames_equal(left, right) -> tuple[int, int]:
    equal = 0
    for a, b in zip(left.records, right.records, strict=True):
        try:
            equal += bool(
                np.array_equal(
                    read_frame(a.path, a.descriptor).pixels,
                    read_frame(b.path, b.descriptor).pixels,
                )
            )
        except PREPARATION_ERRORS:
            continue
    return equal, len(left.records)


def _pair(study: str, rule: str, slot: int, dropped: str, kept: str, series) -> dict:
    a, b = series[dropped], series[kept]
    centers = volume_center(a.geometry), volume_center(b.geometry)
    distance = in_plane = None
    if centers[0] is not None and centers[1] is not None:
        delta = centers[0] - centers[1]
        normal = np.asarray(b.geometry.normal, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        distance = float(np.linalg.norm(delta))
        in_plane = float(np.linalg.norm(delta - (delta @ normal) * normal))
    layouts = _layout(a), _layout(b)
    orientation = spacing = same_frame = None
    if layouts[0] is not None and layouts[1] is not None:
        orientation = float(
            np.abs(
                np.asarray(layouts[0].orientation) - np.asarray(layouts[1].orientation)
            ).max()
        )
        spacing = float(
            np.abs(
                np.asarray(layouts[0].spacing) - np.asarray(layouts[1].spacing)
            ).max()
        )
        same_frame = layouts[0].frame_uid == layouts[1].frame_uid
    equal, total = _frames_equal(a, b)
    return {
        "study": study,
        "rule": rule,
        "slot": slot,
        "dropped": dropped,
        "kept": kept,
        "frames": total,
        "equal_frames": equal,
        "center_distance_mm": distance,
        "in_plane_distance_mm": in_plane,
        "center_x_difference_mm": float(delta[0]) if distance is not None else None,
        "orientation_difference": orientation,
        "spacing_difference_mm": spacing,
        "same_frame_uid": same_frame,
        "dropped_series_number": a.summary.series_number,
        "kept_series_number": b.summary.series_number,
    }


def census_study(task) -> list[dict]:
    study, kinds = task
    slots = {}
    indexed = {}
    for slot, values in enumerate(study.slots):
        for candidate in values:
            if candidate.uid in slots:
                continue
            slots[candidate.uid] = slot
            try:
                indexed[candidate.uid] = index_series(
                    candidate.directory,
                    1,
                    list(candidate.listing) if candidate.listing is not None else None,
                )
            except PREPARATION_ERRORS:
                continue
    coils = non_knee_coils(indexed)
    kept = {uid: value for uid, value in indexed.items() if uid not in coils}
    rows = []
    superseded = superseded_series(kept, slots)
    for uid in sorted(superseded):
        key = (slots[uid], kept[uid].summary.signature, *_layout_key(kept[uid]))
        survivor = next(
            other
            for other, value in sorted(kept.items())
            if other not in superseded
            and value.records
            and other in slots
            and (slots[other], value.summary.signature, *_layout_key(value)) == key
        )
        rows.append(_pair(study.uid, "superseded", slots[uid], uid, survivor, kept))
    kept = {uid: value for uid, value in kept.items() if uid not in superseded}
    assigned = relabelled_slots(kept, slots, kinds)
    copies = exact_copies(kept, assigned)
    for uid in sorted(copies):
        key = (assigned[uid], *_layout_key(kept[uid]))
        record = kept[uid].records[len(kept[uid].records) // 2]
        middle = read_frame(record.path, record.descriptor).pixels
        for other, value in sorted(kept.items()):
            if (
                other in copies
                or not value.records
                or other not in assigned
                or (assigned[other], *_layout_key(value)) != key
            ):
                continue
            other_record = value.records[len(value.records) // 2]
            try:
                same = np.array_equal(
                    middle,
                    read_frame(other_record.path, other_record.descriptor).pixels,
                )
            except PREPARATION_ERRORS:
                continue
            if same:
                rows.append(
                    _pair(study.uid, "exact_copy", assigned[uid], uid, other, kept)
                )
                break
    return rows


def main() -> None:
    settings = Settings()
    config = load_config(settings.config)
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    index = build_series_index(config.paths.data_root, config.data, "train")
    kinds = tuple(
        (kind.plane, kind.fluid_sensitive, kind.fat_suppression)
        for kind in config.data.series_types
    )
    studies = [index[uid] for uid in sorted(index)][: settings.studies]
    with Pool(settings.workers) as pool:
        results = pool.map(
            census_study, [(study, kinds) for study in studies], chunksize=8
        )
    rows = [row for result in results for row in result]
    frame = pl.DataFrame(rows, infer_schema_length=None)
    frame.write_parquet(settings.output_dir / "pairs.parquet")
    summary = {"studies": len(studies)}
    for rule in ("superseded", "exact_copy"):
        part = frame.filter(pl.col("rule") == rule) if rows else frame
        summary[rule] = {
            "pairs": part.height,
            "studies": part["study"].n_unique() if part.height else 0,
            "all_frames_equal": int((part["equal_frames"] == part["frames"]).sum())
            if part.height
            else 0,
            "in_plane_over_5mm": int((part["in_plane_distance_mm"] > 5).sum())
            if part.height
            else 0,
            "in_plane_over_20mm": int((part["in_plane_distance_mm"] > 20).sum())
            if part.height
            else 0,
            "in_plane_over_100mm": int((part["in_plane_distance_mm"] > 100).sum())
            if part.height
            else 0,
            "orientation_over_0.01": int((part["orientation_difference"] > 0.01).sum())
            if part.height
            else 0,
            "spacing_differs": int((part["spacing_difference_mm"] > 1e-4).sum())
            if part.height
            else 0,
            "different_frame_uid": int((~part["same_frame_uid"]).sum())
            if part.height
            else 0,
        }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(settings.config),
        "sources": copy_sources([Path(__file__)], settings.output_dir / "sources"),
        "summary": summary,
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
