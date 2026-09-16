import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import SLICES_PER_INPUT, load_config
from knee_abnormality.data.dataset import KneeStudyDataset, collate_studies
from knee_abnormality.data.labels import STUDY
from knee_abnormality.data.preprocessing.geometry import estimate_reference
from knee_abnormality.data.preprocessing.intensity import _linear_quantiles
from knee_abnormality.data.preprocessing.preparation import decode_series
from knee_abnormality.data.sources import build_series_index
from knee_abnormality.identity import preprocessing_identity
from knee_abnormality.provenance import copy_sources, sha256_file
from scripts.adhoc.analyze_physical_sampling import read_acquisitions
from scripts.adhoc.profile_threaded_loading import ThreadedStudyDataset


class InspectCLI(CLISettings):
    config: Path = Path("configs/training.yaml")
    headers: Path = Path("outputs/analysis/geometry-metadata-20260912/headers.parquet")
    inventory: Path = Path("data/metadata/train_series_image_counts.parquet")
    output_dir: Path
    accepted_studies: int = Field(default=4, ge=1, le=8)


def digest(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {k: digest(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [digest(v) for v in value]
    return value


def inspect(args: InspectCLI) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = load_config(args.config)
    acquisitions = read_acquisitions(args)
    accepted, rejected = [], []
    for uid, values in acquisitions.items():
        reference = estimate_reference(
            tuple(
                (a.uid, a.plane, a.geometry)
                for a in values
                if len(a.records) >= SLICES_PER_INPUT
            )
        )
        (accepted if reference.point is not None else rejected).append(uid)
    uids = sorted(rejected + accepted[: args.accepted_studies])
    studies = pl.DataFrame({STUDY: uids})
    index = build_series_index(config.paths.data_root, config.data, "train")
    source_files = [
        Path(__file__),
        Path("scripts/adhoc/profile_threaded_loading.py"),
        Path("scripts/adhoc/analyze_physical_sampling.py"),
        *Path("src/knee_abnormality").rglob("*.py"),
    ]
    copy_sources(source_files, args.output_dir / "source")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "settings": args.model_dump(mode="json"),
        "study_uids": uids,
        "rejected_center_uids": rejected,
        "preprocessing_identity": preprocessing_identity(config.model.backbone),
        "inputs": [
            {"path": str(p.resolve()), "sha256": sha256_file(p)}
            for p in [args.config, args.headers, args.inventory, *source_files]
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    rows, comparisons = [], []
    for training in (False, True):
        multiplier = 1 if training else config.data.validation_image_multiplier
        serial = KneeStudyDataset(
            studies, index, config.data, (), training, config.seed, multiplier
        )
        threaded = ThreadedStudyDataset(serial, 4)
        reordered = KneeStudyDataset(
            studies.reverse(), index, config.data, (), training, config.seed, multiplier
        )
        for i, uid in enumerate(uids):
            first, second, third = serial[i], threaded[i], reordered[len(uids) - 1 - i]
            first_batch = digest(collate_studies([first]))
            comparisons.append(
                {
                    "study_uid": uid,
                    "training": training,
                    "serial_threaded_equal": first_batch
                    == digest(collate_studies([second])),
                    "row_order_equal": first_batch == digest(collate_studies([third])),
                    "batch": first_batch,
                }
            )
            for slot, values in enumerate(first["series"]):
                for value in values:
                    candidate = next(
                        c for c in index[uid].slots[slot] if c.uid == value.uid
                    )
                    records = {str(r.path): r for r in serial._records(candidate)}
                    unique_paths = sorted(
                        {p for group in value.selected_paths for p in group},
                        key=lambda p: records[p].original_index,
                    )
                    decoded = decode_series(
                        tuple(records[p] for p in unique_paths),
                        config.data.spacing_mm,
                        config.data.image_size,
                    )
                    valid = [v for v in decoded if not isinstance(v, str)]
                    population = np.concatenate(
                        [
                            v.image[t : t + h, l : l + w].reshape(-1)
                            for v in valid
                            for t, l, h, w in [v.rect]
                        ]
                    )
                    bounds = _linear_quantiles(
                        population,
                        config.data.lower_quantile,
                        config.data.upper_quantile,
                    )
                    appearances = {}
                    alignment = True
                    for group_index, paths in enumerate(value.selected_paths):
                        middle = records[paths[1]]
                        alignment &= np.isclose(
                            float(value.positions[group_index]),
                            middle.relative_position_mm,
                        )
                        for channel, path in enumerate(paths):
                            image = value.images[group_index, channel]
                            rect = value.valid_rects[group_index, channel]
                            patient_map = value.patient_maps[group_index, channel]
                            previous = appearances.setdefault(
                                path, (image, rect, patient_map)
                            )
                            alignment &= all(
                                torch.equal(a, b)
                                for a, b in zip(
                                    previous, (image, rect, patient_map), strict=True
                                )
                            )
                            native = np.asarray(
                                [
                                    records[path].pose.layout.orientation[:3],
                                    records[path].pose.layout.orientation[3:],
                                ]
                            ).T
                            native *= np.asarray(records[path].pose.layout.spacing)[
                                ::-1
                            ]
                            expected = (
                                np.column_stack((native, records[path].pose.position))
                                @ valid[unique_paths.index(path)].output_to_native
                            )
                            alignment &= np.allclose(
                                patient_map.numpy(), expected, atol=1e-5
                            )
                    rows.append(
                        {
                            "study_uid": uid,
                            "uid": value.uid,
                            "slot": slot,
                            "training": training,
                            "sampling_path": value.sampling_path,
                            "triplets": len(value.images),
                            "channel_appearances": value.slice_count,
                            "unique_slices": value.unique_slice_count,
                            "reused_appearances": value.slice_count
                            - value.unique_slice_count,
                            "intensity_pixel_count": value.intensity_pixel_count,
                            "independent_pixel_count": len(population),
                            "intensity_bounds": value.intensity_bounds,
                            "independent_bounds": bounds,
                            "population_equal": len(population)
                            == value.intensity_pixel_count
                            and bounds == value.intensity_bounds,
                            "metadata_aligned": bool(alignment),
                            "geometry_valid": value.geometry_valid.tolist(),
                            "paths": value.selected_paths,
                            "failures": value.failures,
                        }
                    )
        threaded.close()
    report = {
        "acquisitions": rows,
        "comparisons": comparisons,
        "limits": "Real CPU preparation only. No model or test harness. No artificial failures or geometry fixtures.",
        "summary": {
            "study_count": len(uids),
            "rejected_center_count": len(rejected),
            "serial_threaded_equal": all(
                v["serial_threaded_equal"] for v in comparisons
            ),
            "row_order_equal": all(v["row_order_equal"] for v in comparisons),
            "intensity_populations_equal": all(v["population_equal"] for v in rows),
            "metadata_aligned": all(v["metadata_aligned"] for v in rows),
            "reused_appearances": sum(v["reused_appearances"] for v in rows),
            "failures": sum(len(v["failures"]) for v in rows),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest["report_sha256"] = sha256_file(args.output_dir / "report.json")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return report["summary"]


def main() -> None:
    try:
        summary = inspect(InspectCLI())
    except (OSError, RuntimeError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Physical preparation inspection error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
