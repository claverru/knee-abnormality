import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import load_config
from knee_abnormality.data.labels import STUDY
from knee_abnormality.data.preprocessing.geometry import (
    NativePose,
    PoseLayout,
    estimate_reference,
    summarize_series,
)
from knee_abnormality.data.preprocessing.sampling import (
    build_pool,
    select_triplets,
    target_depths,
)
from knee_abnormality.data.preprocessing.series import (
    IndexedSeries,
    SeriesSummary,
    SliceRecord,
)
from knee_abnormality.data.sources import KNOWN_BAD_INSTANCES, SERIES
from knee_abnormality.identity import preprocessing_identity
from knee_abnormality.provenance import copy_sources, sha256_file


class AnalyzeCLI(CLISettings):
    config: Path = Path("configs/training.yaml")
    headers: Path = Path("outputs/analysis/geometry-metadata-20260912/headers.parquet")
    inventory: Path = Path("data/metadata/train_series_image_counts.parquet")
    reference: Path = Path(
        "outputs/analysis/physical-sampling-20260913/geometry-v3/summary.json"
    )
    output_dir: Path
    seed: int = 20260913
    training_draws: int = Field(default=32, ge=1)


@dataclass
class Acquisition:
    uid: str
    study: str
    plane: str
    fluid: int
    fat: int
    records: tuple
    geometry: object


def distribution(values):
    a = np.asarray(values, dtype=float)
    return (
        {
            "n": len(a),
            "mean": float(a.mean()),
            **{f"p{q}": float(np.percentile(a, q)) for q in (0, 50, 95, 99, 100)},
        }
        if len(a)
        else {"n": 0}
    )


def read_acquisitions(settings):
    inventory = pl.read_parquet(settings.inventory)
    fat = dict(inventory.select(SERIES, "Fat_Suppression").iter_rows())
    result = defaultdict(list)
    for part in pl.read_parquet(settings.headers).partition_by(SERIES):
        first = part.row(0, named=True)
        records = []
        for row in part.iter_rows(named=True):
            if Path(row["path"]).stem in KNOWN_BAD_INSTANCES:
                continue
            orientation = row["ImageOrientationPatient"]
            normal = np.cross(orientation[:3], orientation[3:])
            position = float(np.dot(normal, row["ImagePositionPatient"]))
            layout = PoseLayout(
                tuple(orientation),
                tuple(row["PixelSpacing"]),
                int(row["Rows"]),
                int(row["Columns"]),
                row["FrameOfReferenceUID"],
            )
            records.append(
                SliceRecord(
                    Path(row["path"]),
                    position,
                    0,
                    position_source="patient",
                    pose=NativePose(tuple(row["ImagePositionPatient"]), layout),
                )
            )
        records.sort(key=lambda r: (r.physical_position, r.path.name))
        origin = (records[0].physical_position + records[-1].physical_position) / 2
        records = tuple(
            SliceRecord(
                r.path,
                r.physical_position,
                i,
                r.physical_position - origin,
                position_source="patient",
                pose=r.pose,
            )
            for i, r in enumerate(records)
        )
        item = Acquisition(
            first[SERIES],
            first[STUDY],
            first["Anatomical_Plane"],
            first["Fluid_Sensitive"],
            fat[first[SERIES]],
            records,
            summarize_series(records),
        )
        result[item.study].append(item)
    return {
        study: sorted(values, key=lambda a: a.uid)
        for study, values in sorted(result.items())
    }


def analyze(settings: AnalyzeCLI) -> None:
    output = settings.output_dir
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(settings.config)
    acquired = read_acquisitions(settings)
    folds = dict(
        pl.read_csv(config.split_dir / "studies.csv").select(STUDY, "fold").iter_rows()
    )
    kinds = {
        (k.plane, k.fluid_sensitive, k.fat_suppression): k
        for k in config.data.series_types
    }
    source_files = [Path(__file__), *Path("src/knee_abnormality").rglob("*.py")]
    inputs = [
        settings.config,
        settings.headers,
        settings.inventory,
        settings.reference,
        config.split_dir / "studies.csv",
        config.split_dir / "protocol.json",
        Path("uv.lock"),
    ]
    copy_sources(source_files, output / "source")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "preprocessing_identity": preprocessing_identity(config.model.backbone),
        "inputs": [
            {"path": str(p.resolve()), "sha256": sha256_file(p)}
            for p in inputs + source_files
        ],
        "versions": {
            name: version(name)
            for name in ("numpy", "scipy", "polars", "torch", "pydantic-settings")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    slots, draws, selected, centers, exposures = [], [], [], [], []
    for number, (study, acquisitions) in enumerate(acquired.items(), 1):
        reference = estimate_reference(
            tuple(
                (a.uid, a.plane, a.geometry)
                for a in acquisitions
                if len(a.records) >= 3
            ),
        )
        centers.append(
            {
                "study_uid": study,
                "accepted": reference.point is not None,
                "reason": reference.reason,
                "rms_mm": reference.rms_mm,
                "fold": folds[study],
            }
        )
        grouped = defaultdict(list)
        for a in acquisitions:
            grouped[a.plane, a.fluid, a.fat].append(a)
        for kind, items in sorted(grouped.items()):
            if kind not in kinds:
                continue
            pool = build_pool(
                tuple(
                    (a.uid, IndexedSeries(a.records, a.geometry, SeriesSummary()))
                    for a in items
                ),
                kind[0],
                reference,
                config.data.spacing_mm,
                config.data.image_size * config.data.spacing_mm / 2,
            )
            identity = {
                "study_uid": study,
                "type": "/".join(map(str, kind)),
                "fold0_training": folds[study] != 0,
                "path": pool.diagnostics["path"],
                "repeated": len(items) > 1,
                "dense": any(
                    np.median(np.diff([r.physical_position for r in a.records])) < 2
                    for a in items
                ),
                "wide": any(
                    a.records[-1].physical_position - a.records[0].physical_position
                    > 120
                    for a in items
                ),
            }
            slots.append(
                {
                    **identity,
                    "candidate_centers": len(pool.candidates),
                    "diagnostics": json.dumps(pool.diagnostics),
                }
            )
            intended = defaultdict(float)
            if pool.candidates:
                for t in target_depths(
                    pool, (np.arange(2048) + 0.5) / 2048, config.data.sampling_sigma_mm
                ):
                    suppliers = {
                        v.uid
                        for v in pool.candidates
                        if v.support[0] <= t <= v.support[1]
                    }
                    for uid in suppliers:
                        intended[uid] += 1 / len(suppliers) / 2048
            for mode, repeats, multiplier in (
                ("validation", 1, config.data.validation_image_multiplier),
                ("training", settings.training_draws, 1),
            ):
                for epoch in range(repeats):
                    selection = select_triplets(
                        pool,
                        kinds[kind].max_images // 3 * multiplier,
                        config.data.sampling_sigma_mm,
                        config.data.image_size * config.data.spacing_mm / 2,
                        study,
                        kind,
                        settings.seed if mode == "training" else None,
                        epoch,
                    )
                    meta = selection.metadata
                    draw_id = {**identity, "mode": mode, "draw": epoch}
                    draws.append(
                        {
                            **draw_id,
                            **{
                                k: meta[k]
                                for k in (
                                    "requested_triplets",
                                    "capacity",
                                    "triplets",
                                    "channel_appearances",
                                    "unique_source_slices",
                                    "distinct_middle_geometry",
                                    "shortfall",
                                )
                            },
                            "target_beyond30": sum(abs(t) > 30 for t in meta["targets"])
                            if pool.physical
                            else None,
                            "target_beyond40": sum(abs(t) > 40 for t in meta["targets"])
                            if pool.physical
                            else None,
                        }
                    )
                    for assignment, triplet in zip(
                        meta["assignments"], selection.triplets, strict=True
                    ):
                        selected.append(
                            {
                                **draw_id,
                                **assignment,
                                "source_paths": [str(r.path) for r in triplet.records],
                            }
                        )
                    for uid in sorted(intended):
                        realized = meta.get("realized_counts", {}).get(uid, 0)
                        exposures.append(
                            {
                                **draw_id,
                                "uid": uid,
                                "intended_share": intended[uid],
                                "realized_count": realized,
                                "triplets": meta["triplets"],
                                "preferred_count": meta.get("preferred_counts", {}).get(
                                    uid, 0
                                ),
                            }
                        )
        print(json.dumps({"studies": number, "total": len(acquired)}), flush=True)
    for name, rows in (
        ("slots", slots),
        ("draws", draws),
        ("selected", selected),
        ("centers", centers),
        ("exposures", exposures),
    ):
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(
            output / f"{name}.parquet"
        )
    comparisons = []
    for mode in ("training", "validation"):
        for path in ("physical", "local"):
            ds = [r for r in draws if r["mode"] == mode and r["path"] == path]
            ss = [r for r in selected if r["mode"] == mode and r["path"] == path]
            shares = defaultdict(lambda: [0.0, 0.0, 0.0])
            for r in exposures:
                if r["mode"] == mode and r["path"] == path and r["repeated"]:
                    v = shares[r["study_uid"], r["type"], r["uid"]]
                    v[0] += r["realized_count"]
                    v[1] += r["triplets"]
                    v[2] = r["intended_share"]
            comparisons.append(
                {
                    "mode": mode,
                    "path": path,
                    "groups": len(ds),
                    "triplets": len(ss),
                    "shortfall": sum(r["shortfall"] for r in ds),
                    "selected_fraction_beyond30mm": sum(
                        abs(r["depth"]) > 30 for r in ss
                    )
                    / len(ss)
                    if ss and path == "physical"
                    else None,
                    "selected_fraction_beyond40mm": sum(
                        abs(r["depth"]) > 40 for r in ss
                    )
                    / len(ss)
                    if ss and path == "physical"
                    else None,
                    "target_fraction_beyond30mm": sum(r["target_beyond30"] for r in ds)
                    / sum(min(r["capacity"], r["requested_triplets"]) for r in ds)
                    if ds and path == "physical"
                    else None,
                    "mean_unique_raw_slices": float(
                        np.mean([r["unique_source_slices"] for r in ds])
                    )
                    if ds
                    else None,
                    "placement_error_mm"
                    if path == "physical"
                    else "placement_error_percentile": distribution(
                        [r["error"] for r in ss]
                    ),
                    "supplier_changes": sum(r["supplier_changed"] for r in ss),
                    "source_share_mean_absolute_error": float(
                        np.mean(
                            [abs(v[0] / v[1] - v[2]) for v in shares.values() if v[1]]
                        )
                    )
                    if shares
                    else None,
                }
            )
    diagnostics = [json.loads(s["diagnostics"]) for s in slots]
    summary = {
        "studies": len(centers),
        "rejected_centers": sum(not r["accepted"] for r in centers),
        "comparisons": comparisons,
        "diagnostics": {
            k: sum(d[k] for d in diagnostics)
            for k in (
                "split_sources",
                "singleton_runs",
                "plane_disagreements",
                "no_valid_run_sources",
            )
        },
        "historical_reference": json.loads(settings.reference.read_text()),
        "limits": "Metadata selection only; no pixels, models, training, tests or simulated failures.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["outputs"] = {
        p.name: sha256_file(p)
        for p in sorted(output.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    try:
        analyze(AnalyzeCLI())
    except (OSError, ValueError, yaml.YAMLError, SettingsError) as error:
        print(f"Physical sampling analysis error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
