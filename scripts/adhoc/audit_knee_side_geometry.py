import ast
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pydicom
from pydantic import Field
from pydantic_settings import SettingsError
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.sources import SERIES
from knee_abnormality.data.splits import STUDY
from knee_abnormality.provenance import copy_sources, sha256_file
from scripts.adhoc.audit_knee_laterality import text_sides

EXTRA_TAGS = [
    "Rows",
    "Columns",
    "PixelSpacing",
    "FrameOfReferenceUID",
    "SequenceName",
    "ReceiveCoilName",
    "TransmitCoilName",
    "InPlanePhaseEncodingDirection",
]


class GeometryCLI(CLISettings):
    metadata_dir: Path = Path("outputs/analysis/knee-laterality-20260914/reviewed")
    output_dir: Path = Path("outputs/analysis/knee-side-geometry-20260914")
    enriched_headers: Path | None = None
    workers: int = Field(default=2, ge=1, le=8)


def enrich(row):
    result = dict(row)
    try:
        dataset = pydicom.dcmread(
            row["path"], stop_before_pixels=True, specific_tags=EXTRA_TAGS
        )
        result.update({tag: str(dataset.get(tag, "")) for tag in EXTRA_TAGS})
        position = np.asarray(
            ast.literal_eval(row["ImagePositionPatient"]), dtype=float
        )
        orientation = np.asarray(
            ast.literal_eval(row["ImageOrientationPatient"]), dtype=float
        ).reshape(2, 3)
        spacing = np.asarray(dataset.PixelSpacing, dtype=float)
        valid = (
            np.isfinite(position).all()
            and np.isfinite(orientation).all()
            and np.isfinite(spacing).all()
            and (spacing > 0).all()
            and np.allclose(orientation @ orientation.T, np.eye(2), atol=1e-3, rtol=0)
        )
        if not valid:
            raise ValueError("Invalid coordinate geometry")
        center = (
            position
            + orientation[0] * spacing[1] * (int(dataset.Columns) - 1) / 2
            + orientation[1] * spacing[0] * (int(dataset.Rows) - 1) / 2
        )
        result.update(
            {
                f"center_{axis}": float(value)
                for axis, value in zip("xyz", center, strict=True)
            }
        )
        result.update(
            {f"direction_{i}": float(v) for i, v in enumerate(orientation.flatten())}
        )
        result["sagittal_anchor_x"] = float(position[0])
        result["geometry_error"] = None
    except (
        OSError,
        ValueError,
        AttributeError,
        TypeError,
        pydicom.errors.InvalidDicomError,
    ) as error:
        result["geometry_error"] = f"{type(error).__name__}: {error}"
    return result


def agreement(frame, column, margin):
    values = frame[column].to_numpy()
    keep = np.isfinite(values) & (np.abs(values) > margin)
    target = (frame["side"] == "L").to_numpy()
    correct = int(((values[keep] > 0) == target[keep]).sum())
    return {
        "available": len(frame),
        "retained": int(keep.sum()),
        "correct": correct,
        "agreement": correct / int(keep.sum()) if keep.any() else None,
        "left_auc": float(roc_auc_score(target[keep], values[keep]))
        if len(np.unique(target[keep])) == 2
        else None,
    }


def analyze(settings):
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = [
        settings.metadata_dir / "headers.parquet",
        settings.metadata_dir / "studies.parquet",
        Path("data/raw/train_series.csv"),
        Path("data/raw/test_series.csv"),
    ]
    headers = pl.read_parquet(inputs[0])
    if headers["error"].null_count() != headers.height:
        raise ValueError("Resolve metadata read errors before geometric analysis")
    inventories = pl.concat(
        [
            pl.read_csv(path).with_columns(
                pl.lit(path.stem.split("_", 1)[0]).alias("split")
            )
            for path in inputs[2:]
        ]
    )
    headers = headers.join(
        inventories.select("split", STUDY, SERIES, "Anatomical_Plane"),
        on=["split", STUDY, SERIES],
        validate="m:1",
    )
    if settings.enriched_headers:
        enriched = pl.read_parquet(settings.enriched_headers)
        inputs.append(settings.enriched_headers)
        if sorted(enriched["path"].to_list()) != sorted(headers["path"].to_list()):
            raise ValueError("Enriched snapshot does not match original header paths")
    else:
        records = []
        with ThreadPoolExecutor(max_workers=settings.workers) as pool:
            for index, row in enumerate(pool.map(enrich, headers.to_dicts()), 1):
                records.append(row)
                if index % 15000 == 0:
                    print(
                        json.dumps({"headers_read": index, "total": len(headers)}),
                        flush=True,
                    )
        enriched = pl.from_dicts(records, infer_schema_length=None)
    enriched.write_parquet(settings.output_dir / "headers.parquet")
    valid = enriched.filter(pl.col("geometry_error").is_null())
    fields = [
        "center_x",
        "center_y",
        "center_z",
        "sagittal_anchor_x",
        *[f"direction_{i}" for i in range(6)],
    ]
    series = valid.group_by(["split", STUDY, SERIES, "Anatomical_Plane"]).agg(
        *[pl.col(name).median().alias(name) for name in fields],
        pl.col("FrameOfReferenceUID").n_unique().alias("frame_count"),
        pl.col("Manufacturer").first(),
        pl.col("ManufacturerModelName").first(),
        pl.col("PatientPosition").first(),
        pl.col("ReceiveCoilName").first(),
        pl.col("TransmitCoilName").first(),
        pl.col("InPlanePhaseEncodingDirection").first(),
    )
    series.write_parquet(settings.output_dir / "series.parquet")
    centers = series.group_by("split", STUDY).agg(
        pl.col("center_x").median().alias("center_x"),
        pl.col("center_x").min().alias("min_center_x"),
        pl.col("center_x").max().alias("max_center_x"),
        pl.col("Manufacturer").unique().sort().str.join("+").alias("raw_manufacturer"),
    )
    for plane in ["Sagittal", "Coronal", "Axial"]:
        p = (
            series.filter(pl.col("Anatomical_Plane") == plane)
            .group_by("split", STUDY)
            .agg(*[pl.col(name).median().alias(f"{plane}_{name}") for name in fields])
        )
        centers = centers.join(p, on=["split", STUDY], how="left", validate="1:1")
    centers = (
        pl.read_parquet(inputs[1])
        .join(centers, on=["split", STUDY], how="left", validate="1:1")
        .with_columns(
            ((pl.col("min_center_x") > 0) | (pl.col("max_center_x") < 0)).alias(
                "all_series_agree"
            ),
            (pl.col("max_center_x") - pl.col("min_center_x")).alias(
                "series_center_range_mm"
            ),
        )
    )
    centers = centers.with_columns(
        pl.when(pl.col("center_x") > 0)
        .then(pl.lit("L"))
        .when(pl.col("center_x") < 0)
        .then(pl.lit("R"))
        .otherwise(None)
        .alias("coordinate_candidate"),
        (
            pl.col("all_series_agree")
            & (pl.col("center_x").abs() > 40)
            & (pl.col("series_center_range_mm") <= 40)
        ).alias("exploratory_gate"),
    )
    centers.write_parquet(settings.output_dir / "studies.parquet")
    reports_path = Path("data/raw/train.csv")
    inputs.append(reports_path)
    reports = pl.read_csv(reports_path).select(STUDY, "Report")
    reports = reports.with_columns(
        pl.Series("report_sides", [text_sides(v or "") for v in reports["Report"]])
    ).drop("Report")
    corroboration = (
        centers.filter(pl.col("split") == "train")
        .join(reports, on=STUDY, validate="1:1")
        .with_columns(
            pl.when(pl.col("report_sides").list.len() == 1)
            .then(pl.col("report_sides").list.first())
            .otherwise(None)
            .alias("report_side")
        )
    )
    corroboration.write_parquet(settings.output_dir / "report_corroboration.parquet")
    report_agreement = []
    for status in ["explicit", "text_inferred", "unknown", "conflict"]:
        frame = (
            corroboration.filter(
                (pl.col("status") == status) & pl.col("report_side").is_not_null()
            )
            .drop("side")
            .rename({"report_side": "side"})
        )
        for vendor in ["all", *sorted(frame["raw_manufacturer"].unique().to_list())]:
            f = (
                frame
                if vendor == "all"
                else frame.filter(pl.col("raw_manufacturer") == vendor)
            )
            for margin in [0, 20, 40, 60]:
                report_agreement.append(
                    {
                        "metadata_status": status,
                        "manufacturer": vendor,
                        "margin_mm": margin,
                        **agreement(f, "center_x", margin),
                    }
                )
    pl.from_dicts(report_agreement, infer_schema_length=None).write_csv(
        settings.output_dir / "report_agreement.csv"
    )
    metrics = []
    orientations = []
    for status in ["explicit", "text_inferred"]:
        labeled = centers.filter(
            (pl.col("split") == "train") & (pl.col("status") == status)
        )
        for vendor in ["all", *sorted(labeled["raw_manufacturer"].unique().to_list())]:
            frame = (
                labeled
                if vendor == "all"
                else labeled.filter(pl.col("raw_manufacturer") == vendor)
            )
            for column in [
                "Sagittal_sagittal_anchor_x",
                "center_x",
                "Sagittal_center_x",
                "Coronal_center_x",
                "Axial_center_x",
            ]:
                for margin in [0, 20, 40, 60]:
                    metrics.append(
                        {
                            "status": status,
                            "manufacturer": vendor,
                            "rule": column,
                            "margin_mm": margin,
                            **agreement(frame, column, margin),
                        }
                    )
            metrics.append(
                {
                    "status": status,
                    "manufacturer": vendor,
                    "rule": "exploratory_gate",
                    "margin_mm": 40,
                    "total_before_gate": frame.height,
                    **agreement(
                        frame.filter(pl.col("exploratory_gate")), "center_x", 0
                    ),
                }
            )
            near = frame.filter(pl.col("center_x").abs() <= 40)
            y = (near["side"] == "L").to_numpy()
            if len(np.unique(y)) != 2:
                continue
            for plane in ["Sagittal", "Coronal", "Axial"]:
                for i in range(6):
                    column = f"{plane}_direction_{i}"
                    values = near[column].to_numpy()
                    good = np.isfinite(values)
                    if len(np.unique(y[good])) == 2:
                        orientations.append(
                            {
                                "status": status,
                                "manufacturer": vendor,
                                "cases_with_abs_center_x_le_40": int(good.sum()),
                                "feature": column,
                                "left_auc": float(roc_auc_score(y[good], values[good])),
                            }
                        )
    pl.from_dicts(metrics, infer_schema_length=None).write_csv(
        settings.output_dir / "agreement.csv"
    )
    pl.from_dicts(orientations, infer_schema_length=None).write_csv(
        settings.output_dir / "orientation_probes.csv"
    )
    coverage = (
        centers.group_by("split", "status", "raw_manufacturer")
        .agg(
            pl.len().alias("studies"),
            pl.col("exploratory_gate").sum().alias("gated_coordinate_candidates"),
            pl.col("all_series_agree").sum().alias("all_series_agree"),
            pl.col("center_x").abs().median().alias("median_abs_center_x"),
        )
        .sort("split", "status", "raw_manufacturer")
    )
    coverage.write_csv(settings.output_dir / "coverage.csv")
    categorical = series.join(
        centers.select("split", STUDY, "side", "status"),
        on=["split", STUDY],
        validate="m:1",
    )
    for field in [
        "ReceiveCoilName",
        "TransmitCoilName",
        "PatientPosition",
        "InPlanePhaseEncodingDirection",
    ]:
        categorical.group_by("split", "Manufacturer", "status", field, "side").agg(
            pl.col(STUDY).n_unique().alias("studies")
        ).sort("split", "Manufacturer", "status", field).write_csv(
            settings.output_dir / f"{field}.csv"
        )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "versions": {
            "numpy": np.__version__,
            "polars": pl.__version__,
            "pydicom": pydicom.__version__,
        },
        "headers": enriched.height,
        "geometry_errors": enriched.filter(
            pl.col("geometry_error").is_not_null()
        ).height,
        "rules": "L if X > 0, R if X < 0; image centers use DICOM rows/columns/spacing/direction; median per series then median across series.",
        "gate": "Exploratory: all series center-X signs agree, absolute study median center X > 40 mm, series-center range <= 40 mm; not calibrated or independently validated.",
        "agreement": metrics,
        "report_corroboration": report_agreement,
        "frame_references_per_study": valid.group_by("split", STUDY)
        .agg(pl.col("FrameOfReferenceUID").n_unique().alias("frames"))
        .group_by("split", "frames")
        .len()
        .sort("split", "frames")
        .to_dicts(),
        "coverage": coverage.to_dicts(),
        "example_test": centers.filter(pl.col("split") == "test")
        .select(
            "raw_manufacturer",
            "status",
            "side",
            "center_x",
            "all_series_agree",
            "series_center_range_mm",
            "coordinate_candidate",
            "exploratory_gate",
        )
        .to_dicts(),
        "limitations": [
            "Agreement is against available metadata side evidence, not independent anatomical ground truth.",
            "Rules and margin thresholds are exploratory; no learned classifier, train/test model validation or hidden-test evaluation was performed.",
            "Coordinate conventions and missing labels depend strongly on manufacturer/model; accuracy on unlabelled groups is unmeasured.",
            "Orientation probes are descriptive multiple comparisons on small subsets, not validated side predictors.",
            "Three headers per series sample slice positions; FOV centers are not anatomical landmarks.",
            "Training reports are an additional weak reference for corroboration only, never an inference input; text parsing and clinical wording can be wrong or ambiguous.",
            "Series have distinct FrameOfReferenceUIDs; pooled coordinates and agreement gates are empirical acquisition heuristics, not guaranteed registered anatomy.",
            "CPU metadata only; no pixels, image-based methods, model execution, tests or GPU access.",
        ],
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    source_snapshot = copy_sources([Path(__file__)], settings.output_dir / "source")
    paths = inputs + [
        Path(__file__),
        Path("src/knee_abnormality/cli.py"),
        Path("src/knee_abnormality/data/labels.py"),
        Path("src/knee_abnormality/data/splits.py"),
        Path("src/knee_abnormality/data/sources.py"),
        Path("src/knee_abnormality/provenance.py"),
        Path("scripts/adhoc/audit_knee_laterality.py"),
        Path("uv.lock"),
    ]
    manifest = {
        "inputs_and_sources": [
            {"path": str(p), "sha256": sha256_file(p)} for p in paths
        ],
        "outputs": [
            {"path": str(p), "sha256": sha256_file(p)}
            for p in sorted(settings.output_dir.iterdir())
            if p.is_file()
        ],
        "source_snapshot": source_snapshot,
    }
    (settings.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "headers": summary["headers"],
                "geometry_errors": summary["geometry_errors"],
                "agreement_all": [
                    r
                    for r in metrics
                    if r["manufacturer"] == "all"
                    and r["rule"] in ["center_x", "exploratory_gate"]
                ],
                "example_test": summary["example_test"],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    try:
        analyze(GeometryCLI())
    except (OSError, ValueError, SettingsError) as error:
        print(f"Knee side geometry error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
