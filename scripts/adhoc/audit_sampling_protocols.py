import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pydicom
from pydantic import Field

from knee_abnormality.cli import CLISettings
from knee_abnormality.provenance import sha256_file

TAGS = [
    "SequenceName",
    "ProtocolName",
    "SeriesDescription",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "RepetitionTime",
    "EchoTime",
    "EchoTrainLength",
    "FlipAngle",
    "MRAcquisitionType",
    "ImageType",
    "AcquisitionNumber",
    "NumberOfAverages",
    "Manufacturer",
]
NUMERIC = {"RepetitionTime", "EchoTime", "EchoTrainLength", "FlipAngle"}
KEYS = ["StudyInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]
REDUCED = [
    "ScanningSequence",
    "SequenceVariant",
    "MRAcquisitionType",
    "RepetitionTime",
    "EchoTime",
    "EchoTrainLength",
    "FlipAngle",
]


class Settings(CLISettings):
    headers: Path = Path("outputs/analysis/geometry-metadata-20260912/headers.parquet")
    inventory: Path = Path("data/metadata/train_series_image_counts.parquet")
    output_dir: Path = Field(description="Output directory for the audit artifacts")


def read_headers(settings: Settings) -> pl.DataFrame:
    headers = (
        pl.read_parquet(settings.headers)
        .select(
            [
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "path",
                "Anatomical_Plane",
                "Fluid_Sensitive",
            ]
        )
        .sort(["SeriesInstanceUID", "path"])
        .unique("SeriesInstanceUID", keep="first", maintain_order=True)
    )
    inventory = (
        pl.read_parquet(settings.inventory)
        .select(["SeriesInstanceUID", "Fat_Suppression"])
        .unique("SeriesInstanceUID")
    )
    rows = []
    for row in headers.join(inventory, on="SeriesInstanceUID").iter_rows(named=True):
        dataset = pydicom.dcmread(
            row["path"], stop_before_pixels=True, specific_tags=TAGS
        )
        record = {
            tag: (
                None
                if dataset.get(tag) is None
                else repr(getattr(dataset.get(tag), "value", dataset.get(tag)))
            )
            for tag in TAGS
        }
        for tag in NUMERIC:
            value = dataset.get(tag)
            record[f"{tag}_num"] = (
                None if value is None else float(getattr(value, "value", value))
            )
        record.update({key: row[key] for key in KEYS + ["SeriesInstanceUID", "path"]})
        rows.append(record)
    return pl.from_dicts(rows, infer_schema_length=1000)


def audit(frame: pl.DataFrame) -> dict:
    supported = frame.filter(
        pl.col("Anatomical_Plane").is_in(["Axial", "Coronal", "Sagittal"])
        & (
            ((pl.col("Fluid_Sensitive") == 0) & (pl.col("Fat_Suppression") == 0))
            | ((pl.col("Fluid_Sensitive") == 1) & (pl.col("Fat_Suppression") == 1))
        )
    )
    repeated = (
        supported.group_by(KEYS)
        .agg(pl.len().alias("acquisitions"))
        .filter(pl.col("acquisitions") > 1)
    )
    repeats = supported.join(repeated.select(KEYS), on=KEYS, how="semi")
    availability = {
        tag: int(frame.select(pl.col(tag).is_not_null().sum()).item()) for tag in TAGS
    }
    variation = {}
    for tag in REDUCED:
        counts = repeats.group_by(KEYS).agg(
            [
                pl.col(tag).n_unique().alias("values"),
                pl.col(tag).is_null().any().alias("missing"),
            ]
        )
        variation[tag] = {
            "varying_groups": int((counts["values"] > 1).sum()),
            "missing_groups": int(counts["missing"].sum()),
        }
    signatures = repeats.with_columns(
        pl.concat_str(
            [pl.col(tag).fill_null("<NULL>") for tag in REDUCED], separator="|"
        ).alias("signature")
    )
    signature_counts = signatures.group_by(KEYS).agg(
        pl.col("signature").n_unique().alias("signatures")
    )
    differences = {}
    for tag in ("RepetitionTime", "EchoTime"):
        grouped = repeats.group_by(KEYS).agg(
            [
                pl.col(f"{tag}_num").is_null().any().alias("missing"),
                pl.col(f"{tag}_num").min().alias("minimum"),
                pl.col(f"{tag}_num").max().alias("maximum"),
            ]
        )
        values = grouped.filter(~pl.col("missing"))
        missing_groups = int(grouped["missing"].sum())
        if tag == "EchoTime":
            differences[tag] = {
                "missing_groups": missing_groups,
                "fully_available_groups": values.height,
                "range_gt_1_ms": int(
                    ((values["maximum"] - values["minimum"]) > 1).sum()
                ),
                "range_gt_5_ms": int(
                    ((values["maximum"] - values["minimum"]) > 5).sum()
                ),
            }
        else:
            differences[tag] = {
                "missing_groups": missing_groups,
                "fully_available_groups": values.height,
                "ratio_gt_1_05": int(
                    ((values["maximum"] / values["minimum"]) > 1.05).sum()
                ),
                "ratio_gt_1_2": int(
                    ((values["maximum"] / values["minimum"]) > 1.2).sum()
                ),
            }
    return {
        "series_read": frame.height,
        "supported_series": supported.height,
        "repeated_groups": repeated.height,
        "field_availability": availability,
        "reduced_signature": {
            "identical_groups": int((signature_counts["signatures"] == 1).sum()),
            "multiple_signature_groups": int(
                (signature_counts["signatures"] > 1).sum()
            ),
        },
        "reduced_field_variation": variation,
        "tr_difference": differences["RepetitionTime"],
        "te_difference": differences["EchoTime"],
        "limitations": [
            "One header per sampled series is inspected; exact signatures are conservative metadata proxies, not clinical duplicate labels.",
            "ProtocolName is absent in this sample, and n_images is not physical density or coverage.",
        ],
    }


def main() -> None:
    settings = Settings()
    if settings.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {settings.output_dir}")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_headers(settings)
    summary = {
        "audited_at": datetime.now(UTC).isoformat(),
        "polars_version": pl.__version__,
        "pydicom_version": pydicom.__version__,
        "config": {
            "headers": str(settings.headers),
            "inventory": str(settings.inventory),
            "output_dir": str(settings.output_dir),
        },
        **audit(frame),
    }
    frame.write_parquet(settings.output_dir / "protocol-headers.parquet")
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    source = Path(__file__).resolve()
    source_snapshot = settings.output_dir / source.name
    shutil.copy2(source, source_snapshot)
    source_files = [
        source,
        Path("src/knee_abnormality/cli.py"),
        Path("src/knee_abnormality/provenance.py"),
        Path("uv.lock"),
    ]
    manifest = {
        "created_at": summary["audited_at"],
        "inputs": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (settings.headers, settings.inventory, *source_files)
        },
        "outputs": {
            name: {
                "sha256": sha256_file(settings.output_dir / name),
                "bytes": (settings.output_dir / name).stat().st_size,
            }
            for name in ("protocol-headers.parquet", "summary.json", source.name)
        },
        "config": summary["config"],
        "limitations": [
            "Original raw DICOM files are not hashed; the selected-header parquet and emitted protocol snapshot are hashed to preserve header-only scope."
        ],
    }
    (settings.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
