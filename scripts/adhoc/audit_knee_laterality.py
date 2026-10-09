import json
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pydicom
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.sources import SERIES
from knee_abnormality.data.splits import STUDY
from knee_abnormality.provenance import copy_sources, sha256_file

TEXT_FIELDS = (
    "StudyDescription",
    "SeriesDescription",
    "ProtocolName",
    "BodyPartExamined",
)
SIDE_FIELDS = ("Laterality", "ImageLaterality", "FrameLaterality")
TAGS = [
    STUDY,
    SERIES,
    *TEXT_FIELDS,
    "Laterality",
    "ImageLaterality",
    "Manufacturer",
    "ManufacturerModelName",
    "PatientPosition",
    "PatientOrientation",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "AnatomicRegionSequence",
    "PrimaryAnatomicStructureSequence",
    "SharedFunctionalGroupsSequence",
    "PerFrameFunctionalGroupsSequence",
]
WORDS = {
    "L": {
        "LEFT",
        "LINKS",
        "LINKE",
        "LINKEN",
        "LINKES",
        "IZQUIERDA",
        "IZQUIERDO",
        "GAUCHE",
        "ESQUERDO",
        "ESQUERDA",
        "SINISTRO",
        "SINISTRA",
    },
    "R": {
        "RIGHT",
        "RECHTS",
        "RECHTE",
        "RECHTEN",
        "RECHTES",
        "DERECHA",
        "DERECHO",
        "DROIT",
        "DROITE",
        "DIREITO",
        "DIREITA",
        "DESTRO",
        "DESTRA",
    },
}
ABBREVIATIONS = {
    "L": {"L", "LT", "LI", "IZQ", "IZDA", "ESQ", "SX"},
    "R": {"R", "RT", "RE", "DER", "DCHA", "DIR", "DX"},
}
ANATOMY = {"KNEE", "KNIE", "RODILLA", "GENOU", "JOELHO", "GINOCCHIO"}
SIDE_SHORTHAND = {"L": {"LT", "IZQ", "IZDA", "ESQ"}, "R": {"RT", "DCHA"}}
MRI_CONTEXT = {
    "SAG",
    "SAGITTAL",
    "COR",
    "CORONAL",
    "AX",
    "AXI",
    "AXIAL",
    "TRA",
    "PD",
    "T1",
    "T2",
    "T1W",
    "T2W",
    "PDW",
    "TSE",
    "FSE",
}


class AuditCLI(CLISettings):
    data_root: Path = Path("data/raw")
    output_dir: Path = Path("outputs/analysis/k/knee-laterality-20260914")
    headers: Path | None = None
    workers: int = Field(default=2, ge=1, le=8)


def text_sides(value):
    normalized = (
        unicodedata.normalize("NFKD", value.upper()).encode("ascii", "ignore").decode()
    )
    tokens = set(re.findall(r"[A-Z0-9]+", normalized))
    return sorted(
        side
        for side in WORDS
        if tokens & (WORDS[side] | SIDE_SHORTHAND[side])
        or (tokens & ANATOMY and tokens & ABBREVIATIONS[side])
        or (tokens & MRI_CONTEXT and side in tokens)
    )


def manufacturer_group(values):
    groups = set()
    for value in values:
        upper = value.upper()
        groups.add(
            next(
                (
                    group
                    for token, group in [
                        ("SIEMENS", "Siemens"),
                        ("PHILIPS", "Philips"),
                        ("GE", "GE"),
                        ("TOSHIBA", "Toshiba"),
                        ("CANON", "Canon"),
                        ("FUJIFILM", "Fujifilm"),
                        ("HITACHI", "Hitachi"),
                    ]
                    if upper.startswith(token)
                ),
                value or "unknown",
            )
        )
    return "+".join(sorted(groups))


def nested_evidence(dataset, prefix=""):
    result = []
    for element in dataset:
        key = f"{prefix}{element.keyword}"
        if element.keyword in SIDE_FIELDS or element.keyword == "CodeMeaning":
            result.append({"field": key, "value": str(element.value)})
        if element.VR == "SQ":
            for index, item in enumerate(element.value):
                result.extend(nested_evidence(item, f"{key}[{index}]."))
    return result


def read_series(row, data_root):
    directory = data_root / f"{row['split']}_series" / row[STUDY] / row[SERIES]
    paths = sorted(directory.glob("*.dcm"))
    base = {
        "split": row["split"],
        STUDY: row[STUDY],
        SERIES: row[SERIES],
        "files_in_series": len(paths),
    }
    if not paths:
        return [{**base, "path": None, "error": "no_dicom_files"}]
    records = []
    for index in sorted({0, len(paths) // 2, len(paths) - 1}):
        path = paths[index]
        record = {**base, "path": str(path), "file_bytes": path.stat().st_size}
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
            record.update(
                {
                    tag: str(dataset.get(tag, ""))
                    for tag in TAGS
                    if not tag.endswith("Sequence") and tag not in {STUDY, SERIES}
                }
            )
            record["nested_evidence"] = json.dumps(
                nested_evidence(dataset), sort_keys=True
            )
            record["header_uid_match"] = (
                str(dataset.get(STUDY, "")) == row[STUDY]
                and str(dataset.get(SERIES, "")) == row[SERIES]
            )
            record["error"] = (
                None if record["header_uid_match"] else "header_uid_mismatch"
            )
        except (OSError, ValueError, pydicom.errors.InvalidDicomError) as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    return records


def summarize(headers, inventory):
    evidence = []
    for row in headers.filter(pl.col("error").is_null()).iter_rows(named=True):
        items = json.loads(row["nested_evidence"])
        items.extend(
            {"field": field, "value": row[field]} for field in TEXT_FIELDS if row[field]
        )
        for item in items:
            field, value = item["field"], item["value"]
            if field.rsplit(".", 1)[-1] in SIDE_FIELDS:
                sides = (
                    [value.strip().upper()]
                    if value.strip().upper() in {"L", "R"}
                    else []
                )
                kind = "explicit"
            elif field.endswith("CodeMeaning"):
                sides = text_sides(value)
                kind = "coded_meaning"
            else:
                sides = text_sides(value)
                kind = "text"
            for side in sides:
                evidence.append(
                    {
                        "split": row["split"],
                        STUDY: row[STUDY],
                        SERIES: row[SERIES],
                        "path": row["path"],
                        "field": field,
                        "value": value,
                        "side": side,
                        "kind": kind,
                    }
                )
    evidence_frame = pl.from_dicts(
        evidence,
        schema={
            k: pl.String
            for k in ("split", STUDY, SERIES, "path", "field", "value", "side", "kind")
        },
    )
    by_study = {}
    for row in evidence:
        by_study.setdefault((row["split"], row[STUDY]), []).append(row)
    records = []
    header_groups = {key: group for key, group in headers.group_by(["split", STUDY])}
    for row in (
        inventory.select("split", STUDY)
        .unique()
        .sort(["split", STUDY])
        .iter_rows(named=True)
    ):
        key = (row["split"], row[STUDY])
        group = header_groups[key]
        values = by_study.get(key, [])
        explicit = sorted({v["side"] for v in values if v["kind"] != "text"})
        textual = sorted({v["side"] for v in values if v["kind"] == "text"})
        combined = sorted(set(explicit + textual))
        status = (
            "conflict"
            if len(combined) > 1
            else "explicit"
            if explicit
            else "text_inferred"
            if textual
            else "unknown"
        )
        records.append(
            {
                **row,
                "status": status,
                "side": combined[0] if len(combined) == 1 else None,
                "explicit_sides": explicit,
                "text_sides": textual,
                "series": group[SERIES].n_unique(),
                "headers_read": group.filter(pl.col("path").is_not_null()).height,
                "header_errors": group.filter(pl.col("error").is_not_null()).height,
                "manufacturers": sorted(
                    set(group["Manufacturer"].drop_nulls().to_list())
                ),
                "manufacturer_group": manufacturer_group(
                    group["Manufacturer"].drop_nulls().to_list()
                ),
                "evidence_fields": sorted({v["field"] for v in values}),
            }
        )
    studies = pl.from_dicts(records, infer_schema_length=None)
    summary = {}
    for split in sorted(studies["split"].unique()):
        frame = studies.filter(pl.col("split") == split)
        h = headers.filter(pl.col("split") == split)
        same = frame.filter(
            (pl.col("explicit_sides").list.len() == 1)
            & (pl.col("text_sides").list.len() > 0)
        )
        summary[split] = {
            "studies": frame.height,
            "series": h[SERIES].n_unique(),
            "header_rows": h.height,
            "header_errors": h.filter(pl.col("error").is_not_null()).height,
            "series_with_conflicting_explicit_tags": evidence_frame.filter(
                (pl.col("split") == split) & (pl.col("kind") == "explicit")
            )
            .group_by(SERIES)
            .agg(pl.col("side").n_unique().alias("sides"))
            .filter(pl.col("sides") > 1)
            .height,
            "status_counts": dict(Counter(frame["status"].to_list())),
            "manufacturer_status_counts": frame.group_by("manufacturer_group", "status")
            .len()
            .sort(["manufacturer_group", "status"])
            .to_dicts(),
            "side_counts": frame.group_by("status", "side")
            .len()
            .sort(["status", "side"])
            .to_dicts(),
            "text_with_single_explicit_side": same.height,
            "text_disagrees_with_single_explicit_side": same.filter(
                pl.col("status") == "conflict"
            ).height,
            "field_nonempty_header_counts": {
                tag: h.filter(pl.col(tag).fill_null("").str.strip_chars() != "").height
                for tag in TAGS
                if tag in h.columns and tag not in {STUDY, SERIES}
            },
        }
    return studies, evidence_frame, summary


def audit(settings: AuditCLI) -> dict:
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = [settings.data_root / f"{split}_series.csv" for split in ("train", "test")]
    inventory = pl.concat(
        [
            pl.read_csv(path).with_columns(
                pl.lit(path.stem.split("_", 1)[0]).alias("split")
            )
            for path in inputs
        ]
    ).sort(["split", STUDY, SERIES])
    if settings.headers:
        headers = pl.read_parquet(settings.headers)
        inputs.append(settings.headers)
    else:
        records = []
        rows = inventory.to_dicts()
        with ThreadPoolExecutor(max_workers=settings.workers) as pool:
            for index, result in enumerate(
                pool.map(lambda row: read_series(row, settings.data_root), rows), 1
            ):
                records.extend(result)
                if index % 2000 == 0:
                    print(
                        json.dumps({"series_read": index, "total_series": len(rows)}),
                        flush=True,
                    )
        headers = pl.from_dicts(records, infer_schema_length=None)
    expected = (
        inventory.select("split", STUDY, SERIES).unique().sort(["split", STUDY, SERIES])
    )
    actual = (
        headers.select("split", STUDY, SERIES).unique().sort(["split", STUDY, SERIES])
    )
    if not expected.equals(actual):
        raise ValueError("Header snapshot does not match the series inventory")
    headers.write_parquet(settings.output_dir / "headers.parquet")
    studies, evidence, summary = summarize(headers, inventory)
    studies.write_parquet(settings.output_dir / "studies.parquet")
    evidence.write_parquet(settings.output_dir / "evidence.parquet")
    descriptions = (
        pl.concat(
            [
                headers.select(
                    "split",
                    STUDY,
                    SERIES,
                    pl.lit(field).alias("field"),
                    pl.col(field).alias("value"),
                )
                for field in TEXT_FIELDS
            ]
        )
        .filter(pl.col("value").fill_null("") != "")
        .group_by("split", "field", "value")
        .agg(
            pl.col(STUDY).n_unique().alias("studies"),
            pl.col(SERIES).n_unique().alias("series"),
        )
        .sort(["split", "field", "studies"], descending=[False, False, True])
    )
    descriptions.write_csv(settings.output_dir / "descriptions.csv")
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "versions": {"polars": pl.__version__, "pydicom": pydicom.__version__},
        "cohorts": summary,
        "limitations": [
            "Three headers per series at first/middle/last lexicographic filename positions, or all files if fewer; remaining instances are not inspected.",
            "Explicit side is metadata evidence, not independently verified anatomy. Text inference is conservative and may miss shorthand or contain errors.",
            "Only local train and public example-test inventories are accessible; hidden Kaggle test metadata cannot be audited here.",
            "Reads metadata before PixelData only; no image decoding, model execution, training, tests or GPU access.",
            "DICOM files are not fully hashed to avoid pixel reads; emitted metadata snapshots are fingerprinted.",
        ],
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    source_snapshot = copy_sources([Path(__file__)], settings.output_dir / "source")
    paths = inputs + [
        Path(__file__),
        Path("src/knee_abnormality/cli.py"),
        Path("src/knee_abnormality/data/labels.py"),
        Path("src/knee_abnormality/data/splits.py"),
        Path("src/knee_abnormality/data/sources.py"),
        Path("src/knee_abnormality/provenance.py"),
        Path("uv.lock"),
    ]
    manifest = {
        "created_at": summary["created_at"],
        "inputs_and_sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(settings.output_dir.iterdir())
            if path.is_file()
        ],
        "source_snapshot": source_snapshot,
    }
    (settings.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return summary


def main() -> None:
    try:
        summary = audit(AuditCLI())
    except (OSError, ValueError, SettingsError) as error:
        print(f"Knee laterality audit error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
