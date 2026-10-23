import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pydicom

TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
PIPELINE_FS = re.compile(
    r"(\bfs\b|_fs|fs_|fs\+|\+fs|fat.?sat|fatsat|spir|spair|stir|tirm|(^|[^a-z])we([^a-z]|$)"
    r"|water|dixon|\bsat\b|\bfs$|pdfs|t2fs|fsa|fse fs|ideal|wats|f/s)",
    re.IGNORECASE,
)
PIPELINE_NEGATED = re.compile(r"(non.?fat|non.?fs|no.?fs|no.?fat)", re.IGNORECASE)
STEADY_NAME = re.compile(r"de3d|tfi", re.IGNORECASE)
STEADY_DESCRIPTION = re.compile(r"dess|de3d|trufi", re.IGNORECASE)
NOT_FS = re.compile(r"non\s*fat|no\s*fat|non\s*fs", re.I)
PDFSE = re.compile(r"pdfse", re.I)
STRONG = re.compile(
    r"fat|ftst|\bfs\b|fs$|_fs|fs_|f/s|fsat|stir|spair|water|fsefw", re.I
)
PLANES = ("Sagittal", "Coronal", "Axial")
MINIMUM_IMAGES = 2


def clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def words(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value.split("\\")) if value else ()
    return tuple(str(item) for item in value)


def number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pipeline_contrast(header, description: str | None = None) -> str | None:
    get = lambda key: getattr(header, key, None)
    if description is None:
        description = str(get("SeriesDescription") or "")
    inversion_time = number(get("InversionTime"))
    echo_time = number(get("EchoTime"))
    repetition_time = number(get("RepetitionTime"))
    options = words(get("ScanOptions"))
    inversion = (
        "IR" in words(get("ScanningSequence"))
        or "IR" in words(get("SequenceVariant"))
        or (inversion_time is not None and 50 < inversion_time < 300)
    )
    evidence = (
        bool({"FS", "WE"} & set(options))
        or inversion
        or (
            bool(PIPELINE_FS.search(description))
            and not PIPELINE_NEGATED.search(description)
        )
    )
    steady = bool(
        STEADY_NAME.search(str(get("SequenceName") or ""))
        or STEADY_DESCRIPTION.search(description)
    )
    t1_timing = (
        echo_time is not None
        and repetition_time is not None
        and echo_time < 20
        and repetition_time < 1000
        and not inversion
        and not steady
    )
    if evidence:
        return None if t1_timing else "fluid_fat_suppressed"
    if echo_time is not None or repetition_time is not None or options:
        return "not_fat_suppressed"
    return None


def describe(series: str) -> dict:
    directory = Path(series)
    files = sorted(directory.glob("*.dcm"))
    row = {"directory": series, "n_images": len(files), "reason": None}
    if not files:
        return {**row, "reason": "no images"}
    try:
        header = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True)
    except Exception as error:
        return {**row, "reason": f"unreadable: {type(error).__name__}"}
    get = lambda key: getattr(header, key, None)
    description = str(get("SeriesDescription") or "")
    options = " ".join(words(get("ScanOptions")))
    row |= {
        "description": description,
        "scan_options": options,
        "echo_time": number(get("EchoTime")),
        "repetition_time": number(get("RepetitionTime")),
        "field_strength": str(get("MagneticFieldStrength") or ""),
        "laterality": str(get("Laterality") or get("ImageLaterality") or ""),
        "rows": int(get("Rows") or 0),
        "columns": int(get("Columns") or 0),
        "modality": str(get("Modality") or ""),
        "image_type": " ".join(words(get("ImageType"))),
        "dicom_study_uid": str(get("StudyInstanceUID") or ""),
        "dicom_series_uid": str(get("SeriesInstanceUID") or ""),
    }
    orientation = get("ImageOrientationPatient")
    spacing = get("PixelSpacing")
    if orientation is None or spacing is None:
        return {**row, "reason": "no geometry"}
    values = [float(value) for value in orientation]
    normal = np.abs(np.cross(values[:3], values[3:]))
    row["plane"] = PLANES[int(normal.argmax())]
    row["tilt_degrees"] = float(np.degrees(np.arccos(min(1.0, float(normal.max())))))
    row["pixel_mm"] = float(spacing[0])
    contrast = pipeline_contrast(header)
    row["pipeline_contrast"] = contrast
    named = bool(STRONG.search(description)) and not NOT_FS.search(description)
    respelled = PDFSE.sub("pd fse", description)
    if contrast is None:
        if not description and row["echo_time"] is None:
            return {**row, "reason": "unknown contrast"}
        contrast = "fluid_fat_suppressed" if named else "not_fat_suppressed"
        row["contrast_source"] = "description"
    elif contrast == "fluid_fat_suppressed" and NOT_FS.search(description):
        contrast = "not_fat_suppressed"
        row["contrast_source"] = "corrected"
    elif (
        contrast == "fluid_fat_suppressed"
        and respelled != description
        and pipeline_contrast(header, respelled) != "fluid_fat_suppressed"
    ):
        contrast = "not_fat_suppressed"
        row["contrast_source"] = "corrected"
    else:
        row["contrast_source"] = "pipeline"
    row["fat_suppressed"] = int(contrast == "fluid_fat_suppressed")
    if row["modality"] != "MR":
        return {**row, "reason": "not MR"}
    if len(files) < MINIMUM_IMAGES:
        return {**row, "reason": "single image"}
    return row


def main() -> None:
    output = Path(sys.argv[1])
    sources = [Path(value) for value in sys.argv[2:]]
    series = sorted(
        str(path)
        for source in sources
        for path in source.glob("*/*/study_*/*")
        if path.is_dir()
    )
    with ProcessPoolExecutor(60) as pool:
        rows = list(pool.map(describe, series, chunksize=64))
    frame = pl.DataFrame(rows, infer_schema_length=None)
    parts = [Path(value).parts[-5:] for value in frame["directory"]]
    frame = frame.with_columns(
        pl.Series("part", [part for part, _, _, _, _ in parts]),
        pl.Series("patient", [clean(patient) for _, _, patient, _, _ in parts]),
        pl.Series(
            "StudyInstanceUID",
            [
                f"fastmri-{clean(patient)}-{clean(study)}"
                for _, _, patient, study, _ in parts
            ],
        ),
    )
    frame = frame.with_columns(
        pl.Series(
            "SeriesInstanceUID",
            [
                f"{study}-{clean(name)}"
                for study, (_, _, _, _, name) in zip(frame["StudyInstanceUID"], parts)
            ],
        )
    )
    frame = (
        frame.sort("n_images", "part", descending=True)
        .with_columns(copy=pl.int_range(pl.len()).over("SeriesInstanceUID"))
        .with_columns(
            reason=pl.when(pl.col("copy") > 0)
            .then(pl.lit("fuller copy in the other part"))
            .otherwise(pl.col("reason"))
        )
        .drop("copy")
        .sort("directory")
    )
    if len(sys.argv) > 2 and Path("/root/fastmri_rejected.csv").is_file():
        rejected = pl.read_csv("/root/fastmri_rejected.csv")[
            "StudyInstanceUID"
        ].to_list()
        frame = frame.with_columns(
            reason=pl.when(
                pl.col("reason").is_null() & pl.col("StudyInstanceUID").is_in(rejected)
            )
            .then(pl.lit("study rejected by the pipeline loader"))
            .otherwise(pl.col("reason"))
        )
    raw = output / "raw"
    (raw / "test_series").mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output / "series_catalog.parquet")
    kept = frame.filter(pl.col("reason").is_null())
    for study, name, directory in kept.select(
        "StudyInstanceUID", "SeriesInstanceUID", "directory"
    ).iter_rows():
        link = raw / "test_series" / study / name
        link.parent.mkdir(exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(directory)
    table = kept.select(
        "StudyInstanceUID",
        "SeriesInstanceUID",
        pl.col("fat_suppressed").alias("Fluid_Sensitive"),
        pl.col("fat_suppressed").alias("Fat_Suppression"),
        pl.col("plane").alias("Anatomical_Plane"),
    )
    table.write_csv(raw / "test_series.csv")
    table.write_csv(raw / "train_series.csv")
    if not (raw / "train_series").exists():
        (raw / "train_series").symlink_to("test_series")
    studies = kept.select("StudyInstanceUID").unique().sort("StudyInstanceUID")
    studies.with_columns(pl.lit(0.5).alias(target) for target in TARGETS).write_csv(
        raw / "sample_submission.csv"
    )
    kept.group_by("StudyInstanceUID").agg(
        pl.col("part").first(), pl.col("patient").first(), pl.len().alias("n_series")
    ).sort("StudyInstanceUID").write_csv(output / "studies.csv")
    key = pl.col("plane").str.slice(0, 3) + pl.col("fat_suppressed").cast(pl.String)
    for part, group in [
        ("all", kept),
        *sorted(kept.partition_by("part", as_dict=True).items()),
    ]:
        name = part if isinstance(part, str) else part[0]
        sets = (
            group.with_columns(k=key)
            .group_by("StudyInstanceUID")
            .agg(pl.col("k").unique().sort())
            .with_columns(pl.col("k").list.join(" "))
        )
        print(
            f"[{name}] series {group.height} studies {group['StudyInstanceUID'].n_unique()} "
            f"patients {group['patient'].n_unique()} images {int(group['n_images'].sum())}"
        )
        print(
            "  types",
            group.group_by("plane", "fat_suppressed")
            .len()
            .sort("plane", "fat_suppressed")
            .rows(),
        )
        print("  sets", sets["k"].value_counts(sort=True).head(6).rows())
        print(
            "  field",
            group.group_by("StudyInstanceUID")
            .agg(pl.col("field_strength").first())["field_strength"]
            .value_counts(sort=True)
            .head(4)
            .rows(),
        )
    print(
        "all series", frame.height, "all studies", frame["StudyInstanceUID"].n_unique()
    )
    print(
        "excluded",
        frame.filter(pl.col("reason").is_not_null())
        .group_by("part", "reason")
        .len()
        .rows(),
    )
    print("contrast source", kept["contrast_source"].value_counts(sort=True).rows())
    print(
        "corrected names",
        kept.filter(pl.col("contrast_source") != "pipeline")
        .group_by("contrast_source", "description")
        .len()
        .sort("len", descending=True)
        .head(14)
        .rows(),
    )
    print(
        "short series",
        frame.filter(pl.col("n_images") < 12)["n_images"]
        .value_counts()
        .sort("n_images")
        .rows(),
    )
    print(
        "patients in both parts",
        kept.group_by("patient").agg(pl.col("part").n_unique())["part"].gt(1).sum(),
    )


if __name__ == "__main__":
    main()
