import hashlib
import itertools
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pydicom
from pydantic import Field
from pydicom.errors import InvalidDicomError
from scipy.ndimage import zoom
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from knee_abnormality.cli import CLISettings
from knee_abnormality.provenance import sha256_file

STUDY = "StudyInstanceUID"
GEOMETRY = (
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "Rows",
    "Columns",
)
FIELDS = (*GEOMETRY, "PatientID", "FrameOfReferenceUID", "Laterality", "PatientSex")


class AuditCLI(CLISettings):
    output_dir: Path
    headers: Path = Path(
        "outputs/analysis/sharing-disagreement-20260914/study_headers.parquet"
    )
    reports: Path = Path(
        "data/metadata/report_normalization/normalized_reports.parquet"
    )
    split_dir: Path = Path("data/folds/all_5fold_v3")
    workers: int = Field(default=8, ge=1)
    pixel_pairs: int = Field(default=40, ge=0)


def read_geometry(row: dict) -> dict:
    result = {STUDY: row[STUDY], "path": row["path"], "error": None}
    try:
        ds = pydicom.dcmread(row["path"], stop_before_pixels=True, specific_tags=FIELDS)
        for field in FIELDS:
            result[field] = str(ds.get(field, "")).strip()
        geometry = [result[field] for field in GEOMETRY]
        result["geometry_signature"] = (
            hashlib.sha256(json.dumps(geometry).encode()).hexdigest()
            if all(geometry)
            else None
        )
    except (OSError, ValueError, TypeError, AttributeError, InvalidDicomError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def pixels_for_series(row: dict) -> tuple[str, list[dict]]:
    records = []
    for path in sorted(Path(row["path"]).parent.glob("*.dcm")):
        record = {"path": str(path), "pixel_sha256": None, "error": None}
        try:
            ds = pydicom.dcmread(path)
            pixels = np.asarray(ds.pixel_array)
            if pixels.ndim != 2 or np.ptp(pixels.astype(np.float64)) == 0:
                record["error"] = "Non-2D or constant image excluded"
            else:
                payload = np.ascontiguousarray(pixels, dtype="<i4")
                record["pixel_sha256"] = hashlib.sha256(
                    str(payload.shape).encode() + payload.tobytes()
                ).hexdigest()
        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            InvalidDicomError,
        ) as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    return row[STUDY], records


def approximate_pixels(decoded: dict, pairs: list[dict], workers: int) -> list[dict]:
    def describe(item):
        uid, records = item
        features = []
        for record in records:
            if not record["pixel_sha256"]:
                continue
            pixels = pydicom.dcmread(record["path"]).pixel_array.astype(np.float32)
            small = zoom(pixels, (32 / pixels.shape[0], 32 / pixels.shape[1]), order=1)
            low, high = np.quantile(small, (0.01, 0.99))
            values = np.clip(small, low, high).reshape(-1)
            values -= values.mean()
            norm = np.linalg.norm(values)
            if norm:
                features.append(values / norm)
        return uid, np.asarray(features, dtype=np.float32).reshape(-1, 1024)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        descriptors = dict(pool.map(describe, decoded.items()))
    results = []
    for pair in pairs:
        left, right = descriptors[pair["left"]], descriptors[pair["right"]]
        if not len(left) or not len(right):
            continue
        similarities = left @ right.T
        nearest_right = similarities.argmax(axis=1)
        nearest_left = similarities.argmax(axis=0)
        scores = sorted(
            (
                float(similarities[i, j])
                for i, j in enumerate(nearest_right)
                if nearest_left[j] == i
            ),
            reverse=True,
        )
        results.append(
            {
                **pair,
                "mutual_slice_matches": len(scores),
                "max_thumbnail_cosine": scores[0],
                "top5_mutual_mean_cosine": float(np.mean(scores[:5])),
                "mutual_matches_above_099": sum(value >= 0.99 for value in scores),
            }
        )
    return sorted(results, key=lambda row: -row["top5_mutual_mean_cosine"])


def main() -> None:
    args = AuditCLI()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output_dir / Path(__file__).name)
    inputs = (args.headers, args.reports, args.split_dir / "studies.csv")
    manifest = {
        "inputs": {str(path): sha256_file(path) for path in inputs},
        "source_sha256": sha256_file(Path(__file__)),
        "settings": args.model_dump(mode="json"),
        "scope": "Candidate retrieval, not patient identification; folds unchanged",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    table = (
        pl.read_parquet(args.headers)
        .join(pl.read_csv(args.split_dir / "studies.csv"), on=STUDY, validate="1:1")
        .join(
            pl.read_parquet(args.reports).select(STUDY, "unicode_punctuation"),
            on=STUDY,
            validate="1:1",
        )
        .sort(STUDY)
    )
    rows = table.to_dicts()
    by_id = {row[STUDY]: row for row in rows}
    print(f"Reading representative geometry for {len(rows)} studies", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        geometry = list(pool.map(read_geometry, rows))
    pl.DataFrame(geometry).write_parquet(
        args.output_dir / "representative_geometry.parquet"
    )
    pairs = {}

    def add(left: str, right: str, reason: str, similarity: float = 0.0) -> None:
        if left == right:
            return
        left, right = sorted((left, right))
        key = left, right
        row = pairs.setdefault(
            key,
            {
                "left": left,
                "right": right,
                "left_fold": by_id[left]["fold"],
                "right_fold": by_id[right]["fold"],
                "same_report_group": by_id[left]["report_group"]
                == by_id[right]["report_group"],
                "same_scanner_model": by_id[left]["ManufacturerModelName"]
                == by_id[right]["ManufacturerModelName"],
                "reasons": [],
                "report_cosine": 0.0,
            },
        )
        if reason not in row["reasons"]:
            row["reasons"].append(reason)
        row["report_cosine"] = max(row["report_cosine"], similarity)

    repeated = {}
    for field in ("geometry_signature", "PatientID", "FrameOfReferenceUID"):
        groups = defaultdict(list)
        for row in geometry:
            if row.get(field):
                groups[row[field]].append(row[STUDY])
        repeated[field] = sum(len(values) > 1 for values in groups.values())
        for values in groups.values():
            for left, right in itertools.combinations(values, 2):
                add(left, right, field)
    print("Retrieving near-duplicate reports", flush=True)
    texts = table["unicode_punctuation"].fill_null("").to_list()
    features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        dtype=np.float32,
    ).fit_transform(texts)
    neighbors = NearestNeighbors(
        n_neighbors=6, metric="cosine", algorithm="brute", n_jobs=1
    )
    distances, indices = neighbors.fit(features).kneighbors(features)
    for i, (values, matches) in enumerate(zip(distances, indices, strict=True)):
        for distance, j in zip(values, matches, strict=True):
            if (
                i != j
                and min(len(texts[i]), len(texts[j])) >= 200
                and distance <= 0.06
                and rows[i]["report_group"] != rows[j]["report_group"]
            ):
                add(
                    rows[i][STUDY],
                    rows[j][STUDY],
                    "near_report",
                    float(1 - distance),
                )
    candidates = sorted(
        pairs.values(),
        key=lambda row: (
            -int(any(reason != "near_report" for reason in row["reasons"])),
            -int(row["left_fold"] != row["right_fold"]),
            -row["report_cosine"],
            row["left"],
            row["right"],
        ),
    )
    (args.output_dir / "candidate_pairs.json").write_text(
        json.dumps(candidates, indent=2) + "\n"
    )
    selected = candidates[: args.pixel_pairs]
    selected_ids = sorted({row[side] for row in selected for side in ("left", "right")})
    print(
        f"Found {len(candidates)} candidates; decoding one series for {len(selected_ids)} shortlisted studies",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        decoded = dict(
            pool.map(pixels_for_series, [by_id[uid] for uid in selected_ids])
        )
    (args.output_dir / "pixel_inventory.json").write_text(
        json.dumps(decoded, indent=2) + "\n"
    )
    evidence = []
    for row in selected:
        left = {
            item["pixel_sha256"]
            for item in decoded[row["left"]]
            if item["pixel_sha256"]
        }
        right = {
            item["pixel_sha256"]
            for item in decoded[row["right"]]
            if item["pixel_sha256"]
        }
        evidence.append(
            {
                **row,
                "left_distinct_slices": len(left),
                "right_distinct_slices": len(right),
                "matching_distinct_slices": len(left & right),
            }
        )
    (args.output_dir / "pixel_pairs.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    approximate = approximate_pixels(decoded, evidence, args.workers)
    (args.output_dir / "approximate_pixel_pairs.json").write_text(
        json.dumps(approximate, indent=2) + "\n"
    )
    summary = {
        "studies": len(rows),
        "header_errors": sum(row["error"] is not None for row in geometry),
        "repeated_representative_fields": repeated,
        "candidate_pairs": len(candidates),
        "cross_fold_candidate_pairs": sum(
            row["left_fold"] != row["right_fold"] for row in candidates
        ),
        "pixel_pairs_checked": len(evidence),
        "pixel_files_read": sum(len(items) for items in decoded.values()),
        "pixel_errors": dict(
            Counter(
                item["error"]
                for items in decoded.values()
                for item in items
                if item["error"]
            )
        ),
        "pairs_with_matching_slices": [
            row for row in evidence if row["matching_distinct_slices"]
        ],
        "limits": [
            "One representative header and at most one series per study; no exhaustive scan or patient equivalence claim",
            "Exact pixel comparison misses new examinations, resampling, intensity changes and nonmatching selected sequences",
            "Report and geometry matches may reflect templates or scanner protocols",
            "Thumbnail cosine is uncalibrated candidate evidence, without registration or patient labels",
            "Report retrieval checks six nearest neighbors at cosine >= 0.94 and at least 200 normalized characters",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
