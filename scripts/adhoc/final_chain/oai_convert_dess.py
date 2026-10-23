import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pydicom
import SimpleITK as sitk

sys.path.insert(
    0, "/home/claudio/src/knee-abnormality-sync-free/scripts/adhoc/meniscus_masks"
)
from convert_dess import FINEST_MM, ORIENTATION, body_box  # noqa: E402

CACHE = Path.home() / ".cache/knee-segmenters"
RAW = Path("/workspace/data/raw/oai")
SELECTION = Path("/workspace/data/oai/selection_dess.csv")
LAYOUT = Path("/workspace/data/oai/raw/test_series")


def frame_of(directory: Path) -> str:
    first = next(path for path in sorted(directory.iterdir()) if path.is_file())
    return str(
        pydicom.dcmread(first, stop_before_pixels=True).get("FrameOfReferenceUID")
    )


def convert(job):
    out, row = Path(job[0]), job[1]
    study = f"oai-{row['src_subject_id']}-{row['side']}"
    directory = RAW / row["DOWNLOAD_ALIAS"].removesuffix(".tar.gz")
    record = {"study": study, "fold": -1, "series": directory.name}
    try:
        names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory))
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(names)
        image = reader.Execute()
    except RuntimeError as error:
        return {**record, "error": str(error)[:120]}
    native = [round(step, 3) for step in image.GetSpacing()]
    image = sitk.BinShrink(
        image, [max(1, int(FINEST_MM / step + 0.05)) for step in image.GetSpacing()]
    )
    start, size = body_box(image)
    image = sitk.DICOMOrient(sitk.RegionOfInterest(image, size, start), ORIENTATION)
    sitk.WriteImage(image, str(out / "images" / f"{study}_0000.nii.gz"))
    dess_frame = frame_of(directory)
    others = (
        {path.name: frame_of(path.resolve()) for path in (LAYOUT / study).iterdir()}
        if (LAYOUT / study).exists()
        else {}
    )
    return {
        **record,
        "slices": len(names),
        "native_spacing": native,
        "size": list(image.GetSize()),
        "spacing": [round(step, 3) for step in image.GetSpacing()],
        "frame_matches": {name: frame == dess_frame for name, frame in others.items()},
    }


def main():
    out, workers = CACHE / sys.argv[1], int(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(SELECTION) as stream:
        rows = sorted(
            csv.DictReader(stream), key=lambda row: (row["src_subject_id"], row["side"])
        )
    rows = rows[:limit] if limit else rows
    with ProcessPoolExecutor(workers) as pool:
        records = list(
            pool.map(convert, [(str(out), row) for row in rows], chunksize=1)
        )
    (out / "cases.json").write_text(json.dumps(records, indent=1))
    good = [record for record in records if "size" in record]
    matches = [
        all(record["frame_matches"].values()) and bool(record["frame_matches"])
        for record in good
    ]
    print("knees", len(rows), "converted", len(good), "errors", len(rows) - len(good))
    print(
        "DESS frame of reference equal to every other series of the knee",
        sum(matches),
        "of",
        len(good),
    )
    print(
        "native spacing",
        sorted({tuple(record["native_spacing"]) for record in good})[:4],
        "converted size",
        good[0]["size"] if good else None,
    )


if __name__ == "__main__":
    main()
