import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path.home() / ".cache/knee-segmenters"
CENSUS = (
    ROOT / "outputs/analysis/s/soft-tissue-segmenters/3d-series/census_3d_series.csv"
)
SERIES = ROOT / "data/raw/train_series"
ORIENTATION = "PIR"
FINEST_MM = 0.5
MASK_MM = 2.0
MARGIN_MM = 10.0


def body_box(image):
    factors = [max(1, int(MASK_MM / step + 1e-6)) for step in image.GetSpacing()]
    coarse = sitk.BinShrink(image, factors)
    pieces = sitk.RelabelComponent(
        sitk.ConnectedComponent(sitk.OtsuThreshold(coarse, 0, 1)),
        sortByObjectSize=True,
    )
    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(pieces == 1)
    box = shape.GetBoundingBox(1)
    start, size = [], []
    for axis in range(3):
        margin = round(MARGIN_MM / image.GetSpacing()[axis])
        low = max(box[axis] * factors[axis] - margin, 0)
        high = min(
            (box[axis] + box[axis + 3]) * factors[axis] + margin,
            image.GetSize()[axis],
        )
        start.append(low)
        size.append(high - low)
    return start, size


def convert(job):
    out, row = Path(job[0]), job[1]
    study, series = row["StudyInstanceUID"], row["SeriesInstanceUID"]
    record = {"study": study, "fold": int(row["fold"]), "series": series}
    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(SERIES / study / series))
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    try:
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
    return {
        **record,
        "slices": len(names),
        "native_spacing": native,
        "size": list(image.GetSize()),
        "spacing": [round(step, 3) for step in image.GetSpacing()],
        "extent_mm": [
            round(count * step, 1)
            for count, step in zip(image.GetSize(), image.GetSpacing())
        ],
    }


def main():
    out, workers = CACHE / sys.argv[1], int(sys.argv[2])
    (out / "images").mkdir(parents=True, exist_ok=True)
    chosen = {}
    with open(CENSUS) as census:
        rows = list(csv.DictReader(census))
    for row in rows:
        if row["family"] != "DESS":
            continue
        best = chosen.get(row["StudyInstanceUID"])
        if best is None or int(row["slices"]) > int(best["slices"]):
            chosen[row["StudyInstanceUID"]] = row
    jobs = [(str(out), chosen[study]) for study in sorted(chosen)]
    with ProcessPoolExecutor(workers) as pool:
        records = list(pool.map(convert, jobs, chunksize=2))
    (out / "cases.json").write_text(json.dumps(records, indent=1))
    good = [record for record in records if "size" in record]
    print("studies", len(jobs), "converted", len(good), "errors", len(jobs) - len(good))


if __name__ == "__main__":
    main()
