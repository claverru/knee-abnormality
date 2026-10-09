import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[3]
OUT = Path("/home/claudio/.cache/knee-segmenters") / sys.argv[1]
JOBS, RESULTS, WORKERS = Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])


def convert(job):
    study, fold, item = job
    target = OUT / "images" / f"{study}_0000.nii.gz"
    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(ROOT / "data/raw/train_series" / study / item["SeriesInstanceUID"]))
    if not target.exists():
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(names)
        try:
            sitk.WriteImage(reader.Execute(), str(target))
        except RuntimeError as error:
            return {"study": study, "fold": fold, "error": str(error)[:120]}
    return {
        "study": study, "fold": fold, "series": item["SeriesInstanceUID"], "fluid": int(item["Fluid_Sensitive"]),
        "fat_suppressed": int(item["Fat_Suppression"]), "slices": len(names),
    }


def main():
    jobs = [tuple(job) for job in json.loads(JOBS.read_text())]
    with ProcessPoolExecutor(WORKERS) as pool:
        records = list(pool.map(convert, jobs, chunksize=4))
    RESULTS.write_text(json.dumps(records, indent=1))
    print("converted", sum("series" in r for r in records), "errors", sum("error" in r for r in records))


if __name__ == "__main__":
    main()
