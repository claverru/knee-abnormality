import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import SimpleITK as sitk

OUT = Path("/home/claudio/.cache/knee-segmenters") / sys.argv[1]
WORKERS = int(sys.argv[2])
SERIES = Path("/workspace/data/fastmri/raw/train_series")


def convert(job):
    study, fold, item = job
    target = OUT / "images" / f"{study}_0000.nii.gz"
    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(SERIES / study / item["SeriesInstanceUID"])
    )
    if not target.exists():
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(names)
        try:
            sitk.WriteImage(reader.Execute(), str(target))
        except RuntimeError as error:
            return {"study": study, "fold": fold, "error": str(error)[:120]}
    return {
        "study": study,
        "fold": fold,
        "series": item["SeriesInstanceUID"],
        "fluid": int(item["Fluid_Sensitive"]),
        "fat_suppressed": int(item["Fat_Suppression"]),
        "slices": len(names),
        "scanner": item["scanner"],
    }


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    jobs = [tuple(job) for job in json.loads((OUT / "pending.json").read_text())]
    with ProcessPoolExecutor(WORKERS) as pool:
        records = list(pool.map(convert, jobs, chunksize=4))
    (OUT / "cases.json").write_text(json.dumps(records, indent=1))
    print(
        "converted",
        sum("series" in r for r in records),
        "errors",
        sum("error" in r for r in records),
    )


if __name__ == "__main__":
    main()
