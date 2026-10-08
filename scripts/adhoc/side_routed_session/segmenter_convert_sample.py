import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[3]
OUT = Path("/home/claudio/.cache/knee-segmenters") / sys.argv[1]
TRAINING, VALIDATION = int(sys.argv[2]), int(sys.argv[3])
SAMPLE_ROWS = (532, 200, 370, 374, 45, 5)
SEED = 20261007


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
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    folds = {row["StudyInstanceUID"]: int(row["fold"]) for row in csv.DictReader(open(ROOT / "data/folds/all_5fold_scanner_v2/studies.csv"))}
    series = {}
    for row in csv.DictReader(open(ROOT / "data/raw/train_series.csv")):
        if row["Anatomical_Plane"] == "Sagittal":
            series.setdefault(row["StudyInstanceUID"], []).append(row)
    validation = sorted(study for study, fold in folds.items() if fold == 1)
    training = sorted(study for study, fold in folds.items() if fold != 1)
    rng = np.random.default_rng(SEED)
    fixed = [validation[row] for row in SAMPLE_ROWS]
    rest = [study for study in validation if study not in fixed]
    chosen = fixed + [rest[i] for i in rng.choice(len(rest), VALIDATION - len(fixed), replace=False)]
    chosen += [training[i] for i in rng.choice(len(training), TRAINING, replace=False)]
    jobs = []
    for study in chosen:
        candidates = series.get(study, [])
        if not candidates:
            continue
        best = max(candidates, key=lambda item: sum(1 for _ in (ROOT / "data/raw/train_series" / study / item["SeriesInstanceUID"]).glob("*.dcm")))
        jobs.append((study, folds[study], best))
    with ProcessPoolExecutor(4) as pool:
        records = list(pool.map(convert, jobs, chunksize=4))
    (OUT / "cases.json").write_text(json.dumps(records, indent=1))
    good = [r for r in records if "series" in r]
    print(
        "studies", len(chosen), "converted", len(good), "errors", len(records) - len(good),
        "| validation", sum(r["fold"] == 1 for r in good), "training", sum(r["fold"] != 1 for r in good),
        "| fat-suppressed", sum(r["fat_suppressed"] for r in good), "T1/PD", sum(1 - r["fat_suppressed"] for r in good),
    )


if __name__ == "__main__":
    main()
