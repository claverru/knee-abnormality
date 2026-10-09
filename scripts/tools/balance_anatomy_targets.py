import json
import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".cache/knee-segmenters"
SERIES = ROOT / "data/raw/train_series"
BUILDERS = ROOT / "scripts/adhoc/side_routed_session"
SEGMENTER_PYTHON = CACHE / "venv/bin/python"


class Settings(CLISettings):
    model_config = describe(
        "Add anatomy masks until every fold of the split holds the same number, then "
        "rebuild the prepared targets: picks studies of the short folds at random, "
        "converts the sagittal series with most slices of each, segments the new ones "
        "with MRAnnotator on the GPU, and exports the bone volumes, the joint classes "
        "and the measurements for the whole set under data/labels/prepared. The "
        "existing source folder is left untouched; its images and masks are linked"
    )
    source: str = Field(
        default="sample550",
        description="Existing folder under ~/.cache/knee-segmenters to extend",
    )
    name: str | None = Field(
        default=None,
        description="New folder and target name; defaults to balanced<total>",
    )
    per_fold: int | None = Field(
        default=None,
        gt=0,
        description="Masks per fold; defaults to the count of the fold with most",
    )
    split: str = Field(default="all_5fold_scanner_v2", description="Split folder")
    seed: int = Field(default=20261008, description="Seed of the random choice")
    workers: int = Field(default=4, gt=0, description="Conversion processes")
    dry_run: bool = Field(
        default=False, description="Print the choice per fold and stop"
    )


def run(module: str, *arguments: str, **environment: str) -> None:
    subprocess.run(
        [str(SEGMENTER_PYTHON), str(BUILDERS / f"{module}.py"), *arguments],
        cwd=ROOT,
        env={**os.environ, **environment},
        check=True,
    )


def main() -> None:
    settings = Settings()
    cases = json.loads((CACHE / settings.source / "cases.json").read_text())
    usable = [case for case in cases if "series" in case]
    folds = dict(
        pl.read_csv(ROOT / "data/folds" / settings.split / "studies.csv")
        .select("StudyInstanceUID", "fold")
        .iter_rows()
    )
    counts = {fold: 0 for fold in sorted(set(folds.values()))}
    for case in usable:
        counts[folds[case["study"]]] += 1
    per_fold = settings.per_fold or max(counts.values())
    sagittal: dict[str, list[dict]] = {}
    for row in (
        pl.read_csv(ROOT / "data/raw/train_series.csv")
        .filter(pl.col("Anatomical_Plane") == "Sagittal")
        .iter_rows(named=True)
    ):
        sagittal.setdefault(row["StudyInstanceUID"], []).append(row)
    tried = {case["study"] for case in cases}
    rng = np.random.default_rng(settings.seed)
    pools = {}
    for fold in counts:
        pool = sorted(
            study
            for study, value in folds.items()
            if value == fold and study not in tried and study in sagittal
        )
        pools[fold] = [pool[index] for index in rng.permutation(len(pool))]
    missing = {fold: max(per_fold - count, 0) for fold, count in counts.items()}
    total = sum(max(count, per_fold) for count in counts.values())
    name = settings.name or f"balanced{total}"
    print(
        json.dumps(
            {
                "masks_per_fold_now": counts,
                "masks_per_fold_after": {
                    fold: max(count, per_fold) for fold, count in counts.items()
                },
                "new_masks": missing,
                "candidates_left": {fold: len(pool) for fold, pool in pools.items()},
                "training_masks_by_validation_fold": {
                    fold: total - max(count, per_fold) for fold, count in counts.items()
                },
                "name": name,
            },
            indent=2,
        )
    )
    if settings.dry_run:
        return
    target = CACHE / name
    for folder in ("images", "mrannotator"):
        (target / folder).mkdir(parents=True, exist_ok=True)
        for path in (CACHE / settings.source / folder).glob("*.nii.gz"):
            link = target / folder / path.name
            if not link.exists():
                link.symlink_to(path)
    records = list(usable)
    jobs_path, results_path = target / "pending.json", target / "converted.json"
    while any(missing.values()):
        jobs = []
        for fold, wanted in missing.items():
            for study in pools[fold][:wanted]:
                best = max(
                    sagittal[study],
                    key=lambda item, study=study: sum(
                        1
                        for _ in (SERIES / study / item["SeriesInstanceUID"]).glob(
                            "*.dcm"
                        )
                    ),
                )
                jobs.append((study, fold, best))
            pools[fold] = pools[fold][wanted:]
        if not jobs:
            break
        jobs_path.write_text(json.dumps(jobs))
        run(
            "segmenter_convert_cases",
            name,
            str(jobs_path),
            str(results_path),
            str(settings.workers),
        )
        for record in json.loads(results_path.read_text()):
            if "series" in record:
                records.append(record)
                missing[record["fold"]] -= 1
    (target / "cases.json").write_text(json.dumps(records, indent=1))
    prepared = f"anatomy-mrannotator-{name}"
    run("segmenter_mrannotator", name, "mrannotator", STEP="1.0")
    run("segmenter_export_targets", name, prepared)
    run("segmenter_export_joint_targets", name, prepared, f"{prepared}-joints")
    run("mask_measurements", name, f"{prepared}-joints")
    print(f"paths.anatomy_targets: ../data/labels/prepared/{prepared}-joints")


if __name__ == "__main__":
    main()
