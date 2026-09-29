import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.config import LabelsConfig, load_config
from knee_abnormality.data.labels import load_fold_labels
from knee_abnormality.data.splits import STUDY, require_unique_ids
from knee_abnormality.provenance import sha256_file


class PrepareCLI(CLISettings):
    config: Path = Path("configs/training.yaml")
    comparison_manifest: Path = Path(
        "configs/labels/public-label-comparison-sources.json"
    )
    output_dir: Path = Path("data/labels/prepared/tonylin-soft-v3")
    output_manifest: Path = Path("configs/labels/tonylin-label-sources.json")


def fingerprint(path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare(settings):
    if settings.output_dir.exists() or settings.output_manifest.exists():
        raise FileExistsError("Preparation requires fresh output paths")
    audited = json.loads(settings.comparison_manifest.read_text())
    table = next(
        table for table in audited["tables"] if table["name"] == "tonylin_soft"
    )
    source = next(
        source
        for source in audited["sources"]
        if source["dataset"] == "tonylin1026/rsna-knee-refined-soft-labels"
        and source["dataset_version"] == 3
    )
    source_record = next(
        item for item in source["files"] if item["path"] == table["path"]
    )
    metadata_record = next(
        item for item in source["files"] if item["path"].endswith("/dataset-view.json")
    )
    official_records = [
        item for item in audited["inputs"] if item["path"].startswith("data/raw/")
    ]
    for record in [source_record, metadata_record, *official_records]:
        path = Path(record["path"])
        if (
            path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"Pinned input changed: {path}")
    targets = pl.read_csv("data/raw/sample_submission.csv").columns[1:]
    raw = pl.read_csv(table["path"], infer_schema_length=None)
    labels = raw.select(
        STUDY, *[pl.col(target + "_soft").alias(target) for target in targets]
    ).sort(STUDY)
    require_unique_ids(labels, "Tonylin soft labels", reject_blank=True)
    values = labels.select(targets).to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Tonylin soft targets must be finite probabilities")
    official = pl.read_csv("data/raw/train.csv", columns=[STUDY])
    if (
        labels.height != official.height
        or labels.join(official, on=STUDY, how="anti").height
    ):
        raise ValueError(
            "Tonylin study membership must match the official training set"
        )
    settings.output_dir.mkdir(parents=True)
    output_path = settings.output_dir / "labels.csv"
    labels.write_csv(output_path)
    prepared_record = {
        **fingerprint(output_path),
        "label_table": "tonylin_soft",
        "derived_from": source_record,
        "target_column_mapping": {target + "_soft": target for target in targets},
    }
    manifest = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "competition": "rsna-knee-abnormality-detection",
        "method": "Select the twelve release-3 *_soft columns, rename to competition targets, sort by StudyInstanceUID, and preserve values. Publisher fold assignments and hard-label columns are not used.",
        "preparer": fingerprint(Path(__file__).resolve()),
        "comparison_manifest": fingerprint(settings.comparison_manifest),
        "polars_version": pl.__version__,
        "inputs": official_records,
        "rows": labels.height,
        "sources": [
            {
                "dataset": source["dataset"],
                "dataset_version": source["dataset_version"],
                "url": source["url"],
                "download_url": source["download_url"],
                "files": [source_record, metadata_record, prepared_record],
            }
        ],
    }
    settings.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    settings.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    config = load_config(settings.config, label_manifest=settings.output_manifest)
    config = config.model_copy(update={"labels": LabelsConfig(source="tonylin_soft")})
    fold_summaries = []
    for fold in config.validation.folds:
        loaded = load_fold_labels(config, fold)
        fold_summaries.append(
            {
                "fold": fold,
                "training_studies": loaded.training.height,
                "training_gold": int(loaded.training["is_gold"].sum()),
                "validation_studies": loaded.validation.height,
                "validation_gold": int(loaded.validation["is_gold"].sum()),
                "frozen_evaluation_targets_sha256": loaded.protocol[
                    "evaluation_targets_sha256"
                ],
                "provenance": loaded.provenance,
            }
        )
    report = {
        "config": fingerprint(settings.config),
        "label_manifest": fingerprint(settings.output_manifest),
        "prepared_labels": prepared_record,
        "split": config.validation.split,
        "folds": fold_summaries,
    }
    (settings.output_dir / "preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    try:
        report = prepare(PrepareCLI())
    except (
        OSError,
        StopIteration,
        TypeError,
        ValueError,
        yaml.YAMLError,
        SettingsError,
    ) as error:
        print(f"Tonylin label preparation error: {error!r}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
