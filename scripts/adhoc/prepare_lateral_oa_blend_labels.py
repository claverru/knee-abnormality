import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD, require_unique_ids
from knee_abnormality.kaggle import COMPETITION
from knee_abnormality.provenance import sha256_file
from scripts.tools.prepare_combined_labels import fingerprint
from scripts.tools.prepare_image_blend_labels import base_labels

TARGET = "Lateral OA"
BLEND_WEIGHT = 0.5


class InputFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    bytes: int
    sha256: str


class BlendInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    label_table: str
    column: str
    competition: str
    dataset: str
    dataset_id: int
    dataset_version: int
    updated_at: str
    checked_at: str
    url: str
    license: str
    download_command: str
    files: list[InputFile] = Field(min_length=1)


class PrepareCLI(CLISettings):
    model_config = describe(
        "Build a Lateral OA blend training-label table: a fixed half public-source, half "
        "combined_v1 blend for silver Lateral OA only, preserving gold rows and other targets"
    )

    input_manifest: Path = Field(
        description="Pinned public-source input: name, label table, column and file hashes",
    )
    output_dir: Path = Field(
        description="Fresh directory for the prepared training-label table and report",
    )
    output_manifest: Path = Field(
        description="Fresh manifest to write for the prepared table",
    )
    base_manifest: Path = Field(
        default=Path("configs/labels/combined-label-sources.json"),
        description="Pinned combined_v1 label manifest",
    )


def read_source(source: BlendInput) -> pl.DataFrame:
    frames = []
    for record in source.files:
        if sha256_file(record.path) != record.sha256:
            raise ValueError(f"Pinned {source.name} input changed: {record.path}")
        frames.append(
            pl.read_csv(record.path, infer_schema_length=None).select(
                pl.col(STUDY).cast(pl.String),
                pl.col(source.column).cast(pl.Float64).alias("source_lateral"),
            )
        )
    return pl.concat(frames)


def prepare(settings: PrepareCLI) -> dict:
    if settings.output_dir.exists() or settings.output_manifest.exists():
        raise FileExistsError("Preparation requires fresh output paths")
    manifest = json.loads(settings.base_manifest.read_text())
    source_manifest = json.loads(settings.input_manifest.read_text())
    source = BlendInput.model_validate(source_manifest)
    base, base_record = base_labels(manifest)
    targets = pl.read_csv("data/raw/sample_submission.csv").columns[1:]
    if base.columns != [STUDY, *targets] or TARGET not in targets:
        raise ValueError("Combined label schema differs from official targets")
    require_unique_ids(base, "combined_v1", reject_blank=True)
    official_path = Path("data/raw/train.csv")
    official = pl.read_csv(official_path, infer_schema_length=None).select(
        STUDY, *targets
    )
    require_unique_ids(official, "Official training table", reject_blank=True)
    if set(base[STUDY]) != set(official[STUDY]):
        raise ValueError("Combined labels must cover exactly the official studies")
    observed = official.select(pl.col(targets).is_not_null()).to_numpy()
    if not np.array_equal(observed, np.repeat(observed[:, :1], len(targets), axis=1)):
        raise ValueError("Official gold rows must contain all targets")
    public = read_source(source)
    require_unique_ids(public, source.name, reject_blank=True)
    if not set(public[STUDY]).issubset(set(official[STUDY])):
        raise ValueError(f"{source.name} labels contain unknown studies")
    values = public["source_lateral"].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(
            f"{source.name} Lateral OA values must be finite probabilities"
        )
    aligned = (
        base.sort(STUDY)
        .join(
            official.select(STUDY, pl.col("ACL").is_not_null().alias("is_gold")),
            on=STUDY,
            validate="1:1",
        )
        .join(public, on=STUDY, how="left", validate="1:1")
        .sort(STUDY)
    )
    labels = aligned.with_columns(
        pl.when(~pl.col("is_gold") & pl.col("source_lateral").is_not_null())
        .then(
            (1 - BLEND_WEIGHT) * pl.col(TARGET)
            + BLEND_WEIGHT * pl.col("source_lateral")
        )
        .otherwise(pl.col(TARGET))
        .alias(TARGET)
    ).select(STUDY, *targets)
    output_values = labels.select(targets).to_numpy()
    if (
        not np.isfinite(output_values).all()
        or ((output_values < 0) | (output_values > 1)).any()
    ):
        raise ValueError("Prepared labels must be finite probabilities")
    other_targets = [target for target in targets if target != TARGET]
    if not labels.select(other_targets).equals(aligned.select(other_targets)):
        raise ValueError("Preparation changed other targets")
    gold_ids = aligned.filter(pl.col("is_gold")).select(STUDY)
    if not labels.join(gold_ids, on=STUDY).equals(
        aligned.select(STUDY, *targets).join(gold_ids, on=STUDY)
    ):
        raise ValueError("Preparation changed gold source rows")
    silver = aligned.filter(~pl.col("is_gold"))
    revised = labels.join(silver.select(STUDY), on=STUDY).sort(STUDY)
    old = silver[TARGET].to_numpy()
    new = revised[TARGET].to_numpy()
    settings.output_dir.mkdir(parents=True)
    settings.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "labels.csv"
    labels.write_csv(output_path)
    prepared = {**fingerprint(output_path), "label_table": source.label_table}
    method = (
        f"For silver Lateral OA with a {source.name} value, use half combined_v1 and "
        f"half {source.name}; otherwise retain combined_v1. Preserve every other "
        "target and gold source row. The loader applies official gold overrides and "
        "uses the existing frozen split evaluation targets."
    )
    output_manifest = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "competition": COMPETITION,
        "method": method,
        "preparer": fingerprint(Path(__file__).resolve()),
        "inputs": {
            "combined_v1": base_record,
            "base_manifest": fingerprint(settings.base_manifest),
            "source_manifest": fingerprint(settings.input_manifest),
            "source": source_manifest,
            "official": fingerprint(official_path),
        },
        "target": TARGET,
        "source_weight": BLEND_WEIGHT,
        "rows": labels.height,
        "sources": [
            *manifest["sources"],
            {
                "dataset": f"local/{source.label_table}",
                "dataset_version": 1,
                "files": [prepared],
            },
        ],
    }
    settings.output_manifest.write_text(json.dumps(output_manifest, indent=2) + "\n")
    report = {
        "label_manifest": fingerprint(settings.output_manifest),
        "prepared_labels": prepared,
        "method": method,
        "rows": labels.height,
        "gold_rows": gold_ids.height,
        "silver_rows": silver.height,
        "source_rows": public.height,
        "missing_silver_source_ids": silver.filter(pl.col("source_lateral").is_null())[
            STUDY
        ].to_list(),
        "silver_lateral_oa": {
            "changed_probabilities": int((old != new).sum()),
            "changed_calls": int(
                ((old >= TARGET_THRESHOLD) != (new >= TARGET_THRESHOLD)).sum()
            ),
            "mean_absolute_change": float(np.abs(old - new).mean()),
            "baseline_positive_rate": float((old >= TARGET_THRESHOLD).mean()),
            "candidate_positive_rate": float((new >= TARGET_THRESHOLD).mean()),
        },
        "gold_source_rows_unchanged": True,
        "other_targets_unchanged": other_targets,
    }
    (settings.output_dir / "preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    print(json.dumps(prepare(PrepareCLI()), indent=2))


if __name__ == "__main__":
    main()
