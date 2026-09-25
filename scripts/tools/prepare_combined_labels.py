import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD, require_unique_ids
from knee_abnormality.kaggle import COMPETITION
from knee_abnormality.provenance import sha256_file

LABEL_TABLE = "combined_v1"
BASE_TABLE = "steven_v4"
CONSENSUS_TARGETS = ("ACL", "MCL", "PF OA", "Fracture")
SNAPSHOT = Path("data/labels/public/{owner}/{slug}/downloaded-20260921/{name}")
PINNED = {
    "pilkwang_v2": ("data/labels/public/pilkwang/version-1/report_labels_v2.csv", ""),
    "laymond_filled": (
        "data/labels/public/laymond/version-4/weak_labels_v2_filled.csv",
        "__confidence",
    ),
    "zhen_v7": ("data/labels/public/zhen931/version-1/report_labels_v7.csv", ""),
}
SNAPSHOTS = {
    "yunus_llm": ("yunusgmsoy", "rsna-knee-llm-report-labels", "yunus_llm_labels.csv"),
    "jev_text": ("riadmohamed42", "jev-knee-labels", "FINAL2_labels_scores.csv"),
    "dreaddev_soft": ("dreaddevelopment", "rsna-knee-labels", "labels_llm_soft.csv"),
    "ramit_luna": ("ramitnandi", "rsna-knee-labels-gpt-luna", "report_labels.csv"),
    "josh_luna": (
        "joshuaziel",
        "rsna-knee-mri-report-labels-gpt-5-6-luna",
        "all_raw_predictions.json",
    ),
    "vmohit_states": ("vmohitrao", "rsna-knee-report-labels", "label_states.csv"),
}


class PrepareCLI(CLISettings):
    model_config = describe(
        "Build the combined_v1 label table (Steven v4 plus a nine-source consensus "
        "mean for ACL, MCL, PF OA and Fracture) and its pinned manifest"
    )

    base_manifest: Path = Field(
        default=Path("configs/labels/public-label-sources.json"),
        description="Pinned Steven v4 label manifest",
    )
    comparison_manifest: Path = Field(
        default=Path("configs/labels/public-label-comparison-sources.json"),
        description="Pinned manifest of the consensus label sources",
    )
    output_dir: Path = Field(
        default=Path("data/labels/prepared/combined-v1"),
        description="Directory for the prepared label table",
    )
    output_manifest: Path = Field(
        default=Path("configs/labels/combined-label-sources.json"),
        description="Manifest to write for the prepared table",
    )


def fingerprint(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def pinned_record(manifest: dict, path: str) -> dict:
    records = [
        item
        for source in manifest["sources"]
        for item in source["files"]
        if item["path"] == path
    ]
    if len(records) != 1:
        raise ValueError(f"Expected one pinned record for {path}")
    record = records[0]
    if sha256_file(Path(path)) != record["sha256"]:
        raise ValueError(f"Pinned input changed: {path}")
    return record


def standard(frame: pl.DataFrame, targets: list[str], suffix: str = "") -> pl.DataFrame:
    return frame.select(
        pl.col(STUDY).cast(pl.String),
        *[pl.col(t + suffix).cast(pl.Float64).alias(t) for t in targets],
    )


def read_snapshot(name: str, path: Path, targets: list[str]) -> pl.DataFrame:
    if name == "ramit_luna":
        classes = {t.replace(" ", "_").replace("'", ""): t for t in targets}
        long = pl.read_csv(path, infer_schema_length=None).with_columns(
            pl.col("CLASS").replace(classes)
        )
        return standard(
            long.pivot(
                on="CLASS", index=STUDY, values="SCORE", aggregate_function="mean"
            ),
            targets,
        )
    if name == "josh_luna":
        records = pl.DataFrame(json.loads(path.read_text()))
        top = records["confidence"].max()
        findings = {t.replace(" ", "_").replace("'s", ""): t for t in targets}
        wide = records.with_columns(
            pl.col("Finding").replace(findings),
            pl.when(pl.col("annotation") == 1)
            .then(0.5 + pl.col("confidence") / (2 * top))
            .otherwise(0.5 - pl.col("confidence") / (2 * top))
            .alias("score"),
        ).pivot(on="Finding", index=STUDY, values="score", aggregate_function="mean")
        return standard(wide, targets)
    if name == "vmohit_states":
        states = pl.read_csv(path, infer_schema_length=None)
        return states.select(
            pl.col(STUDY).cast(pl.String),
            *[
                pl.when(pl.col(t) == "P")
                .then(1.0)
                .when(pl.col(t).is_in(["N", "B"]))
                .then(0.0)
                .otherwise(None)
                .alias(t)
                for t in targets
            ],
        )
    return standard(pl.read_csv(path, infer_schema_length=None), targets)


def consensus_sources(comparison: dict, targets: list[str]) -> tuple[dict, dict]:
    inputs, sources = {}, {}
    for name, (path, suffix) in PINNED.items():
        inputs[name] = pinned_record(comparison, path)
        sources[name] = standard(
            pl.read_csv(path, infer_schema_length=None), targets, suffix
        )
    for name, (owner, slug, file) in SNAPSHOTS.items():
        path = Path(str(SNAPSHOT).format(owner=owner, slug=slug, name=file))
        inputs[name] = fingerprint(path)
        sources[name] = read_snapshot(name, path, targets)
    for name, frame in sources.items():
        require_unique_ids(frame, name)
        values = frame.select(targets).to_numpy()
        finite = np.isfinite(values)
        if ((values[finite] < 0) | (values[finite] > 1)).any():
            raise ValueError(f"{name} values must be probabilities")
    return inputs, sources


def prepare(settings: PrepareCLI) -> dict:
    if settings.output_dir.exists() or settings.output_manifest.exists():
        raise FileExistsError("Preparation requires fresh output paths")
    base_manifest = json.loads(settings.base_manifest.read_text())
    comparison = json.loads(settings.comparison_manifest.read_text())
    targets = pl.read_csv("data/raw/sample_submission.csv").columns[1:]
    official = pl.read_csv("data/raw/train.csv", columns=[STUDY])
    base_source = next(
        source
        for source in base_manifest["sources"]
        if any(item.get("label_table") == BASE_TABLE for item in source["files"])
    )
    base_record = next(
        item for item in base_source["files"] if item.get("label_table") == BASE_TABLE
    )
    if sha256_file(Path(base_record["path"])) != base_record["sha256"]:
        raise ValueError(f"Pinned input changed: {base_record['path']}")
    base = standard(pl.read_csv(base_record["path"], infer_schema_length=None), targets)
    inputs, sources = consensus_sources(comparison, targets)
    inputs = {BASE_TABLE: base_record, **inputs}

    stacked = official.sort(STUDY)
    for name, frame in sources.items():
        stacked = stacked.join(
            frame.select(
                STUDY, *[pl.col(t).alias(f"{name}|{t}") for t in CONSENSUS_TARGETS]
            ),
            on=STUDY,
            how="left",
            validate="1:1",
        )
    consensus = stacked.select(
        STUDY,
        *[
            pl.mean_horizontal([f"{name}|{t}" for name in sources]).alias(t)
            for t in CONSENSUS_TARGETS
        ],
        *[
            pl.sum_horizontal(
                [pl.col(f"{name}|{t}").is_not_null() for name in sources]
            ).alias(f"sources|{t}")
            for t in CONSENSUS_TARGETS
        ],
    )
    if (
        consensus.select(pl.min_horizontal([f"sources|{t}" for t in CONSENSUS_TARGETS]))
        .min()
        .item()
        < len(sources) // 2
    ):
        raise ValueError("Some studies lack a majority of consensus sources")
    labels = (
        official.join(
            base.drop(CONSENSUS_TARGETS), on=STUDY, how="left", validate="1:1"
        )
        .join(
            consensus.select(STUDY, *CONSENSUS_TARGETS),
            on=STUDY,
            how="left",
            validate="1:1",
        )
        .select(STUDY, *targets)
        .sort(STUDY)
    )
    require_unique_ids(labels, LABEL_TABLE, reject_blank=True)
    values = labels.select(targets).to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Combined targets must be finite probabilities")

    settings.output_dir.mkdir(parents=True)
    output_path = settings.output_dir / "labels.csv"
    labels.write_csv(output_path)
    prepared = {**fingerprint(output_path), "label_table": LABEL_TABLE}
    manifest = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "competition": COMPETITION,
        "method": (
            f"Steven v4 for every target except {', '.join(CONSENSUS_TARGETS)}, which take "
            "the per-study mean of the available probabilities from nine independent "
            "non-Steven report extractors. Josh gpt-5.6-luna is scored as 0.5 plus or minus "
            "half its relative confidence; Vmohit states map P to 1, N and B to 0 and leave "
            "U and M missing. Official gold labels still override these values in training "
            "and evaluation."
        ),
        "preparer": fingerprint(Path(__file__).resolve()),
        "consensus_targets": list(CONSENSUS_TARGETS),
        "consensus_sources": list(sources),
        "inputs": inputs,
        "rows": labels.height,
        "sources": [
            base_source,
            {
                "dataset": "local/combined-labels",
                "dataset_version": 1,
                "files": [prepared],
            },
        ],
    }
    settings.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    silver = labels.join(
        pl.read_csv("data/raw/train.csv", infer_schema_length=None)
        .filter(pl.col("ACL").is_null())
        .select(STUDY),
        on=STUDY,
    )
    base_silver = base.join(silver.select(STUDY), on=STUDY)
    report = {
        "label_manifest": fingerprint(settings.output_manifest),
        "prepared_labels": prepared,
        "silver_positive_rate": {
            t: {
                BASE_TABLE: float((base_silver[t] >= TARGET_THRESHOLD).mean()),
                LABEL_TABLE: float((silver[t] >= TARGET_THRESHOLD).mean()),
            }
            for t in CONSENSUS_TARGETS
        },
        "minimum_sources_per_study": {
            t: int(consensus[f"sources|{t}"].min()) for t in CONSENSUS_TARGETS
        },
    }
    (settings.output_dir / "preparation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    report = prepare(PrepareCLI())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
