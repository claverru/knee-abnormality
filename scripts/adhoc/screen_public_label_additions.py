import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from pydantic_settings import SettingsError
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD
from knee_abnormality.provenance import copy_sources, sha256_file

PUBLIC = Path("data/labels/public")
SNAPSHOT = "downloaded-20260922"
TABLES = {
    "achelijndiamantidis_soft": (
        "achelijndiamantidis/knee-phase1-artifacts",
        "soft_labels.parquet",
        "",
    ),
    "davidadekanmi_synthesized": (
        "davidadekanmi/pseudo-labels",
        "synthesized_labels.csv",
        "",
    ),
    "davidadekanmi_report_v2": (
        "davidadekanmi/pseudo-labels",
        "report_labels_v2 (1).csv",
        "",
    ),
    "hosseini_jev_v2": (
        "hosseini7798/rsna-knee-label-extraction-jev",
        "Label_Extraction_Jev_v2.csv",
        "_1",
    ),
    "lvthanhxun_teacher_fused": (
        "lvthanhxun/rsna-knee-v9-teacher",
        "official_teacher_fused_v9.csv",
        "__y",
    ),
    "nekkon_nli": (
        "nekkon/rsna-knee-nli-entailment-scores",
        "nli_entailment_scores.csv",
        "",
    ),
    "nartaa_v0raw": ("nartaa/rsna-knee-hpo-assets", "labels_v0raw.parquet", ""),
    "nartaa_v1_blend": ("nartaa/rsna-knee-hpo-assets", "labels_v1_blend.parquet", ""),
    "nartaa_v1_gemlow": ("nartaa/rsna-knee-hpo-assets", "labels_v1_gemlow.parquet", ""),
    "nartaa_v1_perlabel": (
        "nartaa/rsna-knee-hpo-assets",
        "labels_v1_perlabel.parquet",
        "",
    ),
    "pearsejim01_labels_v3": ("pearsejim01/rsna-2026-eda", "labels_v3.parquet", ""),
    "pearsejim01_frontier": (
        "pearsejim01/rsna-2026-eda",
        "weak_labels_frontier.parquet",
        "",
    ),
    "pearsejim01_weak_v2": ("pearsejim01/rsna-2026-eda", "weak_labels_v2.parquet", ""),
}
STEVEN_V4 = PUBLIC / "stevenleehans/version-6/llm_labels_v4_blend.csv"
PILKWANG_V2 = PUBLIC / "pilkwang/version-1/report_labels_v2.csv"
COMBINED_V1 = Path("data/labels/prepared/combined-v1/labels.csv")


class ScreenCLI(CLISettings):
    output_dir: Path


def read(path: Path, targets: list[str], suffix: str = "") -> pl.DataFrame:
    frame = (
        pl.read_parquet(path)
        if path.suffix == ".parquet"
        else pl.read_csv(path, infer_schema_length=0)
    )
    return frame.select(
        pl.col(STUDY).cast(pl.Utf8),
        *[pl.col(t + suffix).cast(pl.Float64, strict=False).alias(t) for t in targets],
    )


def macro(truth: np.ndarray, scores: np.ndarray) -> float | None:
    values = []
    for column in range(truth.shape[1]):
        known = np.isfinite(scores[:, column]) & np.isfinite(truth[:, column])
        positive = truth[known, column] >= TARGET_THRESHOLD
        if positive.any() and (~positive).any():
            values.append(roc_auc_score(positive, scores[known, column]))
    return float(np.mean(values)) if values else None


def screen(settings: ScreenCLI) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    targets = json.loads(
        Path("data/folds/all_5fold_scanner_v1/protocol.json").read_text()
    )["targets"]
    train = pl.read_csv("data/raw/train.csv", infer_schema_length=None)
    gold = train.filter(pl.col(targets[0]).is_not_null()).select(
        STUDY, *[pl.col(t).cast(pl.Float64) for t in targets]
    )
    gold_ids = gold[STUDY].to_list()
    steven = read(STEVEN_V4, targets)
    pilkwang = read(PILKWANG_V2, targets)
    combined = read(COMBINED_V1, targets)
    files = {
        "steven_v4": STEVEN_V4,
        "pilkwang_v2": PILKWANG_V2,
        "combined_v1": COMBINED_V1,
    }
    rows = []
    for name, (dataset, file, suffix) in TABLES.items():
        path = PUBLIC / dataset / SNAPSHOT / file
        files[name] = path
        table = read(path, targets, suffix)
        on_gold = gold.join(table, on=STUDY, suffix="_table").join(
            steven, on=STUDY, suffix="_steven"
        )
        truth = on_gold.select(targets).to_numpy()
        scores = on_gold.select([f"{t}_table" for t in targets]).to_numpy()
        reference = on_gold.select([f"{t}_steven" for t in targets]).to_numpy()
        finite = np.isfinite(scores)
        silver = (
            table.filter(~pl.col(STUDY).is_in(gold_ids))
            .join(combined, on=STUDY, suffix="_combined")
            .join(pilkwang, on=STUDY, suffix="_pilkwang")
        )
        values = silver.select(targets).to_numpy()
        against_combined = silver.select([f"{t}_combined" for t in targets]).to_numpy()
        against_pilkwang = silver.select([f"{t}_pilkwang" for t in targets]).to_numpy()
        both = np.isfinite(values) & np.isfinite(against_combined)
        both_pilkwang = np.isfinite(values) & np.isfinite(against_pilkwang)
        rows.append(
            {
                "table": name,
                "dataset": dataset,
                "file": file,
                "studies": table.height,
                "gold_rows": on_gold.height,
                "gold_cells_equal_to_gold": float(
                    np.isclose(scores[finite], truth[finite]).mean()
                )
                if finite.any()
                else None,
                "gold_macro_auc": macro(truth, scores),
                "steven_v4_same_rows": macro(
                    truth, np.where(finite, reference, np.nan)
                ),
                "silver_cells_crossing_combined": float(
                    (
                        (values >= TARGET_THRESHOLD)
                        != (against_combined >= TARGET_THRESHOLD)
                    )[both].mean()
                ),
                "silver_cells_equal_to_pilkwang": float(
                    np.isclose(
                        values[both_pilkwang], against_pilkwang[both_pilkwang]
                    ).mean()
                )
                if both_pilkwang.any()
                else None,
            }
        )
    result = pl.DataFrame(rows).sort("gold_macro_auc", descending=True, nulls_last=True)
    result.write_csv(settings.output_dir / "screen.csv")
    manifest = {
        "inputs": {str(path): sha256_file(path) for path in files.values()},
        "sources": copy_sources((Path(__file__),), settings.output_dir / "source"),
    }
    (settings.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=260, float_precision=4):
        print(result.drop("dataset", "file"))


def main() -> None:
    try:
        screen(ScreenCLI())
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Label addition screen error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
