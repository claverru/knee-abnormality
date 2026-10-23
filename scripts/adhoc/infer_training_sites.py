from pathlib import Path

import polars as pl
from langdetect import DetectorFactory, detect_langs
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY

HEADERS = Path("outputs/analysis/s/sharing-disagreement-20260914/study_headers.parquet")
SCRIPT_CHECKS = {
    "mongolian_letters": r"[өүӨҮ]",
    "cyrillic": r"[Ѐ-ӿ]",
    "thai_script": r"[฀-๿]",
    "cjk": r"[一-鿿]",
    "italian_knee_terms": r"(?i)\b(ginocchio|versamento articolare|legamento crociato)\b",
}


class TrainingSitesCLI(CLISettings):
    model_config = describe(
        "Identify the report language of every training study, check for the languages of the "
        "officially listed contributing sites, and tabulate language by vendor, field strength and "
        "scanner model as the basis for inferring training sites"
    )

    raw_dir: Path = Field(
        default=Path("data/raw"), description="Official competition files"
    )
    folds_dir: Path = Field(
        default=Path("data/folds/all_5fold_scanner_v2"),
        description="Split whose folds are tabulated",
    )
    output_dir: Path = Field(description="Fresh output directory")
    min_probability: float = Field(
        default=0.8,
        gt=0,
        lt=1,
        description="Language probability below which a report is uncertain",
    )


def language(report: str, min_probability: float) -> str:
    try:
        best = detect_langs(report[:1500])[0]
    except Exception:
        return "none"
    return best.lang if best.prob >= min_probability else "uncertain"


def run(settings: TrainingSitesCLI) -> None:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory already exists: {settings.output_dir}")
    DetectorFactory.seed = 0
    reports = pl.read_csv(
        settings.raw_dir / "train.csv", infer_schema_length=None
    ).select(STUDY, pl.col("Report").fill_null(""))
    headers = (
        pl.read_parquet(HEADERS)
        .unique(STUDY, keep="first")
        .select(
            STUDY,
            "vendor",
            "ManufacturerModelName",
            pl.col("MagneticFieldStrength")
            .cast(pl.Float64, strict=False)
            .alias("field"),
        )
    )
    folds = pl.read_csv(settings.folds_dir / "studies.csv").select(
        STUDY, "fold", "is_gold"
    )
    studies = (
        reports.with_columns(
            pl.Series(
                "language",
                [language(r, settings.min_probability) for r in reports["Report"]],
            ),
            *[
                pl.col("Report").str.contains(pattern).alias(name)
                for name, pattern in SCRIPT_CHECKS.items()
            ],
        )
        .join(headers, on=STUDY, how="left")
        .join(folds, on=STUDY, how="left")
        .with_columns(
            pl.when(pl.col("field") >= 2)
            .then(pl.lit("3T"))
            .when(pl.col("field") >= 1.4)
            .then(pl.lit("1.5T"))
            .when(pl.col("field").is_null())
            .then(pl.lit("unknown"))
            .otherwise(pl.lit("low"))
            .alias("field_group")
        )
    )
    settings.output_dir.mkdir(parents=True)
    studies.drop("Report").write_parquet(
        settings.output_dir / "study_languages.parquet"
    )
    languages = (
        studies.group_by("language")
        .agg(studies=pl.len())
        .sort("studies", descending=True)
    )
    checks = studies.select([pl.col(name).sum() for name in SCRIPT_CHECKS])
    clusters = (
        studies.group_by("language", "vendor", "field_group")
        .agg(
            studies=pl.len(),
            gold=pl.col("is_gold").sum(),
            models=pl.col("ManufacturerModelName")
            .value_counts(sort=True)
            .head(4)
            .struct.field("ManufacturerModelName"),
        )
        .sort("language", "studies", descending=[False, True])
    )
    languages.write_csv(settings.output_dir / "languages.csv")
    checks.write_csv(settings.output_dir / "script_checks.csv")
    clusters.with_columns(pl.col("models").list.join("; ")).write_csv(
        settings.output_dir / "clusters.csv"
    )
    with pl.Config(
        tbl_rows=-1, tbl_width_chars=220, fmt_str_lengths=90, fmt_table_cell_list_len=4
    ):
        print(languages)
        print(checks)
        print(clusters)


if __name__ == "__main__":
    run(TrainingSitesCLI())
