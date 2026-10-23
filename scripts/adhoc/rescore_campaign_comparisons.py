import json
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field

from knee_abnormality.cli import CLISettings, describe
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD, load_split
from knee_abnormality.evaluation.metrics import aucs, bootstrap_auc


class RescoreCLI(CLISettings):
    model_config = describe(
        "Re-score every decided autoresearch comparison's saved candidate and reference "
        "predictions against another frozen label set with the same paired report-group "
        "bootstrap and adoption rule"
    )

    experiments: Path = Field(
        default=Path("outputs/autoresearch"), description="Campaign outputs root"
    )
    raw_dir: Path = Field(
        default=Path("data/raw"), description="Official competition files"
    )
    labels_split: Path = Field(
        default=Path("data/folds/all_5fold_scanner_v2"),
        description="Split whose frozen targets are used for scoring",
    )
    comparisons: list[Path] = Field(
        default=[],
        description="Extra comparison.json files outside autoresearch, such as the evidence behind main adoptions",
    )
    output: Path = Field(description="Fresh CSV output path")


def prediction_path(comparison: dict, name: str) -> Path:
    return next(
        Path(item["path"])
        for item in comparison["runs"][name]["files"]
        if Path(item["path"]).name == "validation_predictions.csv"
    )


def input_path(comparison: dict, file: str) -> Path:
    return next(
        Path(item["path"])
        for item in comparison["inputs"]
        if Path(item["path"]).name == file
    )


def macro_delta(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    bootstrap: dict,
) -> dict:
    targets = [str(index) for index in range(truth.shape[1])]
    point = (
        aucs(truth, candidate, targets)["macro_auc"]
        - aucs(truth, reference, targets)["macro_auc"]
    )
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.random.default_rng(bootstrap["seed"]).multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=bootstrap["draws"]
    )[:, inverse]
    positive = counts @ truth
    valid = ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(
        axis=1
    )
    draws = bootstrap_auc(truth, candidate, counts[valid]) - bootstrap_auc(
        truth, reference, counts[valid]
    )
    low, high = np.percentile(draws, [10, 90])
    accepted = low > 0
    return {
        "delta": point,
        "low80": float(low),
        "high80": float(high),
        "accepted": bool(accepted),
    }


def rescore(comparison_path: Path, targets_frame: pl.DataFrame) -> dict | None:
    evaluation_path = comparison_path.parent.parent / "evaluation.json"
    evaluation = (
        json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
    )
    if not comparison_path.exists() or (evaluation and "macro_auc" not in evaluation):
        return None
    comparison = json.loads(comparison_path.read_text())
    experiment = comparison_path.parent.parent if evaluation else comparison_path.parent
    bootstrap = comparison["paired_report_group_bootstrap"]
    studies = pl.read_csv(input_path(comparison, "studies.csv")).filter(
        pl.col("fold") == comparison["fold"]
    )
    original = pl.read_csv(input_path(comparison, "targets.csv"))
    targets = [row["target"] for row in comparison["per_target"]]
    predictions = {
        name: pl.read_csv(prediction_path(comparison, name))
        .select(STUDY, *targets)
        .sort(STUDY)
        for name in ("reference", "candidate")
    }
    frame = studies.select(STUDY, "report_group").sort(STUDY)
    if frame[STUDY].to_list() != predictions["reference"][STUDY].to_list():
        raise ValueError(
            f"Prediction membership differs from the recorded fold: {experiment}"
        )
    arrays = {
        name: prediction.select(targets).to_numpy().astype(float)
        for name, prediction in predictions.items()
    }
    groups = frame["report_group"].to_numpy()
    results = {}
    for label, truth_frame in (("recorded", original), ("rescored", targets_frame)):
        truth = (
            frame.join(truth_frame, on=STUDY, how="left")
            .select(targets)
            .to_numpy()
            .astype(float)
        )
        results[label] = macro_delta(
            (truth >= TARGET_THRESHOLD).astype(np.int8),
            arrays["reference"],
            arrays["candidate"],
            groups,
            bootstrap,
        )
    scores = evaluation.get("macro_auc") or {
        name: comparison["runs"][name]["metrics"]["macro_auc"]
        for name in ("reference", "candidate")
    }
    recorded = scores["candidate"] - scores["reference"]
    if abs(results["recorded"]["delta"] - recorded) > 1e-9:
        print(
            f"Skipped, saved predictions do not reproduce the recorded delta: {experiment}"
        )
        return None
    return {
        "campaign": experiment.parent.parent.name if evaluation else "manual",
        "experiment": experiment.name,
        "decision": evaluation.get("eligible_decision", "manual"),
        "fold": comparison["fold"],
        "split": input_path(comparison, "targets.csv").parent.name,
        **{f"recorded_{key}": value for key, value in results["recorded"].items()},
        **{f"rescored_{key}": value for key, value in results["rescored"].items()},
    }


def run(settings: RescoreCLI) -> None:
    if settings.output.exists():
        raise ValueError(f"Output already exists: {settings.output}")
    targets_frame = load_split(settings.raw_dir, settings.labels_split).evaluation
    rows = [
        row
        for comparison in [
            *sorted(
                settings.experiments.glob("*/experiments/*/comparison/comparison.json")
            ),
            *settings.comparisons,
        ]
        if (row := rescore(comparison, targets_frame)) is not None
    ]
    table = pl.DataFrame(rows).with_columns(
        flipped=pl.col("recorded_accepted") != pl.col("rescored_accepted"),
        shift=pl.col("rescored_delta") - pl.col("recorded_delta"),
    )
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(settings.output)
    with pl.Config(tbl_rows=-1, tbl_width_chars=220, fmt_str_lengths=48):
        print(
            table.filter(
                pl.col("rescored_accepted")
                | pl.col("recorded_accepted")
                | (pl.col("rescored_delta") >= 0.001)
            )
            .select(
                "campaign",
                "experiment",
                "decision",
                "recorded_delta",
                "recorded_low80",
                "rescored_delta",
                "rescored_low80",
                "rescored_accepted",
            )
            .sort("rescored_delta", descending=True)
        )
        print(
            table.select(
                pl.len(),
                pl.col("recorded_accepted").sum(),
                pl.col("rescored_accepted").sum(),
                pl.col("flipped").sum(),
                pl.corr("recorded_delta", "rescored_delta"),
                pl.col("shift").std(),
            )
        )


if __name__ == "__main__":
    run(RescoreCLI())
