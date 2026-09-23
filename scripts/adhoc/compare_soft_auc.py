import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD
from knee_abnormality.evaluation.metrics import bootstrap_auc
from knee_abnormality.provenance import copy_sources, sha256_file


class SoftCLI(CLISettings):
    reference: Path
    candidate: Path
    output_dir: Path
    fold: int = 0
    bootstrap_draws: int = Field(default=2000, ge=100)
    seed: int = 20260914


def predictions(run: Path, fold: int, targets: list[str]) -> pl.DataFrame:
    frame = pl.read_csv(run / f"fold_{fold}" / "validation_predictions.csv")
    frame = frame.filter(pl.col("fold") == fold) if "fold" in frame.columns else frame
    return frame.select(STUDY, *targets).sort(STUDY)


def split_dir(run: Path) -> Path:
    config = json.loads((run / "effective_config.json").read_text())
    return Path("data/folds") / Path(config["validation"]["split"]).name


def soft_auc_draws(
    truth: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(weights))
    for target in range(truth.shape[1]):
        p = truth[:, target]
        difference = scores[:, target, None] - scores[None, :, target]
        wins = (difference > 0).astype(float) + 0.5 * (difference == 0)
        np.fill_diagonal(wins, 0.0)
        positive = weights * p
        negative = weights * (1 - p)
        numerator = np.einsum("di,ij,dj->d", positive, wins, negative)
        denominator = positive.sum(axis=1) * negative.sum(axis=1) - (
            positive * negative
        ).sum(axis=1)
        result += numerator / denominator
    return result / truth.shape[1]


def soft_auc_per_target(truth: np.ndarray, scores: np.ndarray) -> list[float]:
    ones = np.ones((1, len(truth)))
    return [
        float(soft_auc_draws(truth[:, [t]], scores[:, [t]], ones)[0])
        for t in range(truth.shape[1])
    ]


def hard_auc_per_target(hard: np.ndarray, scores: np.ndarray) -> list[float]:
    ones = np.ones((1, len(hard)))
    return [
        float(bootstrap_auc(hard[:, [t]], scores[:, [t]], ones)[0])
        for t in range(hard.shape[1])
    ]


def summary(delta: np.ndarray, observed: float) -> dict:
    return {
        "observed": observed,
        "delta_mean": float(delta.mean()),
        "interval_80": [float(np.quantile(delta, q)) for q in (0.1, 0.9)],
        "interval_95": [float(np.quantile(delta, q)) for q in (0.025, 0.975)],
        "share_positive": float((delta > 0).mean()),
    }


def compare(settings: SoftCLI) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    split = split_dir(settings.reference)
    if split != split_dir(settings.candidate):
        raise ValueError("Runs use different splits")
    protocol = json.loads((split / "protocol.json").read_text())
    targets = protocol["targets"]
    reference = predictions(settings.reference, settings.fold, targets)
    candidate = predictions(settings.candidate, settings.fold, targets)
    if reference[STUDY].to_list() != candidate[STUDY].to_list():
        raise ValueError("Runs predict different studies")
    frozen = (
        reference.select(STUDY)
        .join(pl.read_csv(split / "targets.csv"), on=STUDY, how="left")
        .join(
            pl.read_csv(split / "studies.csv").select(STUDY, "report_group"),
            on=STUDY,
            how="left",
        )
    )
    if frozen.select(targets).null_count().sum_horizontal().item():
        raise ValueError("Frozen targets do not cover the validation studies")
    truth = frozen.select(targets).to_numpy().astype(np.float64)
    hard = (truth >= TARGET_THRESHOLD).astype(float)
    runs = {
        "reference": reference.select(targets).to_numpy().astype(np.float64),
        "candidate": candidate.select(targets).to_numpy().astype(np.float64),
    }
    groups, inverse = np.unique(frozen["report_group"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(settings.seed)
    counts = rng.multinomial(
        len(groups),
        np.full(len(groups), 1 / len(groups)),
        size=settings.bootstrap_draws,
    )[:, inverse].astype(float)
    positive = counts @ hard
    valid = ((positive > 0) & (positive < counts.sum(axis=1, keepdims=True))).all(
        axis=1
    )
    counts = counts[valid]
    ones = np.ones((1, len(truth)))
    hard_observed = {
        name: float(bootstrap_auc(hard, values, ones)[0])
        for name, values in runs.items()
    }
    soft_observed = {
        name: float(soft_auc_draws(truth, values, ones)[0])
        for name, values in runs.items()
    }
    hard_delta = bootstrap_auc(hard, runs["candidate"], counts) - bootstrap_auc(
        hard, runs["reference"], counts
    )
    soft_delta = soft_auc_draws(truth, runs["candidate"], counts) - soft_auc_draws(
        truth, runs["reference"], counts
    )
    per_target = pl.DataFrame(
        {
            "target": targets,
            "hard_reference": hard_auc_per_target(hard, runs["reference"]),
            "hard_candidate": hard_auc_per_target(hard, runs["candidate"]),
            "soft_reference": soft_auc_per_target(truth, runs["reference"]),
            "soft_candidate": soft_auc_per_target(truth, runs["candidate"]),
        }
    ).with_columns(
        (pl.col("hard_candidate") - pl.col("hard_reference")).alias("hard_delta"),
        (pl.col("soft_candidate") - pl.col("soft_reference")).alias("soft_delta"),
    )
    per_target.write_csv(settings.output_dir / "per_target.csv")
    result = {
        "reference": str(settings.reference),
        "candidate": str(settings.candidate),
        "fold": settings.fold,
        "studies": len(truth),
        "split": str(split),
        "definition": {
            "hard": "macro ROC-AUC against frozen targets thresholded at 0.5",
            "soft": "macro soft AUC against frozen targets: distinct study pairs weighted p_i(1 - p_j)",
        },
        "hard": {
            **hard_observed,
            "delta": summary(
                hard_delta, hard_observed["candidate"] - hard_observed["reference"]
            ),
        },
        "soft": {
            **soft_observed,
            "delta": summary(
                soft_delta, soft_observed["candidate"] - soft_observed["reference"]
            ),
        },
        "bootstrap": {
            "unit": "report group",
            "draws": settings.bootstrap_draws,
            "valid_draws": int(valid.sum()),
            "seed": settings.seed,
        },
        "inputs": {
            str(path): sha256_file(path)
            for path in (
                settings.reference
                / f"fold_{settings.fold}"
                / "validation_predictions.csv",
                settings.candidate
                / f"fold_{settings.fold}"
                / "validation_predictions.csv",
                split / "targets.csv",
                split / "studies.csv",
            )
        },
        "sources": copy_sources((Path(__file__),), settings.output_dir / "source"),
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(result, indent=2))
    with pl.Config(tbl_rows=20, float_precision=4, tbl_width_chars=200):
        print(
            json.dumps({k: result[k] for k in ("hard", "soft", "bootstrap")}, indent=2)
        )
        print(per_target)


def main() -> None:
    try:
        compare(SoftCLI())
    except (KeyError, OSError, TypeError, ValueError, SettingsError) as error:
        print(f"Soft AUC comparison error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
