import json
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from sklearn.metrics import roc_auc_score

from knee_abnormality.cli import Bootstrap, CLISettings, describe
from knee_abnormality.data.splits import STUDY, TARGET_THRESHOLD

MACRO = "macro"
INTERVAL = (10, 90)


class Settings(CLISettings, Bootstrap):
    model_config = describe(
        "Score the slice-budget sensitivity passes: per-label and macro AUC of every "
        "pass, the baseline's parity with the local validation metrics, the paired "
        "report-group bootstrap of each halved or doubled series type against the "
        "baseline, and the same deltas per realised three-slice input changed."
    )

    directory: Path = Field(
        description="Directory holding predictions_*.csv, pass_*.json and counts_*.csv"
    )
    split_dir: Path = Field(description="Fold directory with targets.csv, studies.csv")
    reference_metrics: Path = Field(
        description="Local epoch metrics JSON of the same checkpoint at multiplier 1"
    )


def pair_kernel(truth: np.ndarray, scores: np.ndarray) -> np.ndarray:
    positive = scores[truth == 1][:, None]
    negative = scores[truth == 0][None, :]
    return (positive > negative) + 0.5 * (positive == negative)


def weighted_auc(
    kernel: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    positive = weights[:, truth == 1]
    negative = weights[:, truth == 0]
    return np.einsum("dp,pn,dn->d", positive, kernel, negative) / (
        positive.sum(axis=1) * negative.sum(axis=1)
    )


def run(settings: Settings) -> dict:
    directory = settings.directory
    records = {
        path.stem.removeprefix("pass_"): json.loads(path.read_text())
        for path in sorted(directory.glob("pass_*.json"))
    }
    failed = sorted(name for name, record in records.items() if record["returncode"])
    names = [name for name in records if name not in failed]
    targets_frame = pl.read_csv(settings.split_dir / "targets.csv")
    studies = pl.read_csv(settings.split_dir / "studies.csv")
    targets = [column for column in targets_frame.columns if column != STUDY]
    predictions = {
        name: pl.read_csv(directory / f"predictions_{name}.csv").sort(STUDY)
        for name in names
    }
    ids = predictions["baseline"][STUDY].to_list()
    for name, frame in predictions.items():
        if frame[STUDY].to_list() != ids:
            raise ValueError(f"{name} predicts different studies than the baseline")
    truth_frame = (
        studies.select(STUDY, "report_group")
        .join(targets_frame, on=STUDY)
        .filter(pl.col(STUDY).is_in(ids))
        .sort(STUDY)
    )
    if truth_frame[STUDY].to_list() != ids:
        raise ValueError("Predictions and frozen targets cover different studies")
    truth = (truth_frame.select(targets).to_numpy() >= TARGET_THRESHOLD).astype(int)
    groups = truth_frame["report_group"].to_numpy()
    scores = {
        name: frame.select(targets).to_numpy() for name, frame in predictions.items()
    }
    columns = [*targets, MACRO]

    def point(values: np.ndarray) -> np.ndarray:
        labels = [
            roc_auc_score(truth[:, index], values[:, index])
            for index in range(len(targets))
        ]
        return np.asarray([*labels, np.mean(labels)])

    points = {name: point(scores[name]) for name in names}
    pl.DataFrame(
        [
            {
                "pass": name,
                "series_type": records[name]["series_type"],
                "scale": records[name]["scale"],
                **dict(zip(columns, points[name], strict=True)),
            }
            for name in names
        ]
    ).write_csv(directory / "pass_auc.csv")

    reference = json.loads(settings.reference_metrics.read_text())
    local = np.asarray(
        [
            *(reference["per_target_auc"][target] for target in targets),
            reference["macro_auc"],
        ]
    )
    parity = pl.DataFrame(
        {
            "target": columns,
            "local": local,
            "kaggle": points["baseline"],
            "difference": points["baseline"] - local,
        }
    )
    parity.write_csv(directory / "parity.csv")

    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(settings.bootstrap_seed)
    draws = rng.multinomial(
        len(unique),
        np.full(len(unique), 1 / len(unique)),
        size=settings.bootstrap_draws,
    )[:, inverse].astype(float)

    def bootstrap(values: np.ndarray) -> np.ndarray:
        labels = np.stack(
            [
                weighted_auc(
                    pair_kernel(truth[:, index], values[:, index]),
                    truth[:, index],
                    draws,
                )
                for index in range(len(targets))
            ],
            axis=1,
        )
        return np.concatenate([labels, labels.mean(axis=1, keepdims=True)], axis=1)

    check = bootstrap(scores["baseline"])
    unweighted = np.stack(
        [
            weighted_auc(
                pair_kernel(truth[:, index], scores["baseline"][:, index]),
                truth[:, index],
                np.ones((1, len(ids))),
            )[0]
            for index in range(len(targets))
        ]
    )
    if not np.allclose(unweighted, points["baseline"][:-1], atol=1e-12):
        raise ValueError("Pair-kernel AUC differs from scikit-learn")
    baseline_counts = pl.read_csv(directory / "counts_baseline.csv").sort(STUDY)
    slots = len(records["baseline"]["types"])
    rows, realised = [], []
    for name in names:
        record = records[name]
        counts = pl.read_csv(directory / f"counts_{name}.csv").sort(STUDY)
        if counts[STUDY].to_list() != ids:
            raise ValueError(f"{name} traced different studies than it predicted")
        change = [
            float(
                (
                    counts[f"triplets_{slot}"] - baseline_counts[f"triplets_{slot}"]
                ).mean()
            )
            for slot in range(slots)
        ]
        realised.append(
            {
                "pass": name,
                "seconds": record["seconds"],
                "precision": record["precision"],
                "triplets_per_study": record["triplets_per_study"],
                **{
                    f"triplets_{kind['series_type']}": kind["mean_triplets_all_studies"]
                    for kind in record["types"]
                },
                **{
                    f"present_{kind['series_type']}": kind["mean_triplets_present"]
                    for kind in record["types"]
                },
            }
        )
        if name == "baseline":
            continue
        slot = record["slot"]
        others = max(abs(value) for index, value in enumerate(change) if index != slot)
        if others:
            raise ValueError(f"{name} changed the triplets of another series type")
        difference = bootstrap(scores[name]) - check
        low, high = np.percentile(difference, INTERVAL, axis=0)
        delta = points[name] - points["baseline"]
        kind = record["types"][slot]
        base_kind = records["baseline"]["types"][slot]
        for index, target in enumerate(columns):
            rows.append(
                {
                    "series_type": record["series_type"],
                    "scale": record["scale"],
                    "target": target,
                    "baseline_auc": points["baseline"][index],
                    "auc": points[name][index],
                    "delta": delta[index],
                    "low_80": low[index],
                    "high_80": high[index],
                    "outside_80": bool(low[index] > 0 or high[index] < 0),
                    "studies_with_type": kind["studies_with_type"],
                    "triplets_present_baseline": base_kind["mean_triplets_present"],
                    "triplets_present": kind["mean_triplets_present"],
                    "triplet_change_per_study": change[slot],
                    "delta_per_triplet": delta[index] / abs(change[slot]),
                    "low_80_per_triplet": low[index] / abs(change[slot]),
                    "high_80_per_triplet": high[index] / abs(change[slot]),
                }
            )
    deltas = pl.DataFrame(rows)
    deltas.write_csv(directory / "deltas.csv")
    pl.DataFrame(realised).write_csv(directory / "realised_triplets.csv")
    for scale in ("half", "double"):
        part = deltas.filter(pl.col("scale") == scale)
        for value, stem in (("delta", "delta"), ("delta_per_triplet", "per_triplet")):
            part.pivot(on="target", index="series_type", values=value).write_csv(
                directory / f"{stem}_table_{scale}.csv"
            )
    return {
        "passes": names,
        "failed": failed,
        "studies": len(ids),
        "report_groups": len(unique),
        "parity_macro_difference": float(parity["difference"][-1]),
        "parity_largest_label": parity.filter(pl.col("target") != MACRO)
        .sort(pl.col("difference").abs(), descending=True)
        .row(0, named=True),
        "outside_80": deltas.filter("outside_80")
        .select("series_type", "scale", "target", "delta", "low_80", "high_80")
        .to_dicts(),
    }


def main() -> None:
    print(json.dumps(run(Settings()), indent=2))


if __name__ == "__main__":
    main()
