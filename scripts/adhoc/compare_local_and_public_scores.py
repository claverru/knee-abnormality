import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field
from pydantic_settings import SettingsError

from knee_abnormality.cli import CLISettings
from knee_abnormality.data.labels import STUDY, TARGET_THRESHOLD, load_split
from knee_abnormality.evaluation.metrics import aucs
from knee_abnormality.provenance import sha256_file

METRICS = (
    ("macro_auc", 1),
    ("report_auc", 1),
    ("gold_auc", 1),
    ("soft_bce", -1),
    ("gold_bce", -1),
)


class CompareCLI(CLISettings):
    manifest: Path = Path("scripts/adhoc/submission_public_scores.json")
    raw_dir: Path = Path("data/raw")
    folds_dir: Path = Path("data/folds")
    output_dir: Path = Path("outputs/analysis/local-public-metrics")
    permutations: int = Field(default=20000, ge=1000)
    seed: int = 20260918


def best_epoch(run: Path) -> tuple[int, dict]:
    selected = None
    for path in sorted((run / "fold_0/epochs").glob("metrics_epoch_*.json")):
        metrics = json.loads(path.read_text())
        epoch = int(path.stem.rsplit("_", 1)[1])
        if selected is None or metrics["macro_auc"] > selected[1]["macro_auc"]:
            selected = (epoch, metrics)
    if selected is None:
        raise ValueError(f"No epoch metrics under {run}")
    return selected


def run_metrics(run: Path, settings: CompareCLI) -> dict:
    config = json.loads((run / "effective_config.json").read_text())
    epoch, stored = best_epoch(run)
    split_dir = settings.folds_dir / config["validation"]["split"]
    split = load_split(settings.raw_dir, split_dir)
    targets = split.protocol["targets"]
    predictions = pl.read_csv(
        run / f"fold_0/epochs/validation_epoch_{epoch:03d}.csv"
    ).sort(STUDY)
    frame = (
        predictions.select(STUDY, *targets)
        .join(split.evaluation.rename({t: f"y_{t}" for t in targets}), on=STUDY)
        .join(split.membership.select(STUDY, "is_gold"), on=STUDY)
    )
    if len(frame) != len(predictions):
        raise ValueError(f"Validation studies missing from the frozen split: {run}")
    probability = np.clip(frame.select(targets).to_numpy().astype(float), 1e-7, 1 - 1e-7)
    truth = frame.select([f"y_{t}" for t in targets]).to_numpy().astype(float)
    hard = (truth >= TARGET_THRESHOLD).astype(np.int8)
    gold = frame["is_gold"].to_numpy().astype(bool)
    bce = -(truth * np.log(probability) + (1 - truth) * np.log1p(-probability))
    recomputed = aucs(hard, probability, targets)["macro_auc"]
    if abs(recomputed - stored["macro_auc"]) > 1e-6:
        raise ValueError(f"Recomputed macro AUC differs from the saved metrics: {run}")
    return {
        "epoch": epoch,
        "studies": len(frame),
        "gold_studies": int(gold.sum()),
        "split": config["validation"]["split"],
        "labels": config["labels"]["source"],
        "validation_image_multiplier": config["data"].get(
            "validation_image_multiplier"
        ),
        "seed": config["seed"],
        "macro_auc": stored["macro_auc"],
        "report_auc": stored["report_derived"]["macro_auc"],
        "gold_auc": stored["gold"]["macro_auc"],
        "soft_bce": stored["soft_target_bce"],
        "gold_bce": float(bce[gold].mean()),
        "report_bce": float(bce[~gold].mean()),
    }


def spearman(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values), dtype=float)
    _, inverse = np.unique(values, return_inverse=True)
    return np.array([ranks[inverse == index].mean() for index in inverse])


def correlate(x: np.ndarray, y: np.ndarray, settings: CompareCLI) -> dict:
    generator = np.random.default_rng(settings.seed)
    result = {}
    for name, left, right in (
        ("pearson", x, y),
        ("spearman", spearman(x), spearman(y)),
    ):
        observed = float(np.corrcoef(left, right)[0, 1])
        draws = np.array(
            [
                np.corrcoef(left, generator.permutation(right))[0, 1]
                for _ in range(settings.permutations)
            ]
        )
        exceed = int((np.abs(draws) >= abs(observed) - 1e-12).sum())
        result[name] = observed
        result[f"{name}_p"] = (exceed + 1) / (settings.permutations + 1)
    concordant = total = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[i] == x[j] or y[i] == y[j]:
                continue
            total += 1
            concordant += int((x[i] > x[j]) == (y[i] > y[j]))
    result["concordant_pairs"] = concordant
    result["ranked_pairs"] = total
    return result


def cohort_summary(
    rows: list[dict], noise: dict[str, float], settings: CompareCLI
) -> dict:
    public = np.array([row["public"] for row in rows])
    summary = {
        "submissions": [row["name"] for row in rows],
        "public_range": [float(public.min()), float(public.max())],
        "metrics": {},
    }
    for metric, sign in METRICS:
        values = np.array([row[metric] for row in rows])
        spread = float(values.std(ddof=1))
        summary["metrics"][metric] = {
            **correlate(sign * values, public, settings),
            "spread": spread,
            "replicate_noise": noise[metric],
            "reliability": max(0.0, spread**2 - noise[metric] ** 2) / spread**2,
        }
    local = np.array([row["macro_auc"] for row in rows])
    slope, intercept = np.polyfit(local, public, 1)
    residual = public - (intercept + slope * local)
    summary["macro_auc_fit"] = {
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_sd": float(np.sqrt((residual**2).sum() / (len(rows) - 2))),
    }
    return summary


def compare(settings: CompareCLI) -> None:
    if settings.output_dir.exists():
        raise ValueError(f"Output directory already exists: {settings.output_dir}")
    manifest = json.loads(settings.manifest.read_text())
    cache: dict[str, dict] = {}
    rows = []
    for entry in manifest["submissions"]:
        if entry["run"] not in cache:
            cache[entry["run"]] = run_metrics(Path(entry["run"]), settings)
        rows.append({**entry, **cache[entry["run"]]})
    cohort = manifest["comparable_cohort"]
    comparable = [
        row
        for row in rows
        if row["excluded"] is None
        and row["split"] == cohort["split"]
        and row["labels"] == cohort["labels"]
        and row["validation_image_multiplier"]
        == cohort["validation_image_multiplier"]
    ]
    excluded = [row["name"] for row in rows if row not in comparable]

    noise, groups = {}, []
    for group in manifest["replicate_groups"]:
        members = [run_metrics(Path(run), settings) for run in group["runs"]]
        groups.append(
            {
                "name": group["name"],
                "runs": group["runs"],
                "metrics": {
                    metric: [member[metric] for member in members]
                    for metric, _ in METRICS
                },
            }
        )
    degrees = sum(len(group["runs"]) - 1 for group in manifest["replicate_groups"])
    for metric, _ in METRICS:
        deviations = [
            value - float(np.mean(group["metrics"][metric]))
            for group in groups
            for value in group["metrics"][metric]
        ]
        noise[metric] = float(np.sqrt(np.square(deviations).sum() / degrees))

    plateau = [row for row in comparable if row["submitted"] >= manifest["plateau_from"]]
    earlier = [row for row in comparable if row["submitted"] < manifest["plateau_from"]]
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "manifest": {
            "path": str(settings.manifest),
            "sha256": sha256_file(settings.manifest),
        },
        "comparable_cohort": cohort,
        "excluded": excluded,
        "replicate_noise": {"degrees_of_freedom": degrees, "sd": noise, "groups": groups},
        "cohorts": {
            "comparable": cohort_summary(comparable, noise, settings),
            f"comparable_from_{manifest['plateau_from']}": cohort_summary(
                plateau, noise, settings
            ),
            f"comparable_before_{manifest['plateau_from']}": cohort_summary(
                earlier, noise, settings
            ),
        },
    }
    settings.output_dir.mkdir(parents=True)
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    table = pl.DataFrame(rows).with_columns(
        comparable=pl.col("name").is_in([row["name"] for row in comparable])
    )
    table.write_csv(settings.output_dir / "submissions.csv")
    print(
        table.filter(pl.col("comparable"))
        .select(
            "name", "public", "macro_auc", "report_auc", "gold_auc", "soft_bce", "gold_bce"
        )
        .sort("public", descending=True)
    )
    for name, block in summary["cohorts"].items():
        print(f"\n{name} (n={len(block['submissions'])})")
        for metric, values in block["metrics"].items():
            print(
                f"  {metric:<11} r={values['pearson']:+.3f} (p={values['pearson_p']:.3f})"
                f"  rho={values['spearman']:+.3f} (p={values['spearman_p']:.3f})"
                f"  concordant {values['concordant_pairs']}/{values['ranked_pairs']}"
                f"  spread {values['spread']:.5f}"
                f"  noise {values['replicate_noise']:.5f}"
                f"  reliability {values['reliability']:.2f}"
            )
    print(f"\nWrote {settings.output_dir}")


if __name__ == "__main__":
    try:
        compare(CompareCLI())
    except SettingsError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error
