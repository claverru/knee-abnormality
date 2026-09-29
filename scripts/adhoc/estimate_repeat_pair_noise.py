import json
import math
import statistics
from pathlib import Path

from pydantic import BaseModel, Field
from scipy.stats import norm

from knee_abnormality.cli import CLISettings, describe

EXPERIMENTS = Path("outputs/autoresearch")


class RepeatPair(BaseModel):
    mechanism: str
    first: Path
    second: Path
    same_split: bool = True


PAIRS = [
    RepeatPair(
        mechanism="label-view-attention-bias",
        first=EXPERIMENTS / "oblique/experiments/label-view-attention-bias-20260922T120824Z-29aa5863/comparison/comparison.json",
        second=EXPERIMENTS / "oblique-x4/experiments/label-view-attention-bias-20260922T134812Z-7b4fa495/comparison/comparison.json",
    ),
    RepeatPair(
        mechanism="meniscus-shared-residual-head",
        first=EXPERIMENTS / "knee-geometry/experiments/meniscus-shared-residual-head-20260924T002657Z-24ba1afc/comparison/comparison.json",
        second=EXPERIMENTS / "near-misses/experiments/meniscus-shared-residual-head-20260924T210640Z-860b1d41/comparison/comparison.json",
    ),
    RepeatPair(
        mechanism="dsu-feature-statistics",
        first=EXPERIMENTS / "augmentation/experiments/dsu-feature-statistics-20260926T034836Z-6298ea10/comparison/comparison.json",
        second=Path("outputs/analysis/dsu-without-smoothing-vs-report-attributes-control/comparison.json"),
    ),
    RepeatPair(
        mechanism="run-level-stratified-jitter",
        first=EXPERIMENTS / "triplet-sampling/experiments/run-level-stratified-jitter-20260919T183826Z-f211c7ce/comparison/comparison.json",
        second=EXPERIMENTS / "sampling/experiments/run-level-stratified-jitter-20260921T224831Z-d1fcb8e3/comparison/comparison.json",
        same_split=False,
    ),
]


class RepeatPairNoiseCLI(CLISettings):
    model_config = describe(
        "Estimate training noise on a paired macro-AUC delta from mechanisms measured twice "
        "and compare it with the paired report-group bootstrap standard error"
    )

    confidence: float = Field(default=0.8, gt=0, lt=1, description="Central interval level of the adoption rule")


def measurement(path: Path) -> dict[str, float]:
    comparison = json.loads(path.read_text())
    runs = comparison["runs"]
    low, high = comparison["paired_report_group_bootstrap"]["intervals"]["all"]["delta_percentile_95"]
    return {
        "delta": runs["candidate"]["metrics"]["macro_auc"] - runs["reference"]["metrics"]["macro_auc"],
        "bootstrap_se": (high - low) / 3.92,
    }


def summary(pairs: list[dict], bootstrap_se: float, z: float) -> dict[str, float]:
    noise = math.sqrt(sum(p["difference"] ** 2 for p in pairs) / len(pairs) / 2)
    total = math.hypot(bootstrap_se, noise)
    return {
        "pairs": len(pairs),
        "training_noise_se": noise,
        "bootstrap_se": bootstrap_se,
        "total_se": total,
        "effective_z": z * bootstrap_se / total,
    }


def run(args: RepeatPairNoiseCLI) -> dict:
    pairs = []
    for pair in PAIRS:
        first, second = measurement(pair.first), measurement(pair.second)
        pairs.append(
            {
                "mechanism": pair.mechanism,
                "same_split": pair.same_split,
                "first": first,
                "second": second,
                "difference": first["delta"] - second["delta"],
            }
        )
    bootstrap_se = statistics.median(p[k]["bootstrap_se"] for p in pairs for k in ("first", "second"))
    z = norm.ppf(0.5 + args.confidence / 2)
    estimates = {
        "all": summary(pairs, bootstrap_se, z),
        "same_split": summary([p for p in pairs if p["same_split"]], bootstrap_se, z),
    }
    for estimate in estimates.values():
        estimate["effective_false_positive"] = float(norm.sf(estimate["effective_z"]))
    return {"pairs": pairs, "estimates": estimates}


if __name__ == "__main__":
    print(json.dumps(run(RepeatPairNoiseCLI()), indent=2))
