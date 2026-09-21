import math
from decimal import Decimal
from pathlib import Path

from knee_abnormality.autoresearch.campaigns import CampaignPolicy
from knee_abnormality.autoresearch.storage import read_json, sha256

LOGISTICAL_KEYS = {"name", "seed", "paths.output_root"}
RUNTIME_FACTOR = 1.15
ACCEPTANCE_RULE = {
    "metric": "candidate_minus_incumbent_macro_auc",
    "minimum_gain": 0.001,
    "bootstrap_confidence_level": 0.8,
    "bootstrap_lower_bound_strictly_above": 0.0,
    "reporting_confidence_level": 0.95,
}


def active_incumbent_ideas(state):
    incumbent = state["experiments"].get(state["incumbent"]["id"])
    if incumbent is None:
        return []
    if "active_idea_ids" not in incumbent:
        raise ValueError(
            "Incumbent has no recorded integration inventory; explicit recovery is required"
        )
    return incumbent["active_idea_ids"]


def dependency_blockers(idea, ideas, active_idea_ids):
    return [
        f"{dependency} (status={ideas.get(dependency, {}).get('status', 'unknown')}, "
        f"decision={ideas.get(dependency, {}).get('decision') or 'undecided'}, "
        f"active={dependency in active_idea_ids})"
        for dependency in idea["dependencies"]
        if ideas.get(dependency, {}).get("decision") != "provisional_accept"
        or dependency not in active_idea_ids
    ]


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {
            key: item
            for name, child in value.items()
            for key, item in flatten(child, f"{prefix}{name}.").items()
        }
    return {prefix.removesuffix("."): value}


def config_differences(reference, candidate):
    left, right = flatten(reference), flatten(candidate)
    return sorted(
        key for key in left.keys() | right.keys() if left.get(key) != right.get(key)
    )


def validate_config(reference, candidate, policy: CampaignPolicy):
    changed = config_differences(reference, candidate)
    allowed = LOGISTICAL_KEYS | set(policy.config_keys) | set(policy.list_item_fields)
    rejected = [
        key
        for key in changed
        if key not in allowed
        and not any(
            key == prefix or key.startswith(f"{prefix}.")
            for prefix in policy.config_prefixes
        )
    ]
    if rejected:
        raise ValueError(f"Recipe changed outside campaign policy: {rejected}")
    for key, fields in policy.list_item_fields.items():
        left, right = flatten(reference)[key], flatten(candidate)[key]
        if len(left) != len(right) or any(
            {k: v for k, v in a.items() if k not in fields}
            != {k: v for k, v in b.items() if k not in fields}
            for a, b in zip(left, right, strict=True)
        ):
            raise ValueError(f"Protected list-item fields changed: {key}")
    if candidate["trainer"]["max_epochs"] != reference["trainer"]["max_epochs"]:
        raise ValueError("Epoch budget must remain fixed")
    if candidate["data"]["validation_image_multiplier"] != 1:
        raise ValueError("Validation must remain x1")
    if candidate["validation"] != {"split": "all_5fold_v3", "folds": [0]}:
        raise ValueError("Only fold 0 of all_5fold_v3 is authorized")
    return changed


def initial_limits(reference: Path, config: dict):
    historical = read_json(reference / "effective_config.json")
    unexpected = set(config_differences(historical, config)) - LOGISTICAL_KEYS
    if unexpected:
        raise ValueError(
            f"Historical control differs from the starting recipe: {sorted(unexpected)}"
        )
    timing = read_json(reference / "fold_0/timing.json")
    seconds = sum(
        timing[key]
        for key in (
            "setup_seconds",
            "training_seconds",
            "validation_prediction_seconds",
        )
    )
    return {
        "runtime_factor": RUNTIME_FACTOR,
        "historical_recorded_seconds": seconds,
        "max_run_seconds": math.ceil(seconds * RUNTIME_FACTOR),
        "epochs": config["trainer"]["max_epochs"],
        "optimizer_steps": timing["optimizer_steps"],
        "triplet_tolerance": 0,
        "reference_predictions_sha256": sha256(
            reference / "fold_0/validation_predictions.csv"
        ),
        "bootstrap_draws": 2000,
        "bootstrap_seed": 20260914,
    }


def evaluate(experiment: Path, limits: dict, receipt: dict):
    reasons = []
    if receipt["exit_code"] != 0 or receipt.get("error"):
        reasons.append(f"Training failed: {receipt}")
    if receipt.get("seconds") is None or receipt["seconds"] > limits["max_run_seconds"]:
        reasons.append("Fixed wall-clock ceiling exceeded or unavailable")
    try:
        ledger = read_json(experiment / "training/fold_0/triplet_accounting.json")
        timing = read_json(experiment / "training/fold_0/timing.json")
        if not ledger["complete"]:
            reasons.append("Triplet accounting is incomplete")
        for key in ("training_triplets", "training_visits"):
            if ledger[key] != limits[key]:
                reasons.append(f"Changed {key}")
        if timing["optimizer_steps"] != limits["optimizer_steps"]:
            reasons.append("Optimizer step count changed")
        if len(ledger["train"]) != limits["epochs"]:
            reasons.append("Epoch count changed")
        if len(ledger["validation"]) != limits["epochs"]:
            reasons.append("Validation pass count changed")
        if not ledger["changed_training_studies"]:
            reasons.append("Training sampling showed no stochastic variation")
        if len({row["selection_sha256"] for row in ledger["validation"]}) != 1:
            reasons.append("Validation sampling changed between epochs")
        for phase in ("train", "validation"):
            expected = limits[
                f"{'training' if phase == 'train' else phase}_cohort_sha256"
            ]
            if any(row["cohort_sha256"] != expected for row in ledger[phase]):
                reasons.append(f"Changed {phase} cohort")
        if any(
            row["triplets"] != limits["validation_triplets"]
            for row in ledger["validation"]
        ):
            reasons.append("Validation x1 budget changed")
    except (OSError, KeyError, ValueError) as error:
        reasons.append(f"Missing or malformed accounting: {error}")
    return reasons


def verdict(report, macro_bootstrap):
    bootstrap = report["paired_report_group_bootstrap"]
    interval = bootstrap["intervals"]["all"]
    bounds = interval["delta_percentile_95"]
    scores = report["family_means"]["all"]
    adoption_bounds = (macro_bootstrap or {}).get("delta_percentile_80")
    if (
        not bounds
        or len(bounds) != 2
        or interval["valid_draws"] < 1
        or not all(math.isfinite(value) for value in bounds)
        or not adoption_bounds
        or len(adoption_bounds) != 2
        or not all(math.isfinite(value) for value in adoption_bounds)
        or not bounds[0] <= adoption_bounds[0] <= adoption_bounds[1] <= bounds[1]
        or macro_bootstrap["valid_draws"] != interval["valid_draws"]
        or not all(
            math.isfinite(scores[key]) and 0 <= scores[key] <= 1
            for key in ("reference", "candidate")
        )
    ):
        return "invalid"
    gain = Decimal(str(scores["candidate"])) - Decimal(str(scores["reference"]))
    if (
        gain >= Decimal(str(ACCEPTANCE_RULE["minimum_gain"]))
        and adoption_bounds[0] > ACCEPTANCE_RULE["bootstrap_lower_bound_strictly_above"]
    ):
        return "provisional_accept"
    return "reject"
