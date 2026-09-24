import copy
import math
from decimal import Decimal
from pathlib import Path

from knee_abnormality.autoresearch.campaign.definition import CampaignPolicy
from knee_abnormality.autoresearch.runtime.storage import read_json, sha256

LOGISTICAL_KEYS = {"name", "seed", "paths.output_root"}
RUNTIME_FACTOR = 1.15
CALIBRATION_EPOCHS = 2
EVALUATION = {
    "labels_source": "combined_v1",
    "validation": {
        "split": "all_5fold_scanner_v1",
        "folds": [0],
        "prediction_image_multiplier": 4,
    },
    "validation_image_multiplier": 1,
    "bootstrap_draws": 2000,
    "bootstrap_seed": 20260914,
}
ACCEPTANCE_RULE = {
    "metric": "candidate_minus_incumbent_macro_auc",
    "minimum_gain": 0.001,
    "bootstrap_confidence_level": 0.8,
    "bootstrap_lower_bound_strictly_above": 0.0,
    "reporting_confidence_level": 0.95,
}


def acceptance_text():
    return (
        f"adopt only when valid macro AUC improvement >= {ACCEPTANCE_RULE['minimum_gain']} "
        f"AND the central {ACCEPTANCE_RULE['bootstrap_confidence_level']:.0%} "
        "paired-bootstrap lower bound is strictly above "
        f"{ACCEPTANCE_RULE['bootstrap_lower_bound_strictly_above']:g}; otherwise discard"
    )


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


def without_fields(record, paths, required=True):
    record = copy.deepcopy(record)
    for path in paths:
        if not remove_field(record, path.split(".")) and required:
            raise ValueError(
                f"Reference exception is absent from the reference: {path}"
            )
    return record


def remove_field(value, parts):
    name, rest = parts[0], parts[1:]
    if name.endswith("[]"):
        items = value.get(name[:-2]) if isinstance(value, dict) else None
        if not isinstance(items, list):
            return False
        removed = [remove_field(item, rest) for item in items]
        return any(removed)
    if not isinstance(value, dict) or name not in value:
        return False
    if rest:
        return remove_field(value[name], rest)
    del value[name]
    return True


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
    if (
        candidate["data"]["validation_image_multiplier"]
        != EVALUATION["validation_image_multiplier"]
    ):
        raise ValueError("Validation multiplier differs from the shared evaluation")
    if candidate["validation"] != EVALUATION["validation"]:
        raise ValueError("Validation split differs from the shared evaluation")
    if candidate["labels"]["source"] != EVALUATION["labels_source"]:
        raise ValueError("Label source differs from the shared evaluation")
    return changed


def initial_limits(
    reference: Path,
    config: dict,
    equivalences: dict,
    config_exceptions: list[str],
    identity_exceptions: list[str],
):
    historical = without_fields(
        read_json(reference / "effective_config.json"), config_exceptions
    )
    without_fields(
        read_json(reference / "fold_0/input_identity.json"), identity_exceptions
    )
    config = without_fields(config, config_exceptions, required=False)
    recorded, current = flatten(historical), flatten(config)
    for key, value in equivalences.items():
        if key in recorded or key not in current or current[key] != value:
            raise ValueError(f"Invalid reference equivalence: {key}")
    unexpected = (
        set(config_differences(historical, config))
        - LOGISTICAL_KEYS
        - set(equivalences)
    )
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
        "reference_equivalences": dict(equivalences),
        "reference_config_exceptions": list(config_exceptions),
        "reference_identity_exceptions": list(identity_exceptions),
        "bootstrap_draws": EVALUATION["bootstrap_draws"],
        "bootstrap_seed": EVALUATION["bootstrap_seed"],
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
