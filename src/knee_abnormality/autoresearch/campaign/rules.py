CALIBRATION_EPOCHS = 2
CALIBRATION_VALIDATION_PASSES = 2
SEED_TERMINAL = {"completed", "blocked", "retired"}
EVALUATION = {
    "labels_source": "rebuild_state_v2",
    "validation": {
        "split": "all_5fold_scanner_v2",
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
