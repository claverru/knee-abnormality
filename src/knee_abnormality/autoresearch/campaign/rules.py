CALIBRATION_EPOCHS = 2
CALIBRATION_VALIDATION_PASSES = 2
SEED_TERMINAL = {"completed", "blocked", "retired"}
EVALUATION = {
    "labels_source": "rebuild_state_v2",
    "validation": {
        "split": "all_5fold_scanner_v2",
        "prediction_image_multiplier": 4,
        "unused_folds": [],
    },
    "validation_image_multiplier": 1,
    "bootstrap_draws": 2000,
    "bootstrap_seed": 20260914,
}
ACCEPTANCE_RULE = {
    "metric": "candidate_minus_incumbent_macro_auc",
    "multipliers": [1, 4],
    "change_at_every_multiplier_strictly_above": 0.0,
    "bootstrap_confidence_level": 0.8,
    "reporting_confidence_level": 0.95,
}


def acceptance_text():
    return (
        "adopt only when the valid macro AUC is strictly above the incumbent's at both image "
        "multipliers, x1 (the selected checkpoint's epoch validation) and x4 (its saved "
        "predictions); otherwise discard. The central "
        f"{ACCEPTANCE_RULE['bootstrap_confidence_level']:.0%} paired-bootstrap interval of the "
        "x4 change is reported and does not decide"
    )
