# Implementation status

Current state and pending work only. Finished changes live in git history; campaign outcomes live
in the [campaigns](campaigns/INDEX.md) and the
[validation results](../docs/validation/results/INDEX.md).

## Current state (2026-09-28)

- **No campaign is running.** Every campaign under `outputs/autoresearch/` is stopped, and the
  watchdog is stopped. The last one, [plane-specialization](campaigns/plane-specialization/INDEX.md),
  was stopped by the user on 2026-09-28 and its line closed.
- **Main recipe:** `configs/training.yaml` trains on the rebuilt labels `rebuild_state_v2` with
  seed 1533697213 on fold 0 of `all_5fold_scanner_v2`
  ([current model](../docs/modeling/current-model.md)); the evaluation gate in
  [`campaign/rules.py`](../src/knee_abnormality/autoresearch/campaign/rules.py) requires both, so
  campaigns sealed on `all_5fold_scanner_v1` or `combined_v1` cannot resume.
- **Fold:** the runner takes the fold from the sealed recipe instead of assuming fold 0
  ([protocol](protocol.md#training-seed-policy)).

## Pending

- **Next campaign reference:** `rebuild-state-v2-control` (fold 0 of `all_5fold_scanner_v2`, seed
  1533697213), ×4 0.928288, trained 2026-09-28; a new campaign can start from it.
- **Near-misses cannot resume:** its sealed recipe no longer validates against its historical
  control on the current main. Its outputs stay a read-only record.
