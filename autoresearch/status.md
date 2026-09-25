# Implementation status

Current state and pending work only. Finished changes live in git history; campaign outcomes live
in the [campaigns](campaigns/INDEX.md) and the
[validation results](../docs/validation/results/INDEX.md).

## Current state (2026-09-25)

- **No campaign is running.** All eleven campaigns under `outputs/autoresearch/` are stopped, and
  the watchdog is stopped. The last one, [near-misses](campaigns/near-misses/INDEX.md), was stopped
  by the user on 2026-09-25.
- **Main recipe:** `configs/training.yaml` carries the three-compartment OA detector at seed
  1485198267 on fold 1 of `all_5fold_scanner_v1`
  ([current model](../docs/modeling/current-model.md)).
- **Fold:** the runner takes the fold from the sealed recipe instead of assuming fold 0
  ([protocol](protocol.md#training-seed-policy)).

## Pending

- **Next campaign reference:** because the runner now takes the fold from the sealed recipe and
  `configs/training.yaml` is on fold 1, the next campaign needs a fold-1 reference run.
- **Near-misses cannot resume:** its sealed recipe no longer validates against its historical
  control on the current main. Its outputs stay a read-only record.
