# Refactors

| Read when | Document |
|---|---|
| Removing `trainer.min_delta` | [Retired options](retired-options.md) |
| Launching the overnight queue with dotted config overrides instead of derived YAML files | [Overnight launcher overrides](overnight-launcher-overrides.md) |
| Ensembling checkpoints from different sources or bag sizes in one data pass instead of one pass per bundle; recorded source parity or separate data and model identities | [Single-pass mixed-source ensembles](single-pass-mixed-source-ensembles.md) |
| Letting CPU and GPU work overlap in the training step by removing its forced GPU synchronisations | [Sync-free training step](sync-free-training-step.md) |
