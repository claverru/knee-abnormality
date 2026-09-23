# Remove retired configuration options

**Proposed, unimplemented.** Four options survive only from rejected or redundant experiments.
Remove each option with its branch and every config that sets it.

| Option | Why it goes | Code |
|---|---|---|
| `trainer.min_delta` | Its only consumer is EarlyStopping, whose patience covers the whole fixed training budget | `config.py`, `scripts/pipeline/train.py`; set in `configs/training.yaml`, `coatnet.yaml` |
| `data.repeated_series_share` | Rejected run; the [triplet-budget investigation](../../history/triplet-budget-allocation.md) owns the evidence | `config.py` (field and its exclusivity check), `data/dataset.py` |
| `model.gated_attention` | +0.0010, at or below the control mean ([overnight](../../validation/results/2026-09-18-overnight-results.md)) | `config.py`, `modeling/model.py` |
| `model.residual_readout` | −0.0005 in the same queue | `config.py`, `modeling/model.py` |

`data/dataset.py` and `modeling/model.py` are hashed preprocessing sources, so removing the
last three changes the preprocessing identity and existing checkpoints refuse the new code. Batch
it with the next identity-changing change.
