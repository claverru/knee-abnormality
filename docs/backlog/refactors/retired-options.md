# Remove retired configuration options

**Proposed, unimplemented.** One option survives only from a redundant experiment. Remove it
with its branch and every config that sets it. `data.repeated_series_share`,
`model.gated_attention`, `model.residual_readout` and `trainer.class_balanced_bce` were removed on
2026-09-25.

| Option | Why it goes | Code |
|---|---|---|
| `trainer.min_delta` | Its only consumer is EarlyStopping, whose patience covers the whole fixed training budget | `config.py`, `scripts/pipeline/train.py`; set in `configs/resnet50.yaml`, `coatnet.yaml` |
