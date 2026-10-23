# 2026-09-18 b0 hyperparameter queues

**Two queues tuned EfficientNet-B0 after the backbone comparison rejected the family. The best b0
run reached 0.885263 with `max_epochs: 15`, against 0.888476 for the matched ReXNet-200 control —
a −0.0032 gap, down from −0.0053. Batch size 16 buys no speed at all, and the learning-rate
response rises monotonically only when both parameter groups scale together.** Backbone
`efficientnet_b0.ra_in1k`, fold 0 of `all_5fold_v3`, validation ×1, `study_triplet_total: 26`, one
unseeded run each. The backbone comparison that preceded these is the
[EfficientNet backbone result](../backbones/2026-09-18-efficientnet-backbone-results.md).

## Runs

All runs use `efficientnet_b0.ra_in1k`, fold 0, validation ×1, `study_triplet_total: 26`, one run
each at a recorded random seed. Queue at batch size 8 in
`outputs/training/overnight-20260918T120806Z-0n2c2uro`, at batch size 16 in
`overnight-20260918T143824Z-i94384yr`. Each run differs from the control of its own queue in one
field; the two controls differ from each other only in `trainer.batch_size`.

| Run | Batch | Seed | Change | Best macro AUC | Δ vs its control | Train |
|---|---:|---:|---|---:|---:|---:|
| `00-control` | 8 | 1673961237 | `backbone_lr: 4e-4` | 0.882349 | — | 1010.4 s |
| `01-b0-lr-6e-4` | 8 | 851705996 | `backbone_lr: 6e-4` | 0.882334 | −0.00002 | 1005.4 s |
| `02-b0-lr-8e-4` | 8 | 2348823138 | `backbone_lr: 8e-4` | 0.881285 | −0.0011 | 1012.4 s |
| **`03-b0-epochs-15`** | 8 | 580750367 | `max_epochs: 15` | **0.885263** | **+0.0029** | 1512.1 s |
| `04-b0-ema-0.9999` | 8 | 2330873637 | EMA decay 0.9999 | *void, see below* | — | 1067.3 s |
| `05-b0-warmup-2` | 8 | 1954420016 | `warmup_epochs: 2` | 0.883033 | +0.0007 | 1023.0 s |
| `00-control` | 16 | 3192045323 | batch size only | 0.880700 | — | 990.2 s |
| `01-epochs-20` | 16 | 2153440652 | `max_epochs: 20` | 0.881089 | +0.0004 | 1974.2 s |
| `02-lr-sqrt2` | 16 | 2651922368 | both LRs ×√2 | 0.883225 | +0.0025 | 991.0 s |
| **`03-lr-linear`** | 16 | 1738295521 | both LRs ×2 | **0.884085** | **+0.0034** | 1000.1 s |

The batch-size-16 queue was stopped after `03-lr-linear`; `04-warmup-2` (seed 2350805628) and
`05-wd-0.02` (seed 271435568) were prepared but never ran.

### What it established

- **More steps is the axis that helps.** `03-b0-epochs-15` is the only bs8 run to move, and the
  only one to reduce training loss (0.4221 against the control's 0.4441). It peaked at epoch 14,
  the last, so it may still not have converged. This matches the family's underfitting diagnosis.
- **`trainer.batch_size` is a pure optimizer knob here and buys no speed.** Epoch time is 99–101 s
  at both batch sizes, because the backbone already receives roughly 160 flattened triplets per
  forward at batch size 8 and the GPU is saturated; doubling to 320 scales per-step cost by 1.96×.
  Batch size changes only how many updates an epoch contains.
- **At matched steps (4380) the two batch sizes tie**, 0.881089 against 0.882349, inside noise.
- **Learning rate helps only when both parameter groups scale.** At batch size 8, raising
  `backbone_lr` alone was flat at 6e-4 and negative at 8e-4. At batch size 16, scaling
  `backbone_lr` *and* `head_lr` together gained monotonically, +0.0025 at ×√2 and +0.0034 at ×2,
  and had not turned over when the queue was stopped.
- **Same seed and unchanged sources reproduce bit-identically.** The batch-size-16 control ran
  twice at seed 3192045323 — once in a queue halted by the source-fingerprint guard, once in the
  queue recorded here — and returned 0.880699634552002 both times, despite
  `trainer.deterministic: false`. This does not contradict the
  [ranked-filtered finding](../sampling/2026-09-17-ranked-filtered-result.md) that a same-seed run can move by
  0.0031: there the *sources* changed. Matched seed plus matched sources is exact; matched seed
  alone is not. The first queue's artifacts were deleted, its control's predictions, metrics,
  timing and config kept in `outputs/analysis/b/bs16-control-20260918/`.
- **`04-b0-ema-0.9999` is void, not a result.** At decay 0.9999 over 4,380 steps the averaged model
  still carries 64.5% of the untrained initialisation (0.9999^4380 = 0.645), so its 0.829248 says
  nothing about EMA. Training loss was normal at 0.4432; only the EMA copy used for validation was
  degenerate. EMA remains untested at this scale; 0.999 leaves 1.3% and is the value to use.

### Open

- **`head_lr` has never been raised in this project.** It has been 6e-4 in every experiment, and the
  only prior test moved it down (`08-slower-head` at 3e-4, −0.0017 on ReXNet). The batch-size-16
  gains scaled it alongside `backbone_lr`, so they do not separate the two. Testing `head_lr`
  upward at batch size 8 is the direct follow-up.
- **The LR response has not turned over.** `03-lr-linear` was the largest rate tried and the best;
  where it peaks is unknown.
- **ReXNet-200 has never been run at 15 epochs on this recipe**, so the −0.0032 gap compares a
  15-epoch b0 against a 10-epoch ReXNet. On the older recipe 15 epochs gave ReXNet +0.0023.
- `04-warmup-2` and `05-wd-0.02` at batch size 16 were prepared but not run.
