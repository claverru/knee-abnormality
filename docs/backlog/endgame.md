# Endgame

Final-phase work, deliberately deferred by the [fold-0 iteration
strategy](../validation/validation-strategy.md); bigger images joined on 2026-09-16 and weight
averaging on 2026-09-17. These entries wait for the final phase; when one starts, move it to
[ongoing work](../ongoing/INDEX.md) and remove it from this file in the same change. Fold-0 results that
bear on them are recorded here as evidence. Each entry says whether its motivation is measured or a
hypothesis.

## Ensembling

**Remaining folds, seeds and diverse runs** (fold-0 averaging measured; the full-ensemble gain is
extrapolated). Train the remaining folds under the
[validation strategy](../validation/validation-strategy.md), then combine models under the
[ensemble interpretation](../validation/ensemble-interpretation.md) rules.

- *Full-data fits:* one model trained on all training studies, with its epoch count fixed from
  cross-validation, is the single-model alternative to a fold ensemble. It is also the natural
  fast candidate for the efficiency prize.
- *Fold models:* besides being ensemble members, they give out-of-fold predictions for choosing
  ensemble weights.
- *Local evidence:* averaging the champion with another fold-0 run gained +0.0028 to +0.0041 over
  the better single run, and a crop model added about as much as a second full-slice run
  ([two-crops averaging](../validation/results/2026-09-15-two-crops-result.md#per-target-comparison-with-the-champion),
  [per-label ensembles](../validation/results/2026-09-16-per-label-crops-result.md#ensembles-with-the-full-triplet-models)).
- *Logit averages:* champion + `00-baseline` scored 0.896032, and champion + `00-baseline` +
  per-label crops scored 0.897562.
- *Expected gain:* the
  [crop-selection research](../research/learned-crop-selection.md#ranked-options) ranks ensembles
  first, at an extrapolated +0.003 to +0.008.

## Bigger images

**Finer resolution at the same field of view** (hypothesis; resolution at a fixed field of view
has never been tested, see [input resolution](../research/input-resolution.md)). Train the
full-slice model at `data.spacing_mm: 0.5` and `data.image_size: 280`, which keeps the 140 mm field
and the sampler's 70 mm in-plane radius.

- *Motivation:* per-label crops at 0.5 mm gained over `00-baseline` on ACL (+0.0111), synovitis
  (+0.0077) and medial OA (+0.0057)
  ([result](../validation/results/2026-09-16-per-label-crops-result.md#per-target-auc)). Learned
  zoom, whose ReXNet-200 evidence was in effect at 0.875 mm, lost heavily on the lateral meniscus
  (−0.1057), MCL (−0.0671) and ACL (−0.0223)
  ([result](../validation/results/2026-09-16-learned-zoom-result.md#per-target-auc)). Across the
  crop runs, score followed slice coverage and a spacing of 0.625 mm or finer
  ([investigation](../history/crop-selection-investigation.md#what-the-runs-established)). The
  architectures differed, so none of this is a controlled resolution test.
- *Acquisition evidence:* 0.5 mm keeps the full acquired band in 59% of series against 23% at
  0.625 mm, and finer than about 0.5 mm adds almost nothing, because stored images are
  interpolated about 2×.
- *Public claims (unreplicated):* one discussion thread reports 0.936 for a small single-fold
  ResNet at 224 px and 0.942 at 288 px
  ([public discussion](../research/public-research.md#validation-and-ensembling)).
- *Cost:* 1.56× the pixels (280² / 224²). ReXNet-200 measured 2.462 against 1.526 GMACs per
  triplet (1.61×), and its token grid grows from 7×7 to 9×9, which gives 81 tokens instead of 49
  ([budget](../modeling/backbone-swaps.md#current-budget)).
- *Memory:* scaling `00-baseline`'s 20.02 GB peak gives an estimated 31 GB at batch 8, so the
  24 GB GPU needs gradient accumulation (for example batch 4 × 2) or activation checkpointing. The
  model allows `model.checkpoint_backbone` only with `model.backbone_eval: true`, which also
  changes BatchNorm behaviour.
- *`model.backbone_eval` alone:* moved here from the [overnight queue](overnight.md) on
  2026-09-16 (the user's decision) because it only matters as the gate for
  `model.checkpoint_backbone` above; an isolated on/off comparison at the current image size is
  not worth a queue slot. `baseline0` (`resnet50d.ra4_e3600_r224_in1k`) and `baseline1`
  (`resnet50d.ra2_in1k`) both trained with `backbone_eval: true` and `checkpoint_backbone: true`,
  but neither has a recorded comparison against `backbone_eval: false` on the same recipe.
- *Host memory:* loader memory grows with pixels
  ([host memory](../modeling/training-runtime-and-memory.md#host-memory-with-larger-inputs)).
- *Inference:* the backbone forward is about a quarter of Kaggle prediction time at 16 loader
  workers ([profiling](../inference/2026-09-15-kaggle-inference-profiling.md)).

## Test-time augmentation

**Limited test-time augmentation** (hypothesis; not evaluated). Evaluate a small number of
transformations with consistent geometry and target handling, recording total inference cost.
Training reflection is implemented;
[reflection TTA](../modeling/schedule-and-augmentation.md#study-level-knee-reflection) has not been
evaluated.

**More evaluation triplets: measured publicly, adopted.** With the attention temperature at 0.75,
`ranked-filtered-seed42` scored 0.938 at ×3 and 0.939 at ×4, against 0.933 at ×2
([submissions](../inference/submissions/2026-09-18-temperature-and-count-submissions.md)). Fold 0
had predicted the temperature and ×3 together at +0.0046 against +0.005 measured.

| Evaluation scale | Mean triplets | Fold-0 macro AUC, τ 1.0 | τ 0.75 | Public, τ 0.75 |
|---|---:|---:|---:|---:|
| ×1 | 25.4 | 0.8873 | 0.8889 | – |
| ×2 | 50.7 | 0.8907 | 0.8930 | 0.933 at τ 1.0 |
| ×3 | 75.9 | 0.8922 | 0.8954 | **0.938** |
| ×4 | 99.3 | 0.8922 | 0.8954 | **0.939** |
| Every candidate | 165.1 | 0.8896 | 0.8942 | – |

- **Open:** where the count stops paying above ×4, and what it costs. ×3 and ×4 differ by one
  rounding step publicly and were tied on fold 0, while inference time grows with the count and
  counts toward the efficiency prize. The submit-to-scored times we have (9–11 minutes) mix
  different batch sizes and queueing noise, so a local profile at fixed batch is the way to
  measure it.
- **Earlier contrary evidence is explained:** on 2026-09-11 baseline4 scored 0.912 at ×3 against
  0.913 at ×2 ([image count](../inference/inference-image-count-and-memory.md)). That was at
  τ 1.0, where extra triplets draw attention away from the good ones; sharper pooling is what
  makes the larger counts pay.
- `validation_image_multiplier` is an integer and separate from the bundle's test multiplier,
  which is what the submissions above set.

## Weight averaging

**EMA or SWA as the final recipe's default** (measured on fold 0; moved here on 2026-09-17, the
user's decision). Both are implemented behind `optimizer.weight_averaging` (`null`, feature off, in
`configs/training.yaml`) and were measured by `03-ema-0.999` and `04-swa-3` of the 2026-09-17
queue ([result](../validation/results/2026-09-17-overnight-results.md#weight-averaging)): averaged
weights beat the live ones by about +0.00015 at the best epoch and +0.0005 at the last, and lower
BCE; EMA costs +153 s per run and SWA +369 s. The EMA checkpoint scored 0.929 public, +0.001 over
its own live weights (0.928). The gain is small against seed noise, so fold-0 experiments train
without it until the final phase.

- *Before adopting it:* validate the live weights only at the last epoch. The second validation
  pass is all of EMA's +153 s (about 17 s per epoch; training steps are unchanged), so this cuts
  the overhead to about 17 s.
- *Follow-ups:* EMA decays 0.998 and 0.9995, EMA with `minimum_lr_ratio: 0.1`, and SWA paired with
  a non-vanishing tail learning rate (`minimum_lr_ratio: 0.1`), which `04-swa-3` does not test.
- *Implementation:* EMA and SWA share one hand-written averager in `KneeModule`
  (`src/knee_abnormality/training/lightning.py`); `optimizer.weight_averaging` takes
  `{method: ema, decay}` or `{method: swa, epochs}` (`src/knee_abnormality/config.py`,
  `EMAConfig`/`SWAConfig`/`WeightAveragingConfig`), and an `ExperimentConfig` validator rejects SWA
  `epochs` greater than `trainer.max_epochs` minus `optimizer.warmup_epochs`. The
  [optimizer research](../research/optimizer-research.md#4-ema-first-additional-training-feature)
  gives the background, the
  ["Weight averaging" schedule section](../modeling/schedule-and-augmentation.md#weight-averaging)
  the semantics as shipped, and
  [configs and experiment artifacts](../architecture/configs-and-experiment-artifacts.md#experiment-artifacts-and-reproducibility)
  the `live_state_dict` checkpoint key, `val_live_*` metrics and `epochs/live_*` files that appear
  only when `weight_averaging` is set. The design choices:

- *Averager:* a hand-written `deepcopy` of the live model inside the Lightning module, updated
  after each optimizer step with a stateless per-step weight. Floating tensors (parameters and
  BatchNorm statistics) are interpolated, integer buffers copied, and the copy is exact while the
  weight is 1. A second constructed model would draw from the global RNG and shift the DataLoader
  shuffle. `AveragedModel` pushes int64 buffers through float arithmetic, timm `ModelEmaV3` cannot
  weight uniformly for SWA, and a callback cannot swap weights before the strict checkpoint load.
- *EMA:* copy the live weights until warm-up ends (step 438), then average with decay 0.999 per
  update. Its 1,000-step (2.3-epoch) horizon matches the cosine tail, where `00-baseline`'s live
  AUC was flat (0.89269, 0.89269, 0.89205, 0.89276 at epochs 7–10). Decay 0.99 (100 steps) would
  stay close to the live weights, and 0.9995 puts half its weight on epochs 5–7. BatchNorm
  statistics are averaged with the weights.
- *SWA:* a uniform per-step average over the last K = 3 epochs (from step 3,066) with the schedule
  unchanged. Lightning's `StochasticWeightAveraging` is unsuitable: it swaps in an epoch-level
  `SWALR`, flattens the 3:1 backbone:head learning-rate ratio, leaves out the final epoch, adds a
  pseudo-epoch that becomes a selection candidate, keeps the average out of the saved `model.*`
  weights, and resets BatchNorm statistics that `backbone_eval` never re-estimates.
- *SWA BatchNorm refresh:* at the end of each window epoch, re-estimate the BatchNorm layers that
  are in training mode (none under `backbone_eval`) with a forward-only pass over an unshuffled
  training loader, under the training precision and a forked RNG so the training random streams
  stay untouched.
- *Saved and validated weights:* checkpoints store the averaged weights as `model.*`, so
  prediction, export and fold-run comparisons use them unchanged, and the live weights alongside
  in `live_state_dict`, which export drops.
  - Both models are validated on the same batches every epoch: `val_macro_auc` scores the averaged
    weights, while `val_live_*` metrics and `live_*` epoch files record the live ones.
  - Checkpoint selection and early stopping use only the averaged score, keeping the control's 10
    candidates.
  - Where the copy equals the live model (EMA epoch 1, SWA epochs 1–7), both metric sets must
    match.
- *Costs (estimated before the runs):* +58 MB of GPU memory (0.3% of the 20.0 GiB peak), checkpoints from 175 to
  about 233 MB, a few seconds of updates, about 3 min per run for the second validation forward,
  and 1–1.5 min per SWA refresh (3–4.5 min for K = 3).
- *External evidence:* one public EfficientNet-B0 fold-0 ablation reported 0.7914 → 0.7940 with
  EMA, judged noise by its author
  ([public findings](../research/public-research.md#geometry-sampling-encoders-and-losses)), and a
  public training notebook averages its top epochs' weights
  ([public code](../research/public-research.md#public-code-worth-inspecting)). Averaging
  baseline1's epoch 7–9 probabilities gained +0.004267 on 12 reused validation studies, which is
  not a validated gain ([baseline1 review](../history/baseline1-review.md#follow-up-experiments)).

## Distillation

**Distilling an ensemble into one fast model** (hypothesis; no speedup or score measured). Train a
single model on the ensemble's predictions, so that inference costs one model. The idea recurs in
the [unverified notes](../modeling/unverified-modeling-notes.md#loss-and-ensemble-diversity) and
the [runtime leads](../inference/runtime-evidence-and-leads.md#runtime-leads).
