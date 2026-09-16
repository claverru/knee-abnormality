# Endgame

Final-phase work, deliberately deferred by the
[fold-0 iteration strategy](../validation/validation-strategy.md); bigger images joined on
2026-09-16. These entries wait for the final phase; when one starts, move it to
[ongoing work](../ongoing.md) and remove it from this file in the same change. Fold-0 results that
bear on them are recorded here as evidence. Each entry says whether its motivation is measured or
a hypothesis.

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

## Distillation

**Distilling an ensemble into one fast model** (hypothesis; no speedup or score measured). Train a
single model on the ensemble's predictions, so that inference costs one model. The idea recurs in
the [unverified notes](../modeling/unverified-modeling-notes.md#loss-and-ensemble-diversity) and
the [runtime leads](../inference/runtime-evidence-and-leads.md#runtime-leads).
