# Endgame

Final-phase work, deliberately deferred by the [fold-0 iteration
strategy](../validation/validation-strategy.md); bigger images joined on 2026-09-16 and weight
averaging on 2026-09-17. These entries wait for the final phase; when one starts, move it to
[ongoing work](../ongoing/INDEX.md) and remove it from this file in the same change. Fold-0 results that
bear on them are recorded here as evidence. Each entry says whether its motivation is measured or a
hypothesis.

## Ensembling

**Round one is executed and is no longer backlog:** ten members, +0.0061 OOF over ReXNet alone,
**0.943 public**, recorded in the
[fold and ensemble results](../validation/results/2026-09-19-fold-ensembles-and-sigma-sweep.md),
the [submissions](../inference/submissions/2026-09-19-ensemble-submissions.md) and the
[project state](../ongoing/state.md).

**What remains here is a second round on the current recipe.** Every member of round one trains
with ordinary BCE, which class-balanced BCE beats by 0.0116, so the whole bundle is built on the
superseded loss; seed ensembles and diverse runs are still untried. Train the remaining folds under
the [validation strategy](../validation/validation-strategy.md), then combine models under the
[ensemble interpretation](../validation/ensemble-interpretation.md) rules.

- *Bundling mechanism (implemented 2026-09-19):* `export.py` and `predict.py` now support a
  multi-checkpoint bundle that averages members' per-label probabilities on one shared decode
  pass, so building the eventual fold/seed ensemble is a matter of exporting with the trained
  checkpoints, not further engineering
  ([bundle contract](../inference/model-bundle-and-offline-environment.md#multi-checkpoint-ensemble-bundles),
  [risk guard](risks.md#numerics-and-scores)). What remains here is which folds/seeds/backbones to
  train and combine and how to weight them, not the export/predict mechanism.

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
- *Seed versus loss diversity:* on saved fold-0 predictions from the losses campaign, three control
  seeds averaged gained +0.0044 over one run. Loss-variant trios of equal member quality added
  about +0.0009 more, led by `shared-bce-ranking` and `shared-shift-projected-bce`. Subset search
  did not beat averaging every run once selection optimism was removed
  ([simulation](../validation/results/2026-09-21-loss-ensemble-simulation.md)).

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

**Mirror TTA evaluated 2026-09-19: +0.001 public for +181 s** on five b0 members at ×4 (0.941
against 0.940), and +0.00046 on fold 0 locally. Both sit inside the ±0.006 public noise, so it is a
marginal win on score and a clear loss on efficiency, where 181 s costs 0.0056 of the runtime term —
on for the score bundle, off for the efficiency bundle
([submissions](../inference/submissions/2026-09-19-ensemble-submissions.md)). Other transformations
remain unevaluated, and the mirror gain should be remeasured once members train with class-balanced
BCE. Evaluate any further transformation with consistent geometry and target handling, recording
total inference cost.

The implemented [reflection TTA mechanism](../inference/model-bundle-and-offline-environment.md#reflection-test-time-augmentation-tta2)
is already used by the submitted bundles; the pending work is its value under the next recipe.

**More evaluation triplets are adopted, not pending.** ×4 at τ 0.75 scores 0.939 public against
0.938 at ×3 and 0.933 at ×2; the per-count fold-0 and public table, and the reason sharper pooling
is what makes larger counts pay, are in the
[submissions record](../inference/submissions/2026-09-18-temperature-and-count-submissions.md).
Where the count stops paying above ×4, and what it costs against the efficiency prize, is tracked
in [next](../ongoing/next.md); external evidence points the same way
([public research](../research/public-research.md#slice-density-and-the-board-2026-09-18-re-sweep)).

## Weight averaging

**EMA or SWA as the final recipe's default** (measured on fold 0; moved here on 2026-09-17, the
user's decision). Both are implemented behind `optimizer.weight_averaging` (`null`, feature off, in
`configs/training.yaml`) and were measured by `03-ema-0.999` and `04-swa-3` of the 2026-09-17
queue ([result](../validation/results/2026-09-17-overnight-results.md#weight-averaging)): averaged
weights beat the live ones by about +0.00015 at the best epoch and +0.0005 at the last, and lower
BCE; EMA costs +153 s per run and SWA +369 s. The EMA checkpoint scored 0.929 public, +0.001 over
its own live weights (0.928). The gain is small against seed noise, so fold-0 experiments train
without it until the final phase. A public CoAtNet pipeline reports the same null for SWA over six
pairs: 0.9167 against 0.9150 on the 58 gold studies and 0.938 against 0.937 on the board
([public research](../research/public-research.md#slice-density-and-the-board-2026-09-18-re-sweep)).

- *Before adopting it:* validate the live weights only at the last epoch. The second validation
  pass is all of EMA's +153 s (about 17 s per epoch; training steps are unchanged), so this cuts
  the overhead to about 17 s.
- *Follow-ups:* EMA decays 0.998 and 0.9995, EMA with `minimum_lr_ratio: 0.1`, and SWA paired with
  a non-vanishing tail learning rate (`minimum_lr_ratio: 0.1`), which `04-swa-3` does not test.
Implementation, original design choices and measured costs live in
[the modeling guide](../modeling/schedule-and-augmentation.md#weight-averaging).

## Distillation

**Distilling an ensemble into one fast model** (hypothesis; no speedup or score measured). Train a
single model on the ensemble's predictions, so that inference costs one model. The idea recurs in
the [unverified notes](../modeling/unverified-modeling-notes.md#loss-and-ensemble-diversity) and
the [runtime leads](../inference/runtime-evidence-and-leads.md#runtime-leads).

## Out-of-fold self-distillation

**Train a second five-fold round on targets blended with the first round's out-of-fold
predictions** (hypothesis; untested here, filed 2026-09-21). Every non-gold training cell becomes
`0.7 · label + 0.3 · teacher`, where the teacher is that study's out-of-fold prediction. Gold labels
and the evaluation targets stay unblended. The aim is robustness to report-label noise; speed is
the [distillation](#distillation) entry above.

- *External evidence:* the Plant Pathology 2020 first-place solution used this recipe: five-fold
  out-of-fold predictions and 0.7 truth + 0.3 prediction
  ([write-up](https://hamonk.github.io/2020/12/05/plant_pathology.html); no isolated gain given).
  Same-architecture children beat their parents on CIFAR
  ([born-again networks](https://arxiv.org/abs/1805.04770)). For linear and logistic models under
  label noise, theory gives when the child beats the parent, and under heavy noise the optimal
  teacher weight exceeds the label's ([Das and Sanghavi, ICML 2023](https://arxiv.org/abs/2301.13304)).
  A participant 26th on the board reported, in a discussion reply of about 2026-09-18 with no
  numbers, that mixing predictions into the extracted labels beat both replacing and masking them.
  No measured gain was found for multi-label AUC on report-derived labels; the closest setting,
  [chest X-rays with NLP labels](https://link.springer.com/chapter/10.1007/978-3-030-87358-5_57),
  is paywalled.
- *Settings:* 0.7/0.3 is the only weight with a winning precedent; per-label weights, with more
  teacher weight on the noisiest labels, only if the first round moves. Children keep their
  augmentation, since removing student noise removed the student's gain in
  [Noisy Student](https://arxiv.org/abs/1911.04252). One round only: a few rounds reduce
  overfitting and more underfit ([Mobahi et al., NeurIPS 2020](https://arxiv.org/abs/2002.05715)).
- *Teacher on hand:* round one's out-of-fold predictions cover all 4,407 studies, one ReXNet + B0
  pair per study ([results](../validation/results/2026-09-19-fold-ensembles-and-sigma-sweep.md)).
  They come from split `all_5fold_v3`, ordinary BCE and the labels before `combined_v1`, and carry
  the folds' calibration drift into the targets.
- *Local-score leak:* child k's target for a study in fold j comes from parent j, which trained on
  fold k, so child out-of-fold scores are biased upward against the parents'. The hidden test is
  unaffected. A clean fold-0 gate: four teachers cross-validated inside folds 1–4, then one child on
  folds 1–4, against the fold-0 incumbent. Otherwise the public board decides.
- *Comparison:* child ensemble against parent ensemble, and parents plus children. Models trained on
  improved labels were reported to be more correlated and to add less to ensembles
  ([supervision](../modeling/supervision.md#downstream-benefit-changed-over-time)).
- *Cost:* a training option that blends a predictions table into the targets, then a second
  five-fold round; the fold-0 gate is five runs.
