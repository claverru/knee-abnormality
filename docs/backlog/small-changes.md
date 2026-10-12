# Small changes

Small code additions an agent can implement in one sitting, each behind one new option on the
[current recipe](../modeling/current-model.md) (read-only diagnostics need none). The user
supervises them: implement one only when the user asks. Once it is implemented, queue its run in the
[overnight queue](overnight.md) (one change against the incumbent, at its fold and recorded
training seed) and remove the entry from this file. Runs follow the
[controlled-comparison rules](../validation/validation-strategy.md#controlled-comparisons). Each
entry says whether its motivation is measured or a hypothesis.

## Removals of active mechanisms

Queued by the [fixes-and-removals campaign](../validation/results/2026-10-05-fixes-and-removals-campaign.md#open) and never run (guesses: each was adopted
on an earlier backbone or in a bundle and never removed alone on the current recipe). Each is a
bypass at one point, specified in the campaign's
[seeds 3](../../autoresearch/campaigns/fixes-and-removals/seed-ideas-3.yaml); one run each, and
an adopted removal deletes the code.

- **Half-knee loss off:** the term is not added in the training step.
- **Meniscus own-side mask off.**
- **Plane-contrast readouts off.**
- **Detail-view embedding off.**
- **Report-attribute heads off** (−0.0006, 80% [−0.0011, −0.0001], on the 2026-10-01 recipe).

## Per-slot budget split between its own series

**Hypothesis; untested, no design yet.** Whether splitting a slot's own budget more evenly between
*its own* series helps. It is the part of the repeated-series evidence that does not starve other
types, so the reallocation rejection does not cover it. Every other question on this axis is closed
in the [triplet-budget investigation](../history/triplet-budget-allocation.md), which owns the
evidence: fixed-total reallocation, the missing-type fill, budgets A/B and the repeated-series
share were all measured and none survived.

## Attention pooling

The hypotheses below remain untested. Acquisition-count-normalised attention was
trained and [rejected](../validation/results/2026-09-17-count-normalized-attention-result.md)
(−0.0035 [−0.0069, −0.0001], menisci worst hit) and its option is gone from the code. Gated
attention and the per-target residual readout were measured at +0.0010 and −0.0005 on 2026-09-18,
at or below the control mean
([overnight](../validation/results/2026-09-18-overnight-results.md)); they are rejected, and
removing their options is a [refactor](refactors/retired-options.md). The
attention temperature τ 0.75 is adopted at inference and carries the best submission
([result](../validation/results/2026-09-17-attention-temperature.md),
[state](../ongoing/state.md)).

- **Several attention heads per label** (hypothesis, no design yet). Listed during the
  [crop-selection investigation](../history/crop-selection-investigation.md#options-not-pursued).
- **Nonlinear classifier after pooling** (hypothesis; no recorded comparison on the current
  recipe). Replace each independent linear readout with its own small `256 → 128 → 1` MLP,
  using GELU between layers and preserving the current attention pooling and supervision.
  This isolates readout capacity; it adds no sharing between disease labels and is distinct
  from the [rejected compartment adapter](../validation/results/2026-09-20-compartment-adapter-result.md).
  A cheap architecture ablation, with weaker project evidence than head-LR or supervision
  comparisons. Implement only when requested, behind one option, against the linear-head control.
- **Training with the attention temperature** (hypothesis; untested). τ 0.75 is an inference-time
  override through prediction, export and the bundle; whether training should use it has never been
  tested, and a shared temperature can be absorbed into the scorer's last layer.

### Pooling dense bags at inference

The head trains on about 30 triplets per study and predicts on 115 to 160 at ×4 to ×6; the only
correction in use is one attention temperature per model. Measured on the ten children of
2026-10-04 ([result](../validation/results/2026-10-04-two-family-oof-distillation.md)) and on the
three-fold ResNet-50 of 2026-10-12
([result](../validation/results/2026-10-12-dense-bag-pooling-on-the-resnet-50.md)); all are
inference-only on existing checkpoints. On the 2026-10-12 model ×4 at τ 0.9 is the best setting
on all three folds, so these matter only for bags larger than ×4 or for a model without a sweep.

- **Each series at its ×1 share of the attention, as an inference option** (measured on fold 0 of
  the ResNet-50: +0.00006 at ×4, +0.00029 at ×6, +0.00084 at ×8 over the best single τ, with
  τ 0.9; on one ResNet-18 earlier +0.0009 at full coverage). It makes the multiplier nearly
  indifferent (spread 0.0005 against 0.0011) and does not lift ×6 or ×8 above ×4. S104 tested the
  per-type form with τ 1 and scored 0.951 against 0.955, which confounds the two. Needs the ×1
  count per series in the engine and an offset on the scores of the main and soft-tissue
  scorers; not built (the earlier `pooling-options` branch is deleted).
- **A temperature per label** (measured, not a recipe). The best τ per label differs between
  trained models: three values fitted on fold 0 give +0.00105 there and −0.00064 and −0.00069 on
  folds 1 and 2; within fold 0 they hold on unseen halves (+0.00076). S147 (fold 0 with its own
  values) scored 0.959 public without a like-for-like reference; S105 (τ 1.1 for effusion and
  synovitis on the ten children) scored 0.954 against 0.955 for S103. Open: values fitted per
  model on its own validation fold, which is the small form of the
  [head refit](big-bets.md#pooling-head-at-the-inference-bag-size). The export option
  (`--label-attention-temperature`) is in the worktree `knee-abnormality-label-temperature`,
  uncommitted.
- **τ = ln N_train / ln N_test as the default for an unswept model or multiplier** (measured by
  interpolation on the ten children). It gives 0.84, 0.83 and 0.81 at ×4, ×5 and ×6 and sits
  0.0001 below one tuned τ per family and 0.0003 to 0.0004 below the joint pair. It replaces
  tuning; it does not beat it. Applied per study from its own token count it loses 0.0002 to
  0.0003 to one τ.
- Measured and closed on 2026-10-12: a separate τ for the soft-tissue read, slices weighted by
  the training sampling density, the top ×1-sized set of tokens, and matching the ×1 attention
  entropy per study.

### Plane/contrast-specific feature adapters

**Accepted for the backlog on 2026-09-21; unimplemented and untested.** The
[pretraining research](../research/pretraining-and-external-data.md#adapting-2d-foundation-models-to-volumes)
describes AnyMC3D's view-specific LoRA adapters. A small residual feature adapter on our current
trainable ReXNet is a project adaptation of that idea, not a reproduction or a measured knee gain.

- **Change:** after the shared 256-dimensional projection, apply a small residual bottleneck
  selected by the token's plane/contrast type, before adding embeddings and computing attention.
  Use the adapted features for both scoring and pooling, and initialize the residual output at
  zero. Keep one backbone pass, the existing type metadata, image/token budgets and disease heads.
  These adapters can change which evidence is selected; type-specific readouts after pooling
  lost −0.0006 ([near-misses campaign](../validation/results/2026-09-25-near-misses-campaign.md)).
- **Comparison:** one adapter-mode option for the current model, a shared-adapter control and
  type-specific adapters. Match the shared control's total adapter parameter budget to separate
  specialization from extra capacity, and record actual runtime because equal parameter counts
  do not imply equal computation. Fix bottleneck sizes before the comparison; no width sweep.
- **Decision:** hold labels, split, incumbent seed, optimizer and training budget fixed. Report
  macro/per-label AUC, paired uncertainty, missing-type and vendor strata, time and memory.
  Confirm that type specialization improves on the shared adapter as well as the current model.
  Implementation and user-launched training remain pending.

## Report-label supervision

- **Balance each target column through training masks** (hypothesis; untested). Independently
  for each label, retain equal numbers of positive and negative observed training targets by
  masking randomly selected majority-class cells out of the loss.
  - *Definition:* positives are target values `>= 0.5` and negatives `< 0.5` for counting;
    retained cells keep their original soft values. Keep all minority cells and sample the same
    number of majority cells across the full training fold, with reproducible masks refreshed
    each epoch from the run's own recorded seed and the epoch. Mask individual study–target cells, preserving
    supervision for other labels on the same study.
  - *Run:* keep study membership, image workload, model and training budget fixed, and leave
    validation targets and masks unchanged. Compare with full supervision, recording retained
    positive and negative counts per target. This is separate from the rejected source-evidence
    mask.
- **Hard versus soft report-derived targets** (untested). Change only report-derived training
  targets; retain original gold labels and frozen validation targets. The completed
  [Tonylin source comparison](../validation/results/2026-09-13-tonylin-label-comparison.md) did
  not answer this question.
- **Source-aware uncertainty weighting** (untested; the binary variant was measured and
  rejected). Compare reduced supervision weights against full soft-target supervision. The
  [binary mask was rejected](../history/sharing/2026-09-13-label-masking-result.md); partial
  weighting remains unimplemented. Preserve gold labels and original evidence indicators; Steven
  v4 probabilities are not calibrated confidence.

### Auxiliary report-state prediction

**Hypothesis; unimplemented and untested.** Add a reporting head alongside the disease head
to learn `{not reported, negative, positive}` without removing existing disease supervision.
The [report-state audit](../public-labels/uncertainty-and-confidence-encoding.md#report-state-audit-and-implications-for-supervision)
owns the measured motivation, missingness semantics and limits of the three-state alternative.

- **Head and targets:** reuse each label's pooled features for three additional logits, with
  a separate softmax per label. In the order above, use `(1, 0, 0)` when the original Steven
  v1 score is exactly `0.5`; otherwise use `(0, 1 - v1, v1)`. This preserves soft polarity
  within addressed findings. Pin v1 through the source manifest; v4 cannot recover coverage.
- **Disease supervision:** retain the current class-balanced BCE, all nonmissing disease
  targets and official gold overrides. Reporting targets continue to describe v1 even on
  eligible gold training studies: a gold-positive finding can be unreported. Use only the
  original twelve disease outputs for validation scoring and submission.
- **Loss:** add soft-target reporting cross-entropy averaged over training cells, with one
  fixed auxiliary coefficient specified before launch. That coefficient remains to be chosen;
  do not automatically transfer binary disease class weights to the reporting states. Record
  per-label state counts and both loss components to expose unreported-state dominance.
- **Comparison:** implement behind one option only when requested, then compare with the current
  control at its fold and seed, preserving membership, frozen validation targets, image
  workload and training budget. Record seeds and source/config identities. Select by frozen
  disease macro ROC-AUC and inspect all twelve label AUCs; reporting accuracy is diagnostic.
  Repeat a promising result under the existing controlled-comparison rules before adoption.
- **Judge risk (measured 2026-09-22):** on fold 0, the negatives the model ranks worst are the
  ones the report never addressed, and 14% of such gold cells are positive
  ([report attributes](../public-labels/report-attributes.md#agreement-with-the-incumbents-predictions)).
  A reporting head learns which reports stay silent, which fold-0 validation rewards and the
  leaderboard need not; confirm any fold-0 gain publicly.

## Augmentation families

The pipeline and its settings are in
[schedule and augmentation](../modeling/schedule-and-augmentation.md#ten-epoch-schedule-and-augmentation).
Strength and on/off comparisons of the original families are in the
[overnight queue](overnight.md#already-measured); the new families, late affine shutdown, MixUp,
bias field, channel reversal and stochastic depth were measured in the
[augmentation campaign](../validation/results/2026-09-26-augmentation-campaign.md), and the
rejected candidates with their reasons are in
[MRI augmentation research](../research/mri-augmentation.md#rejected-or-deferred).

- **Transform ordering and selected interactions** (hypothesis; untested). Nine image families
  now run in a fixed order; the campaign showed their gains do not add. Keep supervision,
  sampling, model and training budget fixed against a matched reference, and record which findings
  benefit or deteriorate.
- **Before changing a family,** inspect sampled and forced transformations across planes,
  acquisition orientations and missing-series cases with the
  [augmentation viewer](../data-preparation/reflection-geometry-and-previews.md#reflection-preview-evidence);
  check anatomical plausibility, slice/channel consistency and spatial-coordinate handling.

Validation inputs stay unaugmented; inference augmentation belongs to the
[endgame](endgame.md#test-time-augmentation).

## BCE weighting and study sampling

**Hypotheses; unimplemented.** Keep BCE while varying one of positive/negative
term weights, fixed whole-target weights, study-resampling probabilities, or batch composition.
The [research note](../research/auc_loss_approaches.md#weighting-and-sampling-design-space) owns
the mechanisms and controls. Fixed class-balanced cell weighting is now
[implemented](../modeling/supervision.md#fixed-class-balanced-bce), with its run in the
[completed result](../validation/results/2026-09-19-balanced-bce-result.md); the separate cell-masking proposal
remains under [report-label supervision](#report-label-supervision).

The four positive-directed plus four random studies example illustrates one sampler, not an
accepted ratio or next run. Choose one concrete arm with the user before implementation, keep the
optimizer-step budget and validation fixed, and record per-label exposure and study repetition.
Feedback-driven allocation ran in the
[noisy-labels campaign](../validation/results/2026-09-23-noisy-labels-campaign.md), none adopted.

### Weighting strength: exponent and prevalence definition

**Damping exponent implemented, 2026-09-21, and set to 0.5 in the recipe; soft-prevalence
comparison untested.** The implemented weights set `w_pos/w_neg = ((1 - r_c) / r_c) ** k` with
`k = trainer.class_balance_exponent`, on the thresholded positive fraction; see
[supervision](../modeling/supervision.md#fixed-class-balanced-bce) for the exact formula. Full
balancing (k = 1) reaches 12.9:1 on Fracture and 6.8:1 on Synovitis, whose report-derived targets are
among the least reliable. The [k=0.5 and unweighted runs](../validation/results/2026-09-21-class-balance-exponent.md)
lost on fold-0 validation but beat k=1 on the public leaderboard. An external
recipe seen on 2026-09-19 uses the square root of that ratio, computed on soft prevalence, as a
`pos_weight`; the entry below isolates that remaining knob separately, keeping our cell weighting
and mean-weight-one normalisation. One change per run against a fresh control.

- **Soft instead of thresholded prevalence** (one new option, off by default). Compute `p_c` as
  the mean of the observed soft targets rather than the `y >= 0.5` fraction, leaving the weight
  formula untouched. This is not a small refinement: the report-derived values sit at 0.25 and
  0.355, so soft prevalence is far above the thresholded fraction on exactly the rare labels
  (Synovitis 0.341 against 0.126, Fracture 0.187 against 0.073, Baker's 0.358 against 0.249,
  Lateral Meniscus 0.228 against 0.158), and it damps their weights accordingly. It confounds
  with the exponent, so run it only against the same control, never in the same arm.

A 1→0 ramped exponent tied on fold 0 and scored 0.941 public (S53) against 0.942 for S48
([noisy-labels campaign](../validation/results/2026-09-23-noisy-labels-campaign.md#class-weighting-changes-need-the-public-board)).

Later [soft-mass weighting](../../autoresearch/campaigns/losses/findings.md#completion-20260920t161203z-14ae6e65)
and [prior-offset](../../autoresearch/campaigns/losses/findings.md#completion-20260920t181036z-45163c52)
runs lost about 0.0092 and 0.0119 macro AUC.
They changed other loss semantics, so neither settles the soft-prevalence alternative. It remains
lower priority: retain one isolated comparison, with original soft targets, fixed training-fold
counts and observed-cell reduction, rather than a sweep.
Both runs, and two ranking losses, gain on soft AUC; they are parked on a
[watch list](../validation/results/2026-09-23-validation-auc-variants.md#losses-campaign-rescored-2026-09-24).

Per-label figures above come from `outputs/training/balanced-bce/fold_0/class_balance.json` for
the thresholded fractions, and from the Steven v4 blend restricted to the fold-0 training rows of
`all_5fold_v3` for the soft means (without gold overrides, which move 47 of 3,503 studies).

The external recipe also leaves negative weights at one and skips renormalisation, so its
per-label total loss mass rises with rarity (1.01 to 1.20 across our labels) and the global loss
scale with it. That is a third difference, deliberately not adopted here: our normalisation is
what keeps the comparison to the control a change of emphasis rather than of effective learning
rate.

## Anatomy-map recipe (from the anatomy-masks campaign, 2026-10-08)

Each is one change on the adopted recipe (the map's joint centre, sped up, with the two-knee pose
fix). Seven others are the forced seeds of the
[anatomy-recipe campaign](../../autoresearch/campaigns/anatomy-recipe/INDEX.md). Evidence and figures are in the
campaign's [run notes](../../autoresearch/campaigns/anatomy-masks/run-notes.md).

- **Region membership over the token's footprint** (source-backed, effect a guess): membership
  is decided at the token's centre though a token is 12 to 13 mm wide; the researcher's
  `regions-integrate-the-token-footprint`, queued and never run.
- **Across axis from the joint mask's shape** (measured on masks, effect a guess): the
  researcher's `joint-map-covariance-sets-the-across-axis`, queued and never run.
- **Lighter Dice in the map's loss** and **a third of the map's gradient to the backbone**
  (guesses): the steps between the two settings each tested in one run.
- **Joint gap as a class or relative to the other side** (guess): in millimetres the head
  recovered a fifth of a target that varies by under a millimetre.
- **One plane and centre per knee in two-knee studies** (source-backed): they are fitted over
  both knees pooled; about eight training studies, unknown in the hidden test set.
- **Which structure each label depends on** (read-only diagnostic, from the working notes of
  2026-10-07): where each head's attention falls by anatomy class is already measured
  (`outputs/analysis/f/frame-and-targets/anatomy-sections-audit/`); what is not is the ablation: at
  validation, drop the tokens of one predicted class at a time and see which targets fall.
- **Cleaning rules checked against the masks** (guess, from the same notes): series in which no
  knee is found point at wrong body parts or bad scans; rule-based checks designed against the
  masks, with no trained model in the loader.
