# Small changes

Small code additions an agent can implement in one sitting, each behind one new option on the
[current recipe](../modeling/current-model.md) (read-only diagnostics need none). The user
supervises them: implement one only when the user asks. Once it is implemented, queue its run in the
[overnight queue](overnight.md) (one change against the incumbent, fold 0, the incumbent's recorded
training seed) and
remove the entry from this file. Runs follow the
[controlled-comparison rules](../validation/validation-strategy.md#controlled-comparisons). Each
entry says whether its motivation is measured or a hypothesis.

## AdamP for ReXNet

**Accepted for the backlog on 2026-09-21; unimplemented and untested.** Compare AdamP with
AdamW on the current ReXNet recipe. The [optimizer research](../research/optimizer-research.md#6-adamp-alternative-for-the-rexnet-branch)
motivates projecting out norm-increasing updates on scale-invariant weights. Published gains
are external evidence, not a measured knee-AUC improvement. AdamP uses one gradient evaluation
per step and adds no inference work.

- **Implementation:** one optimizer-choice option, retaining the backbone/head parameter groups,
  no-decay exclusions, scheduler, clipping and checkpoint/resume behavior. The research proposes
  `timm.optim.AdamP` with `delta=0.1` and `wd_ratio=0.1`; verify its installed implementation
  before fixing the candidate. Do not expose an optimizer sweep or unrelated new settings.
- **Comparison:** fresh AdamW control and AdamP candidate on the same current labels, split,
  incumbent seed, image exposure and training budget. Keep learning rates and betas fixed for
  the first comparison. AdamP's projected-weight decay is not numerically equivalent to AdamW's;
  record the effective policy and treat any subsequent decay tuning as a separate comparison.
- **Decision:** record macro/per-label AUC, paired uncertainty and training time, following the
  current adoption rule. Implementation needs a separate user request; training remains
  user-launched. Move the run to the overnight queue only after implementation.

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

### Attention-token dropout

**Accepted for the backlog on 2026-09-21; unimplemented and untested.** Stochastically mask a
small subset of dominant spatial tokens during training, separately for each study and disease.
[ACMIL](https://arxiv.org/abs/2311.07125) supports this mechanism in histopathology MIL, not a
measured knee-MRI gain. Concentrated attention is not inherently wrong; this is an incremental
regularization experiment, not a missing architectural capability.

- **Change:** after attention scoring and anatomical bias, exclude selected valid tokens from
  that label's softmax and renormalize over the remainder. Always retain a valid token for every
  study–label pair. Disable the masking at evaluation; keep the backbone pass, image budget,
  anatomical bias and existing series dropout unchanged. Series dropout removes whole
  plane/contrast slots; this changes evidence selection within retained acquisitions.
- **Comparison:** one token-masking mode option for off, random masking and attention-challenging
  masking. Compare a fresh current-recipe control with random valid-token masking and masking
  drawn from the highest-scoring tokens, matching the number removed per study and label.
  Fix a conservative candidate-pool size, masking rate and cap before implementation; no sweep.
- **Decision:** follow the controlled-comparison rules with identical labels, split, incumbent
  seed and training budget. Record macro/per-label AUC, paired uncertainty, masking counts and
  runtime. Inspect focal findings explicitly: retaining a valid token does not guarantee that
  lesion evidence remains. Require evidence of benefit beyond random masking before preferring
  attention-challenging masking. Implementation requires a separate user request.

### Plane/contrast-specific readouts

**Untried; preferred small architecture comparison from the 2026-09-20 agenda review.** Current
[pooling](../../src/knee_abnormality/modeling/model.py) collapses all acquisition types before the
linear classifier. Retain six partial vectors per label: `u_cq = sum_{i:type(i)=q} alpha_ci * h_i`.
Compare `z_c = sum_q w_cq · u_cq + b_c` with the current shared `w_c`, keeping the same global
attention denominator. Initialize every `w_cq` from `w_c`, reproducing the original predictions
apart from summation order; missing types contribute zero. Keep image and token budgets unchanged.

- Screen shared and type-specific heads on identical cached partial vectors with a refitted
  shared-head control. Record parent weights, input view, preprocessing and split identities;
  supervised feature learning must exclude confirmation studies. Stop if retaining type identity
  does not improve held-out ranking beyond refitting the original head.
- Confirm a useful screen in a normal-budget run behind one option. Compute type-specific token
  contributions without a large six-way pooling tensor; profile actual overhead before adoption.
- Check per-label AUC, missing-type studies and vendor strata, including a presence/attention-mass
  diagnostic: [acquisition availability tracks prevalence](../validation/results/2026-09-17-plane-contrast-importance.md#availability-and-prevalence-shift).
  Count normalization, gated attention and residual readouts above are different, measured changes;
  do not add separate within-type normalization or a nonlinear stack to this comparison.

### Plane/contrast-specific feature adapters

**Accepted for the backlog on 2026-09-21; unimplemented and untested.** The
[pretraining research](../research/pretraining-and-external-data.md#adapting-2d-foundation-models-to-volumes)
describes AnyMC3D's view-specific LoRA adapters. A small residual feature adapter on our current
trainable ReXNet is a project adaptation of that idea, not a reproduction or a measured knee gain.

- **Change:** after the shared 256-dimensional projection, apply a small residual bottleneck
  selected by the token's plane/contrast type, before adding embeddings and computing attention.
  Use the adapted features for both scoring and pooling, and initialize the residual output at
  zero. Keep one backbone pass, the existing type metadata, image/token budgets and disease heads.
  These adapters can change which evidence is selected; the
  [type-specific readouts](#planecontrast-specific-readouts) operate after pooling.
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
- **Disease supervision:** retain current class-balanced v4 BCE, all nonmissing disease
  targets and official gold overrides. Reporting targets continue to describe v1 even on
  eligible gold training studies: a gold-positive finding can be unreported. Use only the
  original twelve disease outputs for validation scoring and submission.
- **Loss:** add soft-target reporting cross-entropy averaged over training cells, with one
  fixed auxiliary coefficient specified before launch. That coefficient remains to be chosen;
  do not automatically transfer binary disease class weights to the reporting states. Record
  per-label state counts and both loss components to expose unreported-state dominance.
- **Comparison:** implement behind one option only when requested, then compare with a fresh
  current-recipe control on fold 0, preserving membership, frozen validation targets, image
  workload and training budget. Record seeds and source/config identities. Select by frozen
  disease macro ROC-AUC and inspect all twelve label AUCs; reporting accuracy is diagnostic.
  Repeat a promising result under the existing controlled-comparison rules before adoption.
- **Judge risk (measured 2026-09-22):** on fold 0, the negatives the model ranks worst are the
  ones the report never addressed, and 14% of such gold cells are positive
  ([report attributes](../public-labels/report-attributes.md#agreement-with-the-incumbents-predictions)).
  A reporting head learns which reports stay silent, which fold-0 validation rewards and the
  leaderboard need not; confirm any fold-0 gain publicly.

## Augmentation families

Measured so far: stronger intensity (+0.000249) and no noise (+0.000105) on
[1C](../history/sharing/2026-09-14-overnight-hyperparameter-results.md), treated as settled,
strong affine (−0.002389) on the
[current base](../validation/results/2026-09-15-overnight-results.md), and on the restored code
affine off (−0.0041 [−0.0078, −0.0002]) and noise probability 0.3 (−0.0019)
([result](../validation/results/2026-09-17-overnight-results.md)). Intensity-off was dropped by
[user decision](overnight.md#already-measured); the
[later candidates](overnight.md#candidates-for-later-queues) hold the other probability and
strength values of the existing families.

- **Late affine shutdown** (hypothesis; needs epoch-dependent augmentation). Affine probability
  0.7 for epochs 1–8 and 0 for epochs 9–10 (zero-based `current_epoch >= 8`), everything else
  unchanged, including series dropout, whose removal would also change the image workload.
  - *Why:* affine off for the whole run and the stronger affine both lost, so the current affine
    helps overall; this asks whether it still helps in the low-learning-rate tail.
  - *Implementation:* pass the epoch from `src/knee_abnormality/training/lightning.py` to the
    model's augmentation sampling and derive the effective probability without changing the saved
    recipe; log it, restore it on resume, and keep the other families' random draws unchanged where
    feasible.
  - *Run:* report the fixed last epoch as well as the selected one. Two clean epochs may be too
    short to matter; do not sweep the shutdown epoch before the first comparison is informative.

- **Transform ordering and selected interactions** (hypothesis; untested), after the family
  probabilities and strengths are compared. Keep supervision, sampling, model and training budget
  fixed against a matched reference, and record which findings benefit or deteriorate.
- **Simulated lower resolution** (hypothesis; a new family). Degrade through-plane resolution by
  physical spacing rather than slice index
  ([M&Ms evidence](../research/mri-benchmarks.md#mms-scanner-and-protocol-robustness)).
- **Before changing a family,** inspect sampled and forced transformations across planes,
  acquisition orientations and missing-series cases with the
  [augmentation viewer](../data-preparation/reflection-geometry-and-previews.md#reflection-preview-evidence);
  check anatomical plausibility, slice/channel consistency and spatial-coordinate handling.

Validation inputs stay unaugmented; inference augmentation belongs to the
[endgame](endgame.md#test-time-augmentation).

### Feature-space augmentation (DSU)

**Accepted for the backlog on 2026-09-21; unimplemented and untested.** Add conservative
feature-statistics augmentation alongside the existing image augmentations.
[DSU](https://arxiv.org/abs/2202.03958) perturbs feature-channel means and standard deviations
to model domain variation. Its vision results are external motivation, not a measured knee gain
or evidence that scanner shift is currently our bottleneck.

- **Change:** one training-only option at one early or intermediate ReXNet feature map, before
  spatial MIL. Perturb per-channel spatial statistics rather than adding another pixel transform
  or masking tokens. Draw the perturbation coherently across triplets from the same acquisition;
  keep statistics plane/contrast-aware instead of mixing unlike MRI contrasts. Use training-only
  statistics and define stable behavior for small acquisition groups. Fix the insertion point,
  application probability and modest strength before the first comparison; no layer/strength sweep.
- **Control:** disable feature augmentation entirely at evaluation. Preserve image augmentation,
  series dropout, image/token budgets, labels, split, incumbent seed and training schedule.
  Compare against a fresh current-recipe control without combining other pending experiments.
- **Decision:** follow the controlled-comparison rules and record macro/per-label AUC, paired
  uncertainty, runtime and scanner/vendor/contrast diagnostics with subgroup counts. MRI feature
  statistics also contain diagnostic signal: do not assume all variation is nuisance or claim
  scanner robustness from aggregate AUC alone. Implementation needs a separate user request;
  move its run to the overnight queue only after implementation.

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
Feedback-driven allocation is classified separately in
[big bets](big-bets.md#adaptive-training-allocation).

### Weighting strength: exponent and prevalence definition

**Damping exponent implemented, 2026-09-21; soft-prevalence comparison untested.** The
implemented weights set `w_pos/w_neg = (1 - r_c) / r_c` at the default
`trainer.class_balance_exponent = 1`, on the thresholded positive fraction; see
[supervision](../modeling/supervision.md#fixed-class-balanced-bce) for the exact formula. Full
balancing reaches 12.9:1 on Fracture and 6.8:1 on Synovitis, whose report-derived targets are
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

A ramped exponent belongs to [adaptive training allocation](big-bets.md#adaptive-training-allocation).

Later [soft-mass weighting](../../autoresearch/campaigns/losses/findings.md#completion-20260920t161203z-14ae6e65)
and [prior-offset](../../autoresearch/campaigns/losses/findings.md#completion-20260920t181036z-45163c52)
runs lost about 0.0092 and 0.0119 macro AUC.
They changed other loss semantics, so neither settles the soft-prevalence alternative. It remains
lower priority: retain one isolated comparison, with original soft targets, fixed training-fold
counts and observed-cell reduction, rather than a sweep.

Per-label figures above come from `outputs/training/balanced-bce/fold_0/class_balance.json` for
the thresholded fractions, and from the Steven v4 blend restricted to the fold-0 training rows of
`all_5fold_v3` for the soft means (without gold overrides, which move 47 of 3,503 studies).

The external recipe also leaves negative weights at one and skips renormalisation, so its
per-label total loss mass rises with rarity (1.01 to 1.20 across our labels) and the global loss
scale with it. That is a third difference, deliberately not adopted here: our normalisation is
what keeps the comparison to the control a change of emphasis rather than of effective learning
rate.
