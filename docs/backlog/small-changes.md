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

**Tested and rejected, 2026-09-25.** Random masking at 5% of valid tokens lost to the OA-detector
control at every inference τ; top-attention masking lost in the near-misses campaign
([results](../validation/results/2026-09-25-near-misses-campaign.md#follow-up-on-main)).

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

A 1→0 ramped exponent tied on fold 0; its public score is pending
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

## Same-cost backbone candidates

**Proposed 2026-09-25; ReXNet-R is implemented in the `convnext-atto` worktree and lost its first
run (−0.0053) with a measured over-large learning rate; its follow-up at `backbone_lr` 8e-5 is
open. ConvNeXt-Atto, run outside this list, lost −0.0192 and is closed
([backbone swap runs](../validation/results/2026-09-25-backbone-swap-runs.md)). ConvNeXt-nano and
HGNetV2-B3 are unimplemented.** Three timm backbones of about ReXNet-200's
size (13.8M parameters, 1.53 GMACs at 224 px) that could replace it. Each needs only a
`BACKBONE_MAPPINGS` entry with a verified origin; `model.backbone` is the option
([swapping a backbone](../modeling/backbone-swaps.md#code-requirements)). All three checkpoints
declare the configured ImageNet mean/std. Counts are backbone-only, measured on the meta device
with `FlopCounterMode` (timm 1.0.26).

| Candidate | Params | GMACs | Mapping | Motivation | Confidence |
|---|---:|---:|---|---|---|
| `rexnetr_200.sw_in12k_ft_in1k` | 14.0M | 1.56 | `_rexnet_mapping`, verified origin 0, stride 32, 2560 channels | Same family and cost, ImageNet-12k pretraining; lumbar 2024 7th place | Moderate, evidence-backed |
| `convnext_nano.in12k_ft_in1k` | 15.0M | 2.44 | [Offset origin](../modeling/backbone-swaps.md#mapping-pitfalls); the `convnext-atto` worktree's `_convnext_mapping` should only need a registration | Best in lumbar 3rd place's comparison; DINOv3 ConvNeXt-T 2nd in the [backbone search](../modeling/backbone-search.md#finished-model-comparison) | Moderate |
| `hgnetv2_b3.ssld_stage2_ft_in1k` | 14.2M | 1.77 | Offset origin, reported 2 px, unverified | Dense and GPU-friendly; family evidence only as a lumbar localizer | Low, speculative |

- **Excluded:** EfficientNet and EfficientNetV2 (measured losses on this dataset), `tf_` ports
  and reparameterizable or fixed-size families (mapping), `regnetz_c16` (0.5 mean/std),
  MobileNetV4, RepViT and small transformers (no MRI evidence), ResNet-18 (trailed ConvNeXt-nano
  in lumbar 3rd's comparison). See [backbone MRI evidence](../research/backbone-mri-evidence.md).
- **Cost:** GMACs do not predict wall time for depthwise-heavy families; ConvNeXt-nano's 1.6×
  GMACs may cost less than that. Estimate time from the run's own logs, then time any winner on a
  Kaggle T4 in FP16.
- **Comparison:** one run per candidate against the incumbent control, same fold, seed and
  budget. Scale `backbone_lr` by the checkpoint's convolution-weight RMS relative to ReXNet-200's
  ([weight scale](../modeling/backbone-swaps.md#checkpoint-weight-scale)); leave other rates
  unchanged. Record macro/per-label AUC, paired uncertainty, time and memory under the current
  adoption rule. Ensemble contribution belongs to the
  [endgame](endgame.md#ensembling).
