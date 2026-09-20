# Small changes

Small code additions an agent can implement in one sitting, each behind one new option on the
[current recipe](../modeling/current-model.md) (read-only diagnostics need none). The user
supervises them: implement one only when the user asks. Once it is implemented, queue its run in the
[overnight queue](overnight.md) (one change against a fresh control, fold 0, one recorded random
seed per run) and
remove the entry from this file. Runs follow the
[controlled-comparison rules](../validation/validation-strategy.md#controlled-comparisons). Each
entry says whether its motivation is measured or a hypothesis.

## Retired configuration cleanup

Remove the unused `trainer.min_delta` option: its only consumer is EarlyStopping, whose patience
covers the complete fixed training budget. This is implementation cleanup, not a hyperparameter
experiment. The rejected `repeated_series_share`, `gated_attention` and `residual_readout` branches
also remain in code; their removal is recorded with the corresponding findings below.

## Superseded same-protocol retakes

**Hypothesis (measured composition, untested as a change).** Drop the superseded series when a
study has a true same-protocol retake with a later `SeriesNumber` (51 of 661 repeated-type slots
in fold-0 validation, about 258 of 6,868 repeated-slot triplets, 0.6% of all validation triplets),
instead of letting the tie-break arbitrate between them. This is the one lever in the [tie-break
investigation](../data-preparation/physical-sampling/evaluation-tie-break-investigation.md) that
changes realized triplet quality rather than just determinism. The implemented source ranking
already prefers the newest retake when entitlements tie, but a superseded series still supplies its
entitlement share of the targets.

## Per-slot budget split between its own series

**Hypothesis; untested, no design yet.** Whether splitting a slot's own budget more evenly between
*its own* series helps. It is the part of the repeated-series evidence that does not starve other
types, so the reallocation rejection does not cover it. Every other question on this axis is closed
in the [triplet-budget investigation](../history/triplet-budget-allocation.md), which owns the
evidence: fixed-total reallocation, the missing-type fill, budgets A/B and the repeated-series
share were all measured and none survived. `data.repeated_series_share` is still in the code from
its rejected run and is a removal.

## Attention pooling

The hypotheses below remain untested. Acquisition-count-normalised attention was
trained and [rejected](../validation/results/2026-09-17-count-normalized-attention-result.md)
(−0.0035 [−0.0069, −0.0001], menisci worst hit) and its option is gone from the code. Gated
attention and the per-target residual readout were measured at +0.0010 and −0.0005 on 2026-09-18,
at or below the control mean
([overnight](../validation/results/2026-09-18-overnight-results.md)); they are rejected, and
`model.gated_attention` and `model.residual_readout` are removals rather than candidates. The
attention temperature τ 0.75 is adopted at inference and carries the best submission
([result](../validation/results/2026-09-17-attention-temperature.md),
[state](../ongoing/state.md)).

- **Several attention heads per label** (hypothesis, no design yet). Listed during the
  [crop-selection investigation](../history/crop-selection-investigation.md#options-not-pursued).
- **Nonlinear classifier after pooling** (hypothesis; no recorded comparison on the current
  recipe). Replace each independent linear readout with its own small `256 → 128 → 1` MLP,
  using GELU between layers and preserving the current attention pooling and supervision.
  This isolates readout capacity; it adds no sharing between disease labels and is distinct
  from the [compartment-transfer investigation](big-bets.md#medial-and-lateral-disease-transfer).
  A cheap architecture ablation, with weaker project evidence than head-LR or supervision
  comparisons. Implement only when requested, behind one option, against the linear-head control.
- **Training with the attention temperature** (hypothesis; untested). τ 0.75 is an inference-time
  override through prediction, export and the bundle; whether training should use it has never been
  tested, and a shared temperature can be absorbed into the scorer's last layer.

### Per-label attention temperatures

**Untried; first inexpensive investigation from the 2026-09-20 agenda review.** Global τ 0.75
already gained about 0.0020–0.0026 across four frozen checkpoints
([result](../validation/results/2026-09-17-attention-temperature.md)); that gain is already adopted.
Compare twelve static attention temperatures against the global 0.75 control, initially choosing
only between 0.75 and 1.0 per label and retaining 0.75 when the difference is unresolved.

- Screen with cached attention scores and scalar token contributions, keeping the checkpoint,
  sampled images and valid tokens fixed. Existing caches cover older checkpoints and approximate
  intensity windows; confirm a locked candidate with the chosen checkpoint's exact preprocessing.
  A new export has a bounded cost; no retraining is needed.
- Separate selection from confirmation; do not report twelve validation maxima as a combined gain.
  Measure macro and per-label AUC against global 0.75. Stop if specialization does not confirm.
- These are pooling temperatures, not final-logit calibration. Fold static scales into the scorer
  rows; if scores already include τ_old, multiply by τ_old / τ_new. No extra image pass or learned
  temperature network. Implement only when requested, through one per-label temperature option.

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

## Augmentation families

Measured so far: stronger intensity (+0.000249) and no noise (+0.000105) on
[1C](../history/sharing/2026-09-14-overnight-hyperparameter-results.md), treated as settled,
strong affine (−0.002389) on the
[current base](../validation/results/2026-09-15-overnight-results.md), and on the restored code
affine off (−0.0041 [−0.0078, −0.0002]) and noise probability 0.3 (−0.0019)
([result](../validation/results/2026-09-17-overnight-results.md)). Intensity-off was dropped by
[user decision](overnight.md#user-decisions-2026-09-16); the
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

## Contrast slot relabel

**Hypothesis (prevalence and inference-only effect measured; training effect untested).** Move
contrast-mislabelled series to the same plane's other slot: fat-suppressed fluid series out of
T1/PD slots and T2/T2* series without fat suppression out of fluid slots, only when headers and a
cheap intensity check on three decoded slices agree (12 training series, 12/12 reviewed
precision; relabelling the 4 fold-0 series moved macro AUC by +0.009 ×1000 on the control
checkpoints). This is the part of the [series-filtering census
rule](../metadata/series-filtering-census.md#proposed-rule-series-filtering-and-cleaning) that the
[implemented drops](../metadata/series-filtering-census.md#implemented-rule-2026-09-17) left out;
it needs one extra header per series and three decoded slices at Kaggle inference, and it changes
the preprocessing identity. Plane relabelling (one swapped study) stays out: it cost −0.21 ×1000
on all three control checkpoints.

## Per-plane spacing

Every plane shares `data.spacing_mm` and `data.image_size`, but the in-plane axes differ
(measured): sagittal spans patella to posterior capsule and needs the most millimetres (140 mm
clips posterior tissue in 34% of sagittal series, 150 mm in 8%), while axial cross-sections are
the smallest and spend pixels on background at the same scale. The gain is a hypothesis.

1. **Measure first, from metadata only:** native in-plane pixel spacing per plane, to see where
   0.625 mm or 0.5 mm already interpolates, and acquired field of view and body extent per plane,
   to see how much of the 140 mm input is background. The
   [input resolution research](../research/input-resolution.md) owns the existing evidence.
2. **Then, if the measurement supports it, vary spacing only,** as a per-entry
   `data.series_types` field, keeping `data.image_size` fixed so every tensor shape and batching
   are unchanged. Keep the ratio between planes within about 25%.

- **Per-plane pixel sizes are not worth it:** differing input shapes cannot share one batched
  tensor, so the model would need per-plane batches or padding, with new stride and geometry
  checks, for a second-order gain.
- **Tradeoff:** a shared target spacing is what makes a structure the same pixel size across
  scanners and studies. Per-plane spacing keeps that within a plane but asks the backbone to
  represent features at two or three fixed scales; positions stay in millimetres, so only the
  convolutional features change meaning. Affine augmentation already jitters scale by ±5%.

## RadImageNet-pretrained backbone

**Evidence-backed (one external report), untested here.** Load a RadImageNet-pretrained
ResNet-50 in place of its ImageNet weights, behind one option naming the pretraining source, and
compare the two at identical architecture, normalization and recipe. A participant reports that
going the other way — RadImageNet to ImageNet ResNet-50, same architecture, per-corpus
normalization corrected — cost them **−0.063**, the largest in-competition number anyone has
published on medical pretraining ([sweep](../research/public-research.md#the-2026-09-20-re-sweep));
the public 0.942 stack also carries RadImageNet heads as one of five families. It disagrees with
the [benchmark ranking](../research/backbone-mri-evidence.md) that put RadImageNet ResNet-50 19th
and MedDINOv3 25th, which is what makes it worth one measurement rather than an adoption.

Two caveats price the work. RadImageNet publishes ResNet-50, DenseNet-121, InceptionV3 and
IRV2 only, so neither of our backbones (EfficientNet-B0, ReXNet-200) can take these weights; the
comparison needs a ResNet-50 pair that is off the current recipe, and a win would then have to be
re-measured on a backbone we would actually submit. The weights are a third-party download, so
they need a pinned source, hash and licence check before any run.

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

**Exact comparisons untested; damping is secondary priority.** The implemented weights set
`w_pos/w_neg = (1 - r_c) / r_c` on the thresholded positive fraction. No exponent comparison
has been run: full balancing reaches 12.9:1 on Fracture and 6.8:1 on Synovitis, whose
report-derived targets are among the least reliable. An
external recipe seen on 2026-09-19 uses the square root of that ratio, computed on soft
prevalence, as a `pos_weight`; these two entries isolate the two knobs separately, keeping our
cell weighting and mean-weight-one normalisation. One change per run against a fresh control.

- **Damping exponent** (one new option, default reproduces today's run). Set
  `rho_c = ((1 - r_c) / r_c) ** k`, then `w_neg_c = 1 / (r_c * rho_c + 1 - r_c)` and
  `w_pos_c = rho_c * w_neg_c`, which keeps each label's mean weight at one for any `k`. `k = 1`
  algebraically returns the current `1 / (2 * r_c)` and `1 / (2 * (1 - r_c))`;
  `k = 0.5` is the arm to run, taking the square root of the emphasis ratio (Fracture
  12.9:1 to 3.6:1, Synovitis 6.8:1 to 2.6:1) while barely moving PF OA or Effusion. `k = 0` is
  unweighted BCE, already measured as the losing side of the
  [balanced-BCE result](../validation/results/2026-09-19-balanced-bce-result.md).
- **Soft instead of thresholded prevalence** (one new option, off by default). Compute `p_c` as
  the mean of the observed soft targets rather than the `y >= 0.5` fraction, leaving the weight
  formula untouched. This is not a small refinement: the report-derived values sit at 0.25 and
  0.355, so soft prevalence is far above the thresholded fraction on exactly the rare labels
  (Synovitis 0.341 against 0.126, Fracture 0.187 against 0.073, Baker's 0.358 against 0.249,
  Lateral Meniscus 0.228 against 0.158), and it damps their weights accordingly. It confounds
  with the exponent, so run it only against the same control, never in the same arm.

The [loss campaign queue](../../autoresearch/campaigns/losses/backlog.yaml) already owns damping
as `shared-tempered-balance`; this entry records the design, not a second execution queue.
Later [soft-mass weighting](../../autoresearch/campaigns/losses/findings.md#completion-20260920t161203z-14ae6e65)
and [prior-offset](../../autoresearch/campaigns/losses/findings.md#completion-20260920t181036z-45163c52)
runs lost about 0.0092 and 0.0119 macro AUC.
They changed other loss semantics, so neither settles this exponent or the whole-cell
soft-prevalence alternative. They lower priority: retain one isolated damping comparison, with
original soft targets, fixed training-fold counts and observed-cell reduction, rather than a sweep.

Per-label figures above come from `outputs/training/balanced-bce/fold_0/class_balance.json` for
the thresholded fractions, and from the Steven v4 blend restricted to the fold-0 training rows of
`all_5fold_v3` for the soft means (without gold overrides, which move 47 of 3,503 studies).

The external recipe also leaves negative weights at one and skips renormalisation, so its
per-label total loss mass rises with rarity (1.01 to 1.20 across our labels) and the global loss
scale with it. That is a third difference, deliberately not adopted here: our normalisation is
what keeps the comparison to the control a change of emphasis rather than of effective learning
rate.
