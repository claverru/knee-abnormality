# Small changes

Small code additions an agent can implement in one sitting, each behind one new option on the
[current recipe](../modeling/current-model.md) (read-only diagnostics need none). The user
supervises them: implement one only when the user asks. Once it is implemented, queue its run in the
[overnight queue](overnight.md) (one change against a fresh control, seed 42, fold 0) and
remove the entry from this file. Runs follow the
[controlled-comparison rules](../validation/validation-strategy.md#controlled-comparisons). Each
entry says whether its motivation is measured or a hypothesis.

## Superseded same-protocol retakes

**Hypothesis (measured composition, untested as a change).** Drop the superseded series when a
study has a true same-protocol retake with a later `SeriesNumber` (51 of 661 repeated-type slots
in fold-0 validation, about 258 of 6,868 repeated-slot triplets, 0.6% of all validation triplets),
instead of letting the tie-break arbitrate between them. This is the one lever in the [tie-break
investigation](../data-preparation/physical-sampling/evaluation-tie-break-investigation.md) that
changes realized triplet quality rather than just determinism. The implemented source ranking
already prefers the newest retake when entitlements tie, but a superseded series still supplies its
entitlement share of the targets.

## Study-level triplet total

**Hypothesis (test-time ties; training effect untested).** Today each type has a fixed budget, so a
study's total depends on which types it has: fold 0 has 112 studies at 32 training triplets, 529
at 26 and about 260 at 21. Here the total is set per study and split over the present types in
proportion to today's budgets (6/6/5/5/5/5). This applies in training, validation and Kaggle
inference, and changes the preprocessing identity. It is part of the
[allocation investigation](../ongoing.md#triplet-budget-allocation). Each variant is its own run.

- **Top-up** (preferred). A study whose per-type budgets sum below T = 26 training triplets is
  raised to 26, and studies at or above 26 keep their budgets.
  - *Compute:* +5.5% on fold 0 (26.8 against 25.4 training triplets per study).
  - *Why:* proportional renormalisation was the best missing-type rule measured, and no study
    loses triplets. That matters because trained uniform cuts cost 0.004–0.005 (`less_images`,
    `less_less_images`).
  - *Test time, frozen checkpoints, T = 52 evaluation triplets:* +0.0011 on search and −0.0002 on
    confirm (p ≈ 0.006), a tie.
- **Fixed total.** Every study gets T = 26 training triplets, so six-type studies drop from 32 to
  26.
  - *Compute:* +2.5%.
  - *Test time:* +0.0009 on search and −0.0006 on confirm, a tie.
  - *Risk:* the six-type cut is about the size of cut that hurt in training.
- *Implementation:* compute each study's counts from its present slots before `select_triplets`,
  using largest-remainder rounding. Cap each count at the slot's capacity and redistribute the
  excess.
- *Run:* report per-target AUC and realised triplets per availability pattern.

## Repeated-series share

**Hypothesis (the value of a second series is measured on frozen checkpoints; the rule is untested
in training).** A slot with k usable series gets its base budget × (1 + β·(k − 1)). The increase is
taken proportionally from the study's other present slots, so the total stays fixed. Fold 0 has
660 repeated slots in 436 studies.

- *Why:* on the [test-time allocation](../validation/results/2026-09-17-test-time-allocation.md#repeated-series-in-one-slot)
  cache, keeping only the top-ranked series cost 0.0019 on search and 0.0037 on confirm. Extra
  triplets also helped more from the second series than from the top one in all six comparisons.
- *Test time, frozen checkpoints:*

  | β | Search | Confirm |
  |---|---:|---:|
  | 0.5 | +0.0003 | −0.0003 |
  | 1.0 | −0.0010 | −0.0015 |

  A frozen model does not reward the larger share, so any gain would have to come from training.
- *Run:* β = 0.5, one run, with the split between series left to the sampler.

## Per-type budget reallocation

**Hypothesis, evidence from removal and capacity analysis.** Shrink axial T1/PD's `max_images`
budget (rarest type: 179 of 904 fold-0 validation studies, weakest and mixed-sign removal effect)
and grow fluid types' budgets (largest per-type removal effects, and already over-attended
relative to token share — see the updated [attention pooling](#attention-pooling) entry), with a
missing-type fallback order: same plane's other contrast first, then coronal fluid, axial fluid,
sagittal fluid, until a series runs out of positions. Two candidate budgets against the current
18/18/15/15/15/15 (axial T1/PD, axial fluid, coronal T1/PD, coronal fluid, sagittal T1/PD,
sagittal fluid): A = 9/18/15/21/15/18 (totals 96 all-types, 87 typical five-type study); B =
9/18/12/21/12/15 (totals 87 and 78). Series capacity supports both without shortfall
([evidence](../validation/results/2026-09-17-plane-contrast-importance.md#proposed-per-type-budgets)).

- *Caveats:* leave-one-type-out removal on a trained checkpoint measures reliance under the
  current budget, not the marginal value of adding slices; series dropout gets no reallocation
  today; keep-only-plane/contrast checkpoints are off-distribution (trained on all six types,
  evaluated with five zeroed); removal happens after preparation, so these numbers do not capture
  a geometry shift from a genuinely smaller acquisition set. Axial T1/PD's removal effect is also
  confounded with an easier study population — studies that have it show lower intact macro AUC
  (0.8855 vs 0.8920) and different target prevalence (e.g. ACL 26.3% vs 19.5%).
- *Earlier, related evidence:* fewer slices already cost accuracy in the restored-code queue: 20%
  fewer slices (15/15/12/12/12/12) lost 0.0041 macro AUC and trained 20% faster, 35% fewer
  (12/12/9/9/9/9) lost 0.0051 and trained 37% faster
  ([result](../validation/results/2026-09-16-restored-code-hyperparameter-runs.md)), a uniform cut
  rather than a per-type reallocation.
- *Run:* against a fresh control, report per-target AUC (plane/contrast budgets affect targets
  unevenly) and realized triplet-quality proxies, since reallocation is a budget change, not just
  a total-slice change.
- *Test-time evidence (2026-09-17):* on frozen checkpoints,
  [reallocations among present types did not confirm](../validation/results/2026-09-17-test-time-allocation.md).
  For a missing type, the fallback order above scored worse than losing the budget, while
  proportional renormalisation over the present types did better. The proportional gain matches
  uniform scaling at the same compute. A training run is still the only test of a
  training-time allocation.
- **Random per-type allocation in training is retired.** Drawing each study's per-type counts from
  a Dirichlet every epoch tied
  ([result](../validation/results/2026-09-17-allocation-concentration-result.md)) and its purpose,
  making a test-time allocation search on-distribution, came to nothing: that checkpoint responds
  to reallocation exactly like the fixed-split ones
  ([test-time allocation](../validation/results/2026-09-17-test-time-allocation.md#the-allocation-trained-checkpoint-responds-the-same)).
  The option was removed from the code.
- **Reallocation after series dropout** (hypothesis). Give a dropped slot's triplets to the
  study's remaining types by the allocation rule. This needs dropout to move from the model to
  the sampler, because today it drops encoded series in `model.py`.

## Attention pooling

Acquisition-count-normalised attention was implemented, trained and
[rejected](../validation/results/2026-09-17-count-normalized-attention-result.md): equalising each
series' attention mass lost 0.0035 [−0.0069, −0.0001] against the same recipe without it, with the
menisci worst hit, and it was neutral-to-negative at inference. The option was removed from the
code.

- **Several attention heads per label** (hypothesis, no design yet). Listed during the
  [crop-selection investigation](../history/crop-selection-investigation.md#options-not-pursued).
- **Attention temperature** (**measured on frozen checkpoints; not implemented**). Divide the
  scorer output by one shared `tau` before the masked softmax in `_attention_pool`
  (`src/knee_abnormality/modeling/model.py`). The diagnostic is done: on the token caches, `tau`
  0.75 gains +0.0024 [+0.0012, +0.0037] macro AUC over 1.0 at today's allocation, with a smooth
  peak at 0.7–0.75 and every one of four checkpoints gaining 0.0021–0.0026
  ([result](../validation/results/2026-09-17-attention-temperature.md)).
  - *Not logit calibration:* scaling a target's final logits keeps its AUC; attention temperature
    changes the evidence mixture and can change rankings.
  - *Implementation:* one shared value, not one per target. Inference only needs the pooling
    change, so existing checkpoints and their Kaggle bundles can use it; decide whether training
    also uses it, since a trained scorer can absorb a shared temperature into its last layer and
    may move the optimum.
  - *Before adopting:* confirm on the exact path rather than the cache, then on the leaderboard;
    the fold-0 gain is the size that has failed to transfer before.
- **Identity-initialized gated attention** (hypothesis; external evidence only). Multiply the
  scorer's tanh branch by a gate that starts at one:
  `a_t(h) = w_t^T [tanh(V h + b_V) * 2 sigmoid(U h + b_U)] + b_t`, with `U` and `b_U` zero. Keep
  the dropout placement and probability, and the linear head.
  - *Why:* each token's score comes from one hidden transformation (256→128→tanh→12). The gated
    form of [attention-based deep MIL](https://proceedings.mlr.press/v80/ilse18a.html)
    ([authors' code](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py))
    adds a multiplicative selection branch. Gated pooling appeared here only inside the retired
    [crop-selection models](../history/crop-selection-investigation.md), never in isolation. The
    existing scorer may already have enough capacity.
  - *Cost:* one `Linear(256, 128)` per token, 32,896 parameters; no extra backbone pass.
  - *Checks:* construct the branch without shifting the existing initialization or the training
    random stream (zeroing a layer still consumes RNG at construction); logits equal the ungated
    model's at initialization on identical inputs; gate parameters get gradients and sit in the
    head's optimizer group; masking and per-study normalization unchanged.
  - *Run:* alone, not combined with temperature, the readout below, several heads or count
    normalization.
- **Per-target residual nonlinear readout** (hypothesis). Keep the linear head and add, for each
  target, an independent `Linear(256, 32) → GELU → Linear(32, 1)` on that target's pooled vector
  (after `pooled` in `_attention_pool`), with the last layer zero-initialized and no new dropout:
  `s_t = w_t^T z_t + b_t + v_t^T GELU(U_t z_t + c_t) + d_t`.
  - *Why:* attention weights sum to one, so today's logit is an attention-weighted average of
    linear token scores; the pooled vector gets no nonlinear readout. It cannot recover
    information discarded by pooling or add token-to-token geometry.
  - *Cost:* 99,084 parameters over the 12 branches, applied to pooled vectors, not tokens.
  - *Checks:* vectorize the branches but keep them independent and in target order; baseline
    logits at initialization, finite gradients, correct optimizer group. The residual path is
    intentional; do not replace the linear head.
  - *Distinct from* the shared nonlinear disease features of
    [medial and lateral transfer](big-bets.md#medial-and-lateral-disease-transfer): no coupling
    between targets. *Risk:* extra capacity may fit report-label noise, so judge per-target AUC,
    not training BCE. Run it separately from gated attention.

## Gradient-norm logging

**Implemented (2026-09-16), shipped with the weight-averaging change.** `KneeModule` logs the
pre-clip total gradient norm as the `grad_norm` step metric (diagnostic; no option, always on) from
`on_before_optimizer_step`, where Lightning 2.x still sees unclipped gradients
(`torch.nn.utils.get_total_norm` over `self.model.parameters()`, which only reads `.grad`; it draws
no randomness and does not clip, so numerics are unchanged — verified bit-identical against the
pre-change code with the gate off). Nobody yet knows whether `trainer.gradient_clip_val: 1.0` ever
binds; that reading comes from the first run that logs it. It unblocks the
[clipping candidates](overnight.md#candidates-for-later-queues).

## 3D embedding: training-time effect

**Hypothesis; needs multiple seeds.** The retired additive 3D embedding is ignored at inference:
replacing it with the `unknown_3d` vector changed fold-0 macro AUC by +0.000037 [−0.00011,
+0.00017] on the knee-side checkpoint
([result](../validation/results/2026-09-14-knee-side-embedding-result.md)), and the champion scored
0.936 publicly both with and without it
([submission](../inference/submissions/2026-09-15-champion-without-3d-at-inference.md)). Yet the
champion (with 3D, one training) scored 0.936, above all four no-3D trainings of the same recipe
(0.928–0.934, mean 0.931, standard deviation about 0.0025;
[seed controls](../inference/submissions/2026-09-17-seed-control-submissions.md)). The simplest
explanation is a lucky run; a training-time effect is possible but unproven.

- **Possible mechanisms:** scaffolding (position helps attention and the backbone learn
  position-aware features early, after which visual features carry the information and 3D is
  ignored); regularisation (per-study embeddings that flip with reflection change token norms,
  normalisation statistics and attention temperature); a different random stream (adding a
  module changes initialisation draws, so a 3D and a no-3D model at the same seed are effectively
  different seeds, which alone could explain the public gap).
- **Change:** restore the embedding from the champion's source snapshot
  (`outputs/training/reflection-sigma40/source_snapshot.zip`) behind a three-way option: real 3D,
  placebo (same module, every token gets `unknown_3d`, so random-number use and capacity match)
  and none (today's model).
- **Test:** three arms × about three seeds (nine runs, about 4.5 hours, one overnight queue). Real
  versus placebo isolates the information at an equal random stream; placebo versus none isolates
  the module. Fold 0 probably shows a tie again, so the evidence would come mostly from public
  scores (three submissions per arm). To check scaffolding directly, save a checkpoint every epoch
  in the real-3D arm and run the 3D removal on each: removal should hurt early and not at the end;
  logging the gradient norm reaching the embedding shows the same.
- **Priority:** low; do it only to settle the question. Seed-to-seed spread (fold 0 0.0032,
  public up to 0.006) is the context for any result.

## Report-label supervision

- **Balance each target column through training masks** (hypothesis; untested). Independently
  for each label, retain equal numbers of positive and negative observed training targets by
  masking randomly selected majority-class cells out of the loss.
  - *Definition:* positives are target values `>= 0.5` and negatives `< 0.5` for counting;
    retained cells keep their original soft values. Keep all minority cells and sample the same
    number of majority cells across the full training fold, with reproducible masks refreshed
    each epoch using seed 42 and the epoch. Mask individual study–target cells, preserving
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
