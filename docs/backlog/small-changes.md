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

## Per-type budget reallocation (rejected)

Budgets A and B, the missing-type fill and the repeated-series share were all measured and none
survived: reallocation at a fixed total does nothing above the detection floor on frozen
checkpoints ([test-time allocation](../validation/results/2026-09-17-test-time-allocation.md)),
the fill scores worse than losing the budget, and the repeated-series share topped the 2026-09-18
queue and then lost 0.004 publicly
([overnight](../validation/results/2026-09-18-overnight-results.md)). The study triplet total sits
unresolved inside the control spread, and `data.study_triplet_total: 26` was adopted anyway on
2026-09-18 for a structural reason rather than a measured gain: the floor only ever adds triplets,
so a study is never made worse, and it costs 5.5% compute
([recipe](../modeling/current-model.md)). `data.repeated_series_share` remains in the code from its
rejected run.

## Attention pooling

Acquisition-count-normalised attention was implemented, trained and
[rejected](../validation/results/2026-09-17-count-normalized-attention-result.md): equalising each
series' attention mass lost 0.0035 [−0.0069, −0.0001] against the same recipe without it, with the
menisci worst hit, and it was neutral-to-negative at inference. The option was removed from the
code.

- **Several attention heads per label** (hypothesis, no design yet). Listed during the
  [crop-selection investigation](../history/crop-selection-investigation.md#options-not-pursued).
- **Attention temperature is adopted at inference.** τ 0.75 gains +0.0024 [+0.0012, +0.0037] on
  fold 0 and carried the best submission, 0.939 public with ×4 images
  ([result](../validation/results/2026-09-17-attention-temperature.md),
  [submissions](../inference/submissions/2026-09-18-temperature-and-count-submissions.md)). It is
  an inference-time override through prediction, export and the bundle; whether training should
  use it is untested, and a shared temperature can be absorbed into the scorer's last layer.
- **Gated attention and the per-target residual readout are rejected.** Both are implemented
  behind `model.gated_attention` and `model.residual_readout`, off by default, and both landed at
  or below the control mean on 2026-09-18 (+0.0010 and −0.0005)
  ([overnight](../validation/results/2026-09-18-overnight-results.md)).

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
