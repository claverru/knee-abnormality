# Experiment priorities from the broader research

Use these as candidate comparisons after the active baseline work. They add to
the existing [label and geometry candidates](../future-work.md); they are not
queued runs, established defaults, or demonstrated improvements. Implement
training with Lightning and keep tabular preparation in Polars.

## Prioritized comparisons

| Priority | Comparison | Why it is useful | Relative cost |
|---|---|---|---|
| 1 | Simple pooling versus target-specific attention, then a small within-series RNN | Directly separates the aggregation choices supported by MRNet, RSNA solutions, and AnyMC3D | Low with fixed cached features; moderate with fine-tuning |
| 1 | Fixed physical crop/resolution with different slice coverage | Separates seeing the anatomy from merely increasing model capacity | Moderate |
| 1 | Controlled series-dropout and missing-slot stress evaluation | Uses complementary MRI information and exposes reliance on one input slot | Low to moderate |
| 2 | Ablate the implemented intensity, affine and noise families one at a time | Investigates MRI protocol robustness with the original split preserved | Moderate |
| 2 | Frozen encoder versus lightweight adaptation versus full fine-tuning | Tests whether adaptation is the bottleneck before buying a larger backbone | Moderate |
| 3 | Anatomy-guided pooling or localizer plus global branch | Promising when missed anatomy is a recurring error and supervision is available | High, including annotation/transfer validation |
| 3 | External knee pretraining or image-assisted label refinement | Requires verified label mapping, source eligibility, and additional fold isolation | High |

Cost describes additional complexity relative to one established baseline, not
a measured GPU-hour estimate. Current-task
[soft-label comparisons](../public-research.md#report-derived-supervision)
remain an early priority; do not postpone supervision quality merely to explore
an architectural idea from another competition.

## Pooling and coverage

**Pooling comparison.** Fix the encoder checkpoint, input windows, label file,
split, and training budget. Compare mean/max pooling with a small per-target
attention head. Add a small bidirectional RNN only within each ordered series,
then combine series at study level. Use missing-slot and padded-window masks
in every variant. Record standalone fold-0 AUC; ensemble comparisons are deferred
until near the competition end under the current project strategy.
The evidence motivating the comparison is in
[MRNet](mri-benchmarks.md#mrnet-knee-mri-classification),
[RSNA CT](rsna-ct.md), and [AnyMC3D](pretraining-and-external-data.md#adapting-2d-foundation-models-to-volumes).

A frozen-feature head experiment is a cheap screen. It cannot establish the
ranking after end-to-end fine-tuning, especially if the head changes which
image features the encoder learns. Retain a simple pooling baseline even if
the most complex head wins one seed.

**Coverage comparison.** Keep physical field of view and output resolution
fixed while varying where windows are sampled. Separately compare crop size
or resolution with the same anatomical coverage. Report the number of unique
slices as well as windows: overlapping triplets can multiply compute without
adding much anatomy. The different slice-count experiences in
[lumbar solutions](rsna-mri.md#lumbar-spine-degenerative-classification-2024)
motivate a target-by-target analysis, not one universal count.

Review where each target's useful evidence could be excluded: medial/lateral
menisci and OA, patellofemoral OA, peripheral MCL, posterior Baker's cyst, and
potentially scattered contusion/fracture findings. This is a coverage checklist
for review with actual images, not a claim that a particular plane is sufficient
for a diagnosis. Avoid selecting windows using held-out reference labels.

## Series robustness and augmentation

**Series dropout comparison.** Compare the current 0.1 probability with disabled
or tuned values; always retain at least one usable input slot. Compare normal
evaluation with deliberate omission of each slot, and stratify naturally
missing-series cases separately. A model robust to artificial dropout may still
fail on a protocol with different anatomy coverage.

**Augmentation comparison.** Disable or isolate the current contrast/gamma,
affine and noise families before attributing value to their combined recipe.
Test simulated lower resolution separately. Use training-only settings and
inspect the transformed pixels. For through-plane degradation, respect physical
spacing rather than treating every adjacent index as an equal distance.
The [M&Ms evidence](mri-benchmarks.md#mms-scanner-and-protocol-robustness)
supports investigating domain robustness but does not prescribe transform
strengths for knee pathology.

Any left/right or in-plane flip must have an explicit orientation and target
mapping. Image-left is not necessarily anatomical medial. A useful augmentation
in a lumbar or vessel model cannot establish label preservation here.

## Adaptation and localization

**Adaptation comparison.** Use one small pretrained encoder with its documented
normalization. Compare a frozen head, limited adaptation, and full fine-tuning,
recording trainable parameters, memory, and actual study throughput. If adding
LoRA, regenerate features through the adapted network during training. Fix the
backbone before comparing adaptation choices. Revisit a larger encoder only
after a smaller one demonstrates a reproducible limitation.

**Localization comparison.** First examine whether the baseline misses targets
because informative anatomy was not sampled. If so, compare a global branch
with a global-plus-local branch. Do not remove the global view until crop
coverage and target-level failures have been evaluated. Include decoding,
localizer inference, and fallback handling in the runtime measurement.
The [aneurysm evidence](rsna-mri.md#intracranial-aneurysm-detection-2025)
shows why a shared anatomical feature map is worth considering and why a
segmentation dependency can dominate runtime.

A localizer trained on competition annotations must produce held-out validation
crops without using those annotations. If using reference-corrected crops for
training, keep a separately identified deployment-style validation path. If
using a pretrained external localizer, record its source subjects and any
overlap with evaluation data. Attention-selected windows are pseudo-localization,
not newly acquired ground truth.

## Evaluation and decision rules

Follow the current [validation protocol](../validation.md#project-validation-protocol):
iterate on fold 0 with frozen mixed gold/report-derived targets and separate
gold diagnostics. Keep the existing five-fold assignment for later evaluation;
additional-fold training and ensembling are deferred until near the competition
end. Only 58 studies have reference annotations, with nine MCL positives.
This is development evaluation, not an untouched final holdout.

For every meaningful comparison, preserve:

- Configuration, code revision, seed, label-source version, split identity,
  training membership, checkpoint rule, and full preprocessing description.
- Complete held-out-fold predictions in the official target order; full-fold
  macro AUC, per-target AUC, and relevant subgroup coverage. Pooled all-study
  OOF evaluation belongs to the later multi-fold phase.
- Separate gold and report-derived diagnostics; the selected primary metric
  uses all study predictions against their frozen effective labels and is not
  an average of the subgroup macro AUCs.
- Paired study-level uncertainty estimates where used, with the resampling
  method and treatment of single-class resamples recorded. Such intervals do
  not include all variability from retraining or repeated model selection.
- Decode/preprocessing time, model time, total runtime, peak memory, and the
  number of folds, TTA passes, windows, and distinct slices actually used.

Predefine the principal comparison and checkpoint rule. A tiny best-seed gain
on the same reused 58 studies is insufficient reason to expand a model family.
When promising, repeat with an additional fixed seed or independent evidence;
when unresolved, prefer the simpler or cheaper model and retain the uncertainty.
Do not search seeds to obtain an apparent improvement.

For a fixed target and prediction set, a strictly increasing calibration
transform preserves ROC AUC. Threshold tuning for F1, lumbar class weights,
or an any-severe correction is therefore not automatically a knee-AUC gain.
Clipping, ensembling, and different transforms across models can change rankings;
evaluate those changes with held-out predictions rather than assume equivalence.

Before image-assisted label repair or stacking, distinguish current all-study
OOF predictions from the old reference-only protocol's shared report-only
training pool. Fold-isolate every learned component that uses evaluation labels. A final
stacker fitted to OOF predictions also needs held-out or nested evaluation of
its own; evaluating it on its fitting rows is optimistic. The
[code review](code-and-notebooks.md) identifies concrete places where artifact
names or fold loops alone do not establish this isolation.

## Directions to defer

Defer full-volume 3D training from scratch, new reconstruction/denoising systems,
large external-data downloads, and broad leaderboard blend sweeps until a
specific failure or controlled result justifies their cost. Dense auxiliary
tasks require a reliable annotation source. Published segmentation Dice,
reconstruction SSIM, and another competition's rank do not substitute for
reference-label knee AUC.

Move a chosen idea from [future work](../future-work.md) to
[ongoing work](../ongoing.md) when implementation starts. Preserve useful gains
and negative results in the canonical [modeling](../modeling.md),
[supervision](../supervision.md), [preparation](../data-preparation.md), or
[validation](../validation.md) document with the run evidence needed to assess
them. This review itself ran no experiments.
