# Future work

Unstarted experiments retained after the 2026-09-13 review. Follow the
[work-tracking rules](../AGENTS.md#work-tracking) when moving an idea into active work.
The [1C overnight queue](sharing-experiments.md#2026-09-14-overnight-hyperparameter-results)
is complete. No new investigation or training is scheduled by this backlog.

Use the completed [reflection/sigma-40 run](validation.md#2026-09-14-reflection-sigma-40-result)
as the reference: 224 × 224 pixels, 140-mm crop, physical Gaussian sampling,
sigma 40 mm, independent ReXNet-200 heads and Steven v4. Completed crop, sampling,
slice-allocation and model
comparisons remain in [validation](validation.md) and
[modeling](modeling.md#finished-model-comparison).

## Fast baseline experiments

### Evaluation rule

Use **fold 0** of the [current validation protocol](validation.md#project-validation-protocol)
until near the end of the competition. Keep frozen training/validation memberships
and evaluation targets, thresholding the latter at `>= 0.5` for AUC. Record full-fold
and per-target AUC, separate gold diagnostics, BCE, runtime and peak VRAM. Preserve
seed 42, model size, training budget, image dimensions, physical crop, slice budget
and 2× validation density unless that setting is the experimental factor.
Training targets remain soft unless explicitly comparing supervision. Save the
configuration, source, input and checkpoint identities with all 904 validation
predictions. Optional export parity is not required for each run or submission.

### Report-label supervision

- **Balance each target column through training masks:** independently for each
  label, retain equal numbers of positive and negative observed training targets
  by masking randomly selected majority-class cells out of the loss. Define
  positives as target values `>= 0.5` and negatives as `< 0.5` for counting;
  preserve the original soft target values for retained cells. Keep all minority
  cells and sample the same number of majority cells across the full training
  fold, with reproducible masks refreshed each epoch using seed 42 and the epoch.
  Mask individual study–target cells, preserving supervision for other labels
  on the same study. Keep study membership, image workload, model and training
  budget fixed, and leave validation targets and masks unchanged. Compare with
  full supervision using the evaluation rule above, recording retained positive
  and negative counts per target. This class-balancing experiment is separate
  from the rejected source-evidence mask and remains unscheduled.
- **Hard versus soft report-derived targets:** change only report-derived training
  targets; retain original gold labels and frozen validation targets. The completed
  [Tonylin source comparison](validation.md#2026-09-13-tonylin-label-comparison)
  did not answer this question.
- **Source-aware uncertainty weighting:** compare reduced supervision weights
  against full soft-target supervision. The
  [binary mask was rejected](sharing-experiments.md#2026-09-13-label-masking-result); partial
  weighting remains unimplemented. Preserve gold labels and original evidence
  indicators; Steven v4 probabilities are not calibrated confidence.

## Medial and lateral disease transfer

The [transfer research](research/medial-lateral-transfer.md) recommends a weak
meniscus classifier-coupling comparison against the archived reflection/sigma-40
recipe, preserving independent attention, biases and OA classifiers. Evaluate the
intended targets and fixed acquisition/co-occurrence diagnostics alongside the
frozen macro AUC. This is proposed work; no training is scheduled.

If the result warrants another architecture, compare shared nonlinear disease
features with independent final classifiers, or a training-only shared auxiliary
detector with the corresponding independent-auxiliary control. Keep OA supervision
alignment and disease-combination weighting as separate factors. Frozen-feature
readout diagnostics require feature extraction after compute becomes available.
The linked report owns evidence, comparison controls and stopping criteria.

## Deeper augmentation investigation

Investigate intensity, affine, noise and study-level reflection more thoroughly.
Use the stronger-intensity and no-noise results from the
[completed 1C overnight queue](sharing-experiments.md#2026-09-14-overnight-hyperparameter-results).
Those gains are not established on the independent-head sigma-40 baseline.
Compare individual families, their probabilities and strengths, then selected
interactions and transform ordering. Keep supervision, sampling, model and
training budget fixed against a matched reference.

Inspect both sampled and forced transformations across planes, acquisition
orientations and missing-series cases using the
[augmentation viewer](data-preparation.md#reflection-preview-evidence). Check
anatomical plausibility, slice/channel consistency and spatial-coordinate handling.
Record per-target and gold/full-fold AUC, BCE, runtime and memory; examine which
findings benefit or deteriorate. Keep validation inputs unaugmented for this
training comparison; inference augmentation remains a separate experiment.
This investigation is deferred and has not been scheduled.

## Deferred preparation experiments

These remain conditional on the current workload; they are not scheduled runs.

- **Faster exact percentile selection:** first remeasure its cost after the
  [DICOM integration](data-preparation.md#production-dicom-integration).
  The historical profile spent 2.617 aggregate worker-seconds here. Production
  already gathers valid resampled float32 values into contiguous scratch and
  partitions all unique floor/ceil ranks together. Compare a native multi-selection
  loop with NumPy partition, including scratch construction and interpolation.
  If still material, consider exact coarse-histogram assistance followed by
  selection within the rank-containing bins. Keep the population acquisition-wide
  over successfully decoded complete groups. Single-rank partition, full sorting,
  per-slice bounds and approximate quantiles are outside this remaining comparison.
- **Learned acquisition intensity bounds:** compare a shared predictor with the
  current fixed percentile clipping. It must receive unclipped floating-point
  input; uint8 clipping has already discarded information. Parameters indexed by
  training series UID would not generalize to unseen acquisitions.

## Kaggle GPU profiling

Deferred by user decision on 2026-09-14. Leave the current inference notebook
and its GPU, batch and worker configuration unchanged while prioritizing model
generalization. No profiling run or inference change is scheduled.

When resumed, use one separate Kaggle notebook with a fixed checkpoint and a
representative set of available training studies, including expensive acquisition
and codec cases. Compare one/two GPUs and two/four studies per GPU batch with
four total loader workers; measure the old eight-worker condition separately.
Record cold startup, warm throughput, CPU preparation/loader wait, transfer,
backbone/pooling time, actual image/token counts and per-GPU completion tails.
Use device-aware timing and repeated, counterbalanced runs. This requires no
local two-GPU machine or scored competition submission.

Use the results to choose among eliminating duplicated metadata work, improving
CPU/I/O supply and partition balance, FP16, and backbone ONNX/TensorRT export.
The [completed inference investigation](sharing-generalization.md#inference-distinguish-the-measured-clocks)
owns the evidence, measurement limitations and efficiency-prize tradeoff;
submission-to-completion waits and three-study example runs do not establish
hidden-data scaling.

## Additional MRI research candidates

This needs evidence of a localization limitation before implementation;
no improvement over the reflection model is established.

- **Learned anatomical localizer plus global context:** consider only if inspecting
  errors identifies missed anatomy. Evaluate predicted crops without reference
  correction, retain the global view and include localizer/fallback cost. The
  completed [regional-fusion diagnosis](validation.md#2026-09-13-per-plane-fusion-usage-diagnosis)
  does not establish a benefit for this different, learned-localizer proposal.
  [Comparison design](research/experiment-priorities.md#adaptation-and-localization).

## Near the end of the competition

Defer this while improving the single-fold recipe:

- **Limited test-time augmentation:** evaluate a small number of transformations
  with consistent geometry and target handling, recording total inference cost.
  Training reflection is implemented; [reflection TTA](modeling.md#study-level-knee-reflection)
  has not been evaluated.

Additional-fold training and ensembling remain deferred until near the end of
the competition under the [current strategy](ongoing.md). CoAtNet backbone
support is already [implemented](modeling.md#coatnet-224-spatial-backbone);
this cleanup does not launch or select another backbone experiment.
