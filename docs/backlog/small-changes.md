# Small changes

Small code additions an agent can implement in one sitting, each behind one new option on the
[current recipe](../modeling/current-model.md) (read-only diagnostics need none). The user
supervises them: implement one only when the user asks. Once it is implemented, queue its run in the
[overnight queue](overnight.md) (one change against a fresh control, seed 42, fold 0) and
remove the entry from this file. Runs follow the
[controlled-comparison rules](../validation/validation-strategy.md#controlled-comparisons). Each
entry says whether its motivation is measured or a hypothesis.

## Attention pooling

- **Acquisition-count-normalized attention** (hypothesis; external evidence only). Subtract
  `log(tokens contributed by the acquisition)` from each token's attention score, so every series
  starts with the same attention mass whatever its triplet count.
  - *Why:* a study's softmax runs over the spatial tokens of every retained triplet, so a series
    with twelve sampled triplets outweighs one with a single triplet by count alone. The per-plane
    budgets in `data.series_types` (18 axial versus 15 coronal and sagittal, times the image
    multiplier) skew attention by sampling budget rather than evidence. Nothing balances series
    against series.
  - *Evidence (unverified here):* an experiment log from another of the user's competitions kept
    this change ("acquisition-count-normalized MIL attention: subtract log retained windows per
    acquisition", scored 0.92304 with model, sampler and optimizer unchanged), with no control
    delta recorded.
  - *Run:* keep the sampler, supervision and budget fixed, and report per-target AUC, since plane
    budgets differ per target.
- **Several attention heads per label** (hypothesis, no design yet). Listed during the
  [crop-selection investigation](../history/crop-selection-investigation.md#options-not-pursued).

## Weight averaging

**EMA and SWA through one averager** (hypothesis; neither exists in the code or has been tested
here). One new option, `optimizer.weight_averaging`: `null`, `{method: ema, decay}` or
`{method: swa, epochs}`, one method per run. It is required, with `null` in
`configs/training.yaml`, and SWA `epochs` must not exceed `trainer.max_epochs` minus
`optimizer.warmup_epochs`. Once implemented, the runs are `02-ema-0.999` and `03-swa` in the
[overnight queue](overnight.md#next-queue-proposed-pending-the-users-decisions). The
[optimizer research](../research/optimizer-research.md#4-ema-first-additional-training-feature)
gives the background; this design comes from the 2026-09-16 queue design pass.

- *Averager:* a hand-written `deepcopy` of the live model inside the Lightning module, updated
  after each optimizer step with a stateless per-step weight. Floating tensors (parameters and
  BatchNorm statistics) are interpolated, integer buffers copied, and the copy is exact while the
  weight is 1. A second constructed model would draw from the global RNG and shift the DataLoader
  shuffle. `AveragedModel` pushes int64 buffers through float arithmetic, timm `ModelEmaV3` cannot
  weight uniformly for SWA, and a callback cannot swap weights before the strict checkpoint load.
- *EMA:* copy the live weights until warm-up ends (step 438), then average with decay 0.999 per
  update. Its 1,000-step (2.3-epoch) horizon matches the cosine tail, where `00-baseline`'s live
  AUC was flat (0.89269, 0.89269, 0.89205, 0.89276 at epochs 7–10). Decay 0.99 (100 steps) would
  stay close to the live weights, and 0.9995 puts half its weight on epochs 5–7. BatchNorm
  statistics are averaged with the weights.
- *SWA:* a uniform per-step average over the last K = 3 epochs (from step 3,066) with the schedule
  unchanged. Lightning's `StochasticWeightAveraging` is unsuitable: it swaps in an epoch-level
  `SWALR`, flattens the 3:1 backbone:head learning-rate ratio, leaves out the final epoch, adds a
  pseudo-epoch that becomes a selection candidate, keeps the average out of the saved `model.*`
  weights, and resets BatchNorm statistics that `backbone_eval` never re-estimates.
- *SWA BatchNorm refresh:* at the end of each window epoch, re-estimate the BatchNorm layers that
  are in training mode (none under `backbone_eval`) with a forward-only pass over an unshuffled
  training loader, under the training precision and a forked RNG so the training random streams
  stay untouched.
- *Saved and validated weights:* checkpoints store the averaged weights as `model.*`, so
  prediction, export and fold-run comparisons use them unchanged, and the live weights alongside
  in `live_state_dict`, which export drops.
  - Both models are validated on the same batches every epoch: `val_macro_auc` scores the averaged
    weights, while `val_live_*` metrics and `live_*` epoch files record the live ones.
  - Checkpoint selection and early stopping use only the averaged score, keeping the control's 10
    candidates.
  - Where the copy equals the live model (EMA epoch 1, SWA epochs 1–7), both metric sets must
    match.
- *Costs (estimated):* +58 MB of GPU memory (0.3% of the 20.0 GiB peak), checkpoints from 175 to
  about 233 MB, a few seconds of updates, about 3 min per run for the second validation forward,
  and 1–1.5 min per SWA refresh (3–4.5 min for K = 3).
- *Comparisons:* EMA against `00-control` and against its own live weights; SWA against its live
  weights. Report last-epoch values for both models, which involve no selection. EMA averages
  BatchNorm statistics while SWA recomputes them, so part of an EMA-versus-SWA difference is not
  the averaging itself.
- *External evidence:* one public EfficientNet-B0 fold-0 ablation reported 0.7914 → 0.7940 with
  EMA, judged noise by its author
  ([public findings](../research/public-research.md#geometry-sampling-encoders-and-losses)), and a
  public training notebook averages its top epochs' weights
  ([public code](../research/public-research.md#public-code-worth-inspecting)). Averaging
  baseline1's epoch 7–9 probabilities gained +0.004267 on 12 reused validation studies, which is
  not a validated gain ([baseline1 review](../history/baseline1-review.md#follow-up-experiments)).

## Gradient-norm logging

**Log the pre-clip total gradient norm** (diagnostic; no option). Nobody knows whether
`trainer.gradient_clip_val: 1.0` ever binds, so clipping runs at 0.5 or 5.0 may be no-ops. Log the
norm in the Lightning module's `on_before_optimizer_step`, where Lightning 2.x still sees
unclipped gradients; it is read-only and draws no randomness, so numerics are unchanged. Whether
it ships with the weight-averaging change is the user's call. It unblocks the
[clipping candidates](overnight.md#candidates-for-later-queues).

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
[1C](../history/sharing/2026-09-14-overnight-hyperparameter-results.md), treated as settled, and
strong affine (−0.002389) on the
[current base](../validation/results/2026-09-15-overnight-results.md). Affine-off and
intensity-off runs never started; they need no code and are `09-affine-off` and
`15-intensity-off` in the [overnight queue](overnight.md#next-queue-proposed-pending-the-users-decisions),
whose [later candidates](overnight.md#candidates-for-later-queues) hold the other probability and
strength values of the existing families.

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

## Non-target series filtering

**Filter series that are not the target knee** (measured in individual studies; prevalence
unmeasured). Training study `…93672103` contains five ankle series (talus, calcaneus, malleoli)
filed in knee slots, and at least one study appears bilateral (the other knee's axial, coronal and
sagittal series) ([renders](../../outputs/analysis/study-center-robust-20260915/renders/)). All of
them currently reach the model. `BodyPartExamined` cannot flag them: those ankle series are
tagged KNEE, while knee series carry ANKLE, WRIST, HEART or spine tags.

1. Measure how common this is: visually census the 159 series dropped by the
   [robust center fit](big-bets.md#robust-study-center), which are the strongest candidates, then
   check other signals (field of view far from the study's other series, center projection
   outside the acquired range, unusual slice counts or extents).
2. Decide on a rule (geometric consistency with the study's main knee series, or a cheap
   image-based body-part check) and whether excluded series are dropped or down-weighted.
3. Check the test examples for the same pattern. Any rule must run identically at Kaggle
   inference.

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
