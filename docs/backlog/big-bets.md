# Big bets

Investigation-sized plans that need focus sessions with the user: diagnostics, a design, several
runs or a new preprocessing identity. Start one only in a session with the user; when it starts,
move its entry to [ongoing work](../ongoing/INDEX.md) and remove it from this file in the same change.
Entries under [parked](#parked-until-a-specific-need) wait for a concrete need before they become
candidates. Each entry says whether its motivation is measured or a hypothesis.

## Position and knee side

Already measured: the
[knee-side embedding](../validation/results/2026-09-14-knee-side-embedding-result.md) and the
[token graph](../validation/results/2026-09-14-token-graph-result.md) tied and were retired, and
the [attention prior](../validation/results/2026-09-15-attention-prior-result.md) regressed by
0.002177 and was
[retired](../history/model-lineage.md#retired-attention-prior) with the model's
patient points. Remaining (hypotheses), from the
[position research sequence](../research/position-encoding.md#5-recommended-sequence) and the
[knee-side approaches](../research/knee-side-and-location.md):

- **Phase 0 diagnostics:** report laterality read for the persistent wrong-side studies,
  attention and token-position dumps, and frozen token probes. Plane-restricted inference on the
  champion was skipped by user decision.
- **Untied position path** (knee-side approach 5): compute attention from appearance and position
  while pooling appearance alone for the independent classifiers.
- **Common knee orientation** (approach 4) and **learned landmarks or localization supervision**
  (approach 7, which depends on a rules-compatible knee segmenter).
- **Side-conditioned sagittal sampling centre** (measured premise, unmeasured payoff). Sagittal
  slices step along patient L, and the compartment labels sit about 31 mm off the study centre
  while the sagittal draw is centred on it. The [spatial importance
  map](../ongoing/peripheral-coverage.md#spatial-importance-map) shows a central trough on that
  axis alone, flanks minus centre +0.126 against -0.634 coronal and -1.099 axial, carried by the
  compartment labels and symmetric about zero. The symmetry is the problem: pooled left and right
  knees make "medial and lateral matter equally" and "only medial matters, seen twice" produce the
  same picture. Two blockers, in order. First, the export centres and scales L but never mirrors by
  knee side, so it must be re-exported with left knees mirrored before the lobes can be told apart;
  the per-study utilities for that live outside this repository. Second, the offset must flip sign
  per knee side, and knee side does not exist anywhere in `src/` since the embedding was retired,
  so it would have to be derived from the [audited policy](../metadata/knee-side-policy-audit.md)
  at 96.14% training coverage and plumbed into preprocessing, with the abstentions taking no
  offset. That plumbing, not the offset, is the size of this bet.

Approaches 2 and 3 act through the 3D coordinate branch that approach 1's checkpoint barely used;
the user also declined approach 3 on 2026-09-14. Image-based side inference is deferred by user
request. The [coordinate audit](../metadata/knee-side-coordinate-investigation.md) owns the side
evidence.

## Medial and lateral disease transfer

Measured: weak meniscus classifier coupling
[regressed](../validation/results/2026-09-14-meniscus-soft-coupling-result.md). The
[transfer research](../research/medial-lateral-transfer.md) owns the remaining proposals
(hypotheses), their controls and stopping criteria: shared nonlinear disease features with
independent final classifiers, or a training-only shared auxiliary detector with the
corresponding independent-auxiliary control. Keep OA supervision alignment and
disease-combination weighting as separate factors. Frozen-feature readout diagnostics require
feature extraction.

## Learned acquisition intensity bounds

Hypothesis. Compare a shared predictor with the current fixed percentile clipping. It must receive
unclipped floating-point input; uint8 clipping has already discarded information. Parameters
indexed by training series UID would not generalize to unseen acquisitions. Other fixed quantile
levels lost: 0.001/0.999 by 0.0003 and 0.01/0.99 by 0.0021
([result](../validation/results/2026-09-17-overnight-results.md)).

- *Rejected source:* the DICOM `WindowCenter`/`WindowWidth` tags (measured 2026-09-20, 30 series):
  median 1.43× wider than the 0.005/0.995 bounds, vendor-dependent (Toshiba 1.77 against Philips
  1.20) and varying slice to slice in 23 of 30 series, so they are neither a tighter nor a
  sampling-independent bound.

## Sampling-independent intensity bounds

**Code observation; size measured, harm on the metric unmeasured.** `prepare_series`
(`src/knee_abnormality/data/preprocessing/preparation.py`) takes the intensity quantiles over the
unique slices of the selected triplets, so the same source slice gets different normalized pixels
when its companion slices change: between training epochs, and between training and the 2×
validation bag. The variability may also act as useful augmentation.

1. **Measured 2026-09-20** on 40 studies over six epochs
   ([bound stability](../data-preparation/preprocessing-implementation-and-profile.md#clipping-bound-stability-across-draws)):
   the high bound moves 0.072 at the median and 0.245 at p90 between draws of one series, against
   the 0.2 relative range of the contrast augmentation, and 14.2% of series exceed it. The spread
   is a small-sample effect concentrated in series with few triplets (0.150 at two or fewer, 0.025
   above six), not a property of the estimator: robust estimators calibrated to the same clip level
   gain about one point, and per-triplet bounds are four times worse. Not measured: normalized-pixel
   and clipping differences on shared source slices, the 1× versus 2× bags, or any effect on AUC.
2. **Only if material:** estimate the bounds from a deterministic sparse reference sample of each
   acquisition, independent of epoch, bag size and selected windows, identical in training,
   validation and inference, at the current quantile levels. Define short-series and
   decode-failure fallbacks. It changes the preprocessing identity; cache scalar bounds only under
   an identity that prevents stale reuse.

- *Distinct from* the quantile-level runs and the learned bounds above: it changes the pixel
  population, not the levels or the estimator.
- *Cost:* extra decoding, against the
  [select-before-decoding](../architecture/data-flow-and-iteration-speed.md#iteration-speed) rule
  that the [loading speedups](loading-speedup/INDEX.md) rely on.
- *Stop if* the variability is negligible, the runtime cost is disproportionate, or a controlled
  run gains nothing.

## Per-label loss selection and learned mixtures

**Research completed 2026-09-20; implementation and experiments unstarted.** Exploit the trusted
external observation that ranking improves ACL while focal works better for MCL. The user wants
to develop the principle independently of the contact's implementation. The
[research note](../research/per-label-losses.md) owns the results, primary sources, candidate
formulas, model fit, pair-coverage calculation and evaluation design.

1. **Compare independent heads:** cache the current model's per-label pooled features and train
   balanced soft BCE, focal and ranking classifiers for all twelve labels. Share expensive
   feature extraction, allow larger ranking pools and repeat classifier seeds. Use current
   class-balanced BCE as the control.
2. **Assemble per-label winners:** compare stable selections against the original model and
   refitted-BCE controls. Selected linear rows preserve inference cost. Retain the current loss
   when differences are unresolved.
3. **Evaluate joint continuation:** introduce the selected losses after feature learning,
   comparing with an equally long current-loss continuation. Check every label because shared
   features couple their gradients.
4. **Investigate mixtures:** compare fixed BCE-plus-ranking hybrids with a small per-label
   bilevel controller, first on frozen features. Nonnegative coefficients sum to one per label;
   learn them from a fixed tuning objective after virtual training updates, not by minimizing
   the same weighted training loss. Evaluate warmup, slower coefficient updates and shrinkage
   toward a common prior only as needed. Noise-aware ranking follows only if diagnostics justify it.

Preserve soft-label semantics explicitly, define loss scales and usable pairs, and separate
tuning from evaluation according to the research note. Record configuration, revision, parent,
split/label/feature identities, seeds, coefficients, AUCs and compute. Advance when evidence
justifies the extra complexity; a failed frozen-head screen does not rule out representation
gains. Training remains user-launched. Move this entry to ongoing work when experiments start.

This entry owns loss selection within each label. The parked
[adaptive allocation](#adaptive-training-allocation) proposal concerns weights between targets
and study exposure; [static ranking loss](small-changes.md#ranking-aware-loss) remains a separate
small change.

## Intermediate stride-16 feature fusion

**Hypothesis; external evidence only.** ReXNet-200 passes only its final stride-32 map (7×7 at
224 px) to the MIL head. Fuse a stride-16 map from the same backbone,
`F = P16(F16) + resample_to_F16_anchors(P32(F32))`, both projected to 256, and run the existing
embeddings and MIL on the 14×14 grid. It tests whether spatial detail is lost inside the encoder,
not whether the input needs more pixels ([bigger images](endgame.md#bigger-images) stays in the
endgame). The pattern follows [feature pyramid networks](https://arxiv.org/abs/1612.03144), whose
evidence is object detection.

- *Geometry:* verify the intermediate module, channels, stride and anchor origin on the
  instantiated ReXNet rather than assuming a stage name; resample by anchor coordinates, not
  generic resizing; recompute positions and visual validity on the new grid with the affine and
  reflection handling (`src/knee_abnormality/modeling/spatial_tokens.py`,
  [geometry audit](../data-preparation/spatial-token-geometry-and-mapping-audit.md)). The retired
  crop-selection scout already resampled a stride-32 map at stride-16 anchors
  ([plan](../history/crop-selection-plan.md)).
- *Cost:* 196 tokens per triplet instead of 49 (up to 12,544 per validation study), so projection,
  embeddings, attention and activations grow. A user-launched memory and runtime check comes
  first; do not silently shrink the batch or change BatchNorm behaviour. It changes the model
  identity and the inference feature extraction.
- *Keep fixed:* 224 px, 0.625 mm, the 140 mm field and slice budgets; no scout, learned crop or
  second backbone.
- *Order:* after the cheaper [attention and readout changes](small-changes.md#attention-pooling).

## Parked until a specific need

### Study overlap and anatomical matching

**Follow-up investigation; unstarted.** The user-requested bounded
[study-overlap audit](../metadata/study-overlap-investigation.md) found no confirmed
repeat among its shortlisted report/geometry candidates. To look for separate
examinations of the same knee, retrieve neighbors using stable bone anatomy and
verify with physical registration across multiple slices and another plane.
Use existing prepared data where possible; calibrate against unrelated studies
from the same scanner and protocol. Distinctive trabecular patterns or surgical
features should corroborate shape similarity. For potential split-bilateral
sessions, combine sequence inventories, frequency, coil and numbering evidence;
mirrored anatomy or a shared report template alone cannot establish identity.
Keep candidate, likely acquisition overlap and confirmed duplicate evidence
separate. Change fold groups only with convincing evidence and rerun a baseline
on any revised split; never silently rewrite the existing membership.

### Adaptive training allocation

**Hypothesis; unstarted, no chosen controller.** Investigate whether training history can usefully
allocate target weights or study exposure toward learnable ranking errors. The
[research note](../research/auc_loss_approaches.md#adaptive-allocation) owns the candidate signals,
soft-BCE entropy floor, sampling feedback bias and distinction between error and expected gain.
Define a training-only signal, smoothing, coverage and allocation limits before choosing an
implementation. Compare against uniform exposure and a relevant static allocation at the same
training budget. High raw loss alone is not the design. The static alternatives live only in
[small changes](small-changes.md#bce-weighting-and-study-sampling).

### Other parked work

- **External knee data** (MRNet, fastMRI+, SKM-TEA): only for a concrete missing capability, under
  the
  [admission criteria](../research/pretraining-and-external-data.md#admission-criteria-for-an-external-data-experiment).
- **Image-assisted label refinement:** fold-isolate every learned component that uses evaluation
  labels; earlier refinement gains involved
  [circular reasoning](../modeling/supervision.md#refinement-and-earlier-eligibility-uncertainty).
- **Full-volume 3D training from scratch, reconstruction or denoising systems, dense auxiliary
  tasks and broad leaderboard blend sweeps:** only after a specific failure or controlled result
  justifies the cost. Dense auxiliary tasks also need a reliable annotation source; segmentation
  Dice, SSIM or another competition's rank do not substitute for knee AUC.
