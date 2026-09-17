# Big bets

Investigation-sized plans that need focus sessions with the user: diagnostics, a design, several
runs or a new preprocessing identity. Start one only in a session with the user; when it starts,
move its entry to [ongoing work](../ongoing.md) and remove it from this file in the same change.
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

## Sampling-independent intensity bounds

**Code observation; size and harm of the effect unmeasured.** `prepare_series`
(`src/knee_abnormality/data/preprocessing/preparation.py`) takes the intensity quantiles over the
unique slices of the selected triplets, so the same source slice gets different normalized pixels
when its companion slices change: between training epochs, and between training and the 2×
validation bag. The variability may also act as useful augmentation.

1. **Measure first:** on a fixed training-fold subset, compare several training-epoch selections
   and the deterministic 1× and 2× bags: per-acquisition bounds, gain variation, normalized-pixel
   and clipping differences on source slices shared between bags, decoded-slice count and
   preprocessing time (`PreparedSeries` already exposes `intensity_bounds`, `unique_slice_count`
   and `intensity_pixel_count`). Separate changes within one acquisition from changes of supplier
   series; disjoint samples are not a pixel-level comparison.
2. **Only if material:** estimate the bounds from a deterministic sparse reference sample of each
   acquisition, independent of epoch, bag size and selected windows, identical in training,
   validation and inference, at the current quantile levels. Define short-series and
   decode-failure fallbacks. It changes the preprocessing identity; cache scalar bounds only under
   an identity that prevents stale reuse.

- *Distinct from* the quantile-level runs and the learned bounds above: it changes the pixel
  population, not the levels or the estimator.
- *Cost:* extra decoding, against the
  [select-before-decoding](../architecture/data-flow-and-iteration-speed.md#iteration-speed) rule
  that the [loading speedups](../inference/loading-speedup-plan.md) rely on.
- *Stop if* the variability is negligible, the runtime cost is disproportionate, or a controlled
  run gains nothing.

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

## Robust study center

**Robust study center with per-series sampling paths.** It is implemented behind
`data.robust_center_outlier_mm`, and one seed-42 run is queued
([implementation](#implementation-2026-09-17)). Rejection counts and the rescue analysis are
measured on CPU, and an inference-only fold-0 re-score is below. The effect of training with it is
untested.

- *Problem:* 178 of 4,407 train centers are rejected (170 RMS above 10 mm, 8 with one series above
  20 mm), often because of one or two stray series, so the whole study falls back to local
  sampling. An in-plane residual does not affect depth sampling.
- *Proposal:* iteratively drop the worst series while it lies more than 20 mm from the center and
  refit, keeping the RMS, condition and two-plane checks. Each series then samples physically when
  the center projects inside its acquired range, and locally otherwise. The model no longer reads
  3D geometry, so only the previews' `geometry_valid` still depends on `& physical` in
  `src/knee_abnormality/data/dataset.py`.
- *CPU analysis* ([summary](../../outputs/analysis/study-center-robust-20260915/summary.txt)):
  it rescues 131 studies (21 of 31 on fold 0), dropping 159 series, mostly small-FOV axials. Of
  the 47 still rejected, 44 have no series above 20 mm. Centers of accepted studies do not move,
  by construction. Of 24,371 series, 350 would sample locally (75 on fold 0).
- *Gate on range rather than retention:* retained series from the other knee in bilateral
  studies, or from an ankle series filed in a knee slot, project up to 184 mm outside their range
  (see the [series-filtering census](../metadata/series-filtering-census.md#implemented-rule-2026-09-17); the ankle and stray other-knee series are now dropped before the fit).
- *Cost:* setting the option adds a `robust_center` entry to the preprocessing identity, so the
  run starts fresh on fold 0.

### Robust-centre re-score (2026-09-17)

Measured, inference only. The three fold-0 control checkpoints of
`overnight-20260916T214520Z-t8k_kw2z` (runs 00–02) re-scored the rescued fold-0 studies at batch
size 1 with bf16 autocast; the rest of fold 0 keeps its saved predictions. The comparison uses a
paired report-group bootstrap (2,000 draws), with Δ macro AUC ×1000 averaged over the three
checkpoints. Tables: [macro](../../outputs/analysis/series-filtering-20260917/robust-centre-rescore/macro_deltas.csv),
[per target](../../outputs/analysis/series-filtering-20260917/robust-centre-rescore/per_target_deltas.csv),
[series paths](../../outputs/analysis/series-filtering-20260917/robust-centre-rescore/series_sampling.parquet),
[sanity](../../outputs/analysis/series-filtering-20260917/robust-centre-rescore/sanity_slots.csv).
Script: [`robust_rescore.py`](../history/series-and-sampling-sources/series-filtering/robust_rescore.py.txt).

- *Patch:* a `KneeStudyDataset` subclass replaces a rejected reference with the robust fit. Series
  dropped from the fit remain model inputs. A wrapper around `build_pool` passes the robust centre
  only to series whose acquired slice range contains it. Every other series gets a reference with
  no centre, so it samples locally.
- *Paths:* none of the rescued series sample locally. When any series in a slot samples
  physically, `build_pool` keeps only the physical triplets. Each of the 12 out-of-range series
  (all in the 10 range-gated studies, 8 of them axial fluid) shares its slot with an in-range
  series, so it leaves the input. The other 129 of 141 series sample physically. Triplets stay at
  1,102 with no shortfall, no preparation failures and valid geometry.

| Rescued studies | n | Δ ×1000 (95% CI) | Per checkpoint |
|---|---:|---:|---|
| 20 mm robust fit | 21 | −0.107 (−0.256, +0.036) | −0.126, −0.044, −0.150 |
| Range-gated fit | 10 | −0.001 (−0.114, +0.110) | −0.057, +0.060, −0.006 |
| 20 mm fit only | 11 | −0.105 (−0.202, −0.024) | −0.069, −0.104, −0.143 |

- On their 10 shared studies, the two fits give identical predictions. The whole loss therefore
  comes from the 11 studies that only the 20 mm fit rescues. There, the far series project inside
  their range, so all 73 series switch from local to physical sampling. The range-gated rescue,
  which also removes the out-of-range series, is neutral.
- Per target (20 mm fit, 21 studies): medial and lateral OA fall by 0.56 ×1000 each and effusion
  rises by 0.56, with the same sign on all three checkpoints. These studies contain only 1–4
  positives for each of these targets.

### Implementation (2026-09-17)

The option reproduces the 20 mm patch above. The implemented behaviour, the identity effects and
the CPU verification (null against HEAD, 20 mm and 10 mm) are recorded in
[current preparation](../data-preparation/current-preparation.md#robust-study-centre-option).
One run is queued: `robust-center-seed42`, paired with `ranked-filtered-seed42` (0.889573)
([overnight queue](overnight.md#next-queue)).

- *Counts:* after the ankle exclusion, 177 training studies are rejected, and 20 mm rescues 130 of
  them (21 of 31 on fold 0). The 10 mm threshold rescues all 177 and excludes 256 series from the
  fits, against 155 at 20 mm. No inference check has been run at 10 mm.
- *Re-score (measured, inference only):* the `ranked-filtered-seed42` epoch-9 checkpoint was run
  with the option at 20 mm, with the identity check bypassed.
  - The 21 rescued fold-0 studies were re-scored at batch size 1 with bf16 autocast; the other
    883 studies keep their saved predictions.
  - Macro AUC moves from 0.889573 to 0.889228 (−0.00035).
  - A batch-1 recompute without the option gives 0.889574, with a maximum probability difference
    of 0.0019 from the saved predictions.
  - All 105 non-empty slots of these studies switch from local to physical sampling, and the mean
    absolute probability change is 0.028.
  - The three control checkpoints above averaged −0.0001 (from −0.00004 to −0.00015), so this
    checkpoint loses more than any of them. The result comes from one checkpoint and 21 studies
    (`outputs/analysis/robust-center-implementation-20260917/rescore.json`).
- *Expectation:* the run is expected to tie. Seed-to-seed noise is 0.0015–0.0018, so a single run
  cannot resolve a change of this size.

## Parked until a specific need

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
