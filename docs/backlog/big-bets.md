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
indexed by training series UID would not generalize to unseen acquisitions. The fixed-quantile
runs never started; they are `11-clip-0.001` and `12-clip-0.01` in the
[overnight queue](overnight.md#next-queue-proposed-pending-the-users-decisions).

## Robust study center

**Robust study center with per-series sampling paths** (rejection counts and rescue analysis
measured on CPU; the AUC effect is untested).

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
  (see [non-target series filtering](small-changes.md#non-target-series-filtering)).
- *Cost:* it changes the preprocessing identity and needs a fresh fold-0 run.

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
