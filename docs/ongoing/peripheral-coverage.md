# Peripheral coverage of the Gaussian sampler

Where each label's evidence sits relative to the sampler's reach, complementing the
[realized coverage fractions](../data-preparation/physical-sampling/sampling-analysis-coverage-and-exposure.md).

Triplets are placed along each plane's axis by a truncated Gaussian; at the time of this analysis
the width was a fixed `data.sampling_sigma_mm: 40`, so coverage fell with distance from the study
centre. The sampler has since moved to `data.sampling_sigma_fraction`, a per-pool width relative
to each pool's own acquired reach (see the [implemented
sampler](../data-preparation/physical-sampling/implemented-sampler.md)); the measurements below are
against the retired fixed-40 sampler and are not automatically valid at the current per-plane
widths. The fraction was 0.76 when this analysis was written and is **0.867 since 2026-09-19**,
for realized means of axial 45.1, coronal 35.9 and sagittal 39.0 mm
([result](../validation/results/2026-09-19-relative-sampler-width-result.md)). Axial now sits above
the old fixed 40 and sagittal close to it, while coronal at 35.9 sits just outside the 12–35 mm
plateau that [per-plane sampler sigma](../validation/results/2026-09-18-per-plane-sigma-cache.md)
scored as a gain on the cache — at 0.76 its 31.4 mm was inside it. Several labels keep most of
their evidence out there. Measured read-only
from the fold-0 token cache and the depth profiles; scripts and tables in
`outputs/analysis/peripheral-coverage-20260918/`.

- **The sampler thins with depth.** Selected share ÷ candidate share beyond 40 mm is 0.45–0.59 for
  every type, and beyond 50 mm it falls to 0.36 for coronal fluid and 0.14–0.16 for the sagittal
  types. Selection probability drops from about 0.40 at the centre to 0.19 at 40 mm and beyond.
- **14–31% of a series' slices are never sampled** at the default counts, worst for axial T1/PD
  (mean 31%, p90 63%).
- **Share of a label's depth-weighted evidence sitting beyond ±20 mm:** Baker's cyst in coronal
  fluid 80%, lateral meniscus in sagittal fluid 74%, synovitis in axial fluid 67%, effusion in
  axial fluid 65%, medial meniscus in sagittal fluid 59%, MCL in axial fluid 56%. Synovitis peaks
  at 40–50 mm superior, where the selection probability is half the central value; effusion at
  30–40 mm, at 0.28. The medial meniscus in coronal fluid sits close enough that sigma 40 covers
  it.
- **Not a fix that follows for free:** lifting axial fluid's density at 40–50 mm to the central
  level implies a sigma of roughly 65–80 for that type, while global sigma 60, 80 and 120 were all
  measured worse and 30 tied
  ([overnight 2026-09-15](../validation/results/2026-09-15-overnight-results.md)). Both follow-ups
  have since been scored on the caches without the GPU: centre offsets are closed on the map below,
  and per-plane sigma on the token cache makes the axial width itself inert, so widening axial to
  reach this evidence costs nothing measurable and buys nothing measurable
  ([2026-09-18 per-plane sigma](../validation/results/2026-09-18-per-plane-sigma-cache.md)).
- **Limits:** the separation metric behind the evidence shares is small-sample and noisy in tail
  bins, fracture is too sparse to read at all, and geometry cannot recover in-plane tissue
  clipping — the reference point is defined near each series' image centre, so the pixel-based
  figures in [input resolution](../research/input-resolution.md) remain the only source for the 140 mm
  crop (sagittal tissue exceeds it in 34% of series, almost all posterior).

## Spatial importance map

A leave-one-out window-ablation field over the ensemble, exported as a Plotly page outside this
repository and extracted to `outputs/analysis/spatial-importance-20260918/`. All 12 targets share
one 1,092-voxel grid at 9 mm spacing, spanning L [-68, 58], P [-76, 59] and S [-76, 77] mm about
the study centre; the value is positive LOO utility in within-study SD units, density corrected.
Support averages 135 studies per voxel but ranges 25-761, highest at the centre.

- **Read the cross-target contrast, not the per-target field.** Half the absolute fields are
  explained by radius alone (quadratic fit R^2: Contusion 0.80, ACL 0.77, medial OA 0.60, MCL
  0.57), which is the sampler envelope rather than anatomy. Standardizing each voxel across the 12
  targets removes that shared envelope.
- **Single-voxel peaks are noise.** 3.3% of the 13,104 voxel-target cells exceed contrast z = 2,
  about what the standardization yields by construction, and the field is spatially smoothed.
  Hemisphere means, not maxima, are the readable unit.
- **Hemisphere contrasts that survive** (mean contrast z beyond +/-18 mm, superior-inferior /
  anterior-posterior): PF OA +0.43 / **+1.85**, effusion **+1.23** / +0.95, synovitis +0.16 /
  +1.13, Baker's **-0.90** / **-1.23**, medial OA -1.07 / -0.83, lateral OA -1.05 / -0.56, MCL
  +0.29 / -1.32. Patella anterior, effusion anterosuperior, Baker's cyst posteroinferior and the
  tibiofemoral OA pair inferior are anatomically right, from a measurement that never saw anatomy.
- **The medial-lateral axis is unreadable.** The export centres and scales L but never mirrors by
  knee side, so left and right knees sit in one pooled frame in which medial and lateral reverse.
  Fold 0 is 412 left, 450 right and 42 unknown, so a true medial-lateral contrast survives pooling
  at |450-412|/862 = 4.4%. Measured L contrast is 0.00-0.36 in absolute value for every target,
  consistent with that cancellation. The signs do line up (medial meniscus +0.36, lateral meniscus
  -0.32, Baker's +0.23, medial OA +0.09, lateral OA -0.10), but at this size they are not evidence.
  Re-exporting with left knees mirrored carries a sharp prediction: a real contrast grows about
  20-fold, noise stays the same size and moves elsewhere.
- **It gives the depth profiles a direction.** The profiles place effusion's axial evidence 30-40 mm
  off centre at a 0.28 selection probability without saying which way; the map makes it the
  strongest superior contrast in the table.
- **No per-label pull can move the shared sampler.** The contrast is zero-sum by construction:
  standardizing each voxel across the 12 targets forces the mean pull on every axis to 0.000, so
  effusion superior is cancelled by the OA pair inferior, and PF OA and synovitis anterior by
  Baker's and MCL posterior. The quantity one shared sampler can act on is total utility over
  targets, and that is already centred, with a utility-weighted centroid of +1.4 mm on the axial
  through-plane axis, -3.3 mm coronal and -0.9 mm sagittal against a 9 mm grid, flat within
  +/-30 mm and falling only outside it. This closes the centre-offset follow-up without a run.
- **The sagittal axis has a central trough, and it is the only one.** Total utility over all 12
  targets by 9 mm slab, flanks (27-50 mm) minus centre (within 9 mm): sagittal **+0.126**, coronal
  -0.634, axial -1.099. The sagittal profile is near-symmetric, peaking at -31.5 mm (3.758) and
  +31.5 mm (3.725) either side of a 3.406 centre. That is the signature of left and right knees
  superimposed in the export's unmirrored frame, and the per-target split matches: the compartment
  labels carry the trough (lateral meniscus +0.194, medial meniscus +0.064, Baker's +0.057, lateral
  OA +0.041, medial OA +0.021) while the midline ones prefer the centre (PF OA -0.082, effusion
  -0.082, contusion -0.061, MCL -0.055, ACL -0.038). Support falls off centre on all three axes
  identically, so the trough is not an exposure artifact.
- **Sigma is a per-plane quantity, but the map's axis ranking is wrong.** Expected utility per
  sampled slice under a truncated normal falls monotonically with sigma on every axis, matching the
  measured ordering in which 30 tied 40 and 60, 80 and 120 lost. The swing across sigma 20-120 is
  **0.092 sagittal, 0.310 coronal, 0.677 axial**, which made the axial draw look about seven times
  as width-sensitive as the sagittal one. Scoring the real selections on the fold-0 token cache
  reverses that: coronal carries the whole effect and axial is flat from 20 to 120 mm
  ([2026-09-18 per-plane sigma](../validation/results/2026-09-18-per-plane-sigma-cache.md)). Read
  the map for where evidence sits, not for how much a plane's width is worth.
- **Limits:** the generating code and the per-study utilities are not in this repository, so only
  the extracted field can be re-derived; Fracture has the largest field spread (sd 0.15) on the
  sparsest labels and should not be read; and the profiles' synovitis peak at 40-50 mm superior
  does not reproduce as a superior contrast here (+0.16), which is unresolved.

Measured 2026-09-18. **Per-type centre offsets are rejected on the map above, with no GPU spent.**
The proposal was a `centre_offset_mm` field on `SeriesTypeConfig`, shifting the mean of the
truncated normal per type (`target_depths` in
`src/knee_abnormality/data/preprocessing/sampling.py`). One sampler serves all 12 targets, so the
quantity it should place is total utility summed over targets, and that is already centred: the
utility-weighted centroid is +1.4 mm superior on the axial through-plane axis, −3.3 mm coronal and
−0.9 mm sagittal against a 9 mm grid, and marginal total utility is flat within ±30 mm on all three
axes. Per-label pulls cannot overturn it, because the cross-target contrast is zero-sum by
construction. What this does *not* settle: the map is observational, so it cannot show what a
region the sampler rarely reaches would be worth if it were reached, and per-label sampling — as
opposed to a per-type offset on one shared draw — is a different and much larger change. Per-plane sigma was then scored directly on the fold-0 token cache, also with no
GPU: coronal 30 mm is worth +0.00135 macro AUC at the ×1 count, confirmed on both halves and
keeping its sign to ×4, while axial sigma does nothing at matched prediction movement and a
sagittal gain does not survive higher counts
([result](../validation/results/2026-09-18-per-plane-sigma-cache.md)). That measurement, not the
map, now sets the sampler change described below.

The whole map is observational, measured under the current sampler, so rarely-visited depths carry
few ablated windows and their utility is understated. That bias points toward the centre on every
axis, so read the differences between axes, which share the bias, rather than the absolute optimum,
which does not survive it.

## Acquisition geometry

Distance from the study centre to each side, per study, union over that type's series, mean ± sd.
T1/PD is fluid 0 / fat 0, fluid is 1/1.

| Plane | Negative side | Positive side | Span | Midpoint offset |
|---|---|---|---:|---:|
| Axial | inferior 61.8 ± 11.5 | superior 61.5 ± 11.9 | 123.3 | −0.2 |
| Coronal | anterior 59.0 ± 7.9 | **posterior 45.6 ± 10.4** | 104.6 | **−6.7** |
| Sagittal | right 51.3 ± 7.0 | left 51.0 ± 7.2 | 102.2 | −0.1 |

- **Coronal is the only asymmetric axis**, reaching 13 mm further anterior than posterior. A
  Gaussian centred on the study centre is capped by the *shorter* side, so the usable half-reach is
  axial 61.5, coronal 45.6 and sagittal 51.0 — a ratio of 1.35 : 1.00 : 1.12, not the 1.21 : 1.02 :
  1.00 the spans suggest. Sigma in that ratio, with today's average reach preserved, is **axial 47,
  coronal 35, sagittal 39**.
- **That reconciles the geometry with the scores.** On spans alone coronal and sagittal look
  identical, which contradicts the cache preferring a narrower coronal. On limiting reach coronal is
  the shortest axis by 26%, and coronal 35 lands inside the measured-good plateau of 10–35 mm.
- **Contrast does not change the extent.** Within a plane the two contrasts differ by 1.2 mm or
  less (`ax_t1pd` 55.1 against `ax_fluid` 56.3, coronal 47.8/47.6, sagittal 47.3/46.1), against a
  p25–p75 spread across series of about 13 mm. The width is a per-plane quantity, not a per-type
  one.
- **The posterior cap is an acquisition limit, not a sampler one.** Baker's cyst is posteroinferior
  and keeps 80% of its coronal evidence beyond ±20 mm, but no sigma reaches past 45.6 mm because
  the slices were never acquired. Widening coronal buys anterior slices only.
- **The study centre needs no correction.** The coronal acquisition midpoint sits 6.7 mm anterior
  of it, well inside the ±30 mm band over which total utility is flat, consistent with the −3.3 mm
  coronal centroid that already closed the centre-offset follow-up.

## What is settled, and what is not

**Settled.** The *overall scale* of the Gaussian is inert. `sampling_sigma_fraction: 0.867` replaced
the fixed 40 mm and landed at the mean of its two controls, +0.00016 at ×4
([result](../validation/results/2026-09-19-relative-sampler-width-result.md)); a five-point sweep
from 0.60 to 1.10 then came out non-monotone with the control at the bottom, spanning 0.00674, which
is seed noise ([sweep](../validation/results/2026-09-19-fold-ensembles-and-sigma-sweep.md)).
Per-type centre offsets were rejected earlier on the importance map.

**Shape was tried once, and neither arm resolved.** The sweep multiplies all three planes by one
factor, so it measures scale and says nothing about *shape*. Both shape proposals were trained by
the stopped autoresearch campaign on 2026-09-19 — fold 0, validation ×1, ten epochs, 2,000-draw
paired bootstrap — and both came back inconclusive against the historical control at 0.894037:

| Arm | Campaign idea | Macro AUC | Delta (95% paired) |
|---|---|---:|---|
| Per-plane widths | `geometry-informed-plane-widths` | 0.894390 | +0.00041 [−0.0033, +0.0041] |
| Asymmetric coronal draw | `coronal-side-relative-draw` | 0.892728 | −0.00130 [−0.0051, +0.0023] |

These were the two entries this document used to queue as small changes; the campaign took
ownership of them and the general backlog no longer carries them. Neither result establishes harm,
and neither clears the run-to-run floor, so the geometry argument below still stands unanswered —
but **a rerun of the same mechanism is a repeat, not a new experiment**. The campaign owns the
implementations, decisions and limits
([prior work](../../autoresearch/campaigns/triplet-sampling/prior-work.md)); it also supersedes
this document's old 0.008 single-run bar with its paired bootstrap, so the two decision rules
should not be mixed.

The geometry they were testing, unchanged: usable half-reach is axial 61.5, coronal 45.6, sagittal
51.0 mm, a ratio of 1.35 : 1.00 : 1.12, against the relative rule's realized 45.1 / 35.9 / 39.0;
and coronal is the only asymmetric plane, reaching 59.0 mm anterior against 45.6 posterior, so a
symmetric Gaussian centred on the study centre spends mass anteriorly where slices are plentiful
and truncates posteriorly where Baker's cyst and the posterior horns live. A common-factor sweep
cannot see either: if one plane wants more and another less, scaling both together cancels.

**Still open, and the only item here that is not tried:**

- **Re-exporting the slice-importance map with left knees mirrored**, a user action outside this
  repository, which gates
  [side-conditioned sagittal sampling](../backlog/big-bets.md#position-and-knee-side) and is listed
  in [next](next.md).

The [acquisition geometry](#acquisition-geometry) above is the input to all of these, and is
preserved independently in
[accurate sampling centers](../data-preparation/accurate-sampling-centers.md#cohort-wide-center-fit-and-plane-extents).
