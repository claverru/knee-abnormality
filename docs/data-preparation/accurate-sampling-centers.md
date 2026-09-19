# Accurate sampling centers

Midpoint sampling remains selected after the
[completed comparison](../validation/results/2026-09-12-accurate-centers-result.md).
The geometric sampler was not adopted. Its shared foundation remains: the
pooled loader indexes all supported acquisitions before sampling and
fits one study point to their acquisition-center lines, balancing total weight
equally across anatomical planes. It reuses the existing header reads and caches
the result once per study per worker cache lifetime. `dicom.py` retains native
XYZ poses and position provenance; shared layout descriptors avoid per-slice
matrices. `geometry.py` performs the float64 line solve and reliability checks.
The former `shortest` policy has been removed from the production runtime.

Required pose tags must be finite, spacing positive, and orientation axes unit
and orthogonal within `1e-3`. Series orientation changes above `1e-3` or
center-line departures above 0.1 mm exclude that series. At least two usable
planes are required, with condition number at most `1e3`, plane-balanced RMS
at most 10 mm and every contributing residual at most 20 mm. Different frame
IDs across series are recorded; inconsistent IDs within a series are rejected.
These are engineering screening constants, not measured anatomical tolerances.
The refit of rejected studies was removed on 2026-09-18; see
[current preparation](current-preparation.md#robust-study-centre-removed).

Every diagnostic projection requires consistent geometry and patient-derived
positions for all retained slices. Missing/mixed provenance, rejected study
fits and out-of-range projections retain explicit rejection reasons.
An out-of-range projection does not invalidate native poses.
The 140-mm image crop and acquisition-midpoint model position encoding remain
unchanged. Production now uses the shared Gaussian sampler documented above.
The archived 2026-09-12 candidate shifted only the center while retaining the original
half-span, allowing different weights at the two endpoints.

The [real-data audit](../../outputs/analysis/accurate-centers-20260912/summary.json)
reprocessed the saved 128-study, 721-series sample: 124 accepted studies,
four RMS rejections, 696 projected series, 24 study-rejection midpoints and
one out-of-range midpoint. Accepted points differ from the all-slice reference
by at most 0.00056 mm. Validation allocations changed 2,269 triplet centers;
per-series deterministic and seeded selections repeated exactly. This sample
did not exercise missing planes or invalid/mixed-source metadata.

Six training-member [coverage sheets](../../outputs/analysis/accurate-centers-20260912/coverage_00.png)
include the out-of-range case, high residuals and large shifts. Yellow boxes
mark the retained crop and red crosses project the study reference onto native
images. Reviewed cases preserve knee coverage, but acquisition agreement is
not local registration evidence or an accuracy result. The series table records
the selected triplet indices and shifts. The final source audit is saved in
[`accurate-centers-final-20260912`](../../outputs/analysis/accurate-centers-final-20260912/summary.json).
The completed audit utility and candidate sampler are preserved together in
that directory's `source_snapshot.zip` and the training run's source snapshot.
Extract into a separate directory and use its `src` on `PYTHONPATH` to rerun
`scripts/adhoc/audit_sampling_centers.py` with the config/selection/reference
paths recorded in the audit artifacts. The retired candidate is not a
production configuration option.

The preliminary 16-study cProfile runs decoded 1,329 frames on each pass and
indexed 3,865 headers on the first pass, with zero additional indexing headers
on the cached pass. Candidate first/cached wall times were 5.39/0.68 seconds;
the archived control took 4.12/1.73 seconds. The archived source predates the
native-reader optimization, so this comparison includes that change, and
cProfile plus filesystem cache state limits timing interpretation. It does
not isolate geometry overhead. The 128-study serialized index/reference cache
occupied about 7.5 MB; serialized size is not resident worker memory.

At unchanged batch eight and 16 workers, the bounded 64-study loader workflow
measured 22.77 versus 22.93 studies/second (candidate versus archived control),
identical measured input counts and 16.84 GiB peak allocated VRAM. Four measured
steps after two warmups are a preliminary performance check, not full-run
throughput. Artifacts live under `outputs/profiling/accurate-centers-*20260912`.
The controlled fold-0 result belongs in [validation](../validation/INDEX.md).

## Cohort-wide center fit and plane extents

Measured on 2026-09-18 over the whole training split, 4,407 studies and 24,371
series, with [`analyze-study-center-extents`](../../scripts/adhoc/analyze_study_center_extents.py);
artifacts in `outputs/analysis/study-center-extents-20260918/`. It reuses the
production indexer, exclusions, instance guard and reference fit, so the numbers
describe what the sampler actually sees.

- **The fit accepts 4,230 of 4,407 studies (95.98%).** The 177 rejections are 169
  `excessive_rms` and eight `excessive_residual`. Plane-balanced RMS is 2.93 mm at the
  median, 8.64 mm at p95 and 95.06 mm at worst. Every accepted study has all three planes.
- **No series is rejected on geometry.** All 24,371 series pass the orientation, center-line
  and frame checks; the only six ineligible series are dropped by the other-knee split.
- **Distances from the accepted point to each side**, along each plane's through-plane axis,
  per study after taking the union over that plane's series, split by the two supported
  series types (T1/PD is fluid-sensitive 0 without fat suppression, fluid is 1 with it):

  | Plane | Type | Studies | Negative side | Positive side | Span | Midpoint offset |
  |---|---|---|---|---|---|---|
  | Axial | both | 4,230 | inferior 61.8 ± 11.5 | superior 61.5 ± 11.9 | 123.3 | -0.2 |
  |  | T1/PD | 790 | inferior 58.3 ± 13.6 | superior 61.6 ± 12.3 | 119.9 | +1.7 |
  |  | fluid | 4,230 | inferior 60.9 ± 12.8 | superior 61.0 ± 11.6 | 122.0 | +0.0 |
  | Coronal | both | 4,230 | anterior 59.0 ± 7.9 | posterior 45.6 ± 10.4 | 104.6 | -6.7 |
  |  | T1/PD | 3,262 | anterior 58.5 ± 8.1 | posterior 45.3 ± 10.4 | 103.8 | -6.6 |
  |  | fluid | 4,071 | anterior 58.8 ± 8.0 | posterior 44.9 ± 10.4 | 103.7 | -6.9 |
  | Sagittal | both | 4,230 | right 51.3 ± 7.0 | left 51.0 ± 7.2 | 102.2 | -0.1 |
  |  | T1/PD | 4,097 | right 50.6 ± 6.5 | left 50.4 ± 6.5 | 101.0 | -0.1 |
  |  | fluid | 4,001 | right 50.7 ± 6.9 | left 50.3 ± 7.1 | 101.0 | -0.2 |

- **Geometry barely depends on the fluid dimension.** Within the studies that carry both
  types, the fluid-sensitive stack differs from the T1/PD one by -3.7 mm inferior, -1.7 mm
  superior and -5.4 mm span in axial (790 studies), and by under 0.6 mm on every coronal
  (3,103) and sagittal (3,868) figure. Type availability differs far more than extent:
  every accepted study has an axial fluid series but only 790 have axial T1/PD, while
  coronal splits 3,262 / 4,071 and sagittal 4,097 / 4,001.
- **Coronal is the only asymmetric plane.** Its stack midpoint sits 6.7 mm anterior of the
  fitted point on average (p5 -16.8, p95 +2.7 mm), and 32.2% of studies reach less than
  `data.sampling_sigma_mm: 40` posteriorly against 0.9% anteriorly, so a sigma-40 coronal
  draw is truncated posteriorly in a third of studies. Both types carry it, 32.9% T1/PD and
  34.8% fluid, which makes it acquisition geometry rather than a sequence effect. Axial and
  sagittal midpoints are centered within 0.2 mm, with 2.8-4.2% of sides shorter than sigma.
  This is the geometric counterpart of the coronal flank deficit in the [spatial importance
  map](../ongoing/peripheral-coverage.md#spatial-importance-map), which measured coronal
  flanks minus center at -0.634, so it argues for no widening.
- **Axial T1/PD is the one type-specific truncation.** Its stack sits 1.7 mm superior of the
  point and falls under sigma inferiorly in 8.4% of its 790 studies against 1.8% superiorly;
  axial fluid is 5.7% against 3.0%.
- **The point lies inside the union span** of every plane but one axial study (0.02%). Per
  type it falls outside in 0.76% of axial T1/PD studies, 0.02% of axial fluid and 0.03% of
  coronal T1/PD, and never in sagittal. Distance to the nearest side of the union at the
  median is 57.6 mm axial, 44.2 coronal and 47.9 sagittal, with p5 at 39.6, 30.4 and 38.7 mm.
- **This evidence now drives the sampler width directly.** The sigma-40-mm truncation
  fractions above describe the retired fixed-width sampler; the
  [implemented sampler](physical-sampling/implemented-sampler.md) instead sets each candidate
  pool's own Gaussian width to `data.sampling_sigma_fraction` (0.76) times that pool's own
  `limiting_reach`, so the coronal deficit found here no longer needs a fixed threshold to
  show up as truncation risk.
- **Limits.** Distances run slice center to slice center, so the outer faces sit half a
  slice spacing farther out; median spacing is 3.90 mm axial, 3.51 coronal and 3.30 sagittal.
  Sagittal right and left are patient laterality and mix medial with lateral across knee
  sides. Rejected studies contribute no distances.

See [modeling](../modeling/unverified-modeling-notes.md#window-sampling) for slice-window selection and
[inference](../inference/runtime-evidence-and-leads.md#runtime-leads) for reducing decoding work.
