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
The optional refit of rejected studies (`data.robust_center_outlier_mm`) is described in
[current preparation](current-preparation.md#robust-study-centre-option).

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

See [modeling](../modeling/unverified-modeling-notes.md#window-sampling) for slice-window selection and
[inference](../inference/runtime-evidence-and-leads.md#runtime-leads) for reducing decoding work.
