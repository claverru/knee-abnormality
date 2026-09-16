# Repeats, center sensitivity and implications

## Repeats and center sensitivity

Repeated plane/contrast metadata does not establish duplicate scan appearance.
Of 99 repeated groups in the protocol audit, only 25 have identical reduced
signatures using scanning sequence, sequence variant, 2D/3D acquisition type,
TR, TE, echo-train length and flip angle. Two groups have missing fields. Among
the 97 complete groups, 70 have TE ranges greater than 5 ms and 56 have TR
max/min ratios above 1.2. ProtocolName is absent from all 721 headers.

Retain these images as alternatives. Do not classify all repeated types as
duplicate protocols or invent tolerant protocol clusters merely to enforce
uniform source appearances. The present experiment addresses physical coverage
and source-count/spacing effects; it does not establish invariance to copying
one particular appearance among otherwise different acquisitions.

The center audit separates two effects. Adding equal contrast-type weighting
within each plane moves the current reference by a median 0 mm, 95th percentile
0.35 mm, and maximum 0.73 mm across the 124 accepted studies. For 52 accepted
studies with repeated types, omitting each repeated acquisition in turn and
taking the largest accepted displacement per study gives median 0.36 mm,
95th percentile 2.33 mm, and maximum 2.99 mm. This omission analysis is neither
duplicate insertion nor a validation of a new center estimator.

These observations favor keeping the present center estimator fixed for the
first sampling experiment. They do not justify claiming that it is exactly
invariant to acquisition repetition. Geometry alone also does not prove
anatomical registration across the distinct FrameOfReferenceUIDs in this
dataset. Preserve the current reliability gates and explicit unknown geometry.

## Implications for the implementation plan

The following analysis recommendations informed the
[implemented sampler](implemented-sampler.md), which owns the concrete
configuration and edge-case decisions:

1. Preserve budgets by plane/contrast, use union coverage once, and select a
   source independently of slice count before joint placement of overlapping
   triplets with distinct middle geometry.
2. Use the same truncated physical distribution for random stratified training
   and fixed-quantile validation/submission. Keep the existing 2× validation
   multiplier, but record the increased realized CNN count.
3. Start discussion from a broad Gaussian such as sigma 80 mm if maintaining
   peripheral exposure is the priority. Sigma 30 is a materially stronger
   center preference. Neither is a demonstrated accuracy optimum.
4. Keep the current center estimator and avoid protocol grouping in the first
   experiment. The sampling improvement does not require either extra change.
5. Define mandatory behavior for rejected centers, invalid per-series geometry,
   abnormal gaps and decode failures. Retain useful pixels with unknown 3D
   geometry. Such cases are not simulated
   by this accepted-center comparison.
6. Cover all preparation callers, export identity, triplet limits, collation,
   intensity normalization with repeated slice inputs, and documentation in the
   implementation plan. This audit does not exercise those production paths.

This replaces the earlier curve-only proposal that retained per-acquisition
midpoints, non-overlap, pooled capacity allocation and
`exp(log(3) * (1 - u²))` for normalized acquisition position u. That formula
preserved a nominal 3:1 endpoint ratio; it did not promise larger marginal
peripheral inclusion probabilities. The new idea should be compared as one
coherent sampler experiment against the completed spatial-token checkpoint's
recipe, without changing the backbone, crop, split, labels or optimizer budget.

## Earlier subsampling inventory

Read-only counts across all 24,371 training series, covering all folds, from
`data/metadata/train_series_image_counts.parquet`, with the two known bad files
subtracted. Each percentage is the share of individual series whose available
file count exceeds that plane–contrast type's nominal budget.

| Plane | Fluid-sensitive / fat suppression | 1× raw-slice budget | Above 1× | Above 2× | Above 3× |
|---|---|---:|---:|---:|---:|
| Axial | 0 / 0 | 18 | 99.6% | 36.6% | 7.6% |
| Axial | 1 / 1 | 18 | 96.5% | 26.5% | 6.7% |
| Coronal | 0 / 0 | 15 | 97.4% | 29.1% | 0.1% |
| Coronal | 1 / 1 | 15 | 99.7% | 33.9% | 0.2% |
| Sagittal | 0 / 0 | 15 | 98.2% | 40.4% | 7.1% |
| Sagittal | 1 / 1 | 15 | 99.8% | 34.6% | 5.3% |
| Overall | | | 98.4% | 33.3% | 4.3% |

The actual budget is shared across acquisitions within a study/plane/contrast.
An individual series below the nominal cap can therefore still be subsampled.
For example, at 2×, 36.6% of individual axial 0/0 series exceed 36 slices, while
61.7% of corresponding study/type groups exceed 36 complete-triplet slices in
total. The table is not an exact count of series affected after allocation or
of predictions changed by the new weighting. File counts do not replace a
complete usable-pixel audit. See [metadata provenance](../../metadata/acquisition-and-series-profiles.md#images-per-series).
