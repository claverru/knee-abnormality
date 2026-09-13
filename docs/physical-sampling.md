# Physical sampling analysis

Metadata-only investigation completed on 2026-09-13 before implementing the
shared-center sampler. Acquisition choice independent of slice count preserves
the intended source exposure with little additional placement error. A Gaussian
with sigma 30 mm substantially narrows coverage relative to the current sampler;
a much broader Gaussian is needed to approach its peripheral exposure.

The production sampler was implemented on 2026-09-13 using the decisions below.
The historical analysis remains evidence for the initial width and acquisition
policy, not an accuracy result. Its prototype and old-sampler comparison are
preserved with their original source under
`outputs/analysis/physical-sampling-20260913/geometry-v3/source/`.

## Implemented sampler

The active baseline is `spatial-tokens-physical-sampling`, with positive finite
`data.sampling_sigma_mm: 80.0`. It preserves the spatial-token model, accepted
study-center estimator, reliability thresholds, 140-mm crop and acquisition-
midpoint 1D encoding. The old allocator, triangular selection and their three
configuration fields have been removed; old fields are rejected. Historical
checkpoints require their matching archived source and configuration.

Within each plane/contrast type, candidate triplets are every consecutive window
in usable physical runs. Positions must be finite, strictly increasing in the
original acquisition order and use one coordinate source. Runs split at gaps
at most 0.01 mm, non-finite gaps or gaps above `1.5 * median_positive_gap + 0.05`
mm. The median uses original gaps above 0.01 mm. Runs require three slices;
selected channels retain their original order. Decoded failures split these
same original runs without connecting former neighbors or refitting spacing.

Reliable acquisitions with a labelled plane matching their dominant normal and
an accepted center use exact plane intersections with the patient axis through
that center. Each run contributes its eligible middle-depth extent; a single
eligible center contributes the cell bounded by its neighbor midpoints.
Support is the union across runs and acquisitions, so overlapping acquisitions
do not multiply target probability. Centers outside coverage are allowed:
Gaussian masses and inverse quantiles use stable truncated-tail arithmetic.

The per-type triplet budget is `max_images / 3` times the image multiplier:
6 axial or 5 coronal/sagittal inputs per contrast at training density and twice
that for validation/inference. Gaussian probability over the union is divided
into equal bands. Training draws within each band; evaluation uses midpoints.
Training chooses a covering acquisition uniformly. Evaluation allocates each
target's fractional supplier entitlement to the largest outstanding deficit.
Source preference precedes nearest-slice placement, and a target can only use
candidates from a run whose own support contains it.

Alternative middle planes are grouped by deterministic complete-link geometry:
every pair must satisfy `abs(offset_a-offset_b) + R*norm(normal_a-normal_b) <= 0.1`
mm, with positively oriented normals and `R=image_size*spacing_mm/2` (70 mm).
This is geometry grouping, not inferred protocol equivalence. SciPy assignment
maximizes filled targets, then minimizes preferred-supplier changes, then total
squared depth error. Groups are used once. If distinct capacity is below budget,
K is reduced and quantile bands regenerated; any remaining assignment shortfall
is reported without repeated middle planes or transfer to another type.

When a type has no physical candidates, trustworthy acquisition-local millimetre
runs use one affine mapping of each acquisition's total support to [0,1]. Gaps
and singleton cells remain present; a sole center uses 0.5 and full support.
Reliable native ordering or consistently sourced `SliceLocation` can supply
local positions; inconsistent orientation/center-line/frame metadata and
untrustworthy coordinates are excluded. Targets are uniform over the union of
local run support, with the same supplier and assignment rules. Middle indices
are distinct within each acquisition; no cross-series physical deduplication is
claimed. All local inputs are unknown for XYZ embedding. Types without usable
complete inputs remain masked; a completely unusable study fails explicitly.

SHA-256 seeds include the configured seed, study UID, type and operation, plus
epoch only for randomized sampling. Separate streams govern target percentiles,
supplier choices, band order and source ties; retries reuse them against rebuilt
support. Dataset order and decode completion order do not select inputs.
Copying a source does not multiply continuous physical target mass with center
and support fixed. Copies still change supplier alternatives, and the unchanged
reference estimator or changed grids can affect results; full duplicate
invariance and exact final source shares are not claimed.

[Preparation](data-preparation.md#overlapping-triplet-preparation) normalizes each
unique retained source slice once and gathers overlapping appearances. Item-local
resample/failure caches allow fresh clipping populations after retry selection.
The strict identity is `physical_gaussian_overlap_spatial_tokens_v1` and records
fixed rules as well as recursive source hashes. Model fitting and checkpoint
execution are outside this implementation.

## Implementation verification

The maintained metadata command calls the production sampler on all 128 pinned
studies, with seed 20260913 and 32 randomized epochs per type, plus deterministic
2× validation. The 595 accepted-center groups and 20 local-fallback groups are
reported separately. Outputs contain source/config/input hashes, final sampler
identity, target/realized depths, source entitlements/counts, run diagnostics and
placement errors. Historical comparison results remain embedded in the summary.

```bash
rtk proxy uv run --locked python -m scripts.adhoc.analyze_physical_sampling --output-dir outputs/analysis/physical-sampling-new/geometry
```

Final artifacts are in
[`geometry-verified`](../outputs/analysis/physical-sampling-implementation-20260913/geometry-verified/summary.json),
[`preparation-verified`](../outputs/analysis/physical-sampling-implementation-20260913/preparation-verified/report.json)
and the [implementation report](../outputs/analysis/physical-sampling-implementation-20260913/implementation-report.json).
Both manifests match the final production preprocessing identity and preserve
source/config/input identities. Earlier directories in this implementation
folder are intermediate verification passes.

| Production physical path | Randomized, 32 draws | Deterministic validation |
|---|---:|---:|
| Selected triplets | 99,968 | 6,248 |
| Target fraction beyond ±30 mm, weighted by target count | 37.40% | 37.36% |
| Realized middle fraction beyond ±30 mm | 37.29% | 37.68% |
| Realized middle fraction beyond ±40 mm | 19.45% | 19.21% |
| Mean unique source files/type | 14.99 | 27.46 |
| 95th percentile placement error | 1.98 mm | 1.95 mm |
| Mean absolute source-share error in repeated groups | 2.99 percentage points | 0.94 percentage points |
| Supplier changes / missing inputs | 0 / 0 | 0 / 0 |

The original sigma-80 prototype also supplied 6,248 validation triplets, 37.68%
beyond ±30 mm, 27.46 unique files/type and 1.95-mm p95 displacement. Its ±40-mm
fraction was 19.24%. Production uses explicit operation seeds and canonical
source/geometry ordering, so final tie placements and randomized draws can
change. The original `requested_fraction` averaged per-group fractions; the new
target fraction weights actual target counts and should not be compared as the
same aggregation. The physical count remains **10.4% above the old sampler's
5,659 validation inputs**, with no runtime or accuracy inference.

All 721 acquisitions passed the run and plane-agreement rules: zero exclusions
for missing valid runs, gap splits, singleton runs or plane disagreements. The
four rejected-center studies supplied 20 local groups, 210 validation triplets
and 3,360 randomized triplets, with no shortfalls or supplier changes.
Their validation p95 local placement error was 0.0320 on the [0,1] coordinate.
The tables retain per-type budgets, repeated/dense/wide and split-membership
strata, target depths and source-specific intended/preferred/realized counts.

The CPU inspection covered eight studies in both modes: four accepted centers
and all four rejected centers. All serial/four-thread and reversed-row outputs
matched. The 88 prepared acquisitions contained 1,881 channel appearances from
1,508 unique-slice appearances across preparations, including 373 reuses.
Independent clipping populations and bounds matched; positions, native maps,
rectangles and reused images were aligned. All 48 local preparations had zero
valid XYZ flags. No decode failures occurred. Gap/singleton/plane-disagreement,
SliceLocation-only and retry branches therefore remain unexercised by these
real-data cohorts; no artificial fixtures were created.

Ruff lint and format checks passed for the affected scope (32 Python files),
config parsing passed, the catalog check passed for 25 scripts, and
`git diff --check` passed. A broader Ruff check found pre-existing issues in
unrelated scripts, listed in the implementation report; those files were not
changed. No tests, models, training, optimizer steps, checkpoint evaluation or
submission execution were run. The completed plan is retained with the
implementation artifacts; its durable behavior is documented here.

## Cohort and reproduction

The saved geometry sample contains all 26,804 headers from 721 series in 128
studies, originally selected without replacement from sorted training-study UIDs
with NumPy seed 42. Its selection and input identity are in
`outputs/analysis/geometry-metadata-20260912/selection.json`. The current geometry
rules accept all 721 series and the shared reference in 124 studies; four studies
fail the maximum RMS criterion. The sampling comparison therefore contains 595
study/plane/contrast groups, including 95 repeated groups with 197 acquisitions.
There are 31 groups containing an acquisition with median spacing below 2 mm and
98 groups spanning more than 120 mm. These thresholds only define descriptive
strata. Of the original 128 studies, 99 belong to the fold-0 training partition
and 29 to its held-out partition; 461 and 134 accepted groups respectively remain.

The protocol audit includes all 128 studies and therefore 99 repeated groups,
including groups from studies whose shared center was rejected. It reads one
original DICOM header per series, selecting the first sorted path and stopping
before pixel data. The geometric simulation reads only the saved header table.
Neither command decodes images, runs a model, trains, evaluates labels or runs
tests.

The protocol command remains active. Reproduce the original geometric comparison
only inside its archived source/runtime with its recorded original configuration:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.audit_sampling_protocols --output-dir outputs/analysis/physical-sampling-20260913/protocols-v2
```

The original protocol `protocols-v2` and geometry `geometry-v3` directories
already contain the completed runs. Use the matching full historical runtime
and the manifest-recorded config when executing their archived analysis source.
Each manifest pins
the inputs, analysis source and relevant dependencies. The geometry artifacts
include `summary.json`, `centers.parquet`, `slots.parquet`, `eligibility.parquet`,
`draws.parquet`, `selected.parquet`, and `sampling-comparison.png`/`.svg`, with
source snapshots and output hashes. The protocol artifacts include the emitted
field snapshot, summary and source snapshot. Original DICOM files are not hashed;
the selected metadata snapshot is hashed. Earlier directories without `-v2` or
`-v3` are preliminary runs, not the canonical result.

## Compared sampling rules

The archived comparison called its then-production capacity allocator and triangular
sampler directly on reconstructed, physically ordered records. It retains 3:1
selection weights, acquisition-midpoint weighting, pooled allocation, complete
non-overlapping triplets, and the baseline type budgets. Its seeded per-group
random-number streams differ from the live loader's study-wide sequencing; this
is a reproducible distribution comparison, not a replay of saved training inputs.

The prototype holds the current accepted center fixed. Within each supported
plane/contrast type it constructs the union of eligible middle-slice coverage,
counting overlaps once. The second through penultimate slices are candidates.
It maps each slice plane to its intersection with the corresponding fixed
patient-coordinate axis through the shared center:

```text
target_point(t) = center + t * patient_axis
slice_depth = dot(normal, ImagePositionPatient - center) / dot(normal, patient_axis)
```

The axis is superior for axial, posterior for coronal and left for sagittal
acquisitions. This handles oblique orientations explicitly. The Gaussian width
is measured along that shared axis; it is not the perpendicular distance from
the center to each differently tilted plane. Gaussian probability is normalized
once over the union of available interior-depth intervals.

Training divides the cumulative probability into K equal bands and draws one
random percentile per band. Validation uses each band's midpoint. The simulation
seed is 20260913, with 32 training draws per group. Training-mode sampling is
simulated on all sampled studies; partition flags allow separate examination
without reading targets. Validation has twice the configured triplet budget.

Two source-selection rules share the same physical targets:

- **Nearest:** jointly select the distinct available middle planes minimizing
  total squared depth error, allowing any acquisition whose coverage contains
  the target.
- **Balanced:** first select a supplier independently of its slice count.
  Training chooses uniformly among covering acquisitions. Validation accumulates
  each eligible source's fractional entitlement and assigns the source with the
  largest remaining deficit, with a fixed seeded tie order. Joint assignment
  then minimizes missing inputs, supplier changes, and squared depth error in
  that order. No supplier changes or missing inputs were needed by the balanced
  variants in this sample.

Overlapping neighboring triplets are allowed. To avoid buying the same middle
plane twice, candidate planes form complete-link geometry groups when every pair
has at most 0.1 mm of offset difference plus normal-direction deviation over a
70-mm radius. Assignment uses each such group at most once. This is an explicit
analysis tolerance, not a protocol-equivalence classifier or an established
optimal anatomical separation. Different sequence images remain alternatives
within the same geometric group. Adjacent physical planes remain eligible.

## Realized coverage

The following are selected-triplet fractions across all accepted groups at
validation density. Distinct raw slices count source files used per group,
including different acquisitions at coincident geometry; they do not count
unique anatomical locations or measure actual decoder calls.

| Sampling | Beyond ±30 mm | Beyond ±40 mm | Mean distinct raw slices/type | 95th percentile placement error |
|---|---:|---:|---:|---:|
| Current | 40.59% | 22.14% | 28.53 | No shared physical target |
| Balanced Gaussian, sigma 20 mm | 15.03% | 0.05% | 21.61 | 2.01 mm |
| Balanced Gaussian, sigma 30 mm | 22.71% | 10.05% | 24.75 | 1.96 mm |
| Balanced Gaussian, sigma 40 mm | 29.85% | 13.59% | 26.11 | 1.92 mm |
| Balanced Gaussian, sigma 60 mm | 35.60% | 17.51% | 27.14 | 1.97 mm |
| Balanced Gaussian, sigma 80 mm | 37.68% | 19.24% | 27.46 | 1.95 mm |
| Balanced Gaussian, sigma 120 mm | 39.05% | 20.65% | 27.71 | 1.91 mm |

At training density, the current sampler puts 39.13% of selected centers beyond
±30 mm, compared with 23.25% at sigma 30 and 37.40% at sigma 80. Distinct source
slices average 15.75, 14.57 and 14.97 per type respectively. The same narrowing
appears in the training-partition-only strata, so it is not a consequence of the
small held-out subset.

The earlier suggestion that sigma 30 would provide broad arms was unsupported
by the actual baseline distribution. The non-overlap constraint and high
validation budget already force the current sampler to cover much of each stack.
The shared center can also differ from acquisition midpoints. Replacing those
rules together with a moderately concentrated Gaussian narrows exposure despite
its nonzero tails. Sigma 80 or 120 approaches current breadth but still does not
increase peripheral selection in this sample. At these widths the Gaussian is
fairly flat over a typical acquisition; width should reflect the intended
anatomical coverage, not an assumed benefit of the Gaussian name.

Deterministic quantiles also deserve care: sigma 20 places only 0.05% of selected
validation centers beyond ±40 mm, even though its continuous truncated Gaussian
has nonzero tail probability. A finite set of midpoint quantiles does not visit
every nonzero-probability region. At sigma 80, realized training and validation
tail exposure is much closer.

The current validation sampler supplies 5,659 triplets; all prototype variants
supply 6,248, an increase of 10.4% under the same nominal budgets. The current
sampler falls below the nominal budget in 231 of 595 groups because its triplets
cannot overlap. Candidate budgets therefore do not imply equal actual CNN work.
No runtime, VRAM or pixel-decoding cost was measured, and fewer distinct files
does not imply fewer decoder calls in the present preparation implementation.

At sigma 80 the largest assignment error is 3.17 mm in validation and 5.58 mm
over the training draws. Three of 99,968 training triplets exceed 5 mm; none
does in validation. Sigma 20 has a 12.19-mm worst validation displacement when
distinct middle planes must be assigned to tightly concentrated targets. A
fixed-center uniqueness constraint can therefore distort a narrow distribution.

## Acquisition exposure and eligibility

For depth t covered by m(t) acquisitions, the intended source share is the
Gaussian probability at t divided equally among those m(t) sources. Integrating
over depth gives each acquisition's entitlement. This depends on its available
coverage, not its number of slices. The numerical reference uses 2,048 fixed
Gaussian quantiles per group.

At sigma 30, mean absolute deviations from these intended acquisition shares,
across the 197 acquisitions in repeated groups, are:

| Method | Validation deviation | Training deviation over 32 draws |
|---|---:|---:|
| Current | 3.30 percentage points | 4.53 percentage points |
| Nearest across acquisitions | 23.91 percentage points | 18.90 percentage points |
| Choose supplier first | 1.10 percentage points | 2.95 percentage points |

The current method has a different target distribution, so its row is contextual
rather than an isolated source-selection comparison. The nearest and balanced
rows use identical physical targets. Nearest selection can favor denser grids
and systematically resolve coincident geometry to the same source. Supplier
selection corrects both effects in this cohort. Dense acquisitions, defined
here by median spacing below 2 mm, average 10.16 percentage points above their
intended share with nearest validation selection, versus 0.53 points below it
with supplier-first selection.

This improvement costs little placement accuracy: mean validation error changes
from 0.906 mm to 0.920 mm at sigma 30. The balanced method's small remaining
deviation comes from finite target and source counts; its training result also
contains Monte Carlo variation. This is not proof of strict invariance under
every possible duplicate acquisition or missing-data pattern.

The broad sigma-80 variant retains this behavior: mean source-share deviation
is 1.06 percentage points in validation and 2.76 in the training draws, measured
against its own sigma-80 coverage entitlements. Dense acquisitions average
0.22 points below their validation entitlement and 0.05 points above their
training entitlement.

Fixed absolute eligibility thresholds are unsuitable as the main source rule.
At sigma 30, a 1-mm maximum perpendicular distance to the nearest middle plane
leaves an average 41.93% of target probability without any eligible acquisition.
The corresponding figures are 3.78% at 2 mm and 0.052% at 3 mm. Such cutoffs can
also favor dense acquisitions whenever they remain eligible and coarse ones
do not. The prototype instead uses coverage and lets each selected source
represent the target with its locally available slices. Placement error is
measured after assignment.

The existing sample has regular within-series spacing. This does not establish
what to do with abnormal gaps or missing slices outside the cohort. The
implementation plan must define valid continuous runs and failure handling;
the above eligibility result is not permission to bridge arbitrarily large
gaps.

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
[implemented sampler](#implemented-sampler), which owns the concrete
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
complete usable-pixel audit. See [metadata provenance](metadata.md#images-per-series).
