# Implemented sampler

The current recipe in `configs/resnet50.yaml` uses positive finite
`data.sampling_sigma_fraction: 0.867` and independent epoch-specific random
targets within each probability band. It preserves the spatial-token model, accepted
study-center estimator, reliability thresholds, 140-mm crop and acquisition-
midpoint 1D encoding. The old allocator, triangular selection and their three
configuration fields have been removed; old fields are rejected. Historical
checkpoints require their matching archived source and configuration.

## Gaussian width

The Gaussian width is relative to each candidate pool's own acquired reach, not a
fixed millimetre constant: `sigma = sampling_sigma_fraction * limiting_reach(pool)`
(`limiting_reach` in `sampling.py`). `limiting_reach` is the distance from the study
centre (depth 0) to the nearer edge of the pool's union-support envelope — taking
`lo`/`hi` as the min start and max end of `union_support(pool.candidates)`, it is
`min(-lo, hi)`, floored at 1 mm. The floor guards the rare pool whose acquired
coverage does not straddle the study centre at all, where `min(-lo, hi)` would
otherwise be zero or negative; measured on the fold-0 token cache this hit 2 of
4,218 physical pools (a `SeriesTypeConfig` slot's own reach can miss the centre even
when the plane's combined coverage across both fluid-sensitivity slots does not).
At the adopted 0.867 the realized per-plane mean widths are axial 45.1 mm, coronal 35.9 mm and
sagittal 39.0 mm
([result](../../validation/results/2026-09-19-relative-sampler-width-result.md)); at the historical
0.76, which preserved the previous fixed sampler's average width, they measured
axial 39.5 mm, coronal 31.4 mm, sagittal 34.2 mm
(4,218 physical pools, fold-0 cache). Each plane splits into two independent
slots (T1/PD and fluid-sensitive/fat-suppressed) that are sampled separately, so a
slot's own reach is narrower than the plane's two slots combined. This applies only
to physical pools; local (non-physical) pools stay uniform over their support union,
unaffected by sigma. Evidence and rationale: [per-plane sampler sigma
results](../../validation/results/2026-09-18-per-plane-sigma-cache.md) and
[cohort-wide center fit and plane extents](../accurate-sampling-centers.md#cohort-wide-center-fit-and-plane-extents).

## Candidate runs and physical support

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

## Triplet budgets and target draws

The triplet budget of a view is its `slices` times the image multiplier:
6 axial or 5 coronal/sagittal inputs per contrast before the study-total floor
and capacity limits. A series type with several
[views](../../modeling/current-model.md#view-table) selects once per view from the
same pool, each with its own budget and its own training draws. The current recipe uses validation multiplier 1; prediction
uses its separately selected multiplier. Gaussian probability over the union is
divided into equal bands. Evaluation uses each band's midpoint.

Training draws one uniform percentile inside each probability band, independently
at each epoch. All main bands remain represented each epoch. Assignment snaps
targets to available geometry groups, so actual triplets can repeat across epochs.
The experimental ten-epoch subregion schedule was
[reverted after inconclusive confirmation](../../validation/results/2026-09-20-stratified-sampling-promotion.md).
Training chooses a covering acquisition uniformly. Evaluation allocates each
target's fractional supplier entitlement (exact fractions) to the largest
outstanding deficit and breaks exact ties by the pool's source ranking.
Source preference precedes nearest-slice placement, and a target can only use
candidates from a run whose own support contains it.

## Source ranking

Each candidate pool stores one fixed source ranking (`CandidatePool.ranking`,
`source_ranking` in `sampling.py`) over the series that contributed candidates;
both modes list covering acquisitions in this order. The first difference wins:

1. originals before reformats (`ImageType` containing DERIVED, SECONDARY or
   PROJECTION, or `mpr`/`reformat` in the description);
2. fewer newer retakes: the number of other series in the slot with the same
   signature (TR and TE rounded to 1 ms, echo train length, flip angle,
   `MRAcquisitionType`, `SequenceVariant`, `ScanOptions`, rows, columns) and a
   higher `SeriesNumber`, so the latest retake comes first;
3. higher crop coverage, rounded to 0.1: the share of the 140-mm square centred
   on the projection of the accepted study centre that lies inside the middle
   slice's image;
4. finer acquired in-plane resolution (worst axis of field of view ÷
   `AcquisitionMatrix`), clipped at the frame's pixel size (`data.reference_field_mm` over
   `data.image_size`) and rounded to 0.01 mm;
5. `SeriesInstanceUID`.

A value that is missing for any ranked series (no accepted centre, no pose, no
matrix) ties that step for the whole slot; a series without TR/TE or
`SeriesNumber` counts no newer retakes. The tags come from the headers the
indexer already parses ([series summary](../current-preparation.md#series-indexing-guards-and-exclusions)).
Design and evidence: [tie-break investigation](evaluation-tie-break-investigation.md).

## Geometry assignment and local sampling

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

## Seeds and deterministic evaluation

Only training draws random numbers. SHA-256 seeds include the configured seed,
study UID, type, operation and epoch; separate streams govern target
within-band jitter, supplier choices and band order. Retries reuse these streams
against rebuilt support. Evaluation uses midpoints, target order and the ranking, so it
does not depend on the seed: evaluation and prediction datasets receive no seed
(`training_seed=None`). Dataset order and decode completion order do not select
inputs.
Copying a source does not multiply continuous physical target mass with center
and support fixed. Copies still change supplier alternatives, and the unchanged
reference estimator or changed grids can affect results; full duplicate
invariance and exact final source shares are not claimed.

## Preparation and checkpoint identity

[Preparation](../current-preparation.md#overlapping-triplet-preparation) normalizes each
unique retained source slice once and gathers overlapping appearances. Item-local
resample/failure caches allow fresh clipping populations after retry selection.
The strict identity is
`ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10` and
records fixed rules as well as recursive source hashes. Model fitting and checkpoint
execution for existing checkpoints requires their archived source and configuration.
The ten-epoch sampler rollback restores the original target-rule metadata and
sampler source. Its experimental checkpoints retain their archived runtimes;
validation midpoint selection was unchanged throughout.
