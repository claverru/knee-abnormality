# Implemented sampler

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

[Preparation](../current-preparation.md#overlapping-triplet-preparation) normalizes each
unique retained source slice once and gathers overlapping appearances. Item-local
resample/failure caches allow fresh clipping populations after retry selection.
The strict identity is `physical_gaussian_overlap_spatial_tokens_v1` and records
fixed rules as well as recursive source hashes. Model fitting and checkpoint
execution are outside this implementation.
