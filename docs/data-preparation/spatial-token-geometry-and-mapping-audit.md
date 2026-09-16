# Spatial-token geometry and mapping audit

## Spatial-token geometry

Collation retains per-channel valid image rectangles. Study centers and center acceptance
were removed from collation with the additive 3D embedding on 2026-09-14, and the
prepared-pixel-to-patient maps and triplet geometry validity with the
[attention prior](../history/model-lineage.md#retired-attention-prior) on
2026-09-16; prepared series keep both for reflection eligibility and previews, and the dataset
keeps its study reference for physical sampling and previews. The full-slice restoration of
2026-09-16 kept this: batches carry neither tensor. Native positions,
orientation/depth payloads and regional packing slots are no longer transferred to the
model. The series-level mask requires both valid native poses and
`SeriesGeometry.reason is None`; orientation, center-line or within-series frame-ID
rejection therefore marks the prepared geometry invalid.
The shared candidate acceptance path applies this mask after preparation for
both production loading and the threaded loading profiler, at every thread count.
`CenterProjection.reason` is diagnostic only. Different frame IDs across series
remain recorded rather than rejected. Shared-center reliability thresholds and
its independence from selected counts and series dropout are unchanged.

Visual validity requires the retained triplet's inverse-augmented anchor to lie
inside the valid prepared rectangles of all three channels. It alone controls MIL
membership. Until 2026-09-16, an independent geometry validity (reliable triplet/series
geometry and finite mapped coordinates) gated the attention prior's 3D bias; a rejected study
center made it invalid through the local sampling fallback. The reflection preview still
shows it. Unavailable 1D physical positions still follow the existing rejection
behavior, with no slice-index fallback added. A study with no visually valid spatial
tokens fails before softmax.

ReXNet anchors start at pixel zero at stride 32. 2D coordinates use the actual
augmented-image dimensions. The preview's patient points compose the executed inverse affine
with the middle slice's prepared-to-LPS map and stay absolute, in mm, without a study
center, clipping or normalization. Calculations run in float32 outside autocast. An anchor represents a feature with an overlapping receptive field
mixing all three slices, not an isolated voxel or registered correspondence.

## Spatial-fusion mapping audit

The initial spatial-fusion comparison stopped after its
[inference gate failed](../validation/results/2026-09-12-spatial-fusion-engineering-result.md).
The regional implementation is historical; the active spatial-token model reuses
its prepared-pixel and augmentation mapping. The saved audit below describes
the earlier source identity and does not validate the new token path.

The candidate preserved a per-channel mapping from prepared pixel centers to
native pixels using actual rounded resampling dimensions and crop/pad offsets.
Patient coordinates then used native orientation, spacing and origin. Kornia
0.8.3's `warp_affine` normalizes its matrix with the corner-based convention
but executes this pipeline's grid with `align_corners=False`; the inverse
pixel mapping therefore conjugated the inverse affine by those normalization
conversions. It used the exact sampled matrix without additional RNG draws.
Feature anchors followed ReXNet's symmetric-padding stride-32 grid with origin
zero, verified from its spatial convolutions. Anchor validity required all
three channels' inverse-mapped bilinear support to remain within their prepared
rectangles. It did not claim full receptive-field tissue support.

The active implementation constructs affine matrices and performs inverse
coordinate conversion entirely in float32 with autocast disabled, matching
the image warp's precision. Previously, those matrix multiplications could
execute in BF16 during mixed-precision training and round the coordinate map
differently from the image warp. The correction has been reviewed in source
only; its numerical error and training impact have not been measured.

The [16-study audit](../../outputs/analysis/spatial-fusion-20260912/audit/summary.json)
had 15 accepted centers; every study supplied native-valid local tokens from
three planes, with at least 863 valid anchors per study after augmentation.
Saved artifacts include per-triplet paths, pixel hashes, references, geometry
tensors, eight original/augmented overlay sheets, and nearest-plane query
neighborhood sheets. The first and fifth neighborhood sheets were visually
reviewed: central queries remained around the knee joint across views, while
corner queries sometimes fell outside tissue and selected slices could remain
several millimeters from a query. This supports using broad soft guidance,
not exact anatomical-location claims. Obliquity and independent augmentation
were visible; invalid/mixed-source metadata and incomplete decoded triplets
were not established by this sample. No new registration threshold was validated.

The initial candidate's real-data gradients were finite, repeated validation
logits were identical, and a missing-center development pass stayed finite.
Sampling repeated deterministically. Separate paired inference and training
manifests matched the archived control's selected triplets, prepared pixels
and one-dimensional positions exactly. These checks do not establish final
compiled training behavior or an accuracy gain; the performance stop happened
before those stages.
