# Shared geometry contract

`SliceRecord` retains each native XYZ position and provenance (`patient`,
`slice_location`, `instance_number` or `filename`). `NativePose` shares repeated
`PoseLayout` values containing the two original orientation axes, pixel spacing,
dimensions and frame ID. Native values are not rounded to a representative pose.
`IndexedSeries` adds bounds, pose validity, orientation/center-line consistency
and rejection reasons. `StudyReference` records an accepted point or explicit
unavailability, contributors, conditioning, plane-balanced residuals, frame IDs and, for a
robust refit, the series excluded from the fit (`outliers`).
`CenterProjection` records each series' diagnostic projection or rejection reason.
The worker index caches compact CPU values without pixels or dense grids.
Prepared triplets carry per-channel crop-to-patient maps, series/native reliability and
valid rectangles. Collation aligns the valid rectangles with image masks and study
slots; the model composes the executed augmentation transform on each forward and reads no
patient coordinates since the attention prior's retirement on 2026-09-16.
The [reflection contract](../data-preparation/reflection-geometry-and-previews.md#reflection-geometry-and-eligibility)
adds acquisition eligibility and study reduction before dropout. Shared GPU
helpers reverse pixels/channels, compose `F @ U` and negate sagittal positions.
The preview tool uses these helpers, applies LPS reflection to its absolute anchor points
while retaining immutable source maps, and supports bounded augmentation-only profiling.

The [sampling experiment](../validation/results/sampling/2026-09-12-accurate-centers-result.md)
selected midpoint sampling. Projections are diagnostics only and do not change
triplet selection, physical position encoding or the centered image crop.
Native pose validity, reference acceptance and an in-range projection remain
distinct: a projection outside a stack does not invalidate its native poses.
Different cross-series frame IDs are recorded, not automatically rejected.
The fixed [screening thresholds](../data-preparation/accurate-sampling-centers.md)
are not bounds on local registration error. Spatial tokens keep every visually valid
observation; position-aware selection does not establish registration or local anatomical
correspondence.

## Geometry and fusion comparison sequence

The shared geometry foundation and accurate-centers comparison are complete.
The frozen fusion control is the submitted `crop140mm` checkpoint/bundle and
midpoint recipe identified in [validation](../validation/results/sampling/2026-09-12-accurate-centers-result.md).
Use its saved config, seed, input/label/split identities and predictions; the
mutable training YAML is not its recipe. Historical checkpoints run through
their archived source/bundles and never receive invented preprocessing identities.
The spatial-fusion stage was implemented, profiled and
[rejected at the deployment runtime gate](../validation/results/sampling/2026-09-12-spatial-fusion-engineering-result.md).
The user subsequently selected the original 27-query, width-64 architecture
with eager execution, later replaced by the [spatial-token model](../modeling/current-model.md).
The reduced compiled variant remains archived. The selected sampler and
submitted model remain the frozen comparison control; their historical
measurements do not establish an accuracy result for the restored implementation.
