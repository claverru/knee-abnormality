# Current preparation and overlapping triplets

Evidence: unverified working notes, official/public sources checked on
2026-09-09, and the local acquisition audit on 2026-09-10; see
[provenance](../research/provenance.md).

The current baseline prepares pixels on the fly and uses 224-pixel centered
crops at 0.625 mm in-plane spacing, giving 140-mm coverage after the
[completed crop comparison](../validation/results/2026-09-12-140-mm-crop-result.md).
Earlier 0.80 mm inspections and the
historical cached-pipeline notes below are unverified evidence, not current
defaults.

## Overlapping triplet preparation

Production uses the [shared physical sampler](physical-sampling/implemented-sampler.md).
Selected triplets arrive explicitly at preparation. Three records must be
consecutive and distinct within each triplet; neighboring triplets can share
paths, with distinct middle indices per acquisition. The decoder receives a
stable unique-path list in both serial and threaded modes.

Clipping bounds use valid crop pixels from each unique source slice belonging
to a retained complete triplet once. Orphans from failed triplets are excluded.
Each unique image is normalized once; gathering assembles images, native maps,
valid rectangles and paths in the selected channel order. The midpoint supplies
the unchanged acquisition-relative millimetre position. Local-fallback inputs
always have unknown 3D geometry, even if native poses are individually valid.

Unnormalized resamples and failures are cached within the current item's type
preparation attempts. A failed path excludes every containing triplet and splits
the original runs; neither spacing nor the reference or 1D origin is refitted.
Constant acquisitions and unidentified acquisition errors exclude that source.
Every retry makes a new exclusion and rebuilds coverage and quantile bands.
Metadata separates triplets, channel appearances, unique files, distinct middle
geometry, decode attempts, placement errors and missing capacity.

The bounded CPU inspection command includes four accepted-center and all four
rejected-center studies from the pinned metadata cohort, in both dataset modes:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.inspect_physical_preparation --output-dir outputs/analysis/physical-sampling-new/preparation
```

It records serial/four-thread and row-order comparisons, independent clipping
populations, channel metadata and masks without models or a test harness.
See [implementation evidence](physical-sampling/implementation-verification.md).
The former dataloader optimization benchmark is historical and requires its
original source/config runtime; it is no longer an active current-config caller.
