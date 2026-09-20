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

## Series indexing, guards and exclusions

Implemented 2026-09-17; the same `KneeStudyDataset` path runs in training,
validation and Kaggle inference.

- **Headers:** `index_series` (`data/preprocessing/series.py`) parses every
  file's pre-pixel header once. Since 2026-09-18 `read_header` is a raw
  explicit-VR little-endian parser over a fixed 12 KiB prefix, extended and
  retried when the pixel-data element lies beyond it and falling back to pydicom
  for any other encoding; it costs 0.306 ms per header on Kaggle against
  pydicom's 7.6 ms. Prefix reads and frame payloads are fetched on a thread pool
  behind `KneeStudyDataset(io_threads=...)`, defaulting to 1 so training and
  validation stay serial, with inference at `INFERENCE_IO_THREADS = 16`. Prepared
  batches are audited identical to the serial and pydicom paths
  ([loading speedup](../backlog/loading-speedup/attribution.md#what-the-shipped-changes-bought)).
  Besides the pose and pixel descriptor it keeps
  12 series tags (`SeriesTags` in `dicom.py`) and stores the most common tag set
  as a compact `SeriesSummary`: `SeriesNumber`, reformat flag, same-protocol
  signature, acquired in-plane resolution and `ReceiveCoilName`. No extra file
  is read; tag parsing adds about 65 µs per header (30–45% of warm header
  indexing, measured on 337 series).
- **Instance guard:** within a series, instances whose rows or columns differ
  from the most supported layout, or whose pixel spacing or orientation differ
  from it by more than the orientation tolerance (1e-3), are removed before
  sorting. Instances without a
  pose are kept; if no layout is self-consistent, nothing is removed. This is a
  hidden-test defence ([research](../research/series-filtering-signals.md#localizers-scouts-and-calibration));
  it removes no training or example-test instance.
- **Study exclusions** (`data/preprocessing/filtering.py`), decided on all
  indexed series of a study before `estimate_reference` and applied to every
  slot:
  - *Non-knee coil:* when a study uses more than one non-empty
    `ReceiveCoilName` and at least one of them does not name a non-knee part,
    series whose coil name contains ankle, foot, wrist, hand, elbow, shoulder,
    hip, head, neck or breast, and not knee, are dropped. Body and spine coils
    are not anatomy words.
  - *Stray other knee:* when series volume centres spread more than 150 mm in
    patient x, the study is split at the largest x gap, and a side with one
    series is dropped if the other side has at least two. Bilateral exams keep
    both sides.
  - A rule never drops every series of a study; missing tags or geometry keep
    the series. Thigh, calf, patellar, oblique and other far series are kept.
    Slot metadata lists dropped series as `excluded_sources`.
- **Effect (all 4,407 training and 3 example-test studies):** 6 series dropped,
  exactly the 5 ankle series of `…93672103` and the stray sagittal of
  `…42012883`; the 5 body/spine-coil studies are untouched; one study centre
  changes (`…93672103`, RMS 42.05 → 0.48 mm, now accepted); the guard removes
  nothing ([census](../metadata/series-filtering-census.md#implemented-rule-2026-09-17)).

## Robust study centre (removed)

Implemented 2026-09-17 behind `data.robust_center_outlier_mm` and removed on 2026-09-18 after four
training runs at 10, 20 and 30 mm landed inside the control spread
([results](../validation/results/2026-09-18-overnight-results.md)), a subgroup check on the 21
studies it changes showed neither harm nor benefit, and the combined run scored 0.938 publicly
against 0.939 for the checkpoint it replaced. A rejected study centre again samples every series
locally, which is also the flattest coverage in the dataset: local pools draw their targets
uniformly across each slab rather than from the sigma-40 Gaussian, so the refit would have reduced
peripheral coverage for exactly those studies
([coverage](peripheral-coverage.md)). After
the exclusions above, 177 of 4,407 training studies are rejected (31 of 904 on fold 0). The
analysis that produced it is kept in
[history](../history/robust-study-centre.md).

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
