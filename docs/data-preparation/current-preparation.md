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
- **Study cleaning** (`clean_series` in `data/preprocessing/filtering.py`),
  decided on all indexed series of a study before `estimate_reference` and
  applied to every slot. Rules run cheapest first, so free header rules shrink
  the work of the rules that read pixels:
  1. *Non-knee coil:* when a study uses more than one non-empty
     `ReceiveCoilName` and at least one of them does not name a non-knee part,
     series whose coil name contains ankle, foot, wrist, hand, elbow, shoulder,
     hip, head, neck or breast, and not knee, are dropped. Body and spine coils
     are not anatomy words.
  2. *Superseded series:* in one slot, series with the same sequence signature,
     slice count, matrix and sorted positions (0.01 mm) keep one: a non-reformat
     first, then the highest `SeriesNumber`, then the higher UID. These are
     retakes, Philips second reconstructions and exact copies with full headers.
     Headers only
     ([result](../validation/results/2026-09-22-series-cleaning-rules.md#same-signature-groups)).
  3. *Contrast relabel:* a header candidate is a series whose fat-suppression
     evidence contradicts its slot. The evidence is `ScanOptions` FS/WE, an
     inversion-recovery sequence, or a fat-suppression description without T1
     timing in a T1/PD slot; or no such evidence, with TE, TR or scan options
     present, in a fluid slot. Only studies with a candidate decode 3 slices
     (25/50/75%) of each remaining series. For each slice they compute body-mask
     intensity quantiles, bright fraction, and ring, inner and central medians
     relative to the body's 99th percentile. A candidate moves to the same
     plane's other slot when its nearest trusted series in the study belongs to
     the other contrast. Same-plane series are used only when both contrasts
     exist in that plane; otherwise series from every plane are used. Nothing
     is trained.
  4. *Exact copy:* series of one slot, after relabelling, with the same slice
     count, matrix and sorted positions decode their middle-position frame. A
     series whose frame equals an already kept one is dropped (reformats first,
     then the higher UID). Only grouped series read a frame
     ([repeated series](../metadata/repeated-series.md#online-detection-per-study)).
  5. *Stray other knee:* when series volume centres spread more than 150 mm in
     patient x, the study is split at the largest x gap. A side with one series
     is dropped if the other side has at least two. Bilateral exams keep both
     sides. This runs after the redundancy rules, so repeated series do not
     inflate a side.
  - Rules 1 and 5 never drop every remaining series of a study; missing tags or
    geometry keep the series. Thigh, calf, patellar, oblique and other far series
    are kept. Slot metadata lists dropped series as `excluded_sources`.
- **Effect (all 4,407 training and 3 example-test studies):** rules 1 and 5 drop
  6 series, exactly the 5 ankle series of `…93672103` and the stray sagittal of
  `…42012883`; the 5 body/spine-coil studies are untouched; one study centre
  changes (`…93672103`, RMS 42.05 → 0.48 mm, now accepted); the guard removes
  nothing ([census](../metadata/series-filtering-census.md#implemented-rule-2026-09-17)).
  Rules 2 and 4 together drop 237 series. That is every pixel-identical copy (33 in 9
  studies, one per pair, including the Siemens `DESS MPR` copy rather than its
  original) plus 204 superseded series in 163 studies; 52 of those are in 42
  fold-0 studies, the set measured at +0.00015 inference-only. Rule 3 moves
  11 series, exactly 11 of the 12 census-confirmed contrast mislabels and to the
  same slots; it misses one Siemens `pd_tse_fs_sag_np` whose visual verdict is
  ambiguous. It evaluates 53 candidates in 41 studies. No example-test series is dropped or moved.
  Cleaning takes 0.5 ms per study without a contrast candidate and 52 ms on
  average (max 148 ms) with one, local CPU. The near-empty check catches all 61
  frames found by the repeated-series scan, and also some frames that scan
  missed because their central chunk was not exactly constant.

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

**Pixels are finished where the hardware has room.** Loader workers read only
the rows the resampling filter touches, straight into one shared buffer per
decode call, and mask the stored bits in place. `prepare_images` in
`data/preprocessing/device.py` then resamples, takes the quantile bounds from the
series' exact ranks, normalizes and gathers the uint8 triplets. On a CPU batch it
uses the original scipy sparse products and numpy partitions; on a GPU batch it
repeats them tap by tap in float32, bit-identically. Training, validation and
local prediction loaders call it in their workers (`collate_prepared`) and ship
uint8 images, because the local GPU is the bottleneck there; the Kaggle
prediction worker ships the raw rows (`prepare_on_loader=False`) and prepares
them on the T4s, because its four vCPUs are. Frames that are not 8/16-bit, or
whose rescale exceeds 1e30, are resampled on the CPU and shipped as float32
regions. See the
[measurement and parity evidence](../inference/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md#gpu-preparation).

Decoded frames and failures are cached within the current item's type
preparation attempts. A decoded frame with under 1% non-zero stored pixels
(`NEAR_EMPTY_FRACTION` in `preparation.py`) fails like a decode error, so a
near-empty edge slice is never used. The count is over the whole frame: a
trimmed read whose rows already hold enough non-zero pixels passes, and any
other frame is read in full and counted exactly. A failed path excludes every
containing triplet and splits the original runs; neither spacing nor the
reference or 1D origin is refitted. Unidentified acquisition errors exclude that
source, and every retry makes a new exclusion and rebuilds coverage. A constant
acquisition is only known once `prepare_images` computes its quantile bounds, so
it is dropped from the batch there instead of excluded and re-sampled; no train
study at ×4 contains one.
Metadata separates triplets, channel appearances, unique files, distinct middle
geometry, decode attempts, placement errors and missing capacity.

The physical-preparation inspection (`scripts/adhoc/inspect_physical_preparation.py`,
removed; last in commit `df4b6cd`) checked serial/four-thread and row-order agreement and independent clipping
populations against the CPU preparation; it reads prepared images and bounds that
the loader no longer produces.
See [implementation evidence](physical-sampling/implementation-verification.md).
The former dataloader optimization benchmark was removed; its code is in commit `aec290d`.
