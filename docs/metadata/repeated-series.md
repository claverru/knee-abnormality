# Pixel-identical repeated series

The 2026-09-22 exhaustive scan found 33 training series that repeat another series of the same
study pixel for pixel, in 9 of 4,407 studies. It also found one pair of studies that share all five series. No
series repeats slices within itself. The same-study filter is queued in the
[overnight queue](../backlog/overnight.md#next-queue). On 2026-09-22 the user set aside
excluding the cross-study copy from training labels; it is noted under
[training-label exclusions](../backlog/endgame.md#training-label-exclusions). Folds and labels
are unchanged.

## Detection

The [audit script](../../scripts/adhoc/audit_repeated_series.py) reads the 12 KB header prefix
of every train and example-test DICOM file (819,635 files, 24,386 series, all native explicit
little-endian). It also reads an 8 KB chunk from the middle of each pixel payload and hashes it with
BLAKE2b. It takes 64 s on 12 CPU processes. Series that share any non-constant chunk hash
become candidate pairs. Every matched file pair is then decoded in full and compared exactly.

| Check | Result |
|---|---:|
| Constant chunks excluded (62 series, none fully constant) | 554 files |
| Same-study pairs, all chunks shared and all frames equal | 33 |
| Cross-study pairs, all chunks shared and all frames equal | 5 (one study pair) |
| Cross-study pairs sharing one chunk, frames unequal | 2 |
| Within-series repeated chunks (one 3D series, frames unequal) | 1 series |

The three chunk-only matches are near-empty volume-edge slices of 3D sagittal series (about 8%
non-zero pixels) whose central 8 KB coincide. Full-frame verification rejects them. A filter
therefore needs a full-frame hash, or a chunk hash confirmed on the full frame.

Artifacts: `outputs/analysis/r/repeated-series-20260922/`, which holds per-file hashes, per-series
headers, verified pairs, the summary, the manifest and a source snapshot. Pairs are classed as
identical, subset or partial over their distinct informative chunks.

Reproduction (CPU only, fresh output directory; `--studies N` limits the scan to the first N sorted
studies of each split for a trial):

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.audit_repeated_series --output-dir outputs/analysis/repeated-series-new
```

## Same-study repeats

- **8 Philips studies:** every one of their four series (axial, coronal and sagittal fluid-sensitive, and
  sagittal non-fluid) appears twice, with the same `SeriesNumber` (8–11) and
  `SeriesDescription` `DummySeriesDesc!`. This accounts for 32 pairs; the whole study was probably exported twice.
- **1 Siemens study (…782343):** series 1019 `DESS MPR 1mm cor Knie rechts` is a pixel copy of
  series 6 `t2_tse_sag_d`. Both are labelled Sagittal, not fluid-sensitive, in the official CSV.
- All 33 pairs share the official plane and contrast type, so each repeat lands in the same
  sampling slot as its original.
- Folds (`all_5fold_scanner_v1`): 4 studies in fold 0, 2 in fold 1, 1 each in folds 3 and 4.
  None is gold.

## Cross-study repeat

Studies `…99725523560254732094432244556623376711` (fold 2) and
`…99926624968240681772735102303531613209` (fold 1) have identical images in all five series. The
headers say right knee (explicit `R`, patient-left-negative centre x). The fold-1 study's report is
`SAĞ DİZ` (right knee), but the fold-2 study's report is `SOL DİZ` (left knee) with different findings. The
fold-2 study therefore pairs a left-knee report with right-knee images, and its labels do not
describe its pixels. This contradicts the earlier conclusion of the [study overlap investigation](study-overlap-investigation.md),
whose bounded search did not reach this pair.

## Online detection per study

The filter has to run per study at inference, so it is simulated with the
[online script](../../scripts/adhoc/simulate_online_series_dedup.py) on the production
`index_series` output for all 4,407 training studies
(`outputs/analysis/o/online-series-dedup-20260922/`):

1. **Header key (no extra I/O):** series with the same slot, slice count, matrix and sorted slice
   positions (rounded to 0.01 mm) form a bucket. Those headers are already in memory, and bucketing
   takes 0.09 ms per study.
2. **One frame per bucketed series:** decode the middle-position frame and require exact array
   equality.

| Check | Result |
|---|---:|
| Copies found (exhaustive ground truth 33) | 33 |
| False positives | 0 |
| Equal middle frames outside the header key (same count and matrix, any slot or geometry) | 0 |
| Studies needing any frame read | 695 of 4,407 (16%) |
| Extra frame reads per study, mean (max) | 0.42 (12) |

For scale: at x4, prediction already reads about 180 headers and decodes 149 frames per study, at
0.82 s per study cold on Kaggle ([profile](../inference/2026-09-18-kaggle-x4-profile-after-loading-changes.md#cold-loading-stages-100-studies)).
The check therefore adds about 0.3% to frame reads. This is an I/O-count estimate that has not been timed on Kaggle.

Reproduction (CPU only, fresh output directory; ground truth is the audit's
`outputs/analysis/r/repeated-series-20260922/pairs.parquet`). It runs `index_series` per series on one
thread, so its indexing time is not the threaded inference loader's, and its frame-read timings are local:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.simulate_online_series_dedup --output-dir outputs/analysis/online-series-dedup-new
```

The exhaustive scan found every same-study relation to be all-or-nothing, so one frame decides
as well as a full comparison. Exact equality is also well separated from the nearest non-copies. Those
are 38 bucketed pairs with a middle-frame correlation of 0.996–0.99995 but at most 73% equal pixels. Nearly all are
Philips second reconstructions of one acquisition: series *N*01/*N*02, often with an `e`-prefixed
description. They are not literal copies, and a tolerance threshold would absorb them. The
[source ranking](../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#proposed-deterministic-rule)
already orders them as same-signature retakes.

## Header-only alternative

This rule needs no pixel reads: within a study, pair series with equal sorted slice positions
(rounded to 0.01 mm), file count, matrix, TR and TE. It finds all 33 repeats but also 334 non-identical
pairs, which are retakes, GE deep-learning reconstruction pairs and multi-contrast series with the same geometry.
Also requiring equal `SeriesNumber` leaves 32 repeats and no false positives on training, but
misses the Siemens MPR copy. The online check above uses this geometry key only to choose which series to decode.
