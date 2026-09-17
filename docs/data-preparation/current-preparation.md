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
  file's pre-pixel header once. Besides the pose and pixel descriptor it keeps
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

## Robust study centre (option)

Implemented 2026-09-17 behind `data.robust_center_outlier_mm`. The field is required and is
`null` in `configs/training.yaml`; a value in millimetres turns the option on. The option is
untested in training: one seed-42 run is queued
([overnight queue](../backlog/overnight.md#next-queue), [proposal](../backlog/big-bets.md#robust-study-center)).

- **Off (`null`):** `estimate_reference` (`data/preprocessing/geometry.py`) keeps the plain fit and
  its [reliability checks](accurate-sampling-centers.md). A rejected study samples every series
  locally. After the exclusions above, 177 of 4,407 training studies are rejected (31 of 904 on
  fold 0).
- **Refit:** the refit runs only for a study whose plain fit is rejected. `robust_center`
  repeatedly excludes the series with the largest in-plane residual while that residual exceeds
  the value, and refits after each exclusion. The refit is accepted only if it still has two
  planes, a condition number of at most 1e3 and an RMS of at most 10 mm. Accepted studies keep
  their plain reference unchanged.
- **Excluded series:** they are listed in `StudyReference.outliers`, which is `None` for a plain
  fit. They leave only the centre fit and remain in their slots.
- **Sampling:** in a refit study, `build_pool` (`sampling.py`) samples a series physically only
  if the centre lies inside its acquired slice range (`within_acquired_range`). Any other series
  takes the local path, with source status `outside_acquired_range`.
  - The existing slot rule then keeps only the physical triplets when a slot mixes both paths.
  - An out-of-range series therefore leaves the input, unless no series in its slot contains the
    centre.
- **Identity:** when the option is set, the preprocessing identity gains a `robust_center` entry
  (`ROBUST_CENTER_IDENTITY` plus `outlier_mm`).
  - With `null`, every identity field except the source digest is unchanged.
  - The digest changed because this code edits `dataset.py`, `geometry.py` and `sampling.py`. The
    three ranked-filtered checkpoints (`outputs/training/ranked-filtered-seed42`, `-seed43`,
    `-seed44`) therefore no longer validate against the working tree. Prediction or export needs
    their archived sources (`source_snapshot.zip`); their existing Kaggle bundles carry their own
    code and are unaffected.
  - Configs and checkpoints saved before this change lack the required field and do not load. The
    run configs `outputs/training/configs/ranked-filtered-seed43.yaml` and `-seed44.yaml` were
    given `null`.
  - Kaggle inference reads the option from the checkpoint's saved config, like every data option.

Verification (measured, CPU; all 4,407 training studies; evaluation pools, which do not depend on
the seed; [scripts](../history/series-and-sampling-sources/README.md#robust-center-implementation),
`outputs/analysis/robust-center-implementation-20260917/report.json`):

- **`null` against HEAD:** all 4,407 study references are identical. All 904 fold-0 evaluation
  selections (45,846 triplets) are identical.
- **Against the re-score patch:** the 20 mm option rescues the same 21 fold-0 studies at the same
  centres (maximum difference 0 mm), and the same 12 series leave the input.
  - In training it rescues 130 studies, against the census's 131. The missing study is
    `…93672103`, which the ankle exclusion already accepts.
  - For the other 130 studies, centres and exclusion counts match the census exactly.

| Rescued studies (train / fold 0) | 20 mm | 10 mm |
|---|---:|---:|
| Rescued, of 177 / 31 rejected | 130 / 21 | 177 / 31 |
| Series excluded from the fit | 155 / 27 | 256 / 49 |
| Series in the input before | 843 / 141 | 1,109 / 196 |
| Series that leave the input | 58 / 12 | 61 / 12 |
| … of which excluded from the fit | 49 / 8 | 58 / 10 |
| Series left local (no in-range series in the slot) | 4 / 0 | 6 / 0 |
| Evaluation triplets before → after | 7,008 → 7,008 | 9,306 → 9,306 |
| Accepted or still-rejected studies changed | 0 | 0 |

No series joins the input, and no slot loses triplets. The plain fit rejected all 47 studies
that stay rejected at 20 mm for RMS.

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
