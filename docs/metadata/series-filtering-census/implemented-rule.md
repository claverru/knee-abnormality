# Series filtering: implemented rule and deletion-identity census

## Implemented rule (2026-09-17)

Implemented in `src/knee_abnormality/data/preprocessing/filtering.py` and `series.py`, identically
in training, validation and Kaggle inference; the behaviour is described in
[current preparation](../../data-preparation/current-preparation.md#series-indexing-guards-and-exclusions).
Preprocessing identity `ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`.

- **Non-knee coil:** in a study with more than one `ReceiveCoilName` and at least one coil name
  that is not a non-knee part, drop series whose coil name names a non-knee part (ankle, foot,
  wrist, hand, elbow, shoulder, hip, head, neck, breast) and not the knee.
- **Stray other knee:** split studies whose series volume centres spread more than 150 mm in x at
  the largest gap; drop a single-series side when the other side has at least two series.
- **Within-series guard:** keep only instances matching the series' most supported rows, columns,
  pixel spacing and orientation (tolerance 1e-3).
- Missing tags or geometry keep the series; no study loses every series. Far series (thigh, calf),
  patellar, oblique and bilateral series are kept; contrast and plane relabels are not
  implemented (evidence above).

Verification, CPU, all 24,386 series of the 4,407 training and 3 example-test studies (
`index_census.py` (`041bd16:docs/history/series-and-sampling-sources/deterministic-selection/index_census.py.txt`); `outputs/analysis/d/deterministic-selection-20260917/`
`index_report.json`, `index_series.parquet`, `index_studies.parquet`):

| Check | Result |
|---|---|
| Series dropped | 6: the 5 `SENSE_ANKLE_8_AC` series of `…93672103` (fold 2) and the stray sagittal T1/PD of `…42012883` (fold 0) |
| Body/spine-coil studies (5) and other multi-coil studies (3) | Untouched |
| Instances removed by the guard | 0 |
| Study centres changed | 1: `…93672103`, RMS 42.05 → 0.48 mm, rejected → accepted |
| Example-test studies | Nothing dropped or guarded |
| Volume-centre x against the census | Within 3.0 mm for all series (the census used a slightly different centre); no effect at the 150-mm split |

## Deletion-identity census (2026-10-05)

**The superseded and exact-copy rules delete no series of another knee or another region
(measured): of 213 superseded and 24 exact-copy deletions in the 4,407 training studies, the
largest offset between the deleted series and the one kept is 24.9 mm, and every exact copy
matches its kept series on every frame.** The grouping keys omit in-plane position, spacing,
orientation and frame-of-reference UID, and the copy check compares only the middle frame; this
census measures what those omissions let through on the training data.

Method: the loader's own `index_series`, `non_knee_coils`, `superseded_series`, `relabelled_slots`
and `exact_copies` in the `clean_series` order, on `configs/resnet50.yaml` at `ca0f91c`; for each
deleted series, the volume-centre offset to the series kept from its group, the header
differences, and a pixel comparison of every frame. CPU only, 58 s with 6 workers.

| Measure | Superseded | Exact copy |
|---|---:|---:|
| Deleted series / studies | 213 / 166 | 24 / 6 |
| Every frame equal to the kept series | 9 | 24 |
| In-plane centre offset, median / 95th / maximum (mm) | 0.000 / 0.033 / 24.9 | 0.000 / 0.001 / 0.001 |
| In-plane offset over 5 mm / over 20 mm / over 100 mm | 10 / 2 / 0 | 0 / 0 / 0 |
| Orientation difference over 0.01 | 0 (maximum 0.00006) | 0 |
| Pixel spacing differs (maximum 0.104 mm) | 7 | 0 |
| Different frame-of-reference UID | 213 | 24 |

- The ten superseded deletions offset by 5 to 25 mm are repositioned retakes of the same knee
  (the kept series has the higher series number in all ten); the largest lateral shift is 24.6 mm
  against the 150 mm other-knee split.
- Frame-of-reference UID differs in every pair, the nine pixel-identical retakes included, so it
  cannot serve as an identity key here.
- Nine superseded deletions are pixel-identical to the series kept: copies that the superseded
  rule removes before the copy rule sees them.

Script: `scripts/adhoc/census_series_deletions.py`. Outputs:
`outputs/analysis/s/series-deletion-census-20261005/` (`pairs.parquet`, `summary.json`).
