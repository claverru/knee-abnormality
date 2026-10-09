# 2026-10-05 code-audit measurements

**An external read-only audit of `ca0f91c` listed eleven mechanisms; every one exists in the code,
and none measurably hurts the current recipe on the training data.** The largest measured effect
is the reflection-TTA coordinate mismatch, worth +0.000024 pooled out of fold when corrected.
Three items stay open because they need a training run. Nothing was changed in `src/`.

| ID | Mechanism | Outcome |
|---|---|---|
| F01 | Reflection TTA leaves the detail-to-wide map unmirrored | Real; correcting it gives +0.000024 ([measurement](#f01-reflection-tta-detail-coordinates)) |
| F02 | Contrast mixing blends the 84 mm and 93 mm crops by pixel index | Open: needs a run with the donor resampled onto the target grid |
| F03 | Superseded and exact-copy keys omit in-plane position; copies compare the middle frame | No wrong deletion in training ([census](../../metadata/series-filtering-census.md#deletion-identity-census-2026-10-05)) |
| F04 | Knee side from the sign of patient X | Coordinates agree with reports as often as explicit tags ([census](#f04-and-f05-knee-side-and-centre-census)) |
| F05 | Centre fit across different frame-of-reference UIDs | Frames are per series and the fit holds to 2.8 mm median RMS ([census](#f04-and-f05-knee-side-and-centre-census)) |
| F06 | Intensity bounds depend on the slices the bag selected | Same size as epoch-to-epoch training jitter, no direction ([measurement](#f06-bag-dependent-normalization)) |
| F07 | Checkpoint selected at ×1, deployed at ×4 or ×6 | Open: runs keep only the best-epoch checkpoint, so it cannot be measured from existing runs |
| F08 | `load_models` ignores a runtime temperature for a member directory; baking drops the non-persistent `compartment_signs` | Harmless: `export.py` bakes each member and sets the runtime temperature to 1.0, no local script passes a temperature to a directory, and `anatomical_bias` is false in both configs |
| F09 | EMA/SWA resume drops auxiliary and anchor state | Unused path (`weight_averaging: null`) |
| F10 | `load_source` accepts NaN labels | The pinned `rebuild-state-v2` table holds 0 NaN and 0 null |
| F11 | `study_slice_total` only scales budgets up | As [documented](../../modeling/current-model.md) |

## F01: reflection TTA detail coordinates

`reflect_batch` mirrors the images, valid rectangles and patient maps but returns `view_to_wide`
unchanged, so the 2D position input of mirrored detail views (the two coronal types) is not in the
frame the training reflection uses. The correction is `view_to_wide[mirror] @ reversal`, the same
reversal matrix already applied to the patient maps.

Parity against the model's own training reflection (reflection forced, every other augmentation
off, one batch), maximum absolute difference:

| Quantity | TTA as shipped | TTA corrected |
|---|---:|---:|
| 2D position embedding | 0.835 | 0.0004 |
| Medial and anterior axes | 0.00002 | 0.00002 |
| Valid-token mismatches | 0 | 0 |

Scores: the five Refined ResNet-50 children of S106, pooled out of fold over 4,407 studies at ×6,
τ 0.8, all variants from the same prepared batches:

| Variant | Pooled macro AUC | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|---:|
| No TTA | 0.936115 | 0.939985 | 0.934544 | 0.938474 | 0.933660 | 0.933859 |
| TTA as shipped | 0.936273 | 0.940111 | 0.934431 | 0.938826 | 0.933741 | 0.933975 |
| TTA corrected | 0.936297 | 0.940141 | 0.934534 | 0.938865 | 0.933744 | 0.934060 |

Corrected minus shipped: +0.000024, study bootstrap (2,000 draws) 80% [−0.000002, +0.000051],
95% [−0.000015, +0.000065], positive in 89% of draws and in all five folds; no label moves by more
than 0.0001. The shipped figure reproduces the 0.936272 recorded for S106.

`inference/reflection.py` is one of `identity.PREPROCESSING_SOURCES`, so the correction invalidates
every existing checkpoint. It is the first seed of the
[fixes-and-removals campaign](../../../autoresearch/campaigns/fixes-and-removals/INDEX.md),
adopted there by user decision.

Script: `scripts/adhoc/score_tta_detail_coordinates.py`. Outputs:
`outputs/analysis/t/tta-detail-coordinates-20261005b/` (`report.json`, `predictions.parquet`).

## F04 and F05: knee side and centre census

All 4,407 training studies through the loader's `_index_study_geometry` on `configs/training.yaml`.

| Side source | Studies | Report names one side: agree / checked |
|---|---:|---:|
| Explicit tag | 2,155 | 654 / 662 (98.8%) |
| Coordinates, no tag | 2,082 | 462 / 468 (98.7%) |
| Unknown | 170 | |

Coordinates-only by vendor: GE 290/294, Toshiba 76/76, Siemens 64/64, Philips 31/33, Canon 1/1.
The report side is a full-word parse of the report text, so about 1.2% disagreement is the noise
of the check itself. A wrong side needs no explicit tag, no side word in the description fields
and an origin more than 40 mm off the midline; the hidden test sites are not covered.

| Centre measure | Value |
|---|---|
| Full fit / robust refit / none (`excessive_rms`) | 4,231 / 169 / 7 |
| Fit RMS, median / 90th / 99th percentile | 2.8 / 5.8 / 8.8 mm |
| Largest per-series residual, median / 99th percentile | 3.6 / 15.1 mm |
| Studies with trusted geometry where every contributing series has its own frame UID | 4,238 of 4,238 |

Script: `scripts/adhoc/census_knee_side_and_centre.py`. Outputs:
`outputs/analysis/k/knee-side-centre-census-20261005/` (`studies.parquet`, `summary.json`).

## F06: bag-dependent normalization

The first 300 fold-0 validation studies (about 1,670 series), each prepared as five bags; a
series' gain is the width of its quantile window, and the pixel difference is over the slices two
bags share.

| Bag pair | Gain change, median / 90th / 99th pct | Signed mean | Pixel difference of 255, median / 90th pct |
|---|---:|---:|---:|
| Validation ×1 → ×4 | 2.6% / 10.1% / 23.2% | −0.4% | 2.1 / 7.8 |
| Validation ×1 → ×6 | 3.2% / 11.3% / 26.2% | −0.5% | 2.6 / 9.3 |
| Training epoch 0 → epoch 1, ×1 | 2.7% / 14.3% / 36.7% | −0.3% | 1.9 / 9.0 |
| Training epoch 0 → validation ×6 | 3.4% / 12.7% / 31.0% | −1.1% | 2.7 / 10.0 |

The window moves as much between two training epochs as between a training bag and a ×6
inference bag, so the denser bag is inside what training sees. The epoch-to-epoch movement exceeds
the ±10% contrast augmentation in one series out of ten; a window fixed per series would remove it
(not run).

Script: `scripts/adhoc/measure_bag_normalization_drift.py`. Outputs:
`outputs/analysis/b/bag-normalization-drift-20261005/` (`series.parquet`, `summary.json`).
