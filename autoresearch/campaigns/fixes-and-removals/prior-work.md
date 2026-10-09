# Fixes-and-removals prior work

What the 2026-10-05 audit of `ca0f91c` found and what was measured the same day. Full tables are
in the [code-audit measurements](../../../docs/validation/results/2026-10-05-code-audit-measurements.md).

## Precedents

| ID | Mechanism | Setting | Run or measurement | Macro delta | Source |
|---|---|---|---|---|---|
| F01 | Reflection TTA with `view_to_wide @ F` (the training reflection's convention) | Five S106 ResNet-50 children, pooled out of fold, ×6, τ 0.8 | `outputs/analysis/t/tta-detail-coordinates-20261005b` | +0.000024 (95% [−0.000015, +0.000065]) | code-audit measurements, F01 |
| F02 | Contrast mixing off | ResNet-18 fold 3, 2026-09-27, before the 84/93 mm views | `08-contrast-mixing-off` | −0.0013 ×1, −0.0025 ×4 | `docs/validation/results/2026-09-27-b3-overnight-queue.md` |

## Measured and closed: do not propose

- **Series-deletion keys (F03):** 213 superseded and 24 exact-copy deletions in training; largest
  offset to the kept series 24.9 mm, every copy equal on every frame.
- **Knee side from coordinates (F04):** agrees with the report side in 462 of 468 checked
  studies (98.7%), against 98.8% for explicit tags.
- **Centre fit across frame-of-reference UIDs (F05):** every series has its own UID; fit RMS
  median 2.8 mm, 99th percentile 8.8 mm; seven studies rejected by the RMS gate.
- **Bag-dependent intensity window (F06):** the window changes by 3.2% (median) from ×1 to ×6
  and by 2.7% between two training epochs, with no direction.
- **Temperature for member directories (F08):** `export.py` bakes each member and sets the
  runtime temperature to 1.0; no local script passes a temperature to a directory.
- **`study_slice_total` (F11):** scales budgets up only, as documented.

## Open, outside this campaign

- **Checkpoint selected at ×1, deployed at ×4 or ×6 (F07):** the validation multiplier and the
  selection are protected by the harness; it needs a run that keeps every epoch's checkpoint.

## Follow-up audit (2026-10-05, by reading)

- **Detail-view positions under the training reflection:** `encode_batch` gives a mirrored
  coronal detail view x-reversed positions (`view_to_wide @ prepared_xy`, with the flip folded
  into `prepared_xy`), while wide views use the final image grid. The TTA variant measured as F01
  copies that reversed convention, which is why it matches training and barely moves the score.
  Seed 1 corrects both paths to the mirrored window.
- **Read as sound:** collate indexing, CPU and GPU pixel preparation, patient maps and axis
  signs, the meniscus mask and OA regions, losses and class balance, L2-SP, the schedule, label
  joins, checkpoint and bundle validation.

## Candidates not yet seeded

- **Unknown knee side switches off what needs only the centre** (`data/dataset.py`, `valid`
  requires the side sign): coronal and sagittal crops fall back to the image centre and contrast
  mixing never fires, although the view table uses no side-dependent offset. About 162 training
  studies have a fitted centre and no side (4,400 with a centre, 4,238 with trusted geometry).
- **Two instances per slice position:** `valid_runs` breaks a run at every gap of 0.01 mm or
  less, so a series with repeated positions yields no triplets and leaves the centre fit. Its
  frequency is not measured.

- **Contrast mixing on unaligned grids (F02):** the donor is blended by pixel index although the
  T1/PD crop is 84 mm and the fluid crop 93.33 mm. A fix resamples the donor onto the target grid
  through the two patient maps, conjugated by the mirror when the study is reflected.
- **A window fixed per series (F06):** quantile bounds from a bag-independent sample.
- **Feature-statistics augmentation can sample a negative standard deviation**
  (`modeling/augmentation.py`, `resampled_std` has no positivity clamp).

## Removals already measured

| Mechanism switched off | Recipe | Macro delta ×4 | Source |
|---|---|---|---|
| Report-attribute heads | ResNet-18, 2026-10-01, before the 84/93 mm views | −0.0006 (80% [−0.0011, −0.0001]) | revisits campaign |
| Contrast mixing, together with per-slice quantile 0.98 | Current ResNet-18 recipe, 2026-10-05, against this campaign's reference | −0.0013 (95% [−0.0033, +0.0007]); the quantile alone −0.0007, so about −0.0005 for mixing off on top of it | pretraining-input campaign, `rn18-slice-quantile-no-mixing` |
| Contrast mixing | ResNet-18, older recipe (`radimagenet-rn18` campaign) | +0.0006 (80% [−0.0011, +0.0022]) | `rn18-native-contrast-preservation` |
| Contrast mixing | ResNet-10t | +0.0020 (80% [+0.0010, +0.0029]) | pretraining-input campaign |
| Bilateral smoothing | ResNet-10t | +0.0009 (80% [−0.0001, +0.0019]) | pretraining-input campaign |
| GIN | ResNet-10t, mixing already off | −0.0006 (80% [−0.0018, +0.0007]) | pretraining-input campaign |
| Anatomical attention bias | ResNet-18, 2026-10-01 | adopted as a removal | revisits campaign |

## Active mechanisms not re-tested on the current recipe

Reflection (excluded by rule), affine, intensity (gamma and contrast), noise, bilateral smoothing,
resolution degradation, GIN and feature-statistics augmentation; the half-knee loss, the
three-compartment OA detector, the meniscus own-side mask, L2-SP anchoring, the plane-contrast
readouts, the view label bias, the detail-view embedding and the 1D slice-position embedding.
