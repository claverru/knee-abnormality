# 2026-09-21 untied position path

**Rejected; implementation removed.** Removing explicit position from the pooled values
(appearance-only pooling) scored −0.0018 against historical control B, and the comparison is
confounded by repaired data. It tested the
[untied position path](../../../research/geometry-and-views/position-encoding.md#phase-1-head-only-changes) and the
[independent-classifier proposal](../../../research/geometry-and-views/knee-side-and-location.md#5-separate-attention-inputs-from-pooled-appearance).

## Change

`model.appearance_only_pooling` (now removed), applied to all targets:

- Attention logits still read projected image features plus 1D slice, 2D token, plane and
  contrast embeddings, then LayerNorm.
- Pooled values read image features plus plane and contrast embeddings only, normalized
  separately with the same LayerNorm parameters. No 1D or 2D position enters the values.
- Same attention weights, masks and independent classifier rows; no new parameters.

Image features can still carry spatial context, and position still steers attention.

## Result

Candidate `outputs/training/untied-position/candidate`, seed 2263738815 (B's seed), fold 0 of
`all_5fold_v3`, ten epochs, selected epoch 9 of 10 (B also 9), 3,503 training and 904 validation
studies. Training 1,551.92 s against B's 1,541.67 s. Intervals are paired report-group bootstraps,
2,000 draws.

| Metric | Control B | Candidate | Difference |
|---|---:|---:|---:|
| Macro AUC | 0.896582 | 0.894777 | −0.001805 |
| Medial Meniscus | 0.926587 | 0.916540 | −0.010047 |
| Lateral Meniscus | 0.897803 | 0.900537 | +0.002734 |
| Medial OA | 0.907763 | 0.905072 | −0.002691 |
| Lateral OA | 0.868603 | 0.866337 | −0.002266 |
| PF OA | 0.869275 | 0.866800 | −0.002475 |
| Meniscus margin AUC | 0.971248 | 0.971022 | −0.000226 |
| OA margin AUC | 0.866116 | 0.834838 | −0.031278 |

Macro 95% [−0.005212, +0.001388]. Medial Meniscus [−0.017509, −0.002880] and OA margin
[−0.060555, −0.004338] are below zero (exploratory, unadjusted). Neither family mean improved.

## Confound

The candidate's preprocessing restores two repaired DICOM slices that B excluded
([data repair record](../../../data-preparation/public-data-contract-and-decoding.md)); the exclusions
module changed after B. All input-identity fields match except the preprocessing source
fingerprint. Matching the seed does not isolate the pooling change. No fresh control was trained
for this run; the later [control repeat](2026-09-21-anatomical-attention-result.md#control-repeat-on-repaired-data)
on repaired data scored 0.895112, 0.001470 below B.

## Verdict

No basis to promote appearance-only pooling; the recipe keeps position-rich pooling. This does
not show that every positional separation or geometry mechanism fails: the
[anatomical attention](2026-09-21-anatomical-attention-result.md) line on the same data was
adopted.

Artifacts: `outputs/analysis/u/untied-position-vs-control-b/` (`comparison.json`, `per_label.json`).
