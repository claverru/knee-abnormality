# Where the balanced validation gain comes from

For each label, AUC averages correctly ordered positive–negative pairs, with half credit for ties.
Partitioning those pairs gives an exact accounting of the ordinary → control-B macro difference.
Here, “extreme” describes the stored reference value, not independently established confidence.

| Positive reference | Negative reference | Share of macro pair weight | Contribution to macro gain | Share of total gain |
|---|---|---:|---:|---:|
| q ≥ 0.9 | q ≤ 0.1 | 51.09% | −0.000020 | −0.2% |
| q ≥ 0.9 | 0.1 < q < 0.5 | 31.80% | +0.008068 | 62.5% |
| 0.5 ≤ q < 0.9 | q ≤ 0.1 | 11.76% | +0.000952 | 7.4% |
| 0.5 ≤ q < 0.9 | 0.1 < q < 0.5 | 5.35% | +0.003911 | 30.3% |

Pairs with extreme references on both sides contribute essentially no aggregate improvement.
Separately normalizing AUC within extreme cells gives ordinary **0.930556**, control-B
**0.930325**, stratified **0.933304**; these normalized scores use different label weights from
the contribution table and should not be substituted into its accounting.

The original Steven v1 table provides a more interpretable indicator: `q_v1 == 0.5` means the
report extractor marked that finding unaddressed. Gold cells are always treated as addressed.

| Original report coverage of pair | Share of macro pair weight | Contribution to macro gain | Share of total gain |
|---|---:|---:|---:|
| Both addressed | 67.30% | +0.002488 | 19.3% |
| Addressed positive, unaddressed negative | 32.47% | +0.010427 | 80.8% |
| Unaddressed positive, addressed negative | 0.17% | −0.000060 | −0.5% |
| Both unaddressed | 0.05% | +0.000056 | 0.4% |

Requiring both addressed evidence and v1/v4 binary agreement retains 7,792 of 10,848 cells.
Pairs where both cells qualify carry 62.55% of pair weight but only **8.1% of the gain**;
qualifying positives against other negatives account for 90.6%. This condition is a source
consistency check, not a clinical correctness guarantee.

## Per-label AUC and sensitivity to reference strength

The full-fold gain spans eleven labels, yet several gains collapse or reverse on extreme cells.
This is more informative than the full-fold macro alone. Positives refer to the full 904 studies.

| Label | Positives | Ordinary AUC | Control-B AUC | Full-fold Δ | Extreme-cell Δ |
|---|---:|---:|---:|---:|---:|
| ACL | 184 | 0.907428 | 0.917806 | +0.010379 | −0.002023 |
| MCL | 134 | 0.832099 | 0.862192 | +0.030093 | +0.001762 |
| Medial meniscus | 356 | 0.915120 | 0.926587 | +0.011467 | +0.004332 |
| Lateral meniscus | 138 | 0.873557 | 0.897803 | +0.024246 | +0.014570 |
| Medial OA | 329 | 0.898369 | 0.907763 | +0.009393 | +0.001675 |
| Lateral OA | 241 | 0.855010 | 0.868603 | +0.013593 | −0.003544 |
| PF OA | 406 | 0.868973 | 0.869275 | +0.000302 | −0.007978 |
| Effusion | 525 | 0.869235 | 0.864832 | −0.004403 | −0.002441 |
| Synovitis | 115 | 0.896148 | 0.899631 | +0.003483 | −0.000644 |
| Baker's cyst | 216 | 0.885518 | 0.913874 | +0.028356 | +0.004023 |
| Contusion | 153 | 0.883336 | 0.895521 | +0.012184 | −0.011955 |
| Fracture | 63 | 0.919257 | 0.935092 | +0.015835 | −0.000547 |

Among frozen negatives, original v1 marked 96.2% of Synovitis, 66.3% of Baker's and 44.4% of
Fracture cells unaddressed. Restricting to addressed cells leaves only 30 Synovitis negatives,
so its restricted AUC is particularly fragile. Control-B also raises mean prediction
0.328378 → 0.390193 and worsens soft BCE 0.453629 → 0.485663. This is not merely calibration:
a monotonic per-label probability shift cannot change AUC, whereas these rankings changed.

## Alternative label sources and repeated runs

Each endpoint below uses the same selected predictions, official gold overrides and an equal
mean over twelve label AUCs. Masks are per cell, not a single common study subset.

| Endpoint | Ordinary | Control-B | Stratified | Pearson to public, newer five |
|---|---:|---:|---:|---:|
| Frozen v4, all cells | 0.883671 | 0.896582 | 0.897515 | −0.672 |
| Frozen v4, v1-addressed cells | 0.906905 | 0.909593 | 0.910632 | −0.746 |
| Frozen v4, addressed and v1-agreeing | 0.916119 | 0.917182 | 0.918587 | — |
| Frozen v4, extreme cells | 0.930556 | 0.930325 | 0.933304 | −0.624 |
| Original Steven v1 labels, addressed cells | 0.904597 | 0.901975 | 0.904745 | +0.645 |
| Pilkwang labels, verdict other than UNK | 0.898216 | 0.895757 | 0.897696 | +0.282 |
| TonyLin soft labels, all cells | 0.879398 | 0.891273 | 0.892957 | −0.656 |
| Lixin labels, all cells | 0.884878 | 0.897745 | 0.899472 | −0.691 |

Changing the reference labels is different from masking frozen labels: masking alone does not
restore public ordering. Steven v1 addressed and Pilkwang known endpoints reverse the loss
comparison, but are still report-derived and were explored after seeing public results.
Lixin is an ingredient of v4, so agreement there is not independent corroboration.

A broader saved-run check compares two ordinary runs with four balanced runs of the same
numerical recipe, allowing seed and the balancing flag to differ. The six selected epochs are:

| Run beneath `outputs/training/` | Seed | Completed epoch | Full AUC |
|---|---:|---:|---:|
| `sigma-fraction-0867` | 42 | 6 | 0.886486 |
| `overnight-20260918T232636Z-mt7ouybk/runs/rexnet-fold0` | 471048644 | 10 | 0.883671 |
| `balanced-bce` | 42 | 8 | 0.895282 |
| `overnight-20260919T120130Z-i2oqlnls/runs/00-control` | 927831896 | 9 | 0.894037 |
| `overnight-20260919T235434Z-8ol_1c30/runs/00-control` | 1133376617 | 7 | 0.893222 |
| `overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat` | 2263738815 | 9 | 0.896582 |

Balanced-minus-ordinary group means: **+0.009702 full AUC**, +0.000718 extreme-cell AUC,
+0.000768 addressed-and-agreeing AUC, **−0.001834 v1-addressed-label AUC** and
**−0.002491 Pilkwang-known-label AUC**. Thus endpoint sensitivity is not unique to control-B.
These are descriptive group means, not paired effect estimates. The seed-42 references have
different preprocessing digests; no exact same-source, same-seed ordinary control was found.
See the [source limitations](../../../validation/results/labels-and-losses/2026-09-19-balanced-bce-result.md#existing-references-checked-2026-09-19).

## Why the coverage distinction matters

[The label audit](../../../public-labels/uncertainty-and-confidence-encoding.md) establishes that v4 is
half Steven v2 and half Lixin. Of 13,438 original unaddressed cells across all 4,407 studies,
10,085 become 0.25. Steven v2 also infers unaddressed Synovitis from Effusion. Distance from 0.5
therefore does not recover report coverage or calibrated confidence. Pilkwang's UNK is broader,
including unmentioned and explicitly uncertain findings; its scores are described as ranking scores.

The [archived host criteria](../../../metadata/official-label-criteria.md) describe image-based
adjudication, confirmed absence for negatives and severity thresholds. Better ranking against a
report-derived negative can therefore fail to improve image-defined discrimination. That is the
mechanistic hypothesis supported by this audit; identifying actual mislabels needs image review.
Moving the binary threshold alone did not solve the issue: at thresholds 0.3–0.9 the balanced
advantage persisted and the newer-five macro/public correlation remained negative (about −0.49 to −0.69).
