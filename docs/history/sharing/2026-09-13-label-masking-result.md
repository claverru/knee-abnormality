# 2026-09-13 Label-masking result

**Rejected:** `masked-labels` scores **0.879520** full-fold AUC versus the
reflection champion's **0.890976** (−0.011455). Restore full nonmissing soft-target
supervision with gold overrides; the masking implementation and configuration
have been removed. Reflection remains the selected baseline.

| Metric | Reflection | Masked labels | Masked − reflection |
|---|---:|---:|---:|
| Full-fold AUC (904 studies) | 0.890976 | 0.879520 | −0.011455 |
| Gold AUC (11 studies) | 0.922784 | 0.910351 | −0.012434 |
| Report-derived AUC (893 studies) | 0.890228 | 0.878792 | −0.011436 |
| Soft-target BCE (lower is better) | 0.449310 | 0.498855 | +0.049545 |
| Training seconds | 1794.98 | 1675.52 | −119.46 |

Both runs completed ten epochs and selected epoch 9 (zero-based index 8), with
4,380 optimizer steps and peak allocated CUDA memory of 20,368,205,824 bytes.
Only MCL improved; the other eleven findings regressed.

| Finding | Reflection AUC | Masked AUC | Delta |
|---|---:|---:|---:|
| ACL | 0.913949 | 0.902431 | -0.011519 |
| MCL | 0.836247 | 0.849729 | +0.013481 |
| Medial Meniscus | 0.922563 | 0.917401 | -0.005162 |
| Lateral Meniscus | 0.889980 | 0.884134 | -0.005846 |
| Medial OA | 0.903951 | 0.887438 | -0.016514 |
| Lateral OA | 0.868678 | 0.855485 | -0.013193 |
| PF OA | 0.874246 | 0.847869 | -0.026376 |
| Effusion | 0.868662 | 0.853484 | -0.015178 |
| Synovitis | 0.896302 | 0.873963 | -0.022340 |
| Baker's | 0.908141 | 0.882375 | -0.025766 |
| Contusion | 0.888732 | 0.883737 | -0.004996 |
| Fracture | 0.920257 | 0.916200 | -0.004058 |

Saved configurations differ only in experiment name and the masking policy.
Both use seed 42, fold 0 of `all_5fold_v3`, Steven v4, 224 pixels, 140-mm crops
and the same reflection recipe. Input identities match for preprocessing, label
manifest/source, split protocol, membership, validation rows and series metadata
and counts. Training-row identity changes with the observation mask.

The [method and counts](../../modeling/supervision.md#training-label-masking) document the
10,548 removed training entries and 46 wholly masked studies. This single-fold,
single-seed result supports rejecting this implementation; it does not establish
that every uncertainty-weighting approach fails. The 11-study gold diagnostic
is especially small. No new training, inference or public submission was run
for this comparison.

Evidence: [comparison JSON](../../../outputs/analysis/m/masking-vs-reflection-20260913/comparison.json)
contains full metrics, per-target deltas, configurations, input identities and
artifact SHA-256 hashes. The selected masked checkpoint is
`outputs/training/masked-labels/fold_0/best-epoch=08-val_macro_auc=0.87952.ckpt` (deleted on 2026-09-16);
its [source snapshot](../../../outputs/training/masked-labels/source_snapshot.zip),
[effective configuration](../../../outputs/training/masked-labels/effective_config.json) and
[epoch metrics](../../../outputs/training/masked-labels/fold_0/epochs/metrics_epoch_008.json)
remain available alongside the reflection run's corresponding artifacts.
