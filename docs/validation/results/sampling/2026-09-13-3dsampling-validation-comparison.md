# 2026-09-13 3dsampling validation comparison

The selected `3dsampling` checkpoint (epoch 9, index 8) scores **0.889964**
versus **0.889506** for the selected epoch-10 five-embedding spatial-token
previous champion: **+0.000458 macro AUC**. Kaggle submission **56210103**
completed with **0.930 public AUC**, confirmed through the API on 2026-09-13,
versus **0.929** for that previous champion. The **+0.001** public improvement
was subsequently surpassed by [reflection](2026-09-13-reflection-validation-comparison.md)
at **0.931**.

Both runs use seed 42, Steven v4, fold 0 of `all_5fold_v3`, and 2× validation
images. Their saved training/validation row hashes, split protocol, membership,
label source and series metadata/count hashes match. Saved prediction study IDs
also match across all 904 rows. Model, optimizer and trainer settings match;
3dsampling replaces the triangular sampler with physical Gaussian overlap
sampling at sigma 80 mm. Selected epochs differ. This compares saved training
outputs; no fresh inference or training was run.

| Metric | Previous champion | 3dsampling | Delta |
|---|---:|---:|---:|
| Full validation AUC (904) | 0.889506 | 0.889964 | +0.000458 |
| Report-derived AUC (893) | 0.888626 | 0.889051 | +0.000425 |
| Gold AUC (11) | 0.931118 | 0.936673 | +0.005556 |
| Soft-target BCE, lower is better | 0.450625 | 0.452571 | +0.001946 |
| Public AUC | 0.929 | 0.930 | +0.001 |

| Target | Previous champion | 3dsampling | Delta |
|---|---:|---:|---:|
| ACL | 0.903170 | 0.918893 | +0.015723 |
| MCL | 0.844146 | 0.838486 | -0.005660 |
| Medial Meniscus | 0.923086 | 0.919616 | -0.003470 |
| Lateral Meniscus | 0.885326 | 0.896857 | +0.011532 |
| Medial OA | 0.897793 | 0.898729 | +0.000936 |
| Lateral OA | 0.867189 | 0.869179 | +0.001990 |
| PF OA | 0.875082 | 0.867193 | -0.007889 |
| Effusion | 0.865013 | 0.864345 | -0.000668 |
| Synovitis | 0.894440 | 0.890164 | -0.004276 |
| Baker's | 0.908289 | 0.911122 | +0.002833 |
| Contusion | 0.891787 | 0.887201 | -0.004586 |
| Fracture | 0.918748 | 0.917785 | -0.000963 |

Five targets improve and seven decline. ACL and lateral meniscus account for
the largest gains; PF OA and MCL show the largest losses. Overall AUC is nearly
unchanged and BCE worsens. One fold and one training seed do not establish a
repeatable improvement, and the gold subset contains only eleven studies.

The comparison artifact (`outputs/analysis/0-9/3dsampling-vs-spatial-tokens-20260913/comparison.json`, deleted)
records full metrics, deltas, source paths and metric/prediction hashes. Original
configs, input identities and source snapshots remain in each run directory.
The [submission record](../../../../outputs/submission/records/knee-3dsampling-20260913.json)
records the completed public evaluation; see [submission details](../../../inference/submissions/early-submission-details.md).
