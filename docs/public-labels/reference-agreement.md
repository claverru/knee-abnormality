# Reference agreement

These are locally recomputed **label-table macro ROC-AUCs**, using the original
official binary labels without overwriting public predictions. The common
endpoint contains the same 57 reference studies for every source.

| Label table | Rows | Available reference rows | AUC on available reference rows | AUC on common 57 |
|---|---:|---:|---:|---:|
| Steven v1 | 4,407 | 58 | 0.877991 | 0.877731 |
| Steven v2 | 4,407 | 58 | 0.887349 | 0.887402 |
| Steven v4 blend | 4,407 | 58 | 0.892707 | 0.893237 |
| Pilkwang v2 | 4,406 | 57 | 0.870040 | 0.870040 |
| lixin73, either file | 4,407 | 58 | 0.835194 | 0.834699 |

The published Steven scores reproduce to their reported precision. Pilkwang's
reported 0.870 reproduces on **57**, not 58, studies. The published lixin73
0.8347 also matches the common-57 calculation; its score on all 58 is 0.8352.
Earlier summaries describing every comparison as a 58-study evaluation need
this denominator qualification.

| Target | Steven v1, 58 | Steven v2, 58 | Steven v4, 58 | Pilkwang, 57 |
|---|---:|---:|---:|---:|
| ACL | 0.9933 | 0.9933 | 0.9871 | 0.9974 |
| MCL | 0.9637 | 0.9637 | 0.9683 | 0.9757 |
| Medial Meniscus | 0.9543 | 0.9543 | 0.9483 | 0.9429 |
| Lateral Meniscus | 0.8621 | 0.8621 | 0.8789 | 0.8414 |
| Medial OA | 0.9310 | 0.9310 | 0.9318 | 0.9079 |
| Lateral OA | 0.8298 | 0.8298 | 0.8327 | 0.7885 |
| PF OA | 0.9009 | 0.9009 | 0.9015 | 0.8912 |
| Effusion | 0.8534 | 0.8534 | 0.8770 | 0.8299 |
| Synovitis | 0.6780 | 0.7903 | 0.7903 | 0.6938 |
| Baker's | 0.9457 | 0.9457 | 0.9438 | 0.9324 |
| Contusion | 0.8306 | 0.8306 | 0.8596 | 0.7678 |
| Fracture | 0.7931 | 0.7931 | 0.7931 | 0.8713 |

Use the common-endpoint per-target scores in `audit.json` for direct source
comparisons. The table above preserves the complete available endpoint for
each source; it is not a basis for selecting a different teacher per target.

None of Steven's or Pilkwang's available reference rows is wholly binary or
exactly equal to its official target vector. Thus those files have not simply
replaced all reference predictions with the twelve official binary labels.
lixin73 agrees exactly on six studies, but also disagrees on others and has
continuous values on 23 reference rows; agreement alone cannot diagnose copying.
No table's values can prove the absence of indirect reference-label tuning.
