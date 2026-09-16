# 2026-09-13 Disease-and-compartment attention validation comparison

`disease-compartment-attention` scores **0.889430 macro AUC** on all 904 validation
studies, **−0.002593 versus 1B** (`meniscus-oa-classifiers`, 0.892022) and
**−0.001546 versus reflection** (0.890976). This is a competitive regression on
the primary local selection metric. The initial recommendation favored 1B;
subsequent public scores favored original 1C over 1B, and reflection/sigma 40
later became the champion. This section preserves the local comparison, not
a current recommendation to restore 1B.

All three effective configurations and pretrained backbone identities match.
Data input identities match, with only preprocessing/model source identities
differing. Frozen protocol, membership and target hashes were verified. Saved
selected-checkpoint predictions cover exactly the same 904 unique studies, folds
and official target order, with finite probabilities in [0, 1]. Recomputed AUC
matches the best saved epoch metric. The saved 1C identity confirms both shared
classifier pairs and the disease-plus-signed-compartment attention factors.

| Metric | Reflection | 1B | 1C | 1C − 1B |
|---|---:|---:|---:|---:|
| Full-fold macro AUC | 0.890976 | 0.892022 | 0.889430 | -0.002593 |
| Meniscus mean AUC | 0.906272 | 0.909762 | 0.906810 | -0.002952 |
| OA mean AUC | 0.886315 | 0.885661 | 0.880112 | -0.005549 |
| Report-derived AUC (893) | 0.890228 | 0.891061 | 0.888592 | -0.002469 |
| Gold AUC (11) | 0.922784 | 0.941997 | 0.937368 | -0.004630 |
| Soft-target BCE (lower is better) | 0.449310 | 0.453585 | 0.448458 | -0.005127 |

| Target | 1B | 1C | Delta |
|---|---:|---:|---:|
| ACL | 0.911609 | 0.916629 | +0.005020 |
| MCL | 0.834096 | 0.831130 | -0.002966 |
| Medial Meniscus | 0.931077 | 0.921635 | -0.009442 |
| Lateral Meniscus | 0.888447 | 0.891985 | +0.003538 |
| Medial OA | 0.905241 | 0.903650 | -0.001591 |
| Lateral OA | 0.866081 | 0.856574 | -0.009507 |
| PF OA | 0.872871 | 0.874988 | +0.002117 |
| Effusion | 0.867596 | 0.877583 | +0.009986 |
| Synovitis | 0.896192 | 0.890505 | -0.005687 |
| Baker's | 0.900476 | 0.907145 | +0.006669 |
| Contusion | 0.902622 | 0.891152 | -0.011471 |
| Fracture | 0.927958 | 0.910179 | -0.017779 |

Five targets improve and seven decline. The largest losses are fracture
(−0.017779), contusion (−0.011471), lateral OA (−0.009507) and medial meniscus
(−0.009442). Effusion gains 0.009986, Baker's 0.006669 and ACL 0.005020. Both
OA targets decline, as do the mean AUCs of both experimental families. BCE improves
by 0.005127; that is useful evidence about probability quality, but full-fold AUC
remains the selection metric. The gold endpoint has only eleven studies.

Exclusive-positive subsets use the frozen >=0.5 label cutoff. Meniscus has 344
studies (281 medial-only, 63 lateral-only); OA has 198 (143 medial-only, 55
lateral-only). Each contains three gold studies. For each target, negative count
equals the other target's positive count. Margin AUC uses
`logit(p_medial) - logit(p_lateral)` to distinguish medial-only from lateral-only,
clipping probabilities to [1e-7, 1-1e-7] before logits.

| Exclusive-positive diagnostic | 1B | 1C |
|---|---:|---:|
| Medial Meniscus subgroup AUC | 0.906682 | 0.911879 |
| Lateral Meniscus subgroup AUC | 0.819974 | 0.813252 |
| Meniscus margin AUC | 0.980173 | 0.972208 |
| Meniscus correct within-study rankings | 325/344 | 322/344 |
| Medial Meniscus-only correct rankings | 275/281 | 274/281 |
| Lateral Meniscus-only correct rankings | 50/63 | 48/63 |
| Medial OA subgroup AUC | 0.706548 | 0.714812 |
| Lateral OA subgroup AUC | 0.542022 | 0.571647 |
| Oa margin AUC | 0.870184 | 0.854291 |
| Oa correct within-study rankings | 160/198 | 156/198 |
| Medial OA-only correct rankings | 138/143 | 123/143 |
| Lateral OA-only correct rankings | 22/55 | 33/55 |

The OA per-target subgroup AUCs improve, but its within-study margin AUC falls
0.870184 → 0.854291. Correct lateral-only rankings improve 22/55 → 33/55 while
medial-only rankings fall 138/143 → 123/143; total correct rankings fall 160/198
→ 156/198. This improves one subgroup at the expense of the other rather than
improving overall separation. Meniscus margin AUC falls 0.980173 → 0.972208 and
correct rankings fall 325/344 → 322/344. Per-target subgroup AUC and within-study
ranking measure different behavior; neither establishes actual anatomical
localization from these mostly report-derived labels.

1C selected epoch 10 (index 9), versus epoch 7 for 1B. Both completed ten epochs
and 4,380 optimizer steps. 1C's best score occurs at the end of the fixed budget;
this result does not establish its performance with a different training budget.
1B recorded 1667.15s training, 33.75s selected-checkpoint prediction and 18.624 GiB peak allocated CUDA memory.
1C recorded 1811.77s training, 34.45s selected-checkpoint prediction and 18.965 GiB peak allocated CUDA memory.
These are recorded run costs, not a controlled runtime benchmark.

The [comparison artifact](../../../outputs/analysis/disease-compartment-vs-controls-20260913/comparison.json)
preserves all four runs' metrics, subgroup counts, within-study rankings, epoch
curves, configs, input identities, timing and hashes of checkpoints, predictions
and source snapshots. The process finished its selected-checkpoint prediction
export during this review; the analysis uses those final artifacts. No fresh
training, inference or tests were launched. No public score for 1C was checked;
this single-fold result supports model selection without quantifying repeatability.
