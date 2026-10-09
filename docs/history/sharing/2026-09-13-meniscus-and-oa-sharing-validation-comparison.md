# 2026-09-13 Meniscus and OA sharing validation comparison

`meniscus-oa-classifiers` scores **0.892022 macro AUC** on all 904 validation
studies: **+0.000830** over `meniscus-classifier` and **+0.001047** over
`spatial-tokens-reflection`. These are competitively relevant gains on the primary
selection metric. The decision at this stage was to retain both shared classifier
pairs and use this run as the control for 1C. Per-target, subgroup and BCE results
provide context; they do not negate the measured full-fold AUC improvement.
The later [public result](2026-09-13-1b-public-result-and-overnight-code-choice.md)
is **0.925 versus reflection's 0.931**, changing the recommendation for the main
hyperparameter search to reflection.

The selected checkpoint is epoch 7 (index 6); both controls selected epoch 9.
All runs completed ten epochs and 4,380 optimizer steps. Saved effective configs
and pretrained backbone identities match exactly. Data input identities match;
only preprocessing/model source identities differ. The candidate identity records
two separate shared vectors, one for meniscus and one for OA, with independent
biases and attention for all twelve targets. Frozen protocol, membership and target
hashes were verified, and all prediction tables contain the same 904 unique
studies, correct folds, official target order and finite probabilities in [0, 1].
Recomputed selected-prediction AUC matches each run's best saved epoch metric.

| Metric | Reflection | Meniscus sharing (1A) | Both pairs (1B) | 1B − 1A |
|---|---:|---:|---:|---:|
| Full-fold macro AUC | 0.890976 | 0.891192 | 0.892022 | +0.000830 |
| Meniscus mean AUC | 0.906272 | 0.909343 | 0.909762 | +0.000419 |
| OA mean AUC | 0.886315 | 0.884359 | 0.885661 | +0.001302 |
| Report-derived AUC (893) | 0.890228 | 0.890516 | 0.891061 | +0.000545 |
| Gold AUC (11) | 0.922784 | 0.914683 | 0.941997 | +0.027315 |
| Soft-target BCE (lower is better) | 0.449310 | 0.448576 | 0.453585 | +0.005009 |

| Target | Reflection | 1A | 1B | 1B − 1A |
|---|---:|---:|---:|---:|
| ACL | 0.913949 | 0.913240 | 0.911609 | -0.001630 |
| MCL | 0.836247 | 0.830238 | 0.834096 | +0.003857 |
| Medial Meniscus | 0.922563 | 0.925300 | 0.931077 | +0.005777 |
| Lateral Meniscus | 0.889980 | 0.893386 | 0.888447 | -0.004938 |
| Medial OA | 0.903951 | 0.906129 | 0.905241 | -0.000888 |
| Lateral OA | 0.868678 | 0.862589 | 0.866081 | +0.003492 |
| PF OA | 0.874246 | 0.882357 | 0.872871 | -0.009486 |
| Effusion | 0.868662 | 0.863882 | 0.867596 | +0.003714 |
| Synovitis | 0.896302 | 0.897801 | 0.896192 | -0.001609 |
| Baker's | 0.908141 | 0.903323 | 0.900476 | -0.002846 |
| Contusion | 0.888732 | 0.895277 | 0.902622 | +0.007345 |
| Fracture | 0.920257 | 0.920786 | 0.927958 | +0.007172 |

Six targets improve and six decline against 1A. OA mean AUC improves by 0.001302:
lateral OA gains 0.003492 and medial OA loses 0.000888. Medial meniscus, contusion
and fracture also improve; PF OA and lateral meniscus show the largest losses.
BCE increases by 0.005009. Gold AUC improves by 0.027315, but its eleven studies
remain a diagnostic sample. OA mean AUC remains 0.000654 below reflection;
the combined model's advantage is established for full-fold macro AUC, not every
family or target separately.

The exclusive-positive subsets use frozen labels at >=0.5: 344 meniscus studies
(281 medial-only, 63 lateral-only) and 198 OA studies (143 medial-only, 55 lateral-only).
Each contains three gold studies. Negative counts for each target equal the other
target's positive count. Per-target AUCs compare studies; the margin AUC uses
`logit(p_medial) - logit(p_lateral)` to distinguish medial-only from lateral-only
studies, with probabilities clipped to [1e-7, 1-1e-7].

| Exclusive-positive diagnostic | Reflection | 1A | 1B |
|---|---:|---:|---:|
| Medial Meniscus subgroup AUC | 0.917189 | 0.907247 | 0.906682 |
| Lateral Meniscus subgroup AUC | 0.826809 | 0.815511 | 0.819974 |
| Meniscus margin AUC | 0.972321 | 0.972039 | 0.980173 |
| Meniscus correct within-study rankings | 322/344 | 322/344 | 325/344 |
| Medial OA subgroup AUC | 0.714431 | 0.701971 | 0.706548 |
| Lateral OA subgroup AUC | 0.577368 | 0.553465 | 0.542022 |
| Oa margin AUC | 0.856707 | 0.872982 | 0.870184 |
| Oa correct within-study rankings | 156/198 | 159/198 | 160/198 |

Meniscus margin AUC improves 0.972039 → 0.980173 against 1A, with correct medial-only
rankings 276/281 → 275/281 and lateral-only 46/63 → 50/63. OA margin AUC changes
0.872982 → 0.870184; correct medial-only rankings rise 131/143 → 138/143 while
lateral-only rankings fall 28/55 → 22/55. That imbalance is relevant to 1C even
though aggregate OA ranking accuracy rises from 159/198 to 160/198. These mostly
report-derived endpoints do not establish anatomical localization quality.

Training took **27m 47s**, versus 27m 55s for 1A and 29m 55s for reflection.
Selected-checkpoint prediction took **33.75s**, versus 32.29s and 33.43s.
Peak allocated CUDA memory was **18.624 GiB**, versus 18.631 and 18.969 GiB.
Recorded run costs do not isolate an architecture-specific runtime effect.
Candidate AUC peaks at epoch 7, falls to 0.886732 at epoch 8, then recovers to
0.891646 at epoch 10; selection uses the saved best checkpoint.

The [comparison artifact](../../../outputs/analysis/m/meniscus-oa-vs-controls-20260913/comparison.json)
preserves complete metrics, subgroup counts, within-study rankings, epoch curves,
configurations, input identities, timing and SHA-256 hashes of checkpoints,
predictions and source snapshots. One fold and seed do not quantify training
repeatability. Public AUC was unavailable for this initial comparison and is now
recorded in the separate public-result analysis above. No fresh training,
inference or tests were run for this comparison.
