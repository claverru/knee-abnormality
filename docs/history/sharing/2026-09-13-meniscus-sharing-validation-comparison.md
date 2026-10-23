# 2026-09-13 Meniscus-sharing validation comparison

The completed `meniscus-classifier` run scores **0.891192** versus **0.890976**
for `spatial-tokens-reflection`: **+0.000217 macro AUC**. Both select epoch 9
(index 8) from ten epochs. This is a mixed result: both meniscus targets improve
over the full fold, but both worsen on studies labeled positive in exactly one
meniscus. User decision for 1B: retain meniscus classifier sharing and use this
saved run as the control when adding OA classifier sharing. Reflection remains a
secondary reference. The small overall gain and mixed diagnostics do not establish
a repeatable benefit or localization failure; direct within-study separation is
broadly unchanged, as detailed below.

Saved effective configurations match exactly, including seed 42, Steven v4,
fold 0 of `all_5fold_v3`, reflection probability 0.5 and 2× validation density.
The candidate retains the config name `spatial-tokens-reflection`, but its run
ID is `meniscus-classifier` and its recorded architecture identifies sharing.
All data input identities match; preprocessing/model source identities differ.
Frozen protocol, membership and target hashes were verified. Both saved prediction
tables contain the same 904 unique studies, correct folds, official target order
and finite probabilities in [0, 1]. Metrics below were recomputed from those tables.

| Metric | Reflection | Meniscus sharing | Delta |
|---|---:|---:|---:|
| Full-fold macro AUC | 0.890976 | 0.891192 | +0.000217 |
| Meniscus mean AUC | 0.906272 | 0.909343 | +0.003071 |
| Report-derived AUC (893) | 0.890228 | 0.890516 | +0.000288 |
| Gold AUC (11) | 0.922784 | 0.914683 | -0.008102 |
| Soft-target BCE (lower is better) | 0.449310 | 0.448576 | -0.000734 |

| Target | Reflection | Meniscus sharing | Delta |
|---|---:|---:|---:|
| ACL | 0.913949 | 0.913240 | -0.000710 |
| MCL | 0.836247 | 0.830238 | -0.006009 |
| Medial Meniscus | 0.922563 | 0.925300 | +0.002737 |
| Lateral Meniscus | 0.889980 | 0.893386 | +0.003406 |
| Medial OA | 0.903951 | 0.906129 | +0.002178 |
| Lateral OA | 0.868678 | 0.862589 | -0.006090 |
| PF OA | 0.874246 | 0.882357 | +0.008111 |
| Effusion | 0.868662 | 0.863882 | -0.004779 |
| Synovitis | 0.896302 | 0.897801 | +0.001499 |
| Baker's | 0.908141 | 0.903323 | -0.004818 |
| Contusion | 0.888732 | 0.895277 | +0.006545 |
| Fracture | 0.920257 | 0.920786 | +0.000528 |

Seven targets improve and five decline. The largest gains are PF OA and
contusion; the largest losses are lateral OA and MCL. Gold AUC declines by
0.008102, but eleven studies provide only a small diagnostic sample.

For the 344 studies positive in exactly one meniscus at the frozen >=0.5 cutoff,
281 are medial-positive/lateral-negative and 63 the reverse. Only three are gold.
Medial AUC falls **0.917189 → 0.907247** (-0.009942); lateral falls
**0.826809 → 0.815511** (-0.011298). Their mean falls **0.871999 → 0.861379**
(-0.010620). These per-target AUCs compare different studies and do not directly
measure which meniscus receives the higher score within a study.

A subsequent within-study check gives the affected meniscus the higher probability
in **322/344 studies (93.60%) for both models**. The AUC of
`logit(p_medial) - logit(p_lateral)` for medial-only versus lateral-only is
**0.972321 → 0.972039**, essentially unchanged. Correct medial-only rankings rise
273/281 → 276/281 while lateral-only rankings fall 49/63 → 46/63; the aggregate
accuracy is dominated by the medial-only majority. Thus the subgroup regression
does not establish attention collapse or broadly worse compartment separation.
Mostly report-derived labels cannot establish actual anatomical localization.
For completeness, the corresponding OA subgroup has 198 studies (143 medial-only,
55 lateral-only, three gold); mean AUC falls 0.645900 → 0.627718. OA weights
were not shared in this experiment.

Training took **27m 55s**, versus 29m 55s for reflection (-6.7%). Selected-checkpoint
validation prediction took 32.29s versus 33.43s; peak allocated CUDA memory was
18.631 versus 18.969 GiB. These are recorded run costs, not a controlled runtime
benchmark. Both completed 4,380 optimizer steps. Candidate macro AUC peaks at
epoch 9 then declines slightly to 0.890950 at epoch 10.

One fold and one seed do not establish a repeatable gain or loss; no uncertainty
interval or new public score was computed. The
comparison artifact (`outputs/analysis/m/meniscus-vs-reflection-20260913/comparison.json`, deleted)
preserves all metrics, subgroup counts, epoch curves, configuration and input
identities, timing, and hashes of predictions, checkpoints and source snapshots.
No fresh training, inference or tests were run.
