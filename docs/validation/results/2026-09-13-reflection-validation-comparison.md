# 2026-09-13 Reflection validation comparison

The selected `spatial-tokens-reflection` checkpoint scores **0.890976** versus
**0.889964** for `3dsampling`, the previous public champion at **0.930**:
**+0.001012 full-fold macro AUC**. Both runs selected epoch 9 (index 8).
Reflection submission **56211972** completed with **0.931 public AUC**, confirmed
through the Kaggle API on 2026-09-13. This **+0.001** improvement makes reflection
the best recorded public result at that time; see the
[submission record](../../inference/submissions/2026-09-13-reflection-submission.md). The later
[sigma-40 run](2026-09-14-reflection-sigma-40-result.md) reached 0.936 publicly.

Both runs use seed 42, Steven v4, fold 0 of `all_5fold_v3`, ten training epochs
and 2× validation images. Training and validation row hashes, split protocol,
membership, label source and series metadata/count hashes match, as do all 904
prediction study IDs and fold assignments. Saved configurations differ only in
the experiment name and the added reflection probability of 0.5. The source
identity also separates 3D availability from visual validity; added reflection
draws change subsequent stochastic choices. This is a comparison of the saved
implementations, not a paired-randomness isolation of reflection alone.

| Metric | 3dsampling | Reflection | Delta |
|---|---:|---:|---:|
| Full validation AUC (904) | 0.889964 | 0.890976 | +0.001012 |
| Report-derived AUC (893) | 0.889051 | 0.890228 | +0.001177 |
| Gold AUC (11) | 0.936673 | 0.922784 | -0.013889 |
| Soft-target BCE, lower is better | 0.452571 | 0.449310 | -0.003261 |
| Public AUC | 0.930 | 0.931 | +0.001 |
| Training time | 27m 41s | 29m 55s | +2m 14s (+8.1%) |
| Setup, training and selected validation prediction | 28m 13s | 30m 29s | +2m 16s |
| Peak CUDA allocated memory | 18.965 GiB | 18.969 GiB | +3.975 MiB |

| Target | 3dsampling | Reflection | Delta |
|---|---:|---:|---:|
| ACL | 0.918893 | 0.913949 | -0.004944 |
| MCL | 0.838486 | 0.836247 | -0.002239 |
| Medial Meniscus | 0.919616 | 0.922563 | +0.002947 |
| Lateral Meniscus | 0.896857 | 0.889980 | -0.006877 |
| Medial OA | 0.898729 | 0.903951 | +0.005223 |
| Lateral OA | 0.869179 | 0.868678 | -0.000501 |
| PF OA | 0.867193 | 0.874246 | +0.007053 |
| Effusion | 0.864345 | 0.868662 | +0.004317 |
| Synovitis | 0.890164 | 0.896302 | +0.006139 |
| Baker's | 0.911122 | 0.908141 | -0.002981 |
| Contusion | 0.887201 | 0.888732 | +0.001532 |
| Fracture | 0.917785 | 0.920257 | +0.002472 |

Seven targets improve and five decline. PF OA, synovitis and medial OA have the
largest gains; lateral meniscus and ACL have the largest losses. The gold decline
comes from Baker's (-0.166667) and medial meniscus (-0.033333), partly offset by
ACL (+0.033333); other gold targets are unchanged. Eleven gold studies, one fold
and one training seed cannot establish a repeatable gain or predict the public score.

The [comparison artifact](../../../outputs/analysis/r/reflection-vs-3dsampling-20260913/comparison.json)
records complete metrics, deltas, matched input identities, preprocessing identities,
timing and SHA-256 hashes for checkpoints, configurations, metrics, predictions,
provenance and source snapshots. No fresh training, local inference or tests were
run for this comparison.
