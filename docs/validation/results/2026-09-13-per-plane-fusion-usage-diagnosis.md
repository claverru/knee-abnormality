# 2026-09-13 per-plane fusion usage diagnosis

The trained `spatial-fusion-radius` checkpoint uses its regional branch, but
final MIL gives it only **4.94% of attention on average**. Omitting regional
tokens slightly reduces validation AUC and worsens BCE. This explains how
fusion can be active while the aggregate curves closely resemble `crop140mm`:
the global descriptors carry most of the final attention, and the regional
branch makes comparatively small adjustments.

The run selected epoch 8 with 0.88635162 AUC and 0.45363703 BCE on the same
904 fold-0 studies as `crop140mm`. The control selected epoch 8 with 0.88612165
AUC and 0.45296129 BCE. The AUC difference is only +0.00022998 and does not
establish a repeatable improvement. Both runs use seed 42, the same pretrained
backbone initialization, saved data and optimizer settings, and matching
label/split/row/series-input hashes. Strict determinism is disabled for the
per-plane run and enabled for the control.

The diagnosis evaluates the complete saved validation fold with the selected
checkpoint, 2× validation images, batch eight, BF16 and eight loader workers.
One unchanged CNN/fusion forward supplies both variants. The intervention
excludes regional tokens from final MIL and renormalizes attention over the
same global tokens. It does not zero features while leaving their attention
weights in the denominator. Original source identities are checked before
loading; the temporary pooling wrapper lives only in the diagnostic script.

| Same trained checkpoint | Macro AUC | Soft-target BCE |
|---|---:|---:|
| Intact fusion | 0.88635162 | 0.45363703 |
| Regional tokens omitted | 0.88565495 | 0.45451753 |
| Intact minus omitted | +0.00069668 | -0.00088050 |

Intact predictions reproduce the saved CSV to a maximum absolute probability
difference of `2.98e-8`, with identical AUC. Removing regional tokens changes
probabilities by 0.006576 on average, or **0.66 percentage points**; the median
change is 0.29 points and the largest is 11.69 points. These effects use the
same checkpoint and image features, without retraining.

Coverage is usually available: **872 of 904 studies retain all 27 regional
queries**, one retains 25, and 31 have no accepted center and no valid queries.
There are 873 accepted centers and no preparation conditions. Thus the
coverage gate is not broadly disabling the branch on validation. Regional
tokens would receive 36.58% of attention on average under uniform weighting of
valid tokens; learned attention assigns 4.94%, or 5.11% among eligible studies.
The median study/target regional share is 3.68%, with p90 10.33% and maximum
40.25%. The low average reflects final MIL weighting rather than a widespread
lack of eligible regional tokens.

Fusion improves AUC for eight targets and lowers it slightly for four within
this checkpoint. The largest gains are lateral OA (+0.002115), synovitis
(+0.001719) and medial OA (+0.001337). The largest reductions are effusion
(-0.000342) and ACL (-0.000309). Mean regional attention ranges from 1.53% for
medial meniscus to 10.17% for medial OA; synovitis receives 9.65%.
The [per-target plot](../../../outputs/analysis/fusion-usage-20260913/fusion_usage.png)
shows both attention and the omission effect.

These are diagnostics of one jointly trained checkpoint. Attention mass is
not a causal attribution score, and inference-time omission does not recreate
a model trained without fusion or isolate any training influence on the shared
backbone. No repeated-seed estimate or uncertainty interval was computed.
The evidence supports a functioning, lightly weighted regional branch with a
small observed contribution on this fold; it does not establish an advantage
over the independently trained `crop140mm` control or a public-score gain.

The [result record](../../../outputs/analysis/fusion-usage-20260913/results.json)
pins source/config/checkpoint and input identities, dependency/runtime versions,
saved-prediction reproduction, per-target metrics and summary distributions.
Sibling CSVs preserve intact, regional-omitted and original saved predictions;
`per_study.parquet` and `query_attention.npz` retain study/target attention,
validity and query offsets. The selected checkpoint SHA-256 is
`0481ee1228cb687d32f9a703671d9799089435d3b7220c1a65a28e14a2a43c14`;
the executing preprocessing source identity is
`b0f7b06268098dede3866aac113e15fd0544bfbee802684f78c4f70c8ba5762b`.

Reproduce using the matching source snapshot and a fresh output directory:

```bash
rtk proxy env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.diagnose_fusion_usage outputs/training/spatial-fusion-radius --output-dir outputs/analysis/fusion-usage-new --workers 8
```

This was an inference diagnosis, with no training, production model changes,
tests or submission. The 36.09-second diagnostic duration includes extra
pooling, analysis and artifact writing and is not an inference-speed comparison.
