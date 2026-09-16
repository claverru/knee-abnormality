# 2026-09-12 spatial-fusion validation comparison

The user-trained eager 27-query, width-64 model did not improve the selected
`crop140mm` checkpoint on fold 0. Saved predictions cover the same 904 studies;
label, split, membership, training/validation row and series-input hashes match.
Both runs use seed 42 and the same saved data, optimizer and trainer settings.
The selected epochs are 8 for `crop140mm` and 10 for `spatial-fusion`.

| Saved predictions | Macro AUC | Soft-target BCE |
|---|---:|---:|
| `crop140mm` | 0.886122 | 0.452961 |
| `spatial-fusion` | 0.884169 | 0.453061 |
| Equal probability average | 0.888049 | 0.449818 |

Fusion improves ACL, medial meniscus and synovitis individually; nine targets
decline, with the largest drops in fracture and lateral meniscus. A fixed
50/50 average improves 11 of 12 targets over `crop140mm`, with fracture the
exception. This supports complementary predictions on this fold, without
establishing that spatial geometry causes the benefit or that it generalizes.
No blend weights were tuned and no uncertainty estimate was computed.

The [comparison record](../../../outputs/analysis/spatial-fusion-vs-crop140mm-20260912.json)
pins prediction/label hashes, alignment, checkpoint epochs and metrics.
This analysis only read saved CSVs with Polars/NumPy/scikit-learn on CPU;
both individual metrics reproduced their saved results within `1e-6`.
No model execution, tests or ensemble submission were performed.
