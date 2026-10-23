# 2026-09-14 original 1C public result

The original sigma-80 1C checkpoint scored **0.928 public AUC**, improving on
1B by **0.003** despite its lower local AUC. Reflection and sigma-40 1C both
score **0.931**, ahead of original 1C by **0.003**.

| Model | Full-fold AUC | Public AUC | Inference studies per batch |
|---|---:|---:|---:|
| Reflection | 0.890976 | 0.931 | 2 |
| 1B | 0.892022 | 0.925 | 2 |
| Original 1C, sigma 80 | 0.889430 | 0.928 | 4 |
| 1C, sigma 40 | 0.892538 | 0.931 | 2 |

All four submissions use their selected fold-0 checkpoint and 2× images.
The original 1C batch-8 attempt failed without a score; its unchanged-model
batch-4 retry completed as submission `56225269`, script version `349717410`.
See [submission evidence](../../inference/submissions/early-submission-details.md)
for checkpoint identity, failure details and saved API responses. These results
used the previous single-GPU inference code.

The compartment architecture therefore outperformed 1B in this public
comparison; its lower local score did not predict that ranking. Sigma 40 also
improved both local and public scores over original 1C. These trained-model
comparisons cannot isolate the embedding's causal effect or establish
repeatability. Hidden per-target predictions remain unavailable. The
[local intervention findings](2026-09-13-compartment-attention-diagnosis.md)
still describe how the original 1C checkpoint uses its learned factor.
