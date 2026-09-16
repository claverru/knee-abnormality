# Overnight 00-baseline submission

The `00-baseline` run of queue `outputs/training/overnight-20260914T220805Z-ozsz4qbu` is the
champion recipe on the current code: token graph off (`graph_layers: 0`) and no
additive 3D embedding. It selected epoch 10 (index 9) at **0.892763 fold-0 macro
AUC**, 0.000403 above reflection/sigma 40. It was exported on 2026-09-15 to
`outputs/submission/exports/knee-overnight-00-baseline-20260915` with the ported
[dual-GPU runtime](../independent-gpu-inference-processes.md): 2× images, four studies per
GPU and four total loader workers. Its checkpoint SHA-256 is
`da45267edb63846dd35272f17fd36a4a3dd42dbac522219ef7a39f706db48e89`.

Competition submission **56249358 completed with public AUC 0.934**, **−0.002** below
the reflection/sigma-40 champion's 0.936, despite the higher fold-0 score. The
dataset is `claverru/knee-overnight-00-baseline-20260915` and the notebook
`claverru/knee-overnight-00-baseline-20260915-inference`. The
[workflow record](../../../outputs/submission/records/knee-overnight-00-baseline-20260915.json)
holds the transitions. The notebook ran for 2m 01s and scoring took 16m 07s after
submission. `submission.runtime.json` confirms two Tesla T4 processes in FP32 with
batch 4 and two loader workers each; the visible three studies were partitioned 2+1.

The run differs from the champion in two ways: the 3D embedding is removed, and the
unused graph is still built each forward. Its seed, fold and recipe are otherwise
identical. One fold-0 run and one public submission each cannot separate that change
from training-run variability, but the public result does not reproduce 0.936.
