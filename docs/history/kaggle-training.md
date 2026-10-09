# Kaggle training

**Closed 2026-09-28, not worth it (user decision): a full training run on Kaggle took over three
hours, about six times the local time, for the same result.** Tried from 2026-09-24 on branch `kaggle-training`
to reproduce the `configs/training.yaml` fold-0 run of the time (`knee-geometry-merged-2`, ×4
0.900955) on a Kaggle 2×T4 notebook without the local GPU.

## Setup

A `scripts/pipeline/kaggle_train.py` launcher packaged the code, split, series image counts and
label table into the private dataset `claverru/knee-kaggle-training` and pushed the notebook
`claverru/knee-train-kaggle`. Only hardware settings changed from the local config: two T4s with
DDP, batch 4 per device (same effective batch of 8), `16-mixed` precision (T4s have no BF16),
preparation on the GPU (four vCPUs) and 4 loader workers per rank.

## What was measured

- **Smoke run** (one epoch on 32 studies): passed in 1,258 s of notebook time; peak 8.1 GB per T4;
  ×1 validation at 2.8 studies/s, final ×4 prediction at 1.2 studies/s.
- **Full fold 0:** the first two attempts were killed by host memory, the second after 58 min near
  the end of epoch 0 with used host memory rising from 6.1 to 27.9 GB. Disabling cuDNN, using
  non-persistent workers and a precomputed series index got past it.
- **Completed run `kaggle-fold0-20260924d`** (notebook version 5), against the local
  `knee-geometry-merged-2` on the same fold and step count (4,380 optimizer steps):

  | | Kaggle 2×T4, FP16, DDP | Local 3090, bf16 |
  |---|---:|---:|
  | Training | 9,710 s | 1,606 s |
  | ×4 validation prediction | 591 s | 50 s |
  | Peak GPU memory | 11.3 GB per T4 | 21.1 GB |
  | ×4 macro AUC | 0.902557 | 0.900955 |

  The setup reproduces the local result; the cost is about six times the training time.
- **Speed:** about 3.6 training studies/s in the completed run (1.2 before the fixes) against about
  22 locally. Loading does not explain it (0.145 s per cold study on one local core, 0.083 s with a
  precomputed series index). Before the fixes, FP16 cuDNN setup for every new backbone batch shape
  (about 3.4 s per shape,
  [measured for inference](../inference/2026-09-15-kaggle-inference-profiling.md#why-fp16-is-slower))
  was the likely cost, since MIL batches change shape almost every step; the completed run disabled
  cuDNN, so its speed is that of PyTorch's native convolution kernels on two T4s.

## Recovering the code

Never committed. The completed run's output, including its `source_snapshot.zip` (the exact code
that trained on Kaggle), checkpoint, predictions and logs, is in
`outputs/kaggle-training/kaggle-fold0-20260924d-output/`; each attempt's launch record is in
`outputs/kaggle-training/`. The worktree's changes against `70a9a24` are archived in
`outputs/analysis/k/kaggle-training-20260924/`: `worktree.patch` (config, dataset, series index,
checkpoints, Lightning module, train script) and `untracked-files.tar.gz` (the launcher, a
series-index builder and the original status document). The Kaggle datasets
`claverru/knee-kaggle-training` and `claverru/knee-series-index` and the notebook
`claverru/knee-train-kaggle` remain on Kaggle.
