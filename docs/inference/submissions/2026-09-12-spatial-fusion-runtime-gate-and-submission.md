# Spatial-fusion runtime gate and submission

## Spatial-fusion runtime gate

The regional runtime results below are historical. Current sources implement
[spatial-token MIL](../../modeling/current-model.md); its inference cost
has not been measured. Resume, prediction and export fingerprint the new token
module and fixed geometry/encoding/dropout semantics. Old checkpoints must use
their original bundles; no architecture flags or weight migration are provided.
The model uses all reliable 3D at evaluation, with no artificial geometry dropout.
Measure matched inference cost after the user's training run before deployment.

The [spatial-fusion engineering comparison](../../validation/results/2026-09-12-spatial-fusion-engineering-result.md)
retained the submitted control: even the reduced, compiled nine-query/width-32
candidate added 7.47% warm end-to-end inference time at batch two, exceeding
the 5% gate. Its fixed-input model overhead was 4.44%; inference peak allocation
fell 1.45% after redundant image indexing was removed. Neither the larger batch-eight
profile nor memory savings override the deployment-workload failure.

The initial 27-query/width-64 implementation added roughly 16% model time at
batch two. Its trace attributed about 2.0 ms to mapping, 2.4 ms to geometry-bias
construction, 0.9 ms to projection and only 0.3 ms to cross-attention. PyTorch
2.10 selected memory-efficient scaled-dot-product attention with the actual
BF16/additive-bias inputs. Compiling and reducing capacity helped but did not
bring the complete workload within budget; sparsity was not justified by
attention-kernel cost.

Profiles retain every first-visit study time, the first loader pass, warm
batch/pass times and operator traces under
`outputs/analysis/spatial-fusion-20260912/`. First visits are process-cache cold;
the filesystem cache was not flushed. Compiled variants also retain warmup
costs, which include compilation when not already cached. These are bounded
local profiles, not hidden Kaggle runtimes. The final control executes the
submitted bundle itself; preliminary batch-eight comparisons against its
training snapshot include that older reader's overhead.

## Spatial-fusion submission

The user-trained `spatial-fusion` fold-0 run selected epoch 10 (checkpoint index
9), with validation macro AUC 0.88416888 on 904 studies. The
checkpoint `outputs/training/spatial-fusion/fold_0/best-epoch=09-val_macro_auc=0.88417.ckpt` (deleted on 2026-09-16)
has SHA-256 `e9e0154ed75ee2ada0ae8e47c185661d1cd99b90e37c80566bbe7b9be8eaee18`.
Its [bundle](../../../outputs/submission/exports/knee-spatial-fusion-20260912) retains 27 queries,
attention width 64 and eager execution, with 2× images and inference batch two.
Checkpoint metadata, strict CPU model loading and bundle integrity passed;
optional local inference was not requested.

Submission `56190442` completed on 2026-09-12 with public score **0.920**,
confirmed through the Kaggle API. It uses version 1 of private dataset
`claverru/knee-spatial-fusion-20260912` and notebook
`claverru/knee-spatial-fusion-20260912-inference`. The
[workflow record](../../../outputs/submission/records/knee-spatial-fusion-20260912.json)
pins the bundle, dataset/notebook versions, verified output and scored result.
This is 0.005 below `crop140mm` at 0.925, consistent with the lower standalone
validation AUC. The saved-prediction blend described in
[validation](../../validation/results/2026-09-12-spatial-fusion-validation-comparison.md)
has not been submitted. The later
[reflection sigma-40 submission](2026-09-14-reflection-sigma-40-submission.md) holds the
best recorded public score, 0.936.
