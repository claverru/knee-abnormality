# 2026-09-13 max-pooling validation comparison

The completed `spatial-fusion-max` run selected epoch 10 at **0.87907846 macro
AUC**, below the average-pooling `spatial-fusion` reference by 0.00509042 and
`crop140mm` by 0.00704319. All runs use the same 3,503 training and 904 validation
studies, fold 0 of `all_5fold_v3`, Steven v4 labels and seed 42. Average and max
fusion have matching label, split, membership, training/validation row and
series-input identities. Their saved configurations differ in run name,
pooling and strict determinism; the historical model omitted the pooling field
and used timm's average-pooling default.

| Run | Selected epoch | Macro AUC | Soft-target BCE | Training seconds |
|---|---:|---:|---:|---:|---:|
| `crop140mm` | 8 | 0.88612165 | 0.45296129 | 2130.23 |
| `spatial-fusion`, average | 10 | 0.88416888 | 0.45306086 | 1855.21 |
| `spatial-fusion-max` | 10 | 0.87907846 | 0.45606172 | 1665.07 |

Max pooling improves four targets over average pooling: ACL, lateral OA,
effusion and contusion. Eight decline; the largest AUC drops are lateral
meniscus (-0.027538), MCL (-0.017717) and fracture (-0.011758).

Both fusion runs completed 4,380 optimizer steps. Max pooling's recorded
training duration is 10.25% shorter, with peak CUDA allocation of
20,031,506,944 bytes versus 20,077,255,168 bytes for average pooling. Max used
`trainer.deterministic: false`; average used `true`. These separate runs do
not isolate the accuracy or timing effect of pooling, determinism or external
runtime conditions. There is no measured matched nondeterministic average
control, repeated-seed estimate or max-pooling public score in this comparison.

This result gives no reason to promote max pooling for the fusion plan;
average pooling remains the stronger development reference. A parent/control
with matching execution settings is still needed for a controlled architecture
comparison. The active YAML remains set to max until the next pooling decision.

The [comparison record](../../../outputs/analysis/spatial-fusion-max-vs-average-20260913.json)
pins selected checkpoint, configuration and prediction hashes, source identities,
input-identity comparisons, timings and per-target deltas. The max checkpoint is
`outputs/training/spatial-fusion-max/fold_0/best-epoch=09-val_macro_auc=0.87908.ckpt` (deleted on 2026-09-16),
SHA-256 `2d4a934157ed9eb00688aa177813fe301bd104f45e090540bd82b295cd21983c`;
its run contains `source_snapshot.zip` for the executing source.
This review read saved JSON metrics and fingerprinted artifacts on CPU. It did
not recompute predictions, execute a model, run tests or launch a submission.
