# 2026-09-14 sigma-40 higher-LR pair result

`outputs/training/sigma40-higher-lr` completed ten epochs in 1,636.70 seconds (27.3
minutes), selecting epoch 10 (index 9). The higher-LR pair improves on the
three-way combination but does not beat sigma 40 alone. Retain sigma 40's
backbone/head LRs 0.0002/0.0006 on current evidence.

| Recipe | Full-fold AUC | BCE | Gold AUC, 11 studies |
|---|---:|---:|---:|
| Sigma 40, original LRs and normalization | 0.892538 | 0.448244 | 0.917229 |
| Sigma 40 + higher LRs | 0.891760 | 0.448679 | 0.925661 |
| Sigma 40 + higher LRs + shared normalization | 0.891035 | 0.448816 | 0.927216 |

Against sigma 40 alone, AUC falls 0.000779 and BCE worsens 0.000435. Seven
targets improve and five decline. Largest gains are ACL (+0.002997), synovitis
(+0.001962) and medial meniscus (+0.001853); largest losses are effusion
(-0.009298), fracture (-0.003680) and lateral OA (-0.003254). Pair-minus-triple
AUC is +0.000724. These single-seed comparisons do not establish that shared
normalization universally hurts or that higher LR always underperforms.

Effective configs differ from sigma 40 only in backbone/head LRs
0.0003/0.0009, name and paths. Complete input identities match; model, dataset
and training source match. Package differences are confined to inference code
and the added inference-process module. The run retains 1C, seed 42, Steven v4,
fold 0, 904 validation studies, 224 pixels, 140 mm, reflection 0.5, 2× validation
images and 4,380 optimizer steps. Peak CUDA allocation is 20,681,703,936 bytes,
identical to sigma 40 and the triple. Evidence is each run's effective config,
source snapshot, input identity, selected-checkpoint record, epoch metrics and
timing JSON. The pair's checkpoint is
`fold_0/best-epoch=09-val_macro_auc=0.89176.ckpt`. No new training, inference,
tests or submission ran for this comparison.
