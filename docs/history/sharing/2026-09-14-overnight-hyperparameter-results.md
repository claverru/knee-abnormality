# 2026-09-14 overnight hyperparameter results

All eleven runs in
[`overnight-20260913T214822Z-74udtekr`](../../../outputs/training/overnight-20260913T214822Z-74udtekr)
completed ten epochs successfully. Recorded setup, training and selected
validation prediction total 5.536 hours. Every selected checkpoint is epoch 10
(index 9), also the maximum saved epoch AUC for its run.

The executed architecture was **1C**, rather than the originally proposed
reflection architecture. All run preprocessing source hashes match; archived
package Python files match `disease-compartment-attention`. Saved label,
membership, training/validation row, split protocol and series identities match
that control. Effective configs differ only by the intended settings, names and
output locations. The 1C control scores 0.889430 AUC and 0.448458 BCE; reflection
scores 0.890976 AUC and remains the 0.931 public benchmark.

| Run | Full-fold AUC | Delta vs 1C | BCE | Training minutes |
|---|---:|---:|---:|---:|
| 02 central sampling, sigma 40 | 0.892538 | +0.003109 | 0.448244 | 29.0 |
| 06 higher final LR | 0.891216 | +0.001787 | 0.449815 | 29.0 |
| 01 higher peak LRs | 0.891146 | +0.001716 | 0.447721 | 27.9 |
| 09 shared normalization | 0.890768 | +0.001338 | 0.447876 | 29.9 |
| 03 crop 128 mm | 0.889792 | +0.000362 | 0.449161 | 28.9 |
| 05 attention dropout | 0.889782 | +0.000352 | 0.449098 | 29.3 |
| 10 stronger intensity | 0.889678 | +0.000249 | 0.450104 | 31.5 |
| 11 no noise | 0.889535 | +0.000105 | 0.448399 | 30.5 |
| 04 broad sampling, sigma 120 | 0.889060 | -0.000370 | 0.448715 | 29.2 |
| 08 longer warmup | 0.888393 | -0.001036 | 0.450408 | 30.6 |
| 07 effective batch 16 | 0.887154 | -0.002276 | 0.451711 | 30.3 |

The queue used these fixed overrides against its sigma-80 1C baseline.
The [archived launcher](../sharing-sources/train_overnight.py.txt) and each
run's effective config preserve the full settings.

| Run | Changes from baseline |
|---|---|
| `01-higher-lr` | Backbone LR `3e-4`, head LR `9e-4` |
| `02-central-sampling` | Sampling sigma 40 mm |
| `03-crop128mm` | Spacing 0.57142857 mm, retaining 224 pixels |
| `04-broad-sampling` | Sampling sigma 120 mm |
| `05-attention-dropout` | Model dropout 0.1 |
| `06-higher-final-lr` | Minimum LR ratio 0.1 |
| `07-effective-batch16` | Accumulation 2, retaining physical batch 8 and baseline LRs |
| `08-longer-warmup` | Warmup 2 epochs |
| `09-shared-normalization` | Mean `[0.449, 0.449, 0.449]`, std `[0.226, 0.226, 0.226]` |
| `10-stronger-intensity` | Gamma `[0.7, 1.3]`, contrast `[0.8, 1.2]`, retaining probability 0.5 |
| `11-no-noise` | Noise probability 0 |

Sigma 40 improves seven targets versus 1C, led by fracture (+0.019874) and
MCL (+0.015536), with lateral meniscus down 0.006575. Its eleven-study gold
AUC is 0.917229. Higher peak LRs improve eight targets and have the lowest
BCE. Shared normalization improves both aggregate metrics versus 1C. Higher
final LR worsens BCE and jumps from 0.885802 AUC at epoch 9 to 0.891216 at
epoch 10. Effective batch 16 uses 2,190 optimizer steps versus 4,380 otherwise.

Evidence is each run's selected-checkpoint record, saved epoch metrics,
effective config, source snapshot, input identity and timing JSON. No new
training, inference or tests ran for this summary. These single-seed comparisons
on 904 studies do not establish repeatability or transfer to reflection/public
performance. The user requested the
[sigma-40 submission](../../inference/submissions/2026-09-14-sigma-40-overnight-submission.md)
on 2026-09-14; it completed with 0.931 public AUC, tying reflection.
