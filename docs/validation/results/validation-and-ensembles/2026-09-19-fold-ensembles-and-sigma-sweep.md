# 2026-09-19 fold ensembles and the sigma sweep

**The first cross-validated numbers in the project, and the first ensemble that transfers.** Sixteen
runs on queue `overnight-20260918T232636Z-mt7ouybk`, all exit 0, 7.1 h: five ReXNet-200 folds, five
EfficientNet-B0 folds, a four-point `sampling_sigma_fraction` sweep and two config arms.

- **The ReXNet + B0 ensemble is worth +0.0061 OOF and +0.006 public**, positive on all twelve labels.
- **The sigma sweep is unreadable** — non-monotone with the control at the bottom, spread comparable
  to seed noise. Sigma is settled.
- **b0's two known gains do not combine**: 15 epochs with both learning rates doubled is −0.0033
  against 15 epochs alone.

## Setup

Common: split `all_5fold_v3`, ×1 validation, `sampling_sigma_fraction 0.867`,
`study_triplet_total 26`, batch 8, one recorded random seed per run. ReXNet-200 at
`backbone_lr 2e-4` / `head_lr 6e-4`, 10 epochs; EfficientNet-B0 at `backbone_lr 4e-4` /
`head_lr 6e-4`, 15 epochs, patience 15 — the best b0 configuration on record
([b0 queues](../campaigns-and-queues/2026-09-18-b0-hyperparameter-queues.md)). Wall time 26.2 min per ReXNet run and
25.7 min per b0 run, so **b0 at 15 epochs costs the same as ReXNet at 10**; its "third of the
training time" advantage exists only at matched epochs.

## Per fold

| Fold | ReXNet-200 | B0 |
|---|---:|---:|
| 0 | 0.88367 | 0.88635 |
| 1 | 0.87297 | 0.86964 |
| 2 | 0.87892 | 0.87845 |
| 3 | 0.87999 | 0.87523 |
| 4 | 0.87055 | 0.87544 |

**Fold difficulty spans 0.013**, five times the effects this project measures on fold 0 alone. Fold 0
is the easiest for ReXNet and fold 4 the hardest.

## Out of fold, 4,407 studies

| | pooled | mean of per-fold AUCs |
|---|---:|---:|
| ReXNet-200 | 0.87654 | 0.87722 |
| EfficientNet-B0 | 0.87485 | 0.87702 |
| **Mean ensemble** | **0.88269** | — |
| Rank ensemble | 0.88244 | — |

- **Use the mean, not ranks.** Ranks must be taken within fold; pooling ranks across folds mixes
  five differently calibrated models and scored 0.88181 before that was corrected to 0.88244.
- **The two backbones tie on per-fold AUC** (0.87702 against 0.87722) and separate only when pooled.
  The pooled gap measures B0's larger calibration drift between folds, not its accuracy.
- **The ensemble gain is +0.0061 and positive on every label**: MCL +0.0086, Baker's +0.0087,
  lateral meniscus +0.0078, fracture +0.0077, contusion +0.0081, medial OA +0.0067, down to
  lateral OA +0.0029.
- **It is a two-model blend per study, not ten.** In OOF each study is predicted once, by the one
  fold model that held it out. A submission averages all ten over every study, so +0.0061 is a lower
  bound on the deployed ensemble; the extra gain from seed averaging is unmeasured.

## The sigma sweep

Fold 0, ReXNet, every setting identical except the fraction. `rexnet-fold0` supplies the 0.867
centre point.

| Fraction | Axial | Coronal | Sagittal | Fold-0 best |
|---:|---:|---:|---:|---:|
| 0.60 | 31.2 | 24.8 | 27.0 | 0.89009 |
| 0.76 | 39.5 | 31.4 | 34.2 | 0.88612 |
| **0.867** | **45.1** | **35.8** | **39.0** | **0.88367** |
| 0.95 | 49.4 | 39.2 | 42.8 | 0.89041 |
| 1.10 | 57.2 | 45.4 | 49.5 | 0.88659 |

**Non-monotone, with the control at the bottom and a 0.0067 range.** That is not a response curve.
The same queue supplies the reason: `rexnet-fold0` (0.88367) and the previous day's
`sigma-fraction-0867` (0.88649) are the *same configuration at different seeds*, 0.0028 apart, and
fold-0 ReXNet at this recipe now spans 0.88367–0.88848 across four seeds. The sweep's spread is seed
noise.

**Sigma is settled.** No fraction between 0.60 and 1.10 is distinguishable, which closes the arms
left open by the [relative width result](../sampling/2026-09-19-relative-sampler-width-result.md).

## The two config arms

- **`rexnet-epochs-15`: 0.88618** against 0.88367, +0.0025. Same sign as the +0.0023 the older
  recipe showed, but inside seed noise. Still unresolved, and ReXNet has still never been compared
  at 15 epochs against a same-seed control.
- **`b0-epochs15-lr2x`: 0.88305** against `b0-fold0`'s 0.88635, **−0.0033**. Combining b0's two
  measured gains (15 epochs, both learning rates doubled) is worse than 15 epochs alone. That
  question is now answered negatively and needs no further run.

## Limits

- One seed per cell. The per-fold and sweep numbers carry the 0.0037 seed spread measured on fold 0.
- Every model here trains with unbalanced BCE, which
  [class-balanced BCE beats by +0.0116 on fold 0](../labels-and-losses/2026-09-19-balanced-bce-result.md). The ensemble
  is built on a recipe that was superseded the same day.
- The OOF pooling assumes comparable calibration across the five fold models; the per-fold column
  above shows it costs 0.0007 for ReXNet and 0.0022 for B0.
