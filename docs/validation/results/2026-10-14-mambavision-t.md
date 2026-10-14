# 2026-10-14 MambaVision-T

MambaVision-T on fold 2 of `all_5fold_scanner_v2`, seed 1110484336, against the ResNet-50 of the
[targets-and-inputs campaign](2026-10-13-targets-and-inputs-campaign.md) (run 27: ×4 0.939634,
×1 0.930902). Code and recipe: [MambaVision-T](../../modeling/mambavision-backbone.md). Per-run
detail and per-label tables: the campaign's
[run notes](../../../autoresearch/campaigns/mambavision-settings/run-notes.md).

## Result

The full recipe (`configs/mambavision.yaml`, run `outputs/training/mambavision-t-combined-12-epochs`,
best epoch 11) scores ×4 0.937801 and ×1 0.929880: −0.0018 at ×4 against the ResNet-50 alone. The
equal logit blend of the two scores 0.941419, +0.0018 over the ResNet-50 alone. It has no public
score.

Blends on fold 2, ×4, equal logit average. The ConvNeXt-Tiny and second ResNet-50 are the fold-2
members of the joint Kaggle + fastMRI five-fold runs, so they trained on more data:

| Added to the ResNet-50 (0.939634) | Blend | Gain |
|---|---:|---:|
| ConvNeXt-Tiny, joint (0.943274 alone) | 0.942906 | +0.0033 |
| MambaVision-T | 0.941419 | +0.0018 |
| ResNet-50, joint (0.939919 alone) | 0.940584 | +0.0010 |

Added to the joint ConvNeXt, MambaVision-T moves it by −0.0003, and the four models together score
0.943142.

## First run

`mambavision-t-rn50-recipe`, 2026-10-13: the ResNet-50 recipe unchanged with the ImageNet-1k
weights and ImageNet mean and std. ×4 0.922130 (−0.017504), ×1 0.915009; blend 0.936435
(−0.003199). Training 2,594 s, 12.7 GB. It underfit: training loss above the ResNet-50's at every
epoch (0.593 against 0.571 at the end). The cause was the weight scale: pretrained weights 1.5 to
12.7 times larger, so the rate was too small and the anchors too strong for them.

## Settings campaign

[`mambavision-settings`](../../../autoresearch/campaigns/mambavision-settings/INDEX.md), Codex
provider, 2026-10-13 23:05 to 2026-10-14 08:58 CEST, then five seeds on the full recipe until the
user's forced stop at 18:40. Adoption needs the macro AUC strictly above the incumbent's at ×1 and ×4.

| Run | ×4 | ×1 | Decision |
|---|---:|---:|---|
| Stage rates by weight scale | 0.919368 | | reject |
| Backbone rate 2e-4 | 0.924571 | 0.919012 | adopted |
| Anchors by weight scale (0.001, 1e-5) | 0.934657 | 0.928335 | adopted (+0.010) |
| Backbone rate 4e-4 | 0.935009 | | reject |
| Twelve epochs | 0.936187 | 0.930634 | adopted |
| Stage rates on the freed anchors | 0.935863 | | reject |
| Feature-statistics augmentation off | 0.935080 | | reject |
| Gradient clipping at 10 | 0.935853 | | reject |
| Late anchor follows the rate | 0.937634 | 0.930689 | adopted by the user |
| Twelve epochs, linear decay | 0.937208 | 0.931826 | adopted |
| Feature statistics after the first epoch | 0.936888 | | reject |
| Full recipe (all five adoptions, trained outside the campaign) | 0.937801 | 0.929880 | incumbent by the user's decision |
| Anchors per stage by weight scale | 0.937090 | 0.932050 | reject |
| Stochastic depth 0.1 | 0.936885 | 0.931018 | reject |
| Two warm-up epochs | 0.937373 | 0.931222 | reject |
| Twice the head rate | 0.936739 | 0.930309 | reject |

The first two-epoch warm-up run (on the anchors incumbent) and `early-anchor-follows-learning-rate`
were stopped by the user and have no result. The four seeds on the full recipe all beat its ×1 and lost at ×4; it pairs the
lowest ×1 of the five with the highest ×4. Fracture and MCL lost in all four, Lateral Meniscus
gained in all four.

**Early epochs:** a run's first-epoch AUC spreads by 0.03 between runs with identical first-epoch
settings (0.816 and 0.844), and validation trails while the rate sits at its peak, then catches up as
it decays: two warm-up epochs were −0.0155 at their peak epoch against the median of the same
schedule and level from epoch 8. Every run was still rising at its last epoch.

## Kaggle T4

Bundle `outputs/submission/exports/mambavision-t-combined-f2-x4-check` (×4, temperature 1.0), profile
`outputs/profiling/kaggle-mambavision-t-f2-r2`, all 875 fold-2 studies on 2×T4 in FP16: macro AUC
0.937654 against 0.937801 locally in BF16 (−0.000147); mean absolute prediction difference 0.0019,
largest 0.058; 4.27 studies per second against 4.38 for one ResNet-50 at ×4. Nothing was submitted.
