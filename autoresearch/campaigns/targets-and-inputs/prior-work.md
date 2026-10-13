# Targets-and-inputs campaign: prior work

Fold 2 of `all_5fold_scanner_v2`, seed 1110484336, one run per design unless said.

## The reference

Fold 2 of `outputs/training/rn50-aligned-mixing-folds-0-2`: `configs/refined-resnet50.yaml` as on
main at `0bd598f` (the soft-tissue line, the sagittal fluid detail view, aligned contrast mixing,
labels `rebuild_state_v2`, teacher share 0). **×1 0.928460 (epoch 10 of 10), ×4 0.937000, 2,691 s
of training** (about 267 s per epoch), 82 s of ×4 prediction, peak GPU memory 16.2 GB. The same
run holds folds 0 and 1 (×4 0.940906 and 0.933392)
([result](../../../docs/validation/results/2026-10-12-dense-bag-pooling-on-the-resnet-50.md)).

Per label at ×4, weakest first, with the validation positives of the 875 studies:

| Label | AUC | Positives |
|---|---:|---:|
| Synovitis | 0.8954 | 114 |
| PF OA | 0.9155 | 293 |
| Lateral OA | 0.9202 | 94 |
| MCL | 0.9308 | 45 |
| Lateral Meniscus | 0.9360 | 136 |
| Contusion | 0.9367 | 92 |
| Effusion | 0.9368 | 246 |
| Fracture | 0.9372 | 60 |
| Baker's | 0.9552 | 113 |
| Medial Meniscus | 0.9580 | 343 |
| ACL | 0.9582 | 131 |
| Medial OA | 0.9640 | 191 |

One point of AUC on one label is 0.0008 macro.

## The training targets

The 3,486 non-gold training studies of fold 2, table `rebuild_state_v2` (values between 0 and 1,
the mean of calibrated source families). Gold studies train on the official labels.

| Label | Thresholded positives | Mean soft target | Cells between 0.25 and 0.75 | Mean of abs(2t − 1) | Positive weight now |
|---|---:|---:|---:|---:|---:|
| Effusion | 0.272 | 0.347 | 34.0% | 0.649 | 1.38 |
| Synovitis | 0.124 | 0.310 | 33.1% | 0.538 | 2.16 |
| PF OA | 0.329 | 0.336 | 24.3% | 0.681 | 1.25 |
| Medial OA | 0.214 | 0.233 | 20.7% | 0.727 | 1.60 |
| Lateral OA | 0.106 | 0.150 | 15.1% | 0.779 | 2.40 |
| Baker's | 0.127 | 0.181 | 12.0% | 0.784 | 2.16 |
| Contusion | 0.102 | 0.149 | 9.8% | 0.801 | 2.43 |
| MCL | 0.050 | 0.098 | 8.1% | 0.856 | 3.66 |
| Medial Meniscus | 0.386 | 0.361 | 7.9% | 0.785 | 1.14 |
| Lateral Meniscus | 0.150 | 0.194 | 7.7% | 0.829 | 1.95 |
| ACL | 0.145 | 0.174 | 7.5% | 0.846 | 1.98 |
| Fracture | 0.064 | 0.125 | 3.0% | 0.817 | 3.15 |

- **The reference's two weakest labels, Synovitis and PF OA, are among the three with the most
  uncertain training cells**; the third, Effusion, is in the middle on this fold (0.9368) and
  the second weakest on fold 1 (0.9197).
- The class weights (`fixed_class_balance`, exponent 0.5) are computed from the thresholded
  fraction over all training studies of the fold; the weights above are the reference's
  (`fold_2/class_balance.json`).
- With the mean soft target in place of the thresholded fraction the positive weight becomes
  1.29 for Synovitis, 2.55 for MCL, 2.19 for Fracture and 1.21 for Effusion (computed from the
  table above, non-gold studies only).

## Precedents

| Mechanism | Setting | Run | Macro delta | Source |
|---|---|---|---|---|
| Class-balanced BCE | ReXNet-200, fold 0 | exponent 1 against none | about +0.008 | [balanced BCE](../../../docs/validation/results/2026-09-19-balanced-bce-result.md) |
| Exponent 0.5 and no weighting against exponent 1 | ReXNet-200, fold 0 | two runs | lost locally, won publicly | [exponent](../../../docs/validation/results/2026-09-21-class-balance-exponent.md) |
| Exponent ramped from 1 to 0 | ReXNet-200 | one run | tie; 0.941 public against 0.942 | [noisy labels](../../../docs/validation/results/2026-09-23-noisy-labels-campaign.md) |
| Soft-mass weighting, prior offset | ReXNet-200, losses campaign | two runs | −0.0092, −0.0119 (other loss semantics changed too) | [losses findings](../losses/findings.md) |
| Binary mask of low-evidence cells | early recipe | one run | rejected | [label masking](../../../docs/history/sharing/2026-09-13-label-masking-result.md) |
| Shared balanced focal (hard cells up-weighted) | ReXNet-200 | one run | −0.0026 | [losses findings](../losses/findings.md) |
| In-batch ranking term, always on | ReXNet-200 | one run | +0.0007, interval spanning zero | [losses findings](../losses/findings.md) |
| Quantile levels 0.001/0.999 and 0.01/0.99 | ReXNet-200 | two runs | −0.0003, −0.0021 | [overnight](../../../docs/validation/results/2026-09-17-overnight-results.md) |
| Intensity preprocessing variants | ResNet-18, `mri-signal-preservation` | 8 runs | no adoption, most over the time ceiling | [campaign](../mri-signal-preservation/INDEX.md) |
| Bright-end clipping per series, type, slice and plane | ResNet-10t, `pretraining-input` | 32 runs | nothing adopted | [result](../../../docs/validation/results/2026-10-05-pretraining-input-campaign.md) |
| Augmentation families on and off | ReXNet-200, augmentation campaign | 13 seeds | smoothing and DSU adopted, half stack kept | [result](../../../docs/validation/results/2026-09-26-augmentation-campaign.md) |
| Contrast mixing off | ResNet-18 | one run | −0.0025 | [fixes and removals](../../../docs/validation/results/2026-10-05-fixes-and-removals-campaign.md) |
| Anatomy map and posed crops | ResNet-50 | S126 against S119 | +0.0001 at ×4, +0.0017 at ×1, +0.002 public | [state](../../../docs/ongoing/state.md#recipe-in-force) |
| Report-attribute heads off | ResNet-18, anatomy recipe | one run | adopted | [anatomy recipe](../anatomy-recipe/INDEX.md) |
| Type-specific readouts after pooling | ReXNet-200, near misses | one run | −0.0006 | [near misses](../../../docs/validation/results/2026-09-25-near-misses-campaign.md) |
| Plane-contrast readouts | ResNet-18, revisits | one run | adopted (Lateral OA +0.0054) | [revisits](../../../docs/validation/results/2026-10-01-revisits-campaign.md) |

- **Clipping-bound stability (measured 2026-09-20, 40 studies, six epochs):** the high bound of
  a series moves 0.072 at the median and 0.245 at the 90th percentile between draws, against
  the 0.2 relative range of the contrast augmentation; 14.2% of series exceed it; the spread
  sits in series with few triplets (0.150 at two or fewer, 0.025 above six)
  ([bound stability](../../../docs/data-preparation/preprocessing-implementation-and-profile.md#clipping-bound-stability-across-draws)).
  No run has measured its effect on AUC.
- **Intensity and noise augmentation** date from the first recipes and were never removed alone
  on a RadImageNet backbone.

## What does not reach the ResNet-50

Four earlier ResNet-50 recipes on fold 1 (run 24, the anatomy map, the view poses, the soft-tissue reads)
score 0.933209, 0.933265, 0.933256 and 0.933292 at ×4: routing and structure changes on the same
features gained on the ResNet-18 and not here. What reached it: finer fields and the view label
bias (+0.0027), distillation (+0.0023 at 0.25 teacher share), 35 distinct slices (+0.0014 locally,
0.959 public against 0.960). This campaign's subjects are none of those: they change the loss and
the pixel values.

## Dropped before the campaign

- **PF OA trained on the official-threshold rule:** needs a label variant and the label gate
  opened ([backlog](../../../docs/backlog/big-bets.md#patellofemoral-oa-evidence)).
- **Bone masks for every training knee:** needs offline segmentation of about 3,100 knees
  ([backlog](../../../docs/backlog/big-bets.md#anatomy-masks-what-is-left-untried)).
