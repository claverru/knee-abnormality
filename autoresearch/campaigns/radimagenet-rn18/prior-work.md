# RadImageNet ResNet-18 prior work

Fold 0 of `all_5fold_scanner_v2`, `rebuild_state_v2` labels, seed 1533697213 unless noted. Code:
branch `radimagenet` ([ongoing doc](../../../docs/ongoing/radimagenet-backbone.md) once merged).

## Precedent table

| Mechanism | Setting | Run | ×4 macro | vs ReXNet control | Training s |
|---|---|---|---:|---:|---:|
| ReXNet-200 control | current recipe | `20260928-150724-rebuild-state-v2-control` | 0.928288 | — | 1,912 |
| RadImageNet ResNet-18 (the campaign's RadImageNet base) | repeated middle slice, backbone LR 5e-5, ImageNet mean/std | `20260928-185524-radimagenet-resnet18` | 0.925476 | −0.0028 | 1,085 |
| RadImageNet ResNet-10t | same, backbone LR 6e-5 | `20260928-183517-radimagenet-resnet10t` | 0.922579 | −0.0057 | 1,048 |
| RadImageNet ResNet-50 | repeated middle, backbone LR 1.23e-3, [−1, 1] | `20260928-174920-radimagenet-resnet50` | stopped at epoch 3 (×1 0.8972 against 0.9038) | — | — |
| RadImageNet ResNet-50 | backbone LR 3e-4 | `20260928-180252-radimagenet-resnet50-lr3e-4` | ×1 0.91223 only (predictions recovered) | — | — |
| RadImageNet ResNet-18 | three adjacent slices, pretrained conv1 unchanged | `20260928-192554-radimagenet-resnet18-adjacent` | 0.919531 | −0.0088 (−0.0059 against the base) | 1,094 |
| RadImageNet ResNet-18 | cosine floor 0.1 (`minimum_lr_ratio`) | `20260928-195321-radimagenet-resnet18-minlr0.1` | 0.925311 | −0.0030 (−0.0002 against the base) | 1,129 |
| RadImageNet ResNet-18 | 14 epochs (campaign `radimagenet`, seed 1) | `outputs/autoresearch/radimagenet/experiments/rn18-fourteen-epochs-20260928T203926Z-edcce1c4` | 0.926464 | −0.0018 (+0.00099 against the base, 80% ≈ [−0.0001, +0.0020]) | 1,511 |
| RadImageNet ResNet-18 | two warmup epochs (campaign `radimagenet`, seed 2) | `outputs/autoresearch/radimagenet/experiments/rn18-two-warmup-epochs-20260928T204343Z-5a06510a` | 0.924780 | −0.0035 (−0.0007 against the base, 80% ≈ [−0.0015, +0.0001]) | 1,107 |
| EfficientNet-B3 | two warmup epochs | B3 overnight `03-warmup-2` | +0.0012 over its B3 control (fold 3) | — | — |

## Measured: do not propose again as is

Each of these was run on the current labels and split; a proposal that repeats one needs a
concrete change beyond it.

- **RadImageNet ResNet-18 base** (repeated middle slice, backbone LR 5e-5, cosine to 1%, 10 epochs):
  0.925476.
- **ResNet-10t** at the same recipe (backbone LR 6e-5): 0.922579, below ResNet-18 at every epoch.
- **Cosine floor 0.1** (`optimizer.minimum_lr_ratio`): 0.925311, curve unchanged.
- **Three adjacent slices through the unchanged pretrained conv1** (`model.slice_channels: adjacent`):
  0.919531, below the base at every epoch. The mean-initialized first convolution (seed) is a
  different mechanism.
- **ResNet-50** at backbone LR 1.23e-3 ([−1, 1] inputs; stopped behind at epoch 3) and 3e-4 (×1
  0.91223): early-layer drift problem at those rates.
- **14 epochs** (same warmup and cosine shape): 0.926464, +0.00099 against the base with 80% ≈
  [−0.0001, +0.0020]; ×1 validation gained only 0.0009 over its last six epochs, and layer1 still
  drifted 8.8%, so length alone does not keep it learning.
- **Two warmup epochs:** 0.924780, −0.0007 against the base; the B3 precedent did not transfer.
- **Weight averaging (SWA, EMA):** excluded by the user.

## Curves (×1 validation)

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Control | 0.8314 | 0.8845 | 0.8937 | 0.9038 | 0.9131 | 0.9159 | 0.9213 | 0.9232 | 0.9244 | 0.9246 |
| ResNet-18 | 0.8787 | 0.8971 | 0.9073 | 0.9109 | 0.9155 | 0.9176 | 0.9186 | 0.9205 | 0.9213 | 0.9215 |
| ResNet-10t | 0.8710 | 0.8952 | 0.8991 | 0.9088 | 0.9111 | 0.9125 | 0.9142 | 0.9158 | 0.9162 | 0.9164 |
| ResNet-18, adjacent slices | 0.8661 | 0.8881 | 0.9017 | 0.9034 | 0.9114 | 0.9132 | 0.9144 | 0.9160 | 0.9169 | 0.9174 |
| ResNet-18, cosine floor 0.1 | 0.8774 | 0.8963 | 0.9076 | 0.9099 | 0.9150 | 0.9168 | 0.9173 | 0.9196 | 0.9210 | 0.9213 |

A cosine floor of 0.1 (ten times the base's late learning rate) left the curve unchanged, so a too-low
late rate does not explain the flattening (one run).

Training loss: ResNet-18 below the control at epochs 0–1 and above it from epoch 2 (2.327 against
2.297 at epoch 9).

## Drift from the pretrained start (selected checkpoint)

| Backbone | Median conv | First five convs | Last five convs |
|---|---:|---:|---:|
| ReXNet-200 control | 15.8% | 3.2% | 19.2% |
| ResNet-18 | 10.8% | 8.1% | 20.0% |
| ResNet-10t | 9.0% | 6.1% | 18.3% |

ResNet-50 at backbone LR 1.23e-3 had moved 10.4% in its first five layers after four epochs.

## Noise yardstick

Paired Δ training noise about 0.0006–0.0010 on top of a bootstrap standard error of 0.0015
([calibration](../../literature-mechanisms.md#3-calibration-against-repeat-pairs-keep-the-rule));
a fresh seed moves absolute ×4 macro by up to about 0.006.

## Blocked directions

Image size above 224, ensembling and TTA wait for the end phase.
