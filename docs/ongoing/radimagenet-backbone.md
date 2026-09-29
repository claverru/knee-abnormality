# RadImageNet backbones

**ResNet-18 is the working RadImageNet recipe (2026-09-29).** The code is on main. ResNet-50
lost to the control (×1 0.91223 against 0.924637). The Refined-RadImageNet ResNet-18
(`configs/radimagenet-resnet18.yaml`) ran as the `radimagenet-rn18` autoresearch campaign. It
adopted no series dropout (+0.0023) and 36 training triplets (+0.0018, weak, deferred to the
[endgame](../backlog/endgame.md#test-time-augmentation) for iteration speed), so the committed recipe is
26 triplets without series dropout ([campaign](../../autoresearch/campaigns/radimagenet-rn18/INDEX.md)).
The 36-triplet ResNet-18 at ×4, τ 0.9 scored 0.944 public (S79), the same as the ReXNet control (S78). A 50/50 ensemble
of the two scored 0.949 (S80), the best public score
([τ sweep](../validation/results/2026-09-29-rn18-attention-temperature.md),
[ledger](../inference/submissions/ledger.md)). Its training takes 1,202 s against the ReXNet
control's 1,912 s.

Other RadImageNet-pretrained weights are in
[RadImageNet weights](../research/radimagenet-weights.md).

## Why

- **External report, one participant:** swapping RadImageNet for ImageNet weights on the same
  ResNet-50, with normalization corrected, cost them **−0.063**
  ([2026-09-20 re-sweep](../research/public-research.md#the-2026-09-20-re-sweep)). The public 0.943
  stack carries RadImageNet ResNet-50 heads as one of its five families.
- **Input format, a second participant** (fold 0, out-of-fold, n=1,102): RadImageNet ResNet-50
  scored 0.8339 on three adjacent slices as channels and **0.8542 on the centre slice repeated ×3**.
  RadImageNet was pretrained on single-channel images copied across RGB
  ([2026-09-28 re-sweep](../research/2026-09-28-public-resweep.md#radimagenet-input-format)).
- **Earlier local search, unverified and gold-selected:** RadImageNet ResNet-50 scored 0.883363
  against 0.861902 for ImageNet ResNet-50 on the 58 gold studies
  ([backbone search](../modeling/backbone-search.md#finished-model-comparison)).
- **Against:** the [benchmark ranking](../research/backbone-mri-evidence.md) put RadImageNet
  ResNet-50 19th.

## What is implemented

- **Weights:** `model.pretrained_weights: radimagenet` loads the Kaggle mirror
  [`marwanmath/resnet-50-radimagenet-marwan`](https://www.kaggle.com/datasets/marwanmath/resnet-50-radimagenet-marwan)
  `ResNet50.pt`, SHA-256 `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`. That is
  the file the public stack pins. It is a plain state dict (318 tensors, `backbone.{0,1,4–7}.*`)
  that `radimagenet_state_dict` renames to timm names and loads strictly into
  `resnet50.tv_in1k`, the torchvision architecture. The file must sit at
  `~/.cache/torch/hub/checkpoints/radimagenet_resnet50.pt`; construction fails otherwise, and
  the error names the download command. Only training needs it: checkpoints carry the weights,
  so inference ignores `pretrained_weights`.
- **Licence:** the official [RadImageNet repository](https://github.com/BMEII-AI/RadImageNet) is
  MIT. The Kaggle mirror is listed as CC-BY-NC-SA-4.0; the host's rule is that non-commercial
  terms alone do not disqualify a source ([supervision](../modeling/supervision.md#official-labeling-context)).
- **Input:** `model.slice_channels: repeated_middle` replaces each triplet's channels with its
  middle slice after augmentation and before normalization. Visual validity uses only the middle
  slice's valid rectangle. The loader still decodes three slices, so input cost is unchanged.
- **Normalization, measured 2026-09-28:** mean 0.5 and std 0.5, i.e. inputs in [−1, 1]. On 64
  random training slices, the conv1 outputs under this scaling reproduce the checkpoint's own
  `bn1` running means: correlation 1.000, scale 0.83, median variance ratio 1.27. ImageNet
  mean/std gives variance 6.2×. RadImageNet's Keras example (caffe mean, then ÷255) gives scale
  0.32. The public stack also uses [−1, 1].
- **Learning rate, measured 2026-09-28:** RadImageNet's convolution weights have RMS 0.314, with
  every one of the 53 layers between 0.30 and 0.36. That compares with 0.0508 for ReXNet-200 and
  0.0159 for torchvision's ImageNet ResNet-50. Following the
  [weight-scale rule](../modeling/backbone-swaps.md#checkpoint-weight-scale), `backbone_lr` is
  0.0002 × 0.314 / 0.0508 = **0.00123**; `head_lr` is unchanged.
- **DSU:** the feature-statistics hook now also splits ResNets at their stride-8 point, the end
  of `layer2` (512 × 28 × 28). Both backbones' split paths equal `forward_features` exactly, so
  DSU stays at the recipe's p 0.5.
- **Spatial tokens:** `_resnet_mapping` walks the stem, the max pool and every bottleneck and
  checks each residual shortcut; origin 0, stride 32, the same anchors as ReXNet-200.
- **Cost:** ResNet-50 is 4.09 GMACs and 23.5M backbone parameters per 224 px triplet, against
  1.53 GMACs for ReXNet-200. Training time and memory at batch 8 are unmeasured.

## First run: learning rate too high (2026-09-28)

`20260928-174920-radimagenet-resnet50` (`backbone_lr` 0.00123) led the control by +0.015 at epoch 0,
the warmup epoch. It fell behind once the full rate arrived: −0.001, −0.0035 and −0.0066 at epochs
1–3 (×1 validation 0.8835, 0.8902, 0.8972 against 0.8845, 0.8937, 0.9038). Its training loss was
also higher than the control's from epoch 1. After four epochs its backbone convolutions had moved
12.9% from initialization (median relative norm), 10.4% in the first five layers. The control moved
15.8% after all ten epochs, but only 3.2% in its first five layers. Because RadImageNet's weight
scale is uniform across depth, the median-matched rate moves its early layers about three times as
far as ReXNet's, overwriting the low-level pretrained features. One run, so weak evidence.

The run was stopped after epoch 3. The next run uses `backbone_lr` 0.0003, the rate that matches
the control's early-layer movement. The single-slice input remains the other suspect, to test
separately if the lower rate does not close the gap.

## Refined-RadImageNet ResNet-18 and ResNet-10t (integrated 2026-09-28)

`configs/radimagenet-resnet18.yaml` (`resnet18.a1_in1k`) and `configs/radimagenet-resnet10t.yaml`
(`resnet10t.c3_in1k`) load the pinned [Refined-RadImageNet](../research/radimagenet-weights.md#refined-radimagenet-resnets)
checkpoints through the same `pretrained_weights: radimagenet` option. `RADIMAGENET_WEIGHTS` maps each
backbone to its file, hash and source. The configs keep the repeated middle slice, use ImageNet
mean/std (the torchvision recipe that trained them) and set `backbone_lr` by the overall
weight-RMS rule: 5e-5 (RMS 0.0124) and 6e-5 (RMS 0.0158). Their early layers are about 2× (ResNet-18)
and 3× (ResNet-10t) the median scale, against 5× for ReXNet-200. Matching early-layer movement
instead would give 2e-5 and 3.6e-5. `_resnet_mapping` now also walks basic blocks and ResNet-D
average-pool shortcuts, whose anchors sit half a stride off the main branch, the same check as
CoAtNet's.

## Second run and a recovered prediction step

`20260928-180252-radimagenet-resnet50-lr3e-4` finished ten epochs at ×1 0.91223 (epoch 9) against
the control's 0.924637; it fell behind from epoch 3 while its training loss stayed about 0.06 above
the control's. Its final prediction step failed because the worktree sources were edited during
training, which changes the preprocessing identity. The validation predictions were recovered
from the run's own `source_snapshot.zip` (`recovered.json` in the run folder).

## Comparisons

- **Against the current control** (`rebuild-state-v2-control`, fold 0, seed 1533697213,
  [state](state.md#current-control)): this is the decision-relevant one, since a RadImageNet model
  is only useful if it competes with, or ensembles well with, ReXNet-200. It changes backbone,
  pretraining, input format, normalization and learning rate together.
- **Pretraining alone:** the same config with ImageNet weights, at their own normalization and
  learning-rate scale:
  `just train configs/radimagenet-resnet50.yaml --model.pretrained-weights timm
  --data.mean '[0.485,0.456,0.406]' --data.std '[0.229,0.224,0.225]'
  --optimizer.backbone-lr 0.0000627`. It is needed only to attribute a result to the pretraining.
