# RadImageNet ResNet backbones

**The Refined-RadImageNet ResNet-50 (`resnet50.a1_in1k` with the
[Refined-RadImageNet](../../research/backbones-and-pretraining/radimagenet-weights.md#refined-radimagenet-resnets)
weights) is one of the three backbone families of the final solution**, trained from
`configs/resnet50.yaml` and its teacher-student variants ([final solution](../../final-solution.md)).
It carries the repeated middle slice, the checkpoint's own input statistics (mean 0.3162, std
0.3213), `backbone_lr` 5e-5 and L2-SP anchors 0.01 (stem and `layer1`) and 0.001 (the rest), and
average-pool shortcuts since 2026-10-13. Only these weights remain in the code
(`modeling/radimagenet.py`); the original RadImageNet ResNet-50, the ResNet-18 and the ResNet-10t
paths below were removed on 2026-10-14 and are kept here as the record of how the family was
chosen.

How it got there:

- **2026-09-28:** the original RadImageNet ResNet-50 lost to the ReXNet-200 control (×1 0.91223
  against 0.924637; runs below).
- **2026-09-29:** the Refined-RadImageNet ResNet-18 became the working recipe through the
  `radimagenet-rn18` autoresearch campaign, which adopted no series dropout (+0.0023) and found 36
  training triplets +0.0018 (weak, not carried), so the recipe kept 26 triplets without series
  dropout ([campaign](../../../autoresearch/campaigns/radimagenet-rn18/INDEX.md)). The 36-triplet
  ResNet-18 at ×4, τ 0.9 scored 0.944 public (S79), the same as the ReXNet control (S78); a 50/50
  ensemble of the two scored 0.949 (S80)
  ([τ sweep](../../validation/results/pooling-and-attention/2026-09-29-rn18-attention-temperature.md),
  [ledger](../../inference/submissions/ledger.md)). Its training took 1,202 s against the ReXNet
  control's 1,912 s.
- **2026-10-01:** the Refined-RadImageNet ResNet-50 on the ResNet-18 recipe unchanged scored ×4
  0.934423 against 0.929835 (+0.0046, 80% [+0.0027, +0.0064]) at 1,969 s against 1,197 s, and
  0.948 public (S88); with the ReXNet control and reflection TTA 0.951 (S89)
  ([revisits campaign](../../validation/results/campaigns-and-queues/2026-10-01-revisits-campaign.md#backbones)).
  It became `configs/refined-resnet50.yaml` (removed), the submission model, while the ResNet-18 stayed the
  control for iteration speed (user decision).
- **2026-10-13:** the ResNet-50 became the only recipe (`configs/resnet50.yaml`) and the
  ResNet-18 recipe was removed
  ([changes inside the backbone](#changes-inside-the-resnet-50-on-the-same-weights-2026-10-13)).

Other RadImageNet-pretrained weights are in
[RadImageNet weights](../../research/backbones-and-pretraining/radimagenet-weights.md).

## Why

- **External report, one participant:** swapping RadImageNet for ImageNet weights on the same
  ResNet-50, with normalization corrected, cost them **−0.063**
  ([2026-09-20 re-sweep](../../research/competition-and-prior-art/public-research.md#the-2026-09-20-re-sweep)). The public 0.943
  stack carries RadImageNet ResNet-50 heads as one of its five families.
- **Input format, a second participant** (fold 0, out-of-fold, n=1,102): RadImageNet ResNet-50
  scored 0.8339 on three adjacent slices as channels and **0.8542 on the centre slice repeated ×3**.
  RadImageNet was pretrained on single-channel images copied across RGB
  ([2026-09-28 re-sweep](../../research/competition-and-prior-art/2026-09-28-public-resweep.md#radimagenet-input-format)).
- **Earlier local search, unverified and gold-selected:** RadImageNet ResNet-50 scored 0.883363
  against 0.861902 for ImageNet ResNet-50 on the 58 gold studies
  ([backbone search](backbone-search.md#finished-model-comparison)).
- **Against:** the [benchmark ranking](../../research/backbones-and-pretraining/backbone-mri-evidence.md) put RadImageNet
  ResNet-50 19th.

## What is implemented

As built on 2026-09-28 for the original RadImageNet ResNet-50; that weight path was removed on
2026-10-14.

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
  terms alone do not disqualify a source ([supervision](../supervision.md#official-labeling-context)).
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
  [weight-scale rule](backbone-swaps.md#checkpoint-weight-scale), `backbone_lr` is
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

The run was stopped after epoch 3. The second run used `backbone_lr` 0.0003, the rate that matches the control's early-layer
movement.

## Refined-RadImageNet ResNet-18 and ResNet-10t (integrated 2026-09-28)

The ResNet-18 (`resnet18.a1_in1k`, `configs/resnet50.yaml` from 2026-09-29 to 2026-10-13) and ResNet-10t
(`resnet10t.c3_in1k`, config removed 2026-09-29 after it lost in the campaign) load the pinned [Refined-RadImageNet](../../research/backbones-and-pretraining/radimagenet-weights.md#refined-radimagenet-resnets)
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

## Changes inside the ResNet-50 on the same weights (2026-10-13)

From the [targets-and-inputs campaign](../../validation/results/campaigns-and-queues/2026-10-13-targets-and-inputs-campaign.md#modules-inside-the-backbone), on fold 2.

- **Adopted:** average-pool shortcuts (timm's `avg_down`, the shortcut keys renamed before
  the strict load). No new parameter, L2-SP anchor intact.
- **Not adopted:** squeeze-and-excitation, global context and coordinate attention in the
  blocks, started as the identity. Free of the anchor at the head rate they move the function far
  from the pretrained one while the L2-SP penalty reads lower; pulled toward the identity at the
  L2-SP strength the gates stay shut and the model is the incumbent again. Blur-pool at the three
  stride-two convolutions adds nothing on top of the average-pool shortcuts and doubles the ×4
  prediction time.
- **Rule for a new module in the blocks (user decision):** pull its last layer toward the
  identity at `optimizer.anchor_alpha_late` with weight decay 0, log how far it moves, and do not
  adopt a free module on a local fold.
- **How timm switches each on, what loads and what it costs:** the review of 2026-10-13 checked
  `block_args=dict(attn_layer=...)`, `avg_down`, `aa_layer`, `output_stride`, activation swaps and
  stochastic depth on timm 1.0.26; stochastic depth lost 0.0035 on the ResNet-18
  ([revisits](../../validation/results/campaigns-and-queues/2026-10-01-revisits-campaign.md)), activation swaps change the
  function with no recovering initialisation, and output stride 16 or 8 costs 57% or five times
  the backbone.


## Mask-trained ("heated") weights for the ResNet-50 (never finished)

The pretraining-input campaign produced mask-trained checkpoints, loaded with
`model.pretrained_weights: radimagenet_policy`. Their weights are 3.4 times larger than the
standard checkpoint's, so the recipe rescaled the backbone rate and the L2-SP strengths by that
factor. The weight entry and loader were only in the campaign's frozen build,
`outputs/autoresearch/pretraining-input/baseline/source/`; main never carried them. The two
ResNet-50 runs on these weights were stopped before completion, and no ResNet-50 run of them
finished ([evidence and the ResNet-10t blend](../../validation/results/campaigns-and-queues/2026-10-05-pretraining-input-campaign.md#findings),
[idea](../../backlog/untried-ideas.md#ensembles)).
