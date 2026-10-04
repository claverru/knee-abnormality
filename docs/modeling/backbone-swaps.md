# Swapping a backbone

This page covers the cost and code requirements for replacing the backbone (`model.backbone`,
ReXNet-200) of the [full-slice model](current-model.md).

- **Units:** costs are parameter and multiply-add (GMAC) counts, not timings. Time real
  candidates on a Kaggle T4. Measured 2026-09-18: EfficientNet-B3 at −37% GMACs trained **7.2%
  slower** than ReXNet-200 and B0 at −74% trained 33.6% faster, so the conversion is neither
  reliable nor proportional for depthwise-heavy families
  ([result](../validation/results/2026-09-18-efficientnet-backbone-results.md)).
- **Related research:**
  - [Backbone MRI evidence](../research/backbone-mri-evidence.md) covers which families suit MRI.
  - [Reusable image and MRI encoders](../research/huggingface-image-encoders.md) covers Hub
    checkpoints, patch-token adapters and the knee-specific 3D alternatives.
  - [EfficientNet training](../research/efficientnet-training.md) covers EfficientNet
    preprocessing and BatchNorm handling.
- **Candidates:** proposed same-cost replacements are in the
  [section below](#same-cost-backbone-candidates).
- **History:** the crop-era budget, with the MobileNetV3 scout, its delta formula and its code
  requirements, is in the
  learned-zoom snapshot (`f076a09:docs/history/crop-selection-sources/learned-zoom-backbone-swaps.md.txt`).

## Current budget

The ReXNet-200 counts were measured on CPU on 2026-09-16 (timm 1.0.26, torch 2.10) with
`FlopCounterMode`, halved. Head counts are analytic, from layer sizes.

Parameters total about 14.52M:

| Component | Parameters | Share |
|---|---:|---:|
| ReXNet-200 | 13.81M | 95.1% |
| Projection (2560→256) | 0.66M | 4.5% |
| Attention, 2D position, plane and contrast embeddings, LayerNorm, head | 0.06M | 0.4% |

Forward compute per triplet, in GMACs:

| Component | Per triplet | Share |
|---|---:|---:|
| ReXNet-200 at 224 px (7×7 grid) | 1.526 | 97.8% |
| Projection, 49 tokens | 0.032 | 2.1% |
| Attention MLP and 2D position, 49 tokens | 0.0025 | 0.2% |
| **Total** | **1.561** | |

- **Triplets per study (K):** 25.4 at 1× image limits and 50.8 at 2×, averaged over the 4,407
  training studies. Each series type contributes min(images − 2, its triplet limit × the
  multiplier) triplets.
- **Training:** K ≈ 20, because series dropout 0.2 removes about a fifth of the series. That
  makes about 32 GMACs per training study and about 79 per inference study at 2×. Both figures
  are derived from the per-triplet total.
- **Backward pass:** it scales all trained parts roughly alike, so the shares hold for training
  steps.
- **Larger inputs:** ReXNet-200 at 280 px measured 2.462 GMACs with a 9×9 grid, 1.61× the
  224 px cost.

## Delta formulas

C is the feature channel count entering the 256-wide projection. Each channel adds 256
parameters, and 256 multiply-adds per token.

- **Parameters:** Δparameters = ΔP + 256 ΔC, with C = `num_features` (2560 now).
- **Compute:** Δcompute = (ΔG₂₂₄ + T · 256 ΔC / 10⁹) / 1.561 of the current total, where T is
  the token count per triplet (49 at stride 32 and 224 px).
- **Stride:** a different stride changes T, and with it the projection, attention and softmax
  costs.

## Code requirements

In `KneeAbnormalityModel.__init__`:

- **Mapping:** the name must have a `BACKBONE_MAPPINGS` entry, and `backbone_mapping` must
  reproduce it. `_rexnet_mapping`, `_coatnet_mapping`, `_efficientnet_mapping` and
  `_resnet_mapping` exist, so any other family needs its own mapping function. `_resnet_mapping`
  covers `resnet50.tv_in1k`, `resnet18.a1_in1k` and `resnet10t.c3_in1k`, the backbones with
  [RadImageNet weights](../ongoing/radimagenet-backbone.md): single or deep stem, max pool,
  bottleneck or basic blocks, checking each residual shortcut (ResNet-D average-pool shortcuts sit
  half a stride off), origin 0 and stride 32. `_efficientnet_mapping` walks `conv_stem`, then
  each block by type (`DepthwiseSeparableConv`, `InvertedResidual`, `ConvBnAct`, `EdgeResidual`),
  then `conv_head`, naming the convolutions explicitly so squeeze-excite is excluded. Registered
  EfficientNets: `efficientnet_b3.ra2_in1k`, `efficientnet_b0.ra_in1k` and
  `efficientnet_b0.ra4_e3600_r224_in1k`, all at origin 0 and stride 32, the same anchors as
  ReXNet-200 ([result](../validation/results/2026-09-18-efficientnet-backbone-results.md)).
- **Features:** `forward_features` must return NCHW with `num_features` channels. Construction
  probes the backbone once at `data.image_size`.
- **Input size:** the backbone must accept `data.image_size`.
  [CoAtNet](coatnet-224-spatial-backbone.md) receives it explicitly and requires a multiple
  of 32.
- **Anchors:** token anchors come from the mapping, so any verified origin works.
- **Activation checkpointing:** `model.checkpoint_backbone` requires `model.backbone_eval: true`.
- **DSU:** `model.feature_statistics_augmentation` splits the backbone at stride 8, which is
  implemented for ReXNet (after block 5) and ResNet (after `layer2`) only.

## Checkpoint weight scale

Batch-normalized convolutions are scale-invariant, while AdamW's step is about the learning rate
per weight, so a checkpoint with smaller weights takes proportionally larger relative steps at the
same `backbone_lr`. Measured 2026-09-25: `rexnetr_200.sw_in12k_ft_in1k` has convolution-weight RMS
0.0204 against 0.0508 for `rexnet_200.nav_in1k`; at the same rate its gradient norms ran 3–4× higher
and it fitted more slowly
([result](../validation/results/2026-09-25-backbone-swap-runs.md#rexnet-r)). Compare the pretrained convolution-weight
RMS before a swap and scale `backbone_lr` by the ratio.

## Mapping pitfalls

`_spatial_mapping` walks the convolutions it is given, in order. Applying it to every
convolution in module order is exact only for single-path networks with static padding. The
2026-09-16 sweep of timm models found these categories:

- **Native:** a single path with symmetric padding gives origin 0 at strides 16 and 32 for
  MobileNetV3/V4, EfficientNet, ReXNet and EfficientViT-B. Only a mapping entry is needed.
- **Offset origin:** these need a mapping entry with the true origin.
  - *ConvNeXt:* at stride 16, the patchify stem gives a 7.5 px origin (6 px for `_ols`
    variants), and grids floor (18×18 at 300 px).
  - *HGNetV2:* the stem pads with `F.pad` and uses a stride-1 max pool, which the walker does
    not see. Its reported 2 px origin needs verifying.
- **Walker wrong:** these need a family-specific function.
  - *RepGhost and GhostNet:* strided shortcut convolutions are counted twice (reported stride
    128/512). Every strided convolution has an odd kernel and symmetric padding, so the true
    origin is 0.
  - *RegNet:* strided 1×1 shortcuts are counted twice. Origin 0 is expected but unverified.
  - *TensorFlow-ported `tf_` models:* "same" padding is dynamic and asymmetric, so a static walk
    is unreliable. Measured 2026-09-18: `tf_efficientnet_b0.ns_jft_in1k` walks to origin **51.0**
    where its non-`tf_` sibling gives 0. Every AdvProp, AutoAugment and noisy-student/JFT
    EfficientNet checkpoint is a `tf_` port, so all of them are out of reach without a
    dynamic-padding-aware mapping verified per input resolution.
  - *Reparameterizable families (FastViT, MobileOne, EfficientFormer):* parallel branches break
    the walk.
- **Fixed input size:** LeViT, EfficientViT-M and EfficientFormer/V2 run only at their native
  size, so `data.image_size` must match it.
- **ViT-like:**
  - *Stride:* depends on the patch embedding; patch-16 and patch-14 encoders produce different
    grids. Our head can retain either grid after adaptation and verified geometry.
  - *Origin:* for unpadded patchification, 7.5 px at patch size 16 and 6.5 px at patch size 14.
  - *Output:* remove non-spatial prefix tokens and reshape patch tokens to NCHW.
  - *Position embeddings:* interpolated, and `img_size` or `dynamic_img_size` must be set.

## Same-cost backbone candidates

**Not pursued since 2026-10-04: models stay on the Refined-RadImageNet weights
([adoption](current-model.md#adoption-history)). Kept as the record of the candidates and their
mappings.** Proposed 2026-09-25; ReXNet-R is implemented in the `convnext-atto` worktree and lost its first
run (−0.0053) with a measured over-large learning rate; its follow-up at `backbone_lr` 8e-5 is
open. ConvNeXt-Atto, run outside this list, lost −0.0192 and is closed
([backbone swap runs](../validation/results/2026-09-25-backbone-swap-runs.md)). ConvNeXt-nano and
HGNetV2-B3 are unimplemented. Three timm backbones of about ReXNet-200's
size (13.8M parameters, 1.53 GMACs at 224 px) that could replace it. Each needs only a
`BACKBONE_MAPPINGS` entry with a verified origin; `model.backbone` is the option
([swapping a backbone](#code-requirements)). All three checkpoints
declare the configured ImageNet mean/std. Counts are backbone-only, measured on the meta device
with `FlopCounterMode` (timm 1.0.26).

| Candidate | Params | GMACs | Mapping | Motivation | Confidence |
|---|---:|---:|---|---|---|
| `rexnetr_200.sw_in12k_ft_in1k` | 14.0M | 1.56 | `_rexnet_mapping`, verified origin 0, stride 32, 2560 channels | Same family and cost, ImageNet-12k pretraining; lumbar 2024 7th place | Moderate, evidence-backed |
| `convnext_nano.in12k_ft_in1k` | 15.0M | 2.44 | [Offset origin](#mapping-pitfalls); the `convnext-atto` worktree's `_convnext_mapping` should only need a registration | Best in lumbar 3rd place's comparison; DINOv3 ConvNeXt-T 2nd in the [backbone search](backbone-search.md#finished-model-comparison) | Moderate |
| `hgnetv2_b3.ssld_stage2_ft_in1k` | 14.2M | 1.77 | Offset origin, reported 2 px, unverified | Dense and GPU-friendly; family evidence only as a lumbar localizer | Low, speculative |

- **Excluded:** EfficientNet and EfficientNetV2 (measured losses on this dataset), `tf_` ports
  and reparameterizable or fixed-size families (mapping), `regnetz_c16` (0.5 mean/std),
  MobileNetV4, RepViT and small transformers (no MRI evidence), ResNet-18 (trailed ConvNeXt-nano
  in lumbar 3rd's comparison). See [backbone MRI evidence](../research/backbone-mri-evidence.md).
- **Cost:** GMACs do not predict wall time for depthwise-heavy families; ConvNeXt-nano's 1.6×
  GMACs may cost less than that. Estimate time from the run's own logs, then time any winner on a
  Kaggle T4 in FP16.
- **Comparison:** one run per candidate against the incumbent control, same fold, seed and
  budget. Scale `backbone_lr` by the checkpoint's convolution-weight RMS relative to ReXNet-200's
  ([weight scale](#checkpoint-weight-scale)); leave other rates
  unchanged. Record macro/per-label AUC, paired uncertainty, time and memory under the current
  adoption rule. Ensemble contribution belongs to the
  [endgame](../backlog/endgame.md#ensembling).
