# Swapping a backbone

This page covers the cost and code requirements for replacing the backbone (`model.backbone`,
ReXNet-200) of the [full-slice model](current-model.md).

- **Units:** costs are parameter and multiply-add (GMAC) counts, not timings. Time real
  candidates on a Kaggle T4.
- **Related research:**
  - [Backbone MRI evidence](../research/backbone-mri-evidence.md) covers which families suit MRI.
  - [EfficientNet training](../research/efficientnet-training.md) covers EfficientNet
    preprocessing and BatchNorm handling.
- **History:** the crop-era budget, with the MobileNetV3 scout, its delta formula and its code
  requirements, is in the
  [learned-zoom snapshot](../history/crop-selection-sources/learned-zoom-backbone-swaps.md.txt).

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
  reproduce it. Only `_rexnet_mapping` and `_coatnet_mapping` exist, so any other family needs
  its own mapping function.
- **Features:** `forward_features` must return NCHW with `num_features` channels. Construction
  probes the backbone once at `data.image_size`.
- **Input size:** the backbone must accept `data.image_size`.
  [CoAtNet](coatnet-224-spatial-backbone.md) receives it explicitly and requires a multiple
  of 32.
- **Anchors:** token anchors come from the mapping, so any verified origin works.
- **Activation checkpointing:** `model.checkpoint_backbone` requires `model.backbone_eval: true`.

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
    is unreliable.
  - *Reparameterizable families (FastViT, MobileOne, EfficientFormer):* parallel branches break
    the walk.
- **Fixed input size:** LeViT, EfficientViT-M and EfficientFormer/V2 run only at their native
  size, so `data.image_size` must match it.
- **ViT-like:**
  - *Stride:* stride 16 only, so a stride-32 map would need 2×2 pooling.
  - *Origin:* a 7.5 px patch origin.
  - *Position embeddings:* interpolated, and `img_size` or `dynamic_img_size` must be set.
