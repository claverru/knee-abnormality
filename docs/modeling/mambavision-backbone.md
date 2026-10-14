# MambaVision-T and stem-staged backbones

MambaVision-T (`mamba_vision_T`, ImageNet-1k weights) trains from `configs/mambavision.yaml` and
ConvNeXt-Tiny from `configs/convnext.yaml`, beside the ResNet-50 of `configs/resnet50.yaml`.
Results: [2026-10-14 MambaVision-T](../validation/results/2026-10-14-mambavision-t.md).

## Code

- **Stage adapters** (`modeling/backbone_stages.py`): the model reads every backbone through one
  adapter per family, chosen by the backbone's class in `BACKBONE_STAGES`. Each gives
  `to_stride_8`, `deep` (the stride-32 maps and the stride-16 maps), the width of the stride-16
  maps (`middle_width`, for the meniscus map). `ResNetStages` runs `conv1` to `layer4`; `StemStages` runs `stem`,
  `stages[0]` to `stages[3]` and a final norm, for timm's ConvNeXt and for the MambaVision wrapper.
  A new family needs an adapter and its `BACKBONE_MAPPINGS` entries
  ([swapping a backbone](backbone-swaps.md#code-requirements)).
- **MambaVision wrapper** (`modeling/staged_backbones.py`): builds the `mambavision` package's model
  and regroups it as stem, four stages and a final norm. MambaVision ends each level with its
  downsample; the wrapper moves it to the start of the next stage, so the stride-8, 16 and 32 maps
  come out of stages 1, 2 and 3. Token origins are 0 at every level (3×3 convolutions with
  padding 1).
- **`mamba_ssm` import:** the package's `__init__` imports its language models, which need
  `transformers`, `tilelang` and `quack-kernels`. The wrapper registers a bare `mamba_ssm` package
  first, so only the selective-scan interface MambaVision uses is imported.
- **Weight anchors:** the early anchor matches `stem.` and `stages.0.` besides the ResNet names.

## Dependencies

- **Local:** `mamba-ssm` comes from the project's prebuilt GitHub release wheel (2.3.2.post1, CUDA 12,
  torch 2.10, cp312), pinned as a URL source in `pyproject.toml`. `[[tool.uv.dependency-metadata]]`
  gives it and `mambavision` 1.2.0 only `einops`, `torch`, `triton` and `timm`: the published
  metadata would pull in `transformers`, `tilelang`, `quack-kernels` and MambaVision's pins
  (`timm==1.0.15`, `mamba-ssm==2.2.4`). The `mamba-ssm==2.2.4` pin is stale: its release wheels
  stop at torch 2.6, so pip compiles it from source on newer torch. On the 3090 (2026-10-14), the
  trained fold-2 backbone gave bit-identical outputs with 2.3.2 and a 2.2.4 source build in FP32,
  BF16 and FP16; gradients differed only as much as two runs of 2.2.4.
- **Kaggle** (Python 3.13, torch 2.11, CUDA 12.8 in October 2026): no release wheel covers torch
  2.11. `outputs/submission/dependencies/kaggle-py313-mambavision` holds a `mamba_ssm` 2.2.4 wheel
  built on Kaggle for the T4 (compute 7.5, 413 s, notebook `claverru/knee-mamba-ssm-t4-build` with
  internet on) and the pure `mambavision` 1.2.0 wheel. Export adds that folder for every
  checkpoint whose backbone is in `BACKBONE_WHEELS` (`src/knee_abnormality/kaggle.py`); the
  notebook installs all bundled wheels with `--no-deps`. Rebuild the wheel when Kaggle's torch
  changes, or switch to a release wheel once one covers it.

## New batch sizes

Depthwise convolutions make cuDNN plan once per stage, forward and backward, for every unseen
image count. Backbone alone, forward and backward, 3090, BF16, measured 2026-10-13:

| Backbone | Cost per unseen count | Handling |
|---|---:|---|
| ConvNeXt-Tiny | 4.3 s | image counts padded to multiples of 16 (`CONVNEXT_IMAGE_COUNT_STEP`) |
| VMamba-T | 2.3 s | the same padding worked (LayerNorm only) |
| MaViT-S | 8 s | `channels_last: false`; BatchNorm rules out zero padding |
| MambaVision-T, Vim-S, ResNet-50 | none | |

Running the depthwise convolutions outside cuDNN removed only part of the cost; cuDNN off removed
all of it at 2 to 4 times the steady time.

## Training time

The ResNet-50 recipe with each backbone, ImageNet mean and std, 3090, 37 timed steps at batch 8 or
57 at batch 4 (two accumulated), 2026-10-13. Epoch is 442 steps of 8 studies, training only.

| Backbone | Batch | Per 8 studies | Epoch | Peak allocated |
|---|---:|---:|---:|---:|
| ResNet-50 | 8 | 0.49 s | 3.6 min | 15.4 GB |
| MambaVision-T | 8 | 0.46 s | 3.4 min | 11.2 GB |
| ConvNeXt-Tiny | 4 | 0.64 s | 4.7 min | 10.6 GB |
| VMamba-T | 4 | 1.07 s | 7.9 min | 15.5 GB |
| Vim-S | 4 | 1.59 s | 11.7 min | 11.9 GB |
| MaViT-S | 4 | 1.71 s | 12.6 min | 18.5 GB |

VMamba-T and MaViT-S do not fit the 3090 at batch 8. VMamba-T, MaViT-S and Vim-S were built and
timed only, never trained, and their code was removed on 2026-10-14; it was compiled from
[VMamba](https://github.com/MzeroMiko/VMamba), [MALA](https://github.com/qhfan/MALA) and
[Vim](https://github.com/hustvl/Vim). The full recipe trains in 3,106 s for twelve epochs at a
12.7 GB peak.

## Recipe

`configs/mambavision.yaml` is the ResNet-50 recipe with five changes, each from the
[MambaVision settings campaign](../../autoresearch/campaigns/mambavision-settings/INDEX.md):

- **Weight scale:** MambaVision's pretrained weights are 1.5 to 12.7 times larger (RMS) than the
  Refined-RadImageNet ResNet-50's, stage by stage (stem 1.47, then 3.67, 5.26, 6.94, 12.66). On the
  ResNet-50's settings the steps were too small relative to the weights and the anchors too strong,
  and the model underfit. So: one backbone rate of 2e-4 (from 5e-5) and anchors 0.001 early and
  1e-5 late (from 0.01 and 0.001).
- **Schedule:** twelve epochs (patience 12), linear decay to the same floor
  (`optimizer.lr_decay: linear`), and the late anchor scaled by the backbone rate over its peak
  (`optimizer.late_anchor_follows_lr: true`).
- **Inputs:** ImageNet mean and std, `pretrained_weights: timm`.
