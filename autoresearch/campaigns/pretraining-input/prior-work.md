# Pretraining-input prior work

What is known before the first run. Run rows are one run each and weak evidence. Gap
measurements are label-free, made on 2026-10-04 with the frozen Refined-RadImageNet ResNet-10t;
their scripts are in `outputs/analysis/p/pretraining-gap-20261004/`.

## How the gap is measured

Our prepared middle slices go through the frozen pretrained ResNet-10t in evaluation mode with
the checkpoint normalization (mean 0.3162, std 0.3213). For every BatchNorm layer the mean and
variance of its input over the slices are compared with the stored running statistics (a
Gaussian KL per channel, averaged). Reported: the **stem gap** (first BatchNorm), the **overall
gap** (mean over all layers), the **stem variance** relative to the stored one (1.00 matches),
and the **knee-class mass** (probability the 165-class head puts on its 18 knee classes). Lower
gaps are closer to the pretraining statistics. The meter measures closeness, not AUC: the
normalization closed a large frozen-network gap for +0.0006 on the ResNet-18 and +0.00004 on the
ResNet-10t, because BatchNorm in training mode absorbs a global shift or scale of the input. A
difference between series types inside a batch is not absorbed.

## The pretraining recipe

From the authors' code and the settings stored in the checkpoints: torchvision's reference
classification script, 90 epochs, SGD; `RandomResizedCrop(224)` with the default scale 0.08–1 and
aspect 3/4–4/3, horizontal flip 0.5, AutoAugment with the ImageNet policy, normalization with
mean 0.3162 and std 0.3213; MixUp off (alpha 0.0); CutMix alpha 1.0 is set, but the published
loader builds the CutMix collate and then passes the plain one, so as published it is not
applied (the paper calls its baseline CutMix; which code trained the checkpoints is unknown).
Evaluation: resize to 256, centre crop 224. The stored BatchNorm statistics were collected on
augmented training images.

## Where the gap is

600 training slices, 100 per series type:

| Series type | Stem gap | Stem variance | Slice std |
|---|---:|---:|---:|
| Axial, non-fluid | 0.018 | 0.97 | 0.306 |
| Coronal, non-fluid | 0.067 | 0.76 | 0.272 |
| Sagittal, non-fluid | 0.048 | 0.79 | 0.277 |
| Axial, fluid | 0.181 | 0.42 | 0.200 |
| Coronal, fluid | 0.212 | 0.37 | 0.188 |
| Sagittal, fluid | 0.173 | 0.42 | 0.201 |

The fluid-sensitive, fat-suppressed series have about two thirds of the other series' contrast
and under half the pretraining variance at the stem.

## Intensity mapping

Loader upper quantile, 48 training studies through the real loader (1,296 slices):

| Upper quantile | Stem gap, all | Fluid: stem gap / variance / std | Non-fluid: stem gap / variance / std |
|---|---:|---|---|
| 0.999 | 0.122 | 0.334 / 0.28 / 0.163 | 0.053 / 0.70 / 0.261 |
| 0.995 (incumbent) | 0.058 | 0.162 / 0.43 / 0.203 | 0.044 / 0.80 / 0.278 |
| 0.99 | 0.033 | 0.085 / 0.54 / 0.229 | 0.045 / 0.85 / 0.287 |
| 0.98 | 0.020 | 0.031 / 0.69 / 0.257 | 0.052 / 0.92 / 0.297 |

Lower quantile 0.02: no change. On prepared slices (600 slices):

| Variant | Stem gap, all | Fluid stem gap / variance | Non-fluid stem gap / variance |
|---|---:|---|---|
| Current preparation | 0.044 | 0.180 / 0.41 | 0.021 / 0.88 |
| Per-slice upper quantile 0.99 | 0.018 | 0.063 / 0.59 | 0.025 / 0.99 |
| Per-slice upper quantile 0.98 | 0.015 | 0.026 / 0.71 | 0.033 / 1.05 |
| Per-slice upper quantile 0.95 | 0.037 | 0.027 / 0.90 | 0.056 / 1.16 |
| Contrast gain 1.15 / 1.3 (around the slice mean) | 0.021 / 0.014 | — | — |
| Contrast gain 0.85 | 0.095 | — | — |
| Brightness ×1.5 | 0.034 | 0.014 / 0.84 | 0.194 / 1.28 |
| Histogram equalization | 0.068 | 0.091 / 0.94 | 0.050 / 0.94 |
| Half-way to equalized | 0.025 | 0.046 / 0.58 | 0.029 / 0.85 |
| Per-slice autocontrast | 0.041 | 0.174 / 0.41 | 0.021 / 0.89 |
| Gamma 0.8 / 1.25 | 0.040 / 0.069 | — | — |

The fluid series' scale is set by a few very bright pixels; clipping the bright end harder, per
series or per slice, is what closes their gap. Brightening helps the fluid series and overshoots
the others, so it is a per-type setting.

## The pretraining augmentation on our slices

| Our slices after | Knee-class mass | Stem gap, all | Overall gap | Fluid stem gap | Non-fluid stem gap |
|---|---:|---:|---:|---:|---:|
| Nothing | 0.961 | 0.044 | 0.033 | 0.180 | 0.021 |
| Their AutoAugment | 0.953–0.957 | 0.015–0.017 | 0.023–0.024 | 0.017–0.019 | 0.023 |
| Their full training transform | 0.799 | 0.019 | 0.023 | 0.018 | 0.029 |
| Their random resized crop alone | 0.838 | 0.117 | 0.059 | 0.274 | 0.086 |
| Their evaluation crop (1.14 zoom) | 0.956 | 0.081 | 0.045 | 0.225 | 0.056 |
| CutMix with any slice | 0.940 | 0.033 | 0.030 | 0.107 | 0.014 |
| CutMix within the series type | 0.946 | 0.035 | 0.030 | 0.170 | 0.017 |
| MixUp alpha 0.2 / 1.0 | 0.955 / 0.942 | 0.070 / 0.131 | 0.038 / 0.053 | — | — |
| Rotation 15° / 30° | 0.949 / 0.922 | 0.032 / 0.038 | 0.024 / 0.019 | — | 0.006 / 0.004 |
| Width stretch 4/3 | 0.945 | 0.073 | 0.040 | — | — |
| Horizontal flip, JPEG 60–90, 5-bit posterize | 0.956–0.972 | 0.043–0.045 | 0.027–0.034 | — | — |

AutoAugment's single operations on our slices: contrast ×1.5 stem gap 0.017 (variance 1.00),
brightness ×1.5 0.034, sharpness 0.037, equalize 0.082, posterize 0.062, autocontrast 0.055;
contrast ×0.5 0.349, brightness ×0.6 0.434, solarize 0.327, invert 0.259.

## Our augmentations, each alone and always on

32 training studies, captured right before the backbone:

| Augmentation | Knee-class mass | Stem gap | Overall gap | Late-layer gap |
|---|---:|---:|---:|---:|
| None | 0.959 | 0.060 | 0.037 | 0.044 |
| Intensity (gamma, contrast) | 0.960 | 0.061 | 0.035 | 0.041 |
| Affine | 0.961 | 0.063 | 0.031 | 0.037 |
| Noise | 0.953 | 0.053 | 0.039 | 0.041 |
| Resolution | 0.958 | 0.061 | 0.037 | 0.043 |
| Bilateral smoothing | 0.970 | 0.078 | 0.052 | 0.077 |
| Contrast mixing | 0.962 | 0.083 | 0.041 | 0.043 |
| GIN | 0.773 | 0.064 | 0.030 | 0.034 |

Smoothing raises the late-layer gap by 75%, contrast mixing lowers the contrast (stem variance
0.58 against 0.66), and under GIN the frozen network stops reading a fifth of the slices as knee
MRI.

## Field of view

Every view centred at one width (48 studies): the incumbent's 120/84/93 mm views stem gap 0.058
and overall gap 0.037; all 120 mm 0.061 and 0.042; all 140 mm 0.070 and 0.042; all 160 mm 0.094
and 0.044. Wider views are no closer. The pretraining crop zooms in by about 1.5 on average, so
a full knee slice is seen at roughly the width of our 84/93 mm crops. The views are fixed in
this campaign.

## Mask-trained (heated) weights

- **Previous campaign, under the ImageNet normalization:** heated ResNet-10t ×4 0.928777 and
  `_FT` 0.928671 against 0.927543 for the standard weights (+0.0012 and +0.0011, 95% intervals
  including zero), with backbone LR 0.0002 and L2-SP 0.00088 / 0.000088 set from the weights'
  scale (convolution RMS 0.0531 against 0.0158). The two checkpoints trained identically (same
  losses to three decimals), so one of them is enough. Both dipped in epoch 1 (−0.006 on ×1) and
  were ahead from epoch 5.
- **At the start, frozen, on our slices:** linear-probe macro AUC of the untouched features
  0.808 standard, 0.807 heated, 0.805 `_FT` (320 studies): equally informative. Share of final
  features that are zero: 48% standard, 88% heated and `_FT` (ResNet-50: 38% and 78%); epoch-1
  gradient norms about half the standard runs'.
- **Input strength:** the heated checkpoint was trained on normalized images multiplied by a
  mask of mean 0.5. Frozen, it reads our slices as knee MRI with mass 0.223 at full strength and
  0.987 at half strength. In training this makes no difference: two ResNet-50 runs at full and
  half strength had identical losses, because BatchNorm absorbs the scale.

## The mask procedure in our training (previous campaign, ResNet-18)

| Run | ×4 Δ | Note |
|---|---:|---|
| Policy mask, ResNet-10t policy from scratch, all epochs | −0.0023 | The policy stayed uniform: mean Beta parameters 1.000, policy gradient norm 0.0001 |
| Policy mask, policy from pretrained ResNet-10t | −0.0026 | Uniform again, no dependence on the image |
| Random smooth mask (uniform cells, mean 0.5) | −0.0023 | The same mask without a policy |
| Policy mask for six epochs, then four without | −0.0017 | |

The three all-epoch runs are one experiment three times: 0.930766, 0.930468, 0.930733, a
run-to-run spread of 0.0003. The authors' loss weights the reward by `logsumexp` over the 4,096
cells of the log-density and uses no baseline; at a uniform start its gradient per cell
averages to zero. A policy with a usable learning signal, and a mask whose mean is 1, were not
run.

## Other results of the previous campaign

Recorded in [its run notes](../refined-radimagenet/run-notes.md): slice teacher +0.0003 to
+0.0004 for ten or six epochs and −0.0001 for one epoch (Lateral OA +0.003 to +0.007 in all
three); knee-logit attention prior +0.0003; backbone in eval mode −0.0031; Cross-D static
kernels −0.0039; gated adjacent-slice stems −0.0003 and +0.0001 against their bases. An
incumbent replicate through a different code path differed by 0.00003 on ×4 and ±0.0012 per
label.

## Earlier augmentation results

On ReXNet-200 with ImageNet weights ([augmentation campaign](../augmentation/findings.md)):
image MixUp +0.0006, token MixUp −0.0001, pooled-output MixUp +0.0007, none separated; bilateral
smoothing, DSU and contrast mixing adopted there. On the ResNet-18 before the normalization fix
([revisits](../revisits/prior-work.md)): removing bilateral smoothing and resolution degradation
did not pass; GIN replaced by a 13× weaker augmentation −0.0015 on the sites-out split.
AutoAugment and CutMix were never run in our fine-tuning.
