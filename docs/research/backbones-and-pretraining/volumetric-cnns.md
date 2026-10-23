# 3D CNNs for knee abnormality detection

A design brief prepared on 2026-09-16 for testing genuine volumetric ResNet and EfficientNet
backbones against the slice/triplet solution, judged on study-level macro ROC AUC and end-to-end
inference cost. **It was never run**: no volumetric model was trained, no throughput was measured
and no third-party constructor was smoke-tested in the project environment. Library and code
facts were read from upstream source on 2026-09-16 (MONAI from its `dev` branch, so behavior in a
pinned release must be rechecked) and from the repository then. A related fused-cube idea is in
the [big-bets backlog](../../backlog/untried-ideas.md#inputs-beyond-the-2d-recipe).

## Decision and evidence boundary

The brief proposed `timm_3d` ResNet18 and EfficientNet-B0 with inflated 2D pretrained weights, one
compact volume per retained acquisition, and study-level feature aggregation, with MONAI as an
independent 3D implementation and a MedicalNet-pretrained ResNet comparison. It was a proposal,
not a reconstruction of another competitor's solution; whether any competitor used volumetric
CNNs, which library or which initialization was not verified.

The existing model consumed adjacent-slice triplets, extracted 2D spatial features and applied
target-specific study MIL with physical-coordinate information. A 3D positional embedding is not a
3D convolutional encoder. The output contract is twelve study logits with observation masks. Labels,
target order, folds, missing-label handling and study-level supervision were to stay fixed, with a
distinct volume input mode rather than a reinterpretation of `adjacent_triplets` checkpoints.

## Library facts

| Library | Relevant models | Initialization | Proposed role |
|---|---|---|---|
| `ZFTurbo/timm_3d` / `timm-3d` | ResNet family; EfficientNet V1/V2 | Converts supported 2D checkpoints into 3D weights during loading | First PyTorch experiment |
| MONAI | 3D ResNets; EfficientNet B0–B8/L2 | Compatible ResNets load MedicalNet weights; built-in EfficientNet ImageNet loading is 2D-only | Medical-pretraining and implementation controls |
| `ZFTurbo/classification_models_3D` | ResNet and EfficientNet variants | Converted 2D ImageNet weights | Keras 3 alternative, not needed |
| `ZFTurbo/segmentation_models_pytorch_3d` | ResNet/EfficientNet encoders | Encoder pretrained-weight configurations | Secondary source; no segmentation decoder |

`timm_3d` identifiers: first `resnet18.a1_in1k` and `tf_efficientnet_b0.in1k`; later
`resnet10t.c3_in1k`, `tf_efficientnetv2_b0.in1k`, `efficientnetv2_rw_t.ra2_in1k`. The registry has to
be queried in the installed version, and a package would enter through the dependency/lockfile and
offline-wheel workflow.

## Initialization: three different experiments

- **Inflated 2D weights.** The `timm_3d` loader adapts input convolutions to the channel count,
  removes the classifier and inflates eligible 4D conv weights to 5D by repeating the kernel along
  the new last spatial axis, dividing by the repetition count (`W3[..., z] = W2 / kZ`). The depth
  axis starts as an average; this transfers 2D features and is not pretraining on 3D MRI. Native 3D
  weights loaded at inference must not be inflated a second time.
- **MONAI EfficientNet trap.** `EfficientNetBN` loads weights only when `pretrained` is true and
  `spatial_dims == 2`, so `spatial_dims=3, pretrained=True` silently gives no pretrained weights;
  the scratch control needs `pretrained=False` explicitly.
- **MONAI MedicalNet ResNet.** MedicalNet loading needs compatible 3D, single-channel, headless
  configurations (`shortcut_type`, `bias_downsample`, `feed_forward=False`, via
  `get_medicalnet_pretrained_resnet_args(depth)`). It should be compared with the same MONAI
  architecture initialized randomly with identical strides, not with a different `timm_3d` ResNet.

## Tensor and encoder contract

Internal convention `[N_valid, C, Z, Y, X]`, with features `[N_valid, F]`, study features
`[B, V_max, F]` plus a `volume_valid` mask, and logits `[B, 12]`. `Z` is ordered through-plane
position within one acquisition. Acquisitions in different planes or contrasts are separate volumes,
not depth segments or registered channels. `timm_3d` expects `[B, C, H, W, D]`, so an explicit
permutation `[N,C,Z,Y,X] -> [N,C,Y,X,Z]` is needed; it must be tested with unequal dimensions and an
axis-coded phantom, since cubic random tensors cannot reveal a swap. The encoder exposes
`num_features` after removing the classifier with global average pooling, and the projection width
comes from that probed value.

## Volume construction and geometry

Each acquisition is encoded in its native plane and fused afterwards; no common isotropic study cube
and no cross-sequence registration. A volume is built from physically ordered unique slices (position
projected on the slice normal; an instance-number fallback must not be called verified geometry),
checking orientation consistency, duplicates, gaps and decode failures. The prototype depth policy is
interpolation onto a regular grid spanning the accepted coverage, recording coverage and effective
spacing; gap tolerances were to come from the data audit.

In-plane field of view stays that of the control when pixel resolution changes: 224 px at 0.625 mm is
140 mm, which at 160 px implies 0.875 mm per pixel. The proposed initial grid was `Z=32, Y=X=160`,
not a measured optimum. Normalization must be explicit: one scalar normalization for a one-channel
input, or the volume repeated into three channels with the checkpoint's per-channel normalization as
an initialization control. Absent acquisitions are not encoded as zero placeholders; valid volumes
are packed before the CNN, with a defined all-missing fallback.

## Aggregation and augmentation

Per-volume vectors enter the target-specific study attention with plane/contrast metadata and the
valid-volume mask; the existing spatial-token path depends on 2D feature anchors and is not reused
unchanged. Supervision applies after aggregation; study-positive labels are not copied to every
series. One geometric transform applies across all slices of a volume, series dropout cannot remove
all evidence, and unported 2D spatial augmentations are disabled explicitly.

## Depth downsampling and optimization

With five factor-two depth reductions a 32-slice input reaches one depth cell, so the schedule is an
ablation (in-plane-only stride `(1,2,2)` in `[Z,Y,X]`, `(2,2,1)` inside the `timm_3d` layout; changing
stride keeps weight shapes but changes computation). Pilot settings proposed, not verified optima:
pretrained-backbone LR 1e-4, head LR 3e-4, weight decay 0.01, one warmup epoch, clipping 1.0, against
the baseline's 2e-4 and 6e-4. BatchNorm statistics come from each forward, so gradient accumulation
does not combine them; updating versus frozen pretrained BN statistics would be compared when the
volume microbatch is small, and random initial BN statistics are not frozen for scratch runs.

## Proposed experiment table (not run)

| Run | Experiment | Question |
|---|---|---|
| C0 | Existing 2D/2.5D control, unchanged | Reference AUC and deployment cost |
| V1 | `timm_3d` ResNet18, inflated weights, 32×160×160 | Does a small volumetric encoder merit work? |
| V2 | `timm_3d` EfficientNet-B0, same recipe | Backbone-family comparison |
| V3 | Better of V1/V2, random initialization | Value of inflation |
| M0/M1 | Identical MONAI ResNet18, random vs MedicalNet | Value of medical pretraining |
| V4 / V5 / V6 | Better volume model: depth only / in-plane resolution only / early depth preserved | Coverage, detail and downsampling versus latency |
| Later | ResNet10t or compact EfficientNetV2 | Accuracy/runtime trade-offs |

C0 versus V1 is a pipeline comparison, not a pure dimensionality ablation, because sampling and
pooling also change; a 2D encoder control on the same volume slices would attribute it better.

## Efficiency and deployment

Dense 3D convolution cost scales with `output_Z·Y·X·Cin·Cout·kZ·kY·kX`, so a depth kernel multiplies a
layer's cost by `kZ`; this is operation-count reasoning, not wall-clock. Timing has to cover the full
submission-like pipeline (reads, decoding, resampling, transfer, encoder, aggregation, loading,
writing) on the target GPU. At inference the architecture is built with external pretraining
downloads disabled and the trained 3D state dict loaded from the bundle, with the library and custom
axis/stride code in the offline bundle. No throughput or leaderboard gain is claimed.

## Sources

Upstream sources read on 2026-09-16: `timm_3d`
[README](https://github.com/ZFTurbo/timm_3d/blob/main/README.md),
[model list](https://github.com/ZFTurbo/timm_3d/blob/main/docs/models_list.md) and
[loader](https://github.com/ZFTurbo/timm_3d/blob/main/timm_3d/models/_builder.py); MONAI
[EfficientNet](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/efficientnet.py) and
[ResNet](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/resnet.py);
[classification_models_3D](https://github.com/ZFTurbo/classification_models_3D);
[segmentation_models_pytorch_3d](https://github.com/ZFTurbo/segmentation_models_pytorch_3d);
PyTorch [Conv3d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv3d.html) and
[BatchNorm3d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm3d.html).
