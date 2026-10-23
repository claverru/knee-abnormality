# EfficientNet training

Research and source-code inspection of pretrained `timm` 2D EfficientNets used with adjacent-slice
triplets and study-level multilabel aggregation, prepared on 2026-09-16 to diagnose assumed poor
convergence. It was **superseded on 2026-09-18 by executed runs**, reported in
[EfficientNet backbone results](../../validation/results/backbones/2026-09-18-efficientnet-backbone-results.md).
It was not a recipe for a randomly initialized 3D EfficientNet. The code facts below describe the
repository at the 2026-09-16 inspection (`configs/baseline.yaml`, since renamed `configs/resnet50.yaml`).

## Outcome: the premise did not hold (2026-09-18)

This report was written to diagnose **poor convergence**. Three executed runs show no convergence failure: EfficientNet-B3 and B0 train normally and reach 0.879449 and 0.880331 fold-0 macro AUC against 0.888476 for a matched ReXNet-200 control. They converge to a worse point, they do not fail to converge.

The following proposals of the original report depended on the convergence premise and were withdrawn:

- **The proposed backbone LR comparison (2e-4 against 5e-5) and the frozen-BN policy both point the wrong way.** EfficientNet **underfits** here: its training loss is worse than the control's at nearly every epoch while its generalization gap is *smaller*. Both proposals reduce backbone adaptation. Neither was run. Supporting evidence on ReXNet-200 from the same queue: `12-backbone-lr-1e-4` −0.0030, `16-backbone-eval` −0.0088 ([overnight results](../../validation/results/campaigns-and-queues/2026-09-18-overnight-results.md)).
- **Doubling the rate to 4e-4 instead gained +0.0029, below the 0.003 single-run threshold, and not through the predicted mechanism** — training loss barely moved, so the gain is not faster fitting.
- **The startup checks and the tiny-subset fitting test were not needed.** Pretrained loading, normalization match and feature geometry were confirmed from the run artifacts, and the model fits.
- **The feature-geometry integration blocker (below) was resolved.** `_efficientnet_mapping` is implemented and non-`tf_` EfficientNets share ReXNet-200's anchors exactly. The `Conv2dSame` warning in that section was correct and is now measured: `tf_efficientnet_b0.ns_jft_in1k` walks to origin 51.0.
- **`backbone_eval` is exact, not a proxy, for these backbones.** The report called it a "broader proxy" and asked for a BN-only helper. `efficientnet_b3.ra2_in1k` contains 78 BatchNorm layers and **zero** Dropout or DropPath modules, so `.eval()` on the backbone changes BatchNorm behaviour and nothing else. The helper is unnecessary here; it would be needed for a backbone carrying drop-path.

What held: the checkpoint-preprocessing match (followed, and the runs confirm the recorded mean/std) and the pretrained stem and feature contract, both below.

## Repository facts at inspection

| Area | Behavior | Consequence |
|---|---|---|
| Baseline | `rexnet_200.nav_in1k`, 224-pixel images, adjacent triplets, study batch size 8, BF16 mixed precision, 10 epochs | Controls for the backbone comparison |
| Normalization | Mean/std configured explicitly and applied inside `encode_batch`, not resolved from the checkpoint | Changing only the backbone name can keep the wrong normalization |
| Backbone input | Valid triplets of the study batch are flattened and encoded together; padded entries are excluded | The BatchNorm batch is the number of encoded triplets, not the study batch size |
| Backbone features | `forward_features` returns spatial maps; the projection width is derived from the observed channels | Do not hard-code an EfficientNet width |
| Optimizer | AdamW, separate backbone/head LRs, no decay for parameters with fewer than two dimensions | BN affine parameters and biases already avoid weight decay |
| Mode control | `backbone_eval=true` reapplies `.eval()` to the backbone; it does not set `requires_grad=False` | Freezes BN statistics; exact for backbones without drop-path (see outcome) |
| Checkpointing | Activation checkpointing is rejected unless `backbone_eval=true` | Kept off in the EfficientNet comparison |

**Spatial feature geometry.** The mapping registry supported only ReXNet-200 and one CoAtNet, so an
EfficientNet name raised an unsupported-mapping error; a registry entry alone would not do, because
the traversal dispatched between CoAtNet and ReXNet. Copying ReXNet's origin because both give a
stride-32 map would be wrong: `timm.layers.Conv2dSame` reports zero convolution padding but pads
dynamically in forward, so a walker that reads `layer.padding` misreads it
([timm `conv2d_same.py`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/conv2d_same.py)).
The mapping was later implemented; `tf_efficientnet_b0.ns_jft_in1k` walks to origin 51.0.

## Match preprocessing to the exact checkpoint

| Checkpoint | Mean | Standard deviation | Declared input / test size |
|---|---|---|---|
| `efficientnet_b3.ra2_in1k` | `[0.485, 0.456, 0.406]` | `[0.229, 0.224, 0.225]` | 288 / 320 |
| `tf_efficientnetv2_s.in21k_ft_in1k` | `[0.5, 0.5, 0.5]` | `[0.5, 0.5, 0.5]` | 300 / 384 |

Both declare `fixed_input_size=false`, so the listed sizes are checkpoint recipes, not mandatory MRI
resolutions. The baseline normalization matched the first row only. `timm.data.resolve_model_data_config`
gives the checkpoint settings; they must be stored in the experiment configuration and carried into
inference rather than re-resolved from mutable remote metadata. The MRI intensity and
physical-geometry pipeline was kept, with the checkpoint's mean/std applied after the conversion to a
unit-scale tensor and no ImageNet center crop. The model's normalization buffers were non-persistent
and built from configuration, so the resolved values have to be persisted. Keras EfficientNet takes
`[0,255]` inputs because preprocessing is inside it; do not carry that convention into `timm`.

## BatchNorm semantics that mattered

- In training mode BN normalizes with the current minibatch; evaluation mode uses stored estimates.
  `momentum=0` stops running-statistic updates but does not make training use stored statistics, and
  `track_running_stats=false` uses batch statistics at evaluation too.
- `.eval()` changes module behavior without freezing parameters; disabling gradients does not put BN
  in evaluation mode. "Frozen-statistics BN" (stored statistics, affine parameters trainable) is not
  fully frozen BN.
- Gradient accumulation does not merge forward passes into a larger BN batch. Many triplets from few
  studies and mixed acquisition types was the hypothesis for unstable BN batches.
- The `timm` builder converts TensorFlow retention 0.99 to PyTorch momentum 0.01 and uses epsilon
  1e-3 as the TensorFlow reference; BN epsilon is distinct from AdamW epsilon.
- `efficientnet_b3.ra2_in1k` has 78 BatchNorm layers and no Dropout or DropPath modules, which is why
  `backbone_eval` was exact for it.

## Stem and feature contract

The three-channel triplet input kept the pretrained stem; a random convolution is not a backbone
swap. `timm` input-convolution adaptation sums RGB weights for one channel and repeats and rescales
for other counts. `forward_features` output is not interchangeable with the last element of a
`features_only=True` extraction (the V2-S model card shows different channels), so projection
dimensions come from the selected output. A failed download or state-dict load must not fall back to
random initialization silently.

## Soft-label BCE floor

With soft targets BCE does not approach zero. For observed entries `O` the lower bound is
`mean over (i,k) in O of -y log y - (1-y) log(1-y)` (with `0 log 0 = 0`); measured BCE minus this
floor is a fit diagnostic, not a guarantee the architecture can reach it.

## Proposed settings and their fate

The report proposed AdamW with backbone LR 2e-4 against 5e-5, head LR 6e-4, weight decay 0.01,
betas (0.9, 0.999), epsilon 1e-8, one warmup epoch, step-based cosine to a 0.01 minimum ratio,
clipping 1.0, BF16, 224 px and 10 epochs, with a four-run factorial of backbone LR against
BN policy. The executed runs instead showed underfitting; the backbone-LR halving and frozen BN
pointed the wrong way, and doubling the rate to 4e-4 gained +0.0029 (see the outcome).

## Sources

Official `efficientnet_b5.sw_in12k_ft_in1k` model card (AdamW, clipping, EMA, cosine with warmup);
Keras EfficientNet fine-tuning example (preprocessing, BN and optimizer cautions);
[`efficientnet_b3.ra2_in1k` config](https://huggingface.co/timm/efficientnet_b3.ra2_in1k/blob/main/config.json),
[`tf_efficientnetv2_s.in21k_ft_in1k` config](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k/blob/main/config.json);
PyTorch BatchNorm2d and autograd documentation; the `timm` EfficientNet builder
([`_efficientnet_builder.py`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_efficientnet_builder.py)) and
input-convolution adaptation ([`_manipulate.py`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_manipulate.py)).
They support mechanisms and checkpoint properties, not MRI-specific optima.
