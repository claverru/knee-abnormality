# FocalNet-T and EfficientViT-B2 backbones

FocalNet-T SRF (`focalnet_tiny_srf.ms_in1k`) trains from `configs/focalnet.yaml` and
EfficientViT-B2 (`efficientvit_b2.r224_in1k`) from `configs/efficientvit.yaml`, both timm models on
the MambaVision-T recipe ([MambaVision-T and stem-staged backbones](mambavision-backbone.md)).
Results: [2026-10-16 backbone families](../validation/results/2026-10-16-backbone-families.md).

## Configs

Both are `configs/mambavision.yaml` with the backbone changed and ten epochs, patience 10 (the
user's decision for FocalNet-T on 2026-10-16; the campaign incumbent trained twelve, ×4 0.938816
against 0.938154 at ten). EfficientViT-B2 adds `model.stochastic_depth: 0.2` and
`optimizer.anchor_alpha_late: 0.00003`.

## FocalNet-T

- **Stage adapter** (`FocalNetStages` in `modeling/backbone_stages.py`): `stem`, `layers[0]` and
  `layers[1]` to stride 8, `layers[2]` and `layers[3]` with the final `norm` after. Image counts
  are padded to multiples of 16 for the depthwise convolutions.
- **Pointwise convolutions as linear layers:** every 1×1 convolution in `layers` runs as a linear
  layer over the channels-last view, on the same weights and parameter names. It brought training
  from 4,250 s projected to 4,067 s for twelve epochs.
- **Token mapping** (`_focalnet_mapping` in `modeling/spatial_tokens.py`): walks the stem and the
  downsample projections and checks that no block convolution moves the anchors; origins as
  ConvNeXt's (15.5 px at stride 32).
- **Stochastic depth:** timm's FocalNet defaults to a drop-path rate of 0.1, so the model passes
  `model.stochastic_depth` (0.0) explicitly.
- **Early anchor:** `layers.0.` is in `EARLY_ANCHOR_PREFIXES`.

## EfficientViT-B2

- **Stage adapter** (`EfficientVitStages`): `stem`, `stages[0]` and `stages[1]` to stride 8,
  `stages[2]` and `stages[3]` after; the classifier head is replaced by the identity and there is
  no final norm. Image counts are padded to multiples of 16.
- **Real-row BatchNorm** (`RealRowBatchNorm2d`): every BatchNorm takes its batch statistics and its
  running-statistics update from the real images only (`native_batch_norm` on the first rows);
  padded rows are normalised with those statistics without gradient. Evaluation is plain BatchNorm.
- **Stochastic depth** (`EfficientVitResidualBlock`): timm's EfficientViT has no drop
  path, so the adapter wraps every residual block with an identity shortcut in a
  `DropPath` on its main branch, rates rising linearly from 0 to `model.stochastic_depth` over the
  blocks.
- **Activation checkpointing of stage 0** in training, `use_reentrant=False` with the random state
  preserved, so the drop masks repeat in the recompute. The recompute pass marks stage 0's
  BatchNorms as recomputing, which normalises with the batch statistics but leaves the running
  statistics alone. Without it the model runs out of the 3090's memory (backbone alone at 256 images:
  18.5 GB); with stage 0 checkpointed the preflight predicted 17.0 GiB and 0.662 s per step.
- **Attention in float32:** timm's LiteMLA as it is. The five-fold runs trained with a bfloat16
  variant of the two attention products and stage 0's first block checkpointed (0.604 s per step,
  20.5 GiB predicted); their checkpoints load into the merged model with every key matched.
- **Token mapping** (`_efficientvit_mapping`): walks the stem and each stage's downsample block and
  checks the other blocks keep the anchors.

Checked on CPU in float32 against the five-fold runs' source: two training steps with stochastic
depth 0.2 and padded image counts give identical outputs, gradients and BatchNorm running
statistics, and identical evaluation outputs.
