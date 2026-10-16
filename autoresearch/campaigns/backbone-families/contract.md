# Backbone families campaign contract

Written 2026-10-15 at the user's request; started the same evening with Codex.

## Objective

Find a backbone family, different in kind from the ResNet-50, ConvNeXt-T, MambaVision-T, Swin-T and
RadioDINO-S/16 already tried, that matches MambaVision-T on the current recipe and adds to the blend
with the ResNet-50. The campaign mixes backbones: each forced seed brings in one family with the
starting settings its own properties call for ([prior work](prior-work.md#the-families)); later ideas
tune the families' adapters and settings.

The reference is `outputs/training/mamba-silver-fold-2`: MambaVision-T on the current recipe
(`configs/mambavision.yaml`, no mask decoder), fold 2 of `all_5fold_scanner_v2`, seed 1110484336,
twelve epochs, batch 8; ×4 0.935160, ×1 0.930081. It was trained on the rented machine from a
checkout with GPU-synchronisation speedups and no change of behaviour
([prior work](prior-work.md#the-reference)).

Ten forced seeds in this order, then the researcher without end:

1. `sam2-hiera-tiny`: SAM2.1 Hiera-T, rate 1e-4, anchors 0.01 and 0.0001.
2. `hrnet-w18`: HRNet-W18, rate 5e-5, anchors 0.005 and 0.0003.
3. `pvt-v2-b2`: PVTv2-b2 on the recipe's settings.
4. `focalnet-tiny-srf`: FocalNet-T SRF on the recipe's settings.
5. `rdnet-tiny`: RDNet-T on the recipe's settings.
6. `dinov3-convnext-tiny`: DINOv3 ConvNeXt-T, rate 5e-5, anchors 0.01 and 0.0002 (added at the user's request
   on 2026-10-15, after the start).
7. `davit-tiny`: DaViT-T on the recipe's settings.
8. `edgenext-base`: EdgeNeXt-B (ImageNet-21k) on the recipe's settings.
9. `efficientvit-b2`: EfficientViT-B2, padded with BatchNorm statistics from the real images only, rate 2e-4,
   anchors 0.001 and 0.00003.
10. `wtconvnext-tiny`: WTConvNeXt-T through the official Triton wavelet kernels, on the recipe's settings.

Seeds 7 to 10 were added at the user's request on 2026-10-15, after the start. Relaunches of seeds that
failed on the harness (`pvt-v2-b2-retry`, `rdnet-tiny-retry`) are the same seeds.

Each seed is compared with the incumbent at the time, so after an adoption the later families are
compared with the adopted family.

**From 2026-10-16 09:20 CEST, by the user's decision, only EfficientViT-B2 ideas run**; the other families
are closed (results in run-notes.md).

## Settled

- **The ten families and their checkpoints** (prior work): timm's, and the official WTConvNeXt-T checkpoint. No
  other family, size or weights.
- **The recipe around the backbone stays:** labels, views, slices, 224 px, augmentations, the OA
  detector, routing, heads and losses; the reads at stride 16 and 32 and the feature-statistics
  augmentation at the stride-8 point.
- **Varying image counts are handled per family:** families with depthwise convolutions and no
  BatchNorm pad the image count to a multiple of 16 as ConvNeXt does; a BatchNorm family is zero-padded only
  when its BatchNorm takes batch statistics from the real images alone.
- **Prediction settings are not subjects:** ×1 validation and ×4 prediction as the recipe has them.
- **Batch 8, one optimizer step per batch.** Epochs are a family setting: ten by default, as the ResNet-50 and
  ConvNeXt recipes have; twelve was MambaVision-T's own recipe (the reference and the first FocalNet-T run
  used it). Changed on 2026-10-15 at the user's request after the start.

## Principles

1. **One setting per run**, applied to the incumbent or, for a family's own setting, to that family's
   best run. A run that changes two settings names the two runs it combines.
2. **Starting settings come from the family's properties**: weight scale per stage (the relative step
   of the rate and the cost of the anchors, as set for MambaVision-T and checked on RadioDINO), its
   normalisation layers, its depthwise convolutions and its pretraining. Values come from a cited
   source or an earlier run, never from a sweep on the validation labels.
3. **A run can be proved active** by metrics it names, at most six.
4. **A superseded behaviour is removed**, not kept behind an option, except for values the campaign
   varies; the existing backbones build and behave exactly as before.

## How a result is read

One run is weak evidence. A run is read at ×4 and ×1 against the incumbent and the reference, on the
training loss per epoch, per label from largest gain to largest loss, and as an equal logit blend with
the ResNet-50 (targets-and-inputs run 27, ×4 0.939634) and with the reference against each alone: the
purpose is a family that adds to the blend.

## Out of bounds

Other backbone families, model sizes or pretrained weights than the ten;
ensembling, weight averaging, test-time augmentation; image sizes other than 224 px; more slices or
images per study; the labels, the split, distillation; new routing, attention or readout structure
downstream of the backbone's maps; reflection probabilities; gold-study weighting; trained models in
the loader; multi-seed, confirmation or control runs not asked for by the user; choosing anything on
the campaign fold's validation labels; Kaggle submissions.

## Cost

**The budget is 0.35 to 0.85 of the reference's training time** (4,812 s), so 1,684 to 4,090 s.
The reference trained on a slower machine; a local MambaVision-T run takes about 3,100 s, so the
window is about 0.54 to 1.32 of a local MambaVision-T run.

## Duration and adoption

Started with `start` by the user; no end. Adoption is by the runtime under the
[shared rule](../../protocol.md#shared-evaluation-and-adoption).
