# Backbone families 2 campaign contract

Written 2026-10-17 at the user's request; a new campaign after the
[first round](../backbone-families/INDEX.md), whose runtime state was deleted.

## Objective

Find a backbone family that adds to the ensemble with the ResNet-50 more than FocalNet-T SRF does. Each
forced seed brings in one family from the user's list of 2026-10-17 with the starting settings its own
properties call for ([prior work](prior-work.md#the-families)); later ideas tune those families' adapters
and settings.

The reference is `outputs/training/focalnet-silver-fold-2`: FocalNet-T SRF on `configs/focalnet.yaml` at
twelve epochs (`configs/campaign.yaml`), fold 2 of `all_5fold_scanner_v2`, seed 1110484336, batch 8;
×4 0.938816, ×1 0.933241, the first round's incumbent. Blends are read with
`outputs/training/resnet50-silver-fold2` (the ResNet-50 on its silver recipe, same fold and seed, ×4 0.937820).

Six forced seeds in this order, then the researcher without end:

1. `overlock-xt`: OverLoCK-XT, its natten window op replaced by an exact dense PyTorch form, the recipe's
   settings, BatchNorm statistics from the real images.
2. `transnext-micro`: TransNeXt-Micro through its pure-PyTorch aggregated attention, the recipe's settings.
3. `vssd-tiny`: VSSD-Tiny (MESA checkpoint), the recipe's settings, BatchNorm statistics from the real
   images, stem and stage 0 checkpointed.
4. `medsam2-hiera-tiny`: SAM2.1 Hiera-T with the MedSAM2 trunk weights, rate 1e-4, anchors 0.01 and 0.0003.
5. `lingbot-vit-small`: LingBot-Vision ViT-S/16 through timm's DINOv3 ViT, the recipe's settings.
6. `flash-internimage-tiny`: FlashInternImage-T through the DCNv4 CUDA op in float32, the recipe's settings
   (last at the user's order: its op must be built first).

Each seed is compared with the incumbent at the time.

## Settled

- **The six families and their checkpoints** ([prior work](prior-work.md#the-families)). No other family,
  size or weights. MobileCLIP2-S2 was dropped by the user.
- **Added by the user on 2026-10-17:** the original RadImageNet DenseNet-121, InceptionV3 and
  InceptionResNetV2 (Keras weights converted by the keeper,
  [RadImageNet prior work](prior-work-radimagenet.md)), each loaded only from its keeper-converted
  file, with its input in [0, 1] and settings from its weight scale. They are families of this
  campaign like the six; if one did not fit the time bound the user allowed a smaller batch or a
  longer bound for it (all three fit as measured).
- **Added by the user on 2026-10-18:** FocalNet-T LRF, NextViT-S (SSLD), EfficientNet-B3 (JFT noisy student),
  TinyViT-21M (IN-22k), InceptionNeXt-T, HGNetV2-B4 (SSLD) and MambaOut-T with their timm weights
  ([candidates](prior-work-candidates.md)); and in a second batch TResNet-M, CS3DarkNet-L, GhostNetV2-160, Twins-SVT-S,
  Res2Net50d and BiT ResNetV2-50x1 (same document).
- **The recipe around the backbone stays:** labels, views, slices, 224 px, augmentations, the OA detector,
  routing, heads and losses; the reads at stride 16 and 32 and the feature-statistics augmentation at the
  stride-8 point; a plain ViT (LingBot), which has no stride-8 map, takes the augmentation at its stride-16
  map after block 3, as the RadioDINO-S/16 adapter did.
- **Varying image counts are handled per family:** families with depthwise convolutions pad the image count
  to a multiple of 16; a BatchNorm family takes its batch statistics from the real images alone.
- **Prediction settings are not subjects:** ×1 validation and ×4 prediction as the recipe has them.
- **Batch 8, one optimizer step per batch.** Epochs are a family setting, ten by default; the reference's
  twelve is FocalNet-T's own.
- **Kernels:** compiled extensions are prepared by the keeper outside the candidates; a candidate never
  builds or installs anything. Every new operator runs in float32 and float16 (Kaggle's T4s).

## Principles

1. **One setting per run**, applied to the incumbent or, for a family's own setting, to that family's best
   run. A run that changes two settings names the two runs it combines.
2. **Starting settings come from the family's properties**: weight scale per stage (the relative step of the
   rate and the cost of the anchors), its normalisation layers, its depthwise convolutions and its
   pretraining. Values come from a cited source or an earlier run, never from a sweep on the validation
   labels.
3. **A run can be proved active** by metrics it names, at most six.
4. **A superseded behaviour is removed**, not kept behind an option, except for values the campaign varies;
   the existing backbones build and behave exactly as before.

## How a result is read

One run is weak evidence. A run is read at ×4 and ×1 against the incumbent and the reference, on the training
loss per epoch, per label from largest gain to largest loss, the training time and the peak memory, and as an
equal logit blend with the ResNet-50 against the reference's blend (+0.003179 over the ResNet-50 alone): the
purpose is a family that adds to the blend.

## Adoption

By the runtime under the [shared rule](../../protocol.md#shared-evaluation-and-adoption).

## Out of bounds

Other backbone families, model sizes or pretrained weights than the six, the three RadImageNet CNNs and the thirteen of 2026-10-18; ensembling, weight averaging, test-time augmentation; image sizes other than 224 px; more slices or images per
study; the labels, the split, distillation; new routing, attention or readout structure downstream of the
backbone's maps; reflection probabilities; gold-study weighting; trained models in the loader; multi-seed,
confirmation or control runs not asked for by the user; choosing anything on the campaign fold's validation
labels; Kaggle submissions.

## Cost

**Training time 1,684 to 4,881 s** (0.414 to 1.20 of the reference's 4,067 s), up to 120% of the
reference's time as the user set it on 2026-10-18 (the campaign started at 1.006, the first round's
window). The preflight measures each candidate on real batches and stops one projected over the time bound or
22.5 GiB.

## Duration

Started with `start` by the user; no end.
