# MambaVision settings campaign contract

Written 2026-10-13 at the user's request. Started 2026-10-13 23:05 CEST.

## Objective

Raise the macro AUC of MambaVision-T on the current recipe by tuning the training settings that
its differences from the ResNet-50 make suspect. The model was trained once with the ResNet-50's
settings unchanged and finished 0.0175 below it at ×4 while fitting the training set less
([prior work](prior-work.md)). The purpose behind the score is a third architecture kind for the
blend with the ResNet-50 and ConvNeXt-Tiny.

The reference and first incumbent is `outputs/training/mambavision-t-rn50-recipe` (fold 2 of
`all_5fold_scanner_v2`, seed 1110484336, ten epochs, batch 8).

Seven forced seeds in this order, then the researcher without end:

1. `stage-rates-by-weight-scale`: 6e-4 on the last stage, 0.6 times less per stage towards the
   stem, the rates that give each stage the ResNet-50's relative step.
2. `backbone-rate-2e-4`: one rate at four times the reference's.
3. `backbone-rate-4e-4`: one rate at eight times the reference's.
4. `anchors-by-weight-scale`: the pulls to the pretrained weights at 0.001 early and 0.00001 late.
5. `twelve-epochs`: the same schedule over twelve epochs (the user's decision).
6. `feature-statistics-augmentation-off`: the augmentation tuned on ResNet stride-8 features off.
7. `gradient-clipping-at-10`: the clip above the model's usual gradient norm.

Seeds written first and removed on 2026-10-13 after the reference's diagnostics, with the reason
for each, are in [prior work](prior-work.md#seeds-removed-before-the-start).

## Settled

- **The backbone is `mamba_vision_T` with its ImageNet-1k weights.** No other backbone, size or
  pretraining.
- **The recipe around it stays:** labels, views, slices, 224 px, augmentations other than the one
  seed 6 names, the mask decoder, the OA detector, routing, heads and losses.
- **Prediction settings are not subjects:** ×1 validation and ×4 prediction as the recipe has
  them; no test-time augmentation, no ensembling, no weight averaging.
- **Batch 8, one optimizer step per batch.**
- **Never fewer than ten epochs.**

## Principles

1. **One setting per run**, applied to the incumbent. A run that changes two settings names the
   two runs it combines.
2. **A setting is a subject when a difference between MambaVision-T and the ResNet-50 gives a
   reason for it**: ImageNet weights against RadImageNet, LayerNorm and attention in the last two
   stages against BatchNorm throughout, the state-space scan, its pretraining regularisers, or the
   measured underfit. The reason is stated and marked as evidence or guess.
3. **A value comes from a cited source** (the model's authors, the ConvNeXt-Tiny line, an earlier
   run of this campaign), never from a sweep read off the validation labels.
4. **A run can be proved active** by metrics it names, at most six; a setting that only changes a
   number in the optimizer is proved by the logged rates or by the recorded effective config.
5. **A superseded behaviour is removed**, not kept behind an option, except where the setting is
   a value the campaign intends to vary.

## How a result is read

One run is weak evidence. Two trainings of the ResNet-50 recipe on this fold were 0.0010 apart at
×4 and 0.0013 at ×1; MambaVision-T's own spread is not known. A run is read at ×4 and ×1 against
the incumbent, on the training loss per epoch (the reference underfits), and per label from
largest gain to largest loss. Every result is also given against the ResNet-50 on this fold
(×4 0.939634, ×1 0.930902), and as an equal logit blend with it against the ResNet-50 alone.

## Out of bounds

Another backbone, model size or pretrained weights; fewer epochs; ensembling, weight averaging,
test-time augmentation; image sizes other than 224 px; more slices or images per study; the
labels, the split, distillation; new routing, attention or readout structure; reflection
probabilities; gold-study weighting; trained models in the loader; multi-seed, confirmation or
control runs not asked for by the user; choosing anything on the campaign fold's validation
labels; edits to the outside model code under `.venv/third_party`; Kaggle submissions.

## Cost

**The budget is 0.75 to 1.3 of the reference's training time** (2,594 s), so 1,946 to 3,372 s.
The upper bound exists for the twelve-epoch seed; every other seed trains within 1.05 of the
reference's time. The reference peaked at 12.7 GB of GPU memory.

## Duration and adoption

Started with `start` by the user; no end. After the forced seeds the researcher keeps proposing
inside this contract until the user stops it. Adoption is by the runtime under the
[shared rule](../../protocol.md#shared-evaluation-and-adoption). The keeper repairs and requeues;
it does not adopt.
