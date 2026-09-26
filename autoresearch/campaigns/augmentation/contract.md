# Augmentation campaign contract

## Objective

Add training-time input variation to the live incumbent: new image augmentation families, an
affine schedule and a feature-level augmentation. The campaign started from
`report-attributes-control` (seed 2464012069) on fold 2. The thirteen seeds come from the user's review of 2026-09-25
([prior work](prior-work.md)) and run first, in order (`seed_order: forced`). Research starts
only after all of them are completed or blocked.

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged. After an adoption, the supervisor retrains the adopted source at a fresh seed
and uses that run as the reference ([seed policy](../../protocol.md#training-seed-policy)).
Per-label AUC and vendor/field-strength subgroups are diagnostics only; nothing is adopted per
label.

## Fixed recipe

- The recipe is `configs/training.yaml` as sealed at start.
- Fixed: labels, report-attribute heads and their weight, split, folds, epochs, optimizer
  schedule, class balance, image budget, sampler, crop, image size and backbone; the
  backbone changes only through its drop-path rate (seed 13).
- The existing reflection, affine, intensity and noise families and series dropout keep their
  values. A candidate adds its own mechanism, with the settings its idea fixes, under its config
  prefix in `campaign.yaml`, switched on in `configs/training.yaml`.
- No strength sweeps: a seed's settings come from the measured data profile and are not tuned.

## Authorized scope

- Image transforms in `modeling/augmentation.py`, run on the GPU per acquisition on `[0, 1]`
  images before ImageNet normalization, identical across an acquisition's slice channels and
  triplets. Filtering transforms respect the geometric valid region.
- Training-schedule changes to an augmentation family (seed 7), through the epoch passed from
  `training/lightning.py`.
- Blending a study's own two contrasts of one plane before augmentation (seed 11).
- Feature-level augmentation inside the backbone (seed 12).
- Stochastic depth in the backbone at a fixed rate (seed 13).
- MixUp (seeds 4–6) between studies of the same knee side and reflection draw, mixing images or
  tokens slot by slot with depth-matched triplets, or pooled outputs, with a two-target loss over
  labels observed in both studies. Study a keeps its own geometry; in the image and token variants,
  the half-knee and report-attribute losses mix their targets with the same lambda.
- Spatial changes must keep reflection, affine inverse and token geometry consistent.

## Evidence requirements

- **Activity logging:** log every mechanism's activity per epoch: acquisitions or studies
  affected, mean absolute pixel or logit change, mixed-loss component, effective probability.
- **Idle mechanisms:** a candidate whose mechanism never activates is invalid evidence, not a
  result. Report it as blocked instead.
- **Disabled parity:** with its probability at zero, a new family draws no random numbers and
  reproduces the incumbent.

## Exclusions

- Strength sweeps of any family, existing or new.
- Sampler, crop, image size, preprocessing-identity or label changes.
- Extra forward passes (two-view consistency losses, auxiliary augmented views).
- Unaligned MixUp (across knee sides or unmatched slots), CutMix, cutout and elastic or grid
  distortion.
- Per-label tuning, gold-study weighting and trained models in preprocessing.
- Bigger images, ensembling and test-time augmentation.
