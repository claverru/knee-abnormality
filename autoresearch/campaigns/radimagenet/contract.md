# RadImageNet campaign contract

## Objective

Keep RadImageNet-pretrained ResNets learning. Refined-RadImageNet ResNet-18 and ResNet-10t start
far ahead of the ReXNet-200 control (+0.047 ×1 at epoch 0) and flatten from about epoch 5, finishing
below it, while their training loss stays above the control's: they underfit, and their first
layers drift 2–2.5× more than ReXNet's from the pretrained start ([prior work](prior-work.md)).
Find schedules, per-layer rates, inputs, regularization and structure that keep them improving
to the end of training and beat the ReXNet-200 control. The campaign's reference and starting
incumbent is that control, `rebuild-state-v2-control`
(`outputs/training/20260928-150724-rebuild-state-v2-control`, seed 1533697213, fold 0 of
`all_5fold_scanner_v2`, ×4 0.928288); a RadImageNet candidate is adopted only when it beats it.

## RadImageNet base

Every seed applies the RadImageNet base on top of the incumbent recipe, then its own change: the
differences of `configs/radimagenet-resnet18.yaml` from `configs/training.yaml`, that is
`model.backbone: resnet18.a1_in1k`, `model.pretrained_weights: radimagenet`,
`model.slice_channels: repeated_middle`, `optimizer.backbone_lr: 5e-5` and the name. The base
alone was measured outside the campaign (`radimagenet-resnet18`, ×4 0.925476, −0.0028 against the
control) and is not rerun. Once a RadImageNet candidate is adopted, the base is part of the
incumbent.

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged against the campaign incumbent, which starts as the ReXNet control. Every
result also reports its ×4 macro AUC against the RadImageNet base run (0.925476) and its training
seconds against the control's 1,912 s.

## Budget

Wall-clock range: each candidate's training time must lie between 50% and 150% of the
RadImageNet base run's 1,085 s (user, 2026-09-28), that is 543–1,628 s, or 0.28–0.85 of the ReXNet
reference's 1,912 s. The range is the only budget rule: the epoch count, batch size, optimizer
steps and the amount of input per study may change; the studies seen per epoch may not. At about 108 s per epoch, ResNet-18 fits up to about
15 epochs at 26 triplets per study.

## Fixed

Split, fold, evaluation labels (`rebuild_state_v2`), gold overrides, class-balanced BCE weights,
half-knee loss, report-attribute heads, scanner confusion head, image size 224, crop and
sampling geometry, and the evaluation and prediction multipliers.

## Permitted

- **Learning-rate schedule:** epochs, warmup, decay shape and floor, restarts, separate head and
  backbone schedules.
- **What moves:** per-stage or per-layer rates, freezing and gradual unfreezing, head-first
  training (linear probe then fine-tune), decay toward the pretrained weights, BatchNorm handling.
- **Optimizer:** rates, weight decay, betas, clipping, sharpness-aware or lookahead variants.
- **Regularization and augmentation strength:** series dropout, DSU, the augmentation
  probabilities and strengths, curricula over training.
- **Input:** repeated middle slice or several slices per triplet, including first-convolution
  initialization for more channels; more triplets per study or more images per series type;
  intensity normalization.
- **Structure:** token readout from more than one stage, token projection width, one backbone per
  plane with per-plane readouts and fusion, and the backbone within the Refined-RadImageNet family
  (ResNet-10t, ResNet-18, ResNet-50).
- **Configuration-only ideas are allowed** (a learning rate alone, an epoch count alone), but every
  researcher refill must also carry at least one idea that needs a source change.

## Excluded

Weight averaging of any kind (SWA, EMA; user, 2026-09-28). Image sizes above 224, ensembling and
test-time augmentation (end phase); other backbone families
and ImageNet weights; label-source changes; gold-study weighting; multi-seed or confirmation runs.

## Evidence requirements

- Every candidate logs, per epoch, the relative weight change from the pretrained start and the
  gradient RMS for the stem, each ResNet stage and the head, next to its own mechanism metrics and
  its activation signature.
- The researcher starts each refill from the per-epoch curves (×1 validation, training loss,
  per-label AUC, per-stage drift and gradients) and effective configurations of every decided run
  and the reference, and names the curve behaviour each proposal targets.
