# Crop views campaign contract

Written 2026-10-16 at the user's request, from the per-label and complementarity analysis of every
earlier crop and pose run ([prior work](prior-work.md)). Not started.

## Objective

Find view framings (field sizes, anchoring, posing, the per-knee scale, the pose estimate, how a
type's slices are split between views) that make a ResNet-50 run **add to the recipe in an
ensemble**. A run does not have to beat the recipe on its own.

The control is `outputs/training/resnet50-silver-fold2`: `configs/resnet50.yaml` (no mask decoder,
no teacher labels), fold 2 of `all_5fold_scanner_v2`, seed 1110484336, ten epochs; ×4 0.937820,
training 2,617 s, peak 16.4 GB ([prior work](prior-work.md#the-control)).

Ten forced seeds in this order, then the researcher without end:

1. `two-views-wide-and-posed`: coronal T1/PD, coronal fluid and sagittal T1/PD each on their posed
   view (3 slices) plus an unposed, image-centred 120 mm view (2 slices).
2. `unposed-120-mm-everywhere`: one unposed 120 mm view per series type, no detail view.
3. `per-knee-scale-off`: the recipe with every view at scale 1.
4. `coronal-crops-a-ninth-wider`: coronal T1/PD at 93.33 mm and coronal fluid at 103.70 mm.
5. `soft-bone-shares-for-the-joint-plane`: the joint plane fitted with continuous bone shares.
6. `canvas-locator-resnet10t`: a ResNet-10t reads the whole canvas, learns the anatomy map and
   poses the crops in place of the ResNet-50 first look from epoch 3.
7. `wide-fluid-stream-resnet10t`: a ResNet-10t reads unposed 120 mm views of the fluid slices; its
   tokens join the pools without a gate.
8. `canvas-locator-with-context-tokens`: seed 6 with the locator's tokens also in the pools.
9. `wide-fluid-stream-own-loss`: seed 7 with the stream also pooled alone under the target loss.
10. `canvas-context-tokens-resnet10t`: a ResNet-10t reads the whole canvas and its tokens join the
    pools; crops posed as today.

Seeds 6 to 10 were added at the user's request on 2026-10-16, before the start. Each forced seed is
the control's recipe plus its own change, even after an adoption; seeds 8 and 9 extend the code of
6 and 7.

## Adoption: the complement rule

**This campaign runs a modified runtime (uncommitted, worktree `knee-abnormality-crop-views`
only):** the [shared rule](../../protocol.md#shared-evaluation-and-adoption) is replaced.

- **Complement gain:** ×4 macro AUC of the equal logit blend of the run's saved predictions with the
  control's, minus the control's ×4 macro AUC (`campaign/complement.py`).
- **Adoption:** the run's complement gain is strictly above the incumbent's. While the control is
  the incumbent, its gain is the bar **+0.00059**, the median complement gain of the 29 fold-2
  ResNet-50 runs of the targets-and-inputs campaign, each one setting away from the recipe outside
  the views ([prior work](prior-work.md#yardstick)).
- **Reported, not deciding:** the run's own ×4 and ×1 against the incumbent with the 80% interval,
  and per label.
- The gain counts quality and diversity together: a run equal to the control in quality and as
  different as a one-setting change adds about +0.0006; half of a solo change passes into it.

## Settled

- **Backbone, labels, split, fold, seed, epochs, batch:** the control's. The one addition allowed is
  a Refined-RadImageNet ResNet-10t auxiliary backbone under `model.auxiliary`, off by default. Since
  2026-10-17 (user decision, recorded in the campaign's `decisions/`) the ResNet-50 may also take
  timm's ImageNet `resnet50.a1_in1k` weights with ImageNet input statistics
  (`model.pretrained_weights`, `data.mean`, `data.std`), to test whether a framing's effect depends
  on the pretraining with the architecture fixed.
- **The recipe outside the views stays:** augmentations, routing, heads, the OA detector, the
  meniscus map, losses and readouts.
- **224 px images and ×4 prediction** as the recipe has them.

## Principles

1. **One setting per run** against the control, or a named combination of earlier runs.
2. **Framings come from measured geometry** (coverage of the structures, knee size, the pose fit),
   never from a sweep on the validation labels.
3. **A run is proved active** by the metrics it names, at most six.
4. **A new option exists only for a value the campaign varies**; at its default the code behaves
   as the control's.

## How a result is read

One run is weak evidence. Each run is read on the complement gain and its per-label gains (largest
gain to largest loss), the gain minus half of its own ×4 change (the part that is not quality), its
own ×4 and ×1, and its gain as a sixth member of the fold-2 five-family blend
([prior work](prior-work.md#fold-2-family-runs)).

## Out of bounds

Other backbones or weights than the ResNet-50 (Refined-RadImageNet or ImageNet `resnet50.a1_in1k`) and
the auxiliary Refined-RadImageNet ResNet-10t; image sizes other than 224 px; more images per study than the time and
memory limits allow; the labels, the split, distillation; routing, attention or readout changes;
ensembling weights, weight averaging, test-time augmentation; reflection probabilities; gold-study
weighting; trained models in the loader; multi-seed, confirmation or control runs not asked for by
the user; choosing anything on the campaign fold's validation labels; Kaggle submissions.

## Cost

**0.3 to 1.2 of the control's training time** (2,617 s): 785 to 3,141 s. Peak GPU memory under
19 GB; the control uses 16.4 GB.

## Duration

Started with `start` by the user from the worktree; no end.
