# Plane-specific backbones

**Closed 2026-09-28, not adopted (user decision).** One backbone per plane (axial, coronal,
sagittal), each seeing only its own plane's images, tested on EfficientNet-B0 in the
plane-specialization campaign and transferred to ReXNet-200. The best structure ties the
single-ReXNet control publicly (0.945 against 0.946) at about 35% more training time and twice the
Kaggle inference time. Results, runs and the τ sweep are in the
[campaign result](../validation/results/2026-09-28-plane-specialization-campaign.md).

## Why it was tried

The [single-plane inference](../validation/results/2026-09-27-single-plane-inference.md) on the
ReXNet control showed each label leaning on one plane (ACL sagittal, MCL coronal, Effusion and
Baker's axial) and fusion adding little over the best single plane for most labels. The hypothesis:
one shared backbone spreads its knowledge across planes, so backbones forced to extract the most
from their own plane should fuse into a better model.

## What was built

- **`model.plane_backbones`:** `PlaneBackbones` holds three timm backbones from the same pretrained
  weights; each triplet goes to its plane's backbone and the feature maps return in the original
  order. The extra backbones initialize under a forked RNG, so everything after them starts as in the
  shared model. FP16 inference captures one CUDA graph per backbone.
- **Dense per-plane readouts and learned fusion** (the campaign's adopted incumbent): each label's
  attention pools one plane's tokens at a time, every plane's logits take the class-balanced BCE,
  and a per-label softmax over present planes, trained on detached logits, fuses them at inference.
- **DSU per plane:** a stride-8 split for ReXNet (`stem` + `features[:5]`) and EfficientNet
  (`conv_stem` + `bn1` + `blocks[:3]`), used in the three-ReXNet run.

## Recovering the code

Nothing entered `main`. The code is in:

- every campaign run's frozen source,
  `outputs/autoresearch/plane-specialization/experiments/<run>/source/`, and the three-ReXNet run's
  `outputs/training/20260928-105516-rexnet-plane-dense-fusion/source_snapshot.zip`.

## What carries forward

- Averaging near-equal runs gained +0.0026 locally, a seed-ensemble effect for the
  [endgame](../backlog/endgame.md#ensembling).
- τ 0.75 stays optimal on plane models as on the single backbone.
- The campaign harness gained two fixes kept in `main`: reused idea IDs are rejected at the researcher
  step, and proposed ideas count as covering their family.
