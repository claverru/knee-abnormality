# Plane-specialization campaign contract

## Objective

Find how plane-specific backbones should be pooled, supervised and fused so that each backbone
extracts the most from its own plane and the fused model beats a shared backbone. The campaign
starts from `b0-plane-backbones` (three EfficientNet-B0 backbones, one per plane, seed
703791404, fold 4). The user's seeds run first, in order (`seed_order: forced`); research
starts only after all of them are completed or blocked ([prior work](prior-work.md)).

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged against the campaign incumbent. Every result also reports its ×4 macro AUC
next to the ReXNet control `scanner-confusion-control` (0.909733 on the same fold, seed and
labels), which the reference trails by 0.0113. Per-plane logit AUCs, fusion weights and
assignment shares are diagnostics; nothing is adopted per label.

## Fixed recipe

- The recipe is `configs/training.yaml` as sealed at start: `efficientnet_b0.ra_in1k` with
  `model.plane_backbones: true`, batch 16, `backbone_lr` 8e-4, `head_lr` 1.2e-3, 10 epochs, DSU
  off and series dropout 0.
- Fixed: backbones and their plane routing, split, fold, labels, gold
  overrides, class-balanced BCE weights, half-knee loss, report-attribute heads, scanner
  confusion head, optimizer and schedule, image budget, sampler, crop, image size and
  augmentation.

## Definitions

Every rule below applies per study and label unless stated otherwise.

- **Plane-exclusive pooling:** for label *c* and plane *q*, the incumbent's attention pooling
  restricted to the study's tokens of plane *q*. The anatomical score bias, the meniscus
  own-side mask (falling back to the plane's tokens when its own side is empty) and the
  three-compartment OA-detector correction (regions taken within the plane's trusted tokens)
  apply as in the incumbent. It gives a pooled vector *u_cq* and a plane logit
  *z_cq = w_c · u_cq + b_c* plus the OA correction. A plane the study lacks has no logit.
- **Present-plane mean:** the mean of *z_cq* over the study's present planes.
- **Learned fusion:** per label and plane a weight *a_cq* (initialized at 0). The fused logit is
  the softmax of *a_c* over the study's present planes applied to *z_cq*. It trains with the
  class-balanced BCE on detached *z_cq*, so no gradient reaches the backbones, and it is the
  model's output at inference.
- **Radiology mapping** (fixed by the user on 2026-09-27; the only label-to-plane mapping not
  learned or random):

  | Label | Planes |
  |---|---|
  | ACL | Sagittal, Coronal |
  | MCL | Coronal, Axial |
  | Medial Meniscus, Lateral Meniscus | Sagittal, Coronal |
  | Medial OA, Lateral OA | Coronal, Sagittal |
  | PF OA | Axial, Sagittal |
  | Effusion, Synovitis | Axial, Sagittal |
  | Baker's | Axial |
  | Contusion, Fracture | Axial, Coronal, Sagittal |

  A study with none of a label's mapped planes pools that label over all its present planes.

## Authorized scope

- **Pooling:** plane-exclusive pooling, joint pooling over a label's permitted planes, and the
  incumbent's joint pooling.
- **Label-to-plane assignment in training:** random partitions, dense (every plane predicts every
  label), learned or competitive routing, and the radiology mapping. Random draws derive from the
  run seed and the global step.
- **Fusion at inference:** present-plane mean, learned fusion, or joint pooling.
- **Heads:** shared across planes, one classifier per plane, or a full head per plane
  (projection, token LayerNorm, attention scorer and classifier). Per-plane modules start as
  copies of the shared module's initialization; position, plane and contrast embeddings,
  anatomical strengths and the OA detector stay shared.
- **Training-only per-plane objectives:** auxiliary per-plane readouts with a coefficient fixed
  before the run.

## Evidence requirements

- **Activity logging:** per epoch, every new loss term, realized assignment shares per label and
  plane, fallback counts, fusion weights, and the validation macro AUC of each plane's own
  logits.
- **Idle mechanisms:** an assignment that never reaches a plane, or fusion weights that never
  move, make invalid evidence. Report it as blocked instead.
- **Disabled parity:** with the new option off, the run reproduces the incumbent.

## Exclusions

- Backbone family, sizes and the number of backbones; image size, epochs, batch, learning rates
  and augmentation.
- Series dropout (fixed at 0 by the user) and token dropout.
- Label-to-plane rules chosen from label results; the radiology mapping above is the only fixed
  mapping.
- Redundancy penalties between plane embeddings: the shared projection lets backbones satisfy
  them by moving features apart without removing shared information.
- Coefficient sweeps, extra forward passes, test-time augmentation, ensembling and bigger images.
