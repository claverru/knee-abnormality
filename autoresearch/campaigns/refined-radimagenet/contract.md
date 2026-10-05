# Refined-RadImageNet campaign contract

## Objective

Get more out of what Refined-RadImageNet pretraining already knows. The Refined-RadImageNet
ResNet-18 and ResNet-50 are our strongest models, and the refinement itself only cleans
ultrasound: for MRI the images are the original RadImageNet's, so the strength comes from the
authors' training recipe and from how we use its result ([prior work](prior-work.md)). The
campaign tests four ways to use it better:

1. **Input statistics.** The checkpoints were trained with mean 0.3162 and std 0.3213, not the
   ImageNet values our recipe uses.
2. **Mask regularization**, the authors' policy-gradient method: their mask-trained weights as
   initialization, and the mask in our own fine-tuning.
3. **The 165-class head**, which we discard: it has 18 knee MRI classes.
4. **Adjacent slices through pretrained kernels**: Cross-D-Conv and ACS convolutions.

The reference and starting incumbent is `rn18-view-label-bias`
(`outputs/training/20261003-100844-rn18-view-label-bias`, the `configs/training.yaml` recipe, seed
1533697213, fold 0 of `all_5fold_scanner_v2`, ×4 0.932477, ×1 0.929026, 1,181 s, 0.951 public as
S101). Experiments run on the ResNet-18; what is adopted is carried to the ResNet-50 afterwards
by the user.

## Evaluation

The [shared evaluation and adoption rule](../../protocol.md#shared-evaluation-and-adoption)
applies unchanged against the campaign incumbent. Every result also reports ×1 and the per-label
deltas.

## Seeds

Forced, in file order ([seeds 1](seed-ideas.yaml), [seeds 2](seed-ideas-2.yaml),
[seeds 3](seed-ideas-3.yaml)). Each applies to the incumbent at its turn, so an accepted seed
carries into the later ones, with the exceptions stated below.

1. `radimagenet-input-statistics`: the checkpoints' own normalization.
2. `rn10t-policy-weights` and 3. `rn10t-policy-ft-weights`: the authors' mask-trained ResNet-10t
   checkpoints as initialization.
4. `policy-mask-rn10t`, 5. `policy-mask-two-stage`, 6. `policy-mask-pretrained-policy`: the
   policy-gradient mask in our fine-tuning, with a ResNet-10t policy.
7. `random-smooth-mask`: the same mask without a policy.
8. `knee-logit-token-features`, 9. `knee-classifier-readout`, 10. `pretrained-slice-teacher`,
   11. `knee-logit-attention-prior`: the pretrained head.
12. `crossd-resnet18-drop-in`, 13. `crossd-3d-stem`, 14. `acs-stem-layer1`: adjacent slices.

**ResNet-10t seeds (2, 3) are measurements.** The mask-trained checkpoints exist only for
ResNet-10t and ResNet-50, so these two change the backbone. Their reference is
`rn10t-view-label-bias` (`outputs/training/20261004-094416-rn10t-view-label-bias`, the incumbent
recipe on the standard Refined-RadImageNet ResNet-10t at backbone LR 6e-5, same fold and seed:
×4 0.927543, ×1 0.921848, 1,141 s). They keep that run's normalization (ImageNet mean and std)
whatever seed 1 decided, so the pair differs only in the starting weights and the rate and
anchor strengths their weight scale implies. The runtime compares them with the ResNet-18
incumbent; the keeper reports them against the ResNet-10t reference. If one passes the adoption
rule against the ResNet-18, the keeper restores the ResNet-18 incumbent by manual adoption and
reports it to the user.

**Seeds 5, 6 and 7 build on seed 4's code**, and seeds 8, 9 and 11 share the token-level knee
logits. Forced seeds declare no dependencies: a later seed whose parent was rejected ports the
parent's code from its frozen source under
`outputs/autoresearch/refined-radimagenet/experiments/<run>/source/` and applies its own change.

**Seeds 8, 9 and 11 need the checkpoint's own normalization** (seed 1): the head reads our slices
as knee MRI only under it ([prior work](prior-work.md#pretrained-head-on-our-slices)). If seed 1
was not accepted, each of them sets `data.mean` and `data.std` to the RadImageNet values as part
of its change and is reported against both the incumbent and seed 1's run.

## User seeds added during the run

`backbone-eval-pretrained-statistics` ([seed](backbone-eval-ideas.yaml)), forced by the user on
2026-10-04 after seed 1 was accepted and placed after the two ResNet-10t measurements: the
existing `model.backbone_eval` option, so the backbone's BatchNorm layers use the pretrained
running statistics during training. Configuration only.

`crossd-static-kernels` and `crossd-gated-depth-stem` ([seeds](crossd-static-ideas.yaml)), forced
by the keeper on 2026-10-04 as repairs of the two Cross-D seeds: the per-slice layer ran ten times
over the time bound, and `crossd-3d-stem` was specified with unshifted kernels
([run notes](run-notes.md)). They use one kernel per layer, shifted by the layer's mean shift
measured without labels, and bring the stem's neighbouring depth slices in through a gate that
starts at zero.

`acs-gated-stem-block1` ([seed](acs-gated-ideas.yaml)), forced by the keeper on 2026-10-04 in place
of `acs-stem-layer1`, which would have started from a broken network (frozen knee-class mass 0.256
against 0.990, [run notes](run-notes.md)): the same ACS reorientation behind gates that start at
zero, in the stem and the first block of layer1.

`acs-gated-stem` ([seed](acs-gated-stem-ideas.yaml)), forced by the keeper on 2026-10-04 after
`acs-gated-stem-block1` matched the incumbent at epoch 1 but ran at 250 s per epoch, over the
bound: the gated ACS form in the stem only, at the middle depth.

## Budget

Wall-clock range: each candidate's training time must lie between 75% and 200% of the
reference's 1,181 s, that is 886–2,362 s (user, 2026-10-04). The upper bound is 2× because the
mask policy adds a ResNet-10t pass per slice and the Cross-D and ACS seeds a heavier backbone,
while inference is unchanged for all but those. Ten epochs, batch size 8, the studies seen per
epoch and the slices per study are fixed.

## Fixed

Split, fold, evaluation labels (`rebuild_state_v2`), class-balanced BCE and its exponent 0.5,
image size 224, the views and sampling geometry, 26 slices per study, ten epochs, batch size 8,
the evaluation and prediction multipliers, the half-knee and report-attribute terms and their
weights.

## Permitted

- The fourteen seeds, each under its own option.
- After the seeds, the researcher proposes only two kinds of idea (user, 2026-10-04):
  - **Inside the Refined-RadImageNet scope:** ideas that use something the pretraining produced
    (its weights, classifier, statistics, training recipe or the authors' published methods):
    other uses of the pretrained classifier or of a frozen pretrained network, the mask family,
    input statistics and intensity mappings, adjacent-slice and 3D uses of pretrained kernels,
    and values, schedules and placements of a mechanism of this campaign. A generic recipe
    change that would apply equally to an ImageNet backbone is out of scope.
  - **Combinations of good prospects from this campaign's own runs:** two mechanisms that each
    showed promise here (accepted, rejected with a positive delta, a clear per-label lift, or
    curves a named change could turn into a gain), applied together, two per run.

  One value per knob.
- Per-label rules, with every choice fixed before the run.

## Excluded

Pretraining on RadImageNet images or any external data (user, 2026-10-04: too costly for a
campaign); weight averaging (SWA, EMA); image sizes above 224, more slices per study, ensembling
and test-time augmentation (endgame); fewer epochs; reflection-probability changes; the
class-balance exponent; label-source changes and gold-study weighting; multi-seed or confirmation
runs; trained models in the loader; choosing any class mapping or setting on fold-0 validation
labels; ideas already measured on ResNet-18 at the same value
([prior work](prior-work.md#measured-on-resnet-18)).

## Build facts

What a developer needs before writing code. File and function names are this worktree's.

- **Where the input is normalized.** `KneeAbnormalityModel.encode_batch`
  (`modeling/model.py`): slices arrive as `uint8`, are divided by 255, augmented, expanded to
  three channels (`model.slice_channels: repeated_middle` keeps the middle slice and repeats
  it), then `valid_images.sub(self.mean).div_(self.std)` and `self._encode_images(...)`.
  `self.mean` and `self.std` come from `data.mean` and `data.std`.
- **Tokens.** `_encode_images` returns `maps`, N × 512 × 7 × 7 (N = slices in the batch, about
  208). `features = self.projection(maps.flatten(2).transpose(1, 2))` is N × 49 × 256; the 49
  positions of a slice are its tokens. `_pooling_tokens` adds position, plane, contrast and
  view embeddings and applies `mil_norm`; `_attention_pool` scores tokens per target
  (`self.attention`, plus `view_label_bias`), restricts and softmaxes them per study, and sums
  the per-token evidence of `self.head` (a `PlaneContrastReadout`) with those weights.
- **Training-only modules** live on the Lightning module (`training/lightning.py`), like
  `report_attribute_heads`, and reach the optimizer through the `heads` tuple of
  `build_optimizer` (head learning rate, weight decay as for the model). Inference loads only
  `model.*` keys, strictly, into a model built from the saved config with `load_pretrained=False`
  (`inference/checkpoint.py`): anything the model needs at prediction is a model parameter or a
  persistent buffer created from the config alone, never read from a pretrained file at
  prediction time.
- **Loss.** `training_step`: `masked_binary_cross_entropy_with_logits` (class-balanced, mean
  over observed targets of the batch) plus the half-knee term, the report-attribute term and the
  L2-SP penalty (`optimizer.anchor_alpha_early` on `conv1.`, `bn1.` and `layer1.` weights,
  `anchor_alpha_late` on the rest, squared distance to the initial weights).
- **Clipping.** Lightning clips the global gradient norm of every optimized parameter at
  `trainer.gradient_clip_val` 1.0. A training-only network with large gradients would shrink
  the model's update: clip it separately (override `configure_gradient_clipping`), the model
  and existing heads together at 1.0 as now.
- **Mixed precision** is on. Distributions, FFTs and logarithms are computed in float32 with
  autocast disabled.
- **Built for this campaign** (uncommitted, checked on CPU):
  - `modeling/radimagenet.py`: `PRETRAINED_WEIGHTS[weights][backbone]` for
    `model.pretrained_weights` `radimagenet`, `radimagenet_policy`, `radimagenet_policy_ft` and
    `radimagenet_crossd`, each with file, hash and source; `radimagenet_state_dict(backbone,
    weights)` (backbone tensors without `fc`; `weights` defaults to `radimagenet`);
    `radimagenet_classifier(backbone, weights)` (the 165 × C `fc.weight` and `fc.bias`); `RADIMAGENET_KNEE_CLASSES` (class name to index,
    83–100); `RADIMAGENET_MEAN`, `RADIMAGENET_STD`.
  - `modeling/crossd.py`: `CrossDConv` and `replace_convolutions`, equal to the authors' layer
    to rounding (outputs within 2e-6, gradients within 2e-7 relative), with the per-sample
    kernels built in chunks of 16 slices under gradient checkpointing so they are not stored.
    A timm `resnet18.a1_in1k` after `replace_convolutions` loads
    `radimagenet_state_dict("resnet18.a1_in1k", "radimagenet_crossd")` strictly.
  - `modeling/model.py` is byte-identical to main, because it is a hashed preprocessing source
    and the baseline must reproduce the reference's identity. It loads only
    `pretrained_weights: radimagenet`. The config accepts `radimagenet_policy`,
    `radimagenet_policy_ft` (entries for `resnet10t.c3_in1k` and `resnet50.a1_in1k`) and
    `radimagenet_crossd` (`resnet18.a1_in1k`), and `scripts/pipeline/train.py` already records
    their source and hash; the seeds that use them make the model load them (seeds 2 and 3
    state the two-line change, seed 12 builds the Cross-D backbone).
- **Weight files** are in `~/.cache/torch/hub/checkpoints/` with the hashes in
  `PRETRAINED_WEIGHTS`. No download is needed or allowed.
- **Pretrained knee classes and our targets.** The mapping used by seeds 9 and 11, fixed before
  any run from the class names alone:

  | Target | RadImageNet knee class | Index |
  |---|---|---:|
  | ACL | `acl_pathology` | 83 |
  | MCL | `mcl_pathology` | 90 |
  | Medial Meniscus, Lateral Meniscus | `meniscal_abnormality` | 91 |
  | Medial OA, Lateral OA, PF OA | `chondral_abnormality` | 85 |
  | Effusion, Synovitis, Baker's | `soft_tissue_fluid_collection` | 99 |
  | Contusion | `bone_inflammation` | 84 |
  | Fracture | `fracture` | 87 |

## Evidence requirements

- Every candidate logs its own mechanism scalars per epoch (listed in its idea) next to the
  per-stage drift and gradient columns the Lightning module already writes.
- Every result records ×4, ×1, the per-label deltas, the training seconds and the mechanism
  scalars, and for the ResNet-10t seeds the delta against the ResNet-10t reference.
- The researcher starts each refill from the per-epoch curves and per-label deltas of every
  decided run and names the behaviour each proposal targets.
