# RadioDINO settings campaign contract

Written 2026-10-14 at the user's request. Not started.

## Objective

Raise the macro AUC of RadioDINO-S/16 on the current recipe by tuning the training settings that
its differences from MambaVision-T make suspect ([prior work](prior-work.md#how-the-model-differs-from-mambavision-t)).
The model enters on the MambaVision-T recipe unchanged. The purpose behind the score is a
medical-pretrained architecture for the blend with the ResNet-50 and MambaVision-T.

The reference is `outputs/training/mambavision-t-combined-12-epochs`, the MambaVision-T incumbent
(fold 2 of `all_5fold_scanner_v2`, twelve epochs, batch 8; ×4 0.937801, ×1 0.929880). The recipe
is `configs/mambavision.yaml`. The baseline source is this worktree's, which holds the RadioDINO
adapter.

Six forced seeds in this order, then the researcher without end:

1. `radiodino-on-mamba-recipe`: RadioDINO-S/16 in place of MambaVision-T, nothing else changed.
   **Adopted by the user's decision whatever its verdict**; every later seed applies to it.
2. `layer-rate-decay-from-radimagenet-profile`: 4e-4 on the last block, 0.85 per block towards
   the stem.
3. `anchors-by-weight-scale`: both anchors twice as strong, 0.002 and 0.00002.
4. `feature-statistics-before-position-embedding`: the augmentation on the patch projection.
5. `middle-read-at-block-10`: the stride-16 read after block 10.
6. `stochastic-depth-0-1`: drop path 0.1.

## Settled

- **The backbone is `radiodino_s16` with the `Snarcy/RadioDino-s16` weights** from seed 1 on.
  No other backbone, size or pretraining.
- **The adapter is a subject.** How RadioDINO's tokens become the stride-8, stride-16 and stride-32
  maps (CLS token handling, the stage split, the read depths, the final downsampling) belongs to
  the backbone, not to the readout: the keeper fixes gaps that runs or diagnoses expose there
  with forced seeds.
- **The recipe around it stays:** labels, views, slices, 224 px, augmentations other than the
  placement seed 4 names, the OA detector, routing, heads and losses.
- **Prediction settings are not subjects:** ×1 validation and ×4 prediction as the recipe has
  them; no test-time augmentation, no ensembling, no weight averaging.
- **Batch 8, one optimizer step per batch, twelve epochs.**

## Principles

1. **One setting per run**, applied to the incumbent. A run that changes two settings names the
   two runs it combines.
2. **A setting is a subject when a difference between RadioDINO-S/16 and MambaVision-T gives a
   reason for it**: RadImageNet self-supervised weights against supervised ImageNet-1k, the weight
   scale, 21.7M against 31.8M parameters, twelve equal pre-norm attention blocks over every token
   against a convolution, Mamba and attention hierarchy, LayerNorm only, the learned position
   embedding, the adapter's read depths. The reason is stated and marked as evidence or guess.
3. **A value comes from a cited source** (the model's authors, the ViT fine-tuning literature,
   the ResNet-50 and MambaVision-T lines, an earlier run of this campaign), never from a sweep read
   off the validation labels.
4. **A run can be proved active** by metrics it names, at most six; a setting that only changes a
   number in the optimizer is proved by the logged rates or by the recorded effective config.
5. **A superseded behaviour is removed**, not kept behind an option, except where the setting is
   a value the campaign intends to vary.

## How a result is read

One run is weak evidence. Two trainings of the ResNet-50 recipe on this fold were 0.0010 apart at
×4 and 0.0013 at ×1; RadioDINO's own spread is not known. A run is read at ×4 and ×1 against the
incumbent, on the training loss per epoch, and per label from largest gain to largest loss.
Every result is also given against the MambaVision-T reference and the ResNet-50 on this fold
(×4 0.939634, ×1 0.930902), and as an equal logit blend with each against that model alone.

## Out of bounds

Another backbone, model size or pretrained weights; more or fewer epochs;
ensembling, weight averaging, test-time augmentation; image sizes other than 224 px; more slices
or images per study; the labels, the split, distillation; new routing, attention or readout
structure downstream of the backbone's maps (heads, routing, readouts); reflection probabilities; gold-study weighting; trained models in the loader;
multi-seed, confirmation or control runs not asked for by the user; choosing anything on the
campaign fold's validation labels; Kaggle submissions.

## Cost

**The budget is 0.5 to 1.1 of the reference's training time** (3,106 s), so 1,553 to 3,417 s,
with twelve epochs fixed. The reference peaked at 12.7 GB of GPU memory.

## Duration and adoption

Started with `start` by the user; no end. After the forced seeds the researcher keeps proposing
inside this contract until the user stops it. Adoption is by the runtime under the
[shared rule](../../protocol.md#shared-evaluation-and-adoption), with one exception the user
ordered: **the keeper adopts seed 1 whatever its verdict**, unless the runtime already adopted it.
Once seed 1 has launched, the keeper
runs `stop` (training finishes and its comparison is recorded), waits for `STOPPED`, runs
`adopt radiodino-settings --experiment <seed 1's experiment> --rationale "User decision: the
campaign starts from RadioDINO-S/16"`, and resumes. Seed 2 may be prepared on the old incumbent
meanwhile; adoption reconciles it. Otherwise the keeper repairs and requeues; it does not adopt.

## Scope note of 2026-10-15 11:10 CEST

The learned final merge of the gap diagnosis (run-notes.md, "Diagnosis of the gap to MambaVision-T")
was blocked at review as "new readout structure". It is an adapter fix and in scope (Settled, "The
adapter is a subject"); the user confirmed that fixing such gaps is the keeper's job and ordered it
next. The blocked experiment `final-read-by-learned-two-by-two-merge-20261015T070028Z-b9442810`,
built on the current incumbent, was returned to `prepared` at the queue head for a fresh review
(campaign stopped with `--abort` at 11:22 by the user's order, which made stochastic-depth-0-1
invalid at epoch 12). Record:
`outputs/autoresearch/radiodino-settings/decisions/user-authorize-final-merge-20261015T0905Z/`.

## Current line (user decision, 2026-10-15 13:40 CEST)

The campaign now pursues one line: **giving RadioDINO's tokens the local, sub-patch detail its
16×16 non-overlapping patches lack**, the cause the gap diagnosis points to (run-notes.md). In it:
the convolutional spatial prior, the overlapping patch embedding, the depthwise convolution in the
block MLPs, and further changes of the same kind (local mixing, convolutional stems or side branches,
overlap, learned token resampling), each started as the incumbent. Optimizer, schedule, anchor,
augmentation and regulariser settings are out of this line; the researcher proposes only ideas in it.
Queued ideas outside it were removed on 2026-10-15.
