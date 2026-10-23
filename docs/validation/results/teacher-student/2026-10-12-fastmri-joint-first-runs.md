# fastMRI joint training: first runs (2026-10-12 to 2026-10-13)

**Training a fold model on Kaggle and pseudo-labelled fastMRI studies together gave the best
public score of its time (S148, 0.962) although the fold score did not move (×4 −0.000278 against
its control); a ConvNeXt-Tiny on the same setup beat the ResNet-50 locally, and the two together
scored 0.963 (S152). Pretraining on fastMRI first gave a head start that was gone by the last
epoch.** Fold 3 of `all_5fold_scanner_v2`, seed 1110484336, one run per arm, on the rented
machine. The setup (data, exclusions, pairing) is in
[external data](../../../modeling/external-data-and-distillation/external-data.md); the joint mode
itself was later retired
([retired pieces](../../../history/fastmri-chain-retired-pieces.md#joint-training-with-external-batches)).

## Setup of the first round

- **Pseudo-labels:** five ResNet-50 fold models of main's recipe at `0bd598f` (seed 1110484336;
  folds 0 to 2 trained locally, 3 and 4 on the rented machine), ×4, τ 0.9, mirror TTA. One row per
  study and teacher in `/workspace/data/labels/prepared/fastmri-rn50-main-x4-tau09-tta/`; the
  scripts that made them are beside the predictions in
  `/workspace/predictions/fastmri/rn50-main-5fold-x4-tau09-tta/source/`. Mean spread between
  teachers 0.018 to 0.029 per label.
- **Joint training:** every Kaggle batch was followed by a fastMRI batch, one optimizer step per
  pair, equal weight. The fold-k run read teacher k's labels. fastMRI studies were drawn at random
  without repeats until all were used. L2-SP was applied on the Kaggle batch only; class weights
  came from the Kaggle rows only. Validation was the Kaggle fold alone. Code on branch
  `fastmri-pretraining` (`extra_training` option).

## ResNet-50 on fold 3, `rn50-joint-fastmri-fold-3` against `rn50-main-0bd598f-folds-3-4`

Same machine, recipe, seed and fold. One run each.

| | Joint | Control | Δ |
|---|---|---|---|
| ×1 macro AUC (epoch 9) | 0.926211 | 0.925655 | +0.000556 |
| ×4 macro AUC | 0.930745 | 0.931023 | −0.000278 |
| ×4 soft macro AUC | 0.787786 | 0.787299 | +0.000487 |
| ×4 soft BCE | 0.4021 | 0.4006 | +0.0015 |
| Training time | 75.5 min | 37.4 min | |
| Public, ×4 τ 0.9 | **0.962** (S148) | not submitted | |

×1 macro AUC by epoch, joint minus control: +0.0082, +0.0063, +0.0018, +0.0017, +0.0033, +0.0018,
+0.0006, +0.0008, +0.0002, +0.0006. Ahead in every epoch, with the lead shrinking as the learning
rate falls.

×4 AUC by label, joint minus control: Lateral Meniscus +0.0032, Baker's +0.0031, Synovitis +0.0028,
Contusion +0.0021, Lateral OA +0.0014, Medial OA +0.0008, ACL −0.0016, Medial Meniscus −0.0016,
Fracture −0.0024, Effusion −0.0024, MCL −0.0027, PF OA −0.0060.

**S148, 0.962 public on 2026-10-12**, was the best public score at the time: above the five-teacher
ensemble S140 (0.961) and every single ResNet-50 (S143 0.960, fold 1 with 0.25 teacher share). The
fold 3 control was never submitted, so the fold's own share of that is unknown.

PF OA is the largest loss; it is also the label with the highest mean pseudo-label (0.48) and the
largest spread between teachers (0.029).

## Variants discussed and not run

Kaggle-only final epochs while the learning rate still moves the weights; L2-SP off in the joint
run (it was adopted on the ResNet-18 for unseen sites and never tested on the ResNet-50; in the
joint run its pull relative to the task gradient is already lower, since each step sums two
batches); class weights over Kaggle and fastMRI together; children distilled from the joint
teachers' held-out Kaggle predictions; mask predictions as extra fastMRI targets. They became
seeds of the [fastMRI joint training campaign](../../../../autoresearch/campaigns/fastmri-joint/INDEX.md),
which was written and never started. L2-SP off was submitted separately as a second "S150"
(0.961) and the learning-rate-weighted fastMRI loss as S149 (0.961), both below S148.

## ConvNeXt-Tiny on the joint setup

Backbone settings and code (token positions, image-count padding) are in
[ConvNeXt-Tiny](../../../modeling/backbones/convnext-tiny-backbone.md).

**Runs, fold 3, seed 1110484336, same pseudo-labels as the ResNet-50 joint run.**

| Run | Settings | ×1 | ×4 | Training |
|---|---|---:|---:|---:|
| `rn50-joint-fastmri-fold-3` (S148, 0.962) | ResNet-50 recipe | 0.926211 | 0.930745 | 75.5 min, compressed fastMRI |
| `convnext-tiny-joint-fastmri-fold-3` | ResNet recipe unchanged; stopped after epoch 0 (0.889449 against 0.908666) | | | |
| `convnext-tiny-tuned-joint-fastmri-fold-3` | backbone rate 1e-4 falling by 0.7 per stage toward the stem, late L2-SP 0.0001, stochastic depth 0.1 (`configs/convnext-tiny-tuned.yaml` (removed)) | 0.928074 | 0.933246 | 46.6 min, uncompressed fastMRI |

Tuned run, ×1 by epoch against the ResNet-50 joint run: −0.0212, −0.0056, −0.0025, +0.0039,
−0.0005, +0.0006, +0.0014, +0.0017, +0.0025, +0.0019. Its epoch 0 was level with the untuned run's
(0.887509 against 0.889449); the two were not compared beyond that epoch.

×4 by label against the ResNet-50 joint run: Fracture +0.0084, MCL +0.0076, Effusion +0.0062, ACL
+0.0055, PF OA +0.0044, Lateral Meniscus +0.0035, Synovitis +0.0017, Baker's +0.0015, Medial
Meniscus −0.0001, Contusion −0.0008, Medial OA −0.0026, Lateral OA −0.0054. ×4 soft macro
+0.0015, prediction 91 s, peak GPU memory 18.9 GB. An equal blend of the two models' logits
scores 0.933287 at ×4, the ConvNeXt's own score; their logits correlate at 0.98.

The ConvNeXt alone was never submitted.

### S152: the ConvNeXt-Tiny and the ResNet-50 joint models together, 0.963

Equal logit average of `convnext-tiny-tuned-joint-fastmri-fold-3` and `rn50-joint-fastmri-fold-3`
(S148's model), ×4, τ 0.9, bundle `convnext-rn50-joint-f3-x4-tau09-b`: **0.963 public, first in
the score-sorted list**, above S148 (0.962). The hidden rerun scored in 9.5 minutes. The bundle
scores ×4 0.933288 on fold 3 (ConvNeXt alone 0.933246, ResNet-50 alone 0.930745).

On fold 3 the two complement weakly: logits correlate 0.96 to 0.99 by label, the best ConvNeXt
weight is 0.7 (0.933640, picked on the scored fold), and the equal blend beats the better single
model only on Medial Meniscus (+0.0009), Synovitis (+0.0008), Contusion and Lateral Meniscus.
The submission does not separate the ConvNeXt's part from the blend's.

Export notes: the ResNet-50 checkpoint predates the ConvNeXt edits to the hashed model files;
the current source gives bitwise-equal predictions on 24 validation studies, and the bundle
carries a copy stamped with the current identity (`fold_3-current-source`, SHA `da61f29b…`, the
original is `19db4060…`). Ensemble members may now differ in `data.mean` and `data.std`, which
each member applies itself.

### Run B: pretraining on fastMRI, then joint (2026-10-13)

`extra_training.alone` trains on the pseudo-labelled studies only, one epoch a full pass over the
9,828 studies (`configs/convnext-tiny-tuned-fastmri-only.yaml` (removed), run
`convnext-tiny-tuned-fastmri-only-fold-3`); `train --initial-weights` starts the joint run
`convnext-tiny-tuned-pretrained-joint-fastmri-fold-3` from its last epoch, with the L2-SP anchor
on those weights.

| Run, fold 3, seed 1110484336 | ×1 | ×4 | Training |
|---|---:|---:|---:|
| `convnext-tiny-tuned-joint-fastmri-fold-3` (run A, from ImageNet) | 0.928074 | 0.933246 | 46.6 min |
| `convnext-tiny-tuned-fastmri-only-fold-3` (stage 1, no Kaggle label) | 0.918316 | 0.921158 | 55.8 min |
| `convnext-tiny-tuned-pretrained-joint-fastmri-fold-3` (run B) | 0.928659 | 0.933862 | 45.4 min |

Run B against run A: ×1 +0.0006, ×4 +0.0006, ×4 soft macro +0.0013, best epoch 8. ×1 by epoch
against run A: +0.0299, +0.0122, +0.0100, +0.0027, +0.0056, +0.0034, +0.0020, +0.0012, +0.0007,
+0.0006. ×4 by label: Baker's +0.0052, Synovitis +0.0037, Lateral OA +0.0033, PF OA +0.0021,
Medial Meniscus +0.0017, Lateral Meniscus +0.0013, Contusion +0.0008, Fracture −0.0006, ACL
−0.0012, Medial OA −0.0013, Effusion −0.0019, MCL −0.0057. Equal logit blends at ×4: run B with
run A 0.934366 (logit correlation 0.989), run B with the ResNet-50 joint model 0.933382 (0.982).

The whole lead of the head start is gone by the end: the pretrained start gives +0.03 at epoch 0
and +0.0006 at epoch 9, for 56 more minutes of training.

### Run C: pretraining on fastMRI, then Kaggle only (2026-10-13)

`configs/convnext-tiny-tuned-kaggle-only.yaml` (removed) (the tuned recipe without `extra_training`) from
stage 1's last checkpoint, run `convnext-tiny-tuned-pretrained-kaggle-only-fold-3`.

| ConvNeXt-Tiny, fold 3, seed 1110484336 | ×1 | ×4 | Training |
|---|---:|---:|---:|
| A: joint, from ImageNet | 0.928074 | 0.933246 | 46.6 min |
| B: fastMRI only, then joint | 0.928659 | 0.933862 | 55.8 + 45.4 min |
| C: fastMRI only, then Kaggle only | 0.928769 | 0.934310 | 55.8 + 29.4 min |

C against B at ×4: +0.0004 macro, +0.0011 soft macro; by label Lateral OA +0.0078, ACL +0.0056,
Lateral Meniscus +0.0039, PF OA +0.0025, Contusion +0.0015, Medial OA +0.0010, Effusion +0.0008,
Medial Meniscus −0.0007, Synovitis −0.0020, Baker's −0.0025, Fracture −0.0045, MCL −0.0081.
×1 by epoch against B: +0.0005, −0.0011, −0.0040, −0.0009, −0.0047, −0.0018, +0.0004, +0.0002,
−0.0003, +0.0001. Equal logit blends at ×4: C with A 0.935048 (correlation 0.979), C with B
0.934785 (0.989), C with the ResNet-50 joint model 0.934397 (0.971).

**The pretraining ran with untrained poses.** fastMRI has no mask targets, so in stage 1 the bone
map received no supervision (0 mask tokens against 678,281 in run A's last epoch; validation
agreement 0.08) and the frame and crop poses built from it were arbitrary within their limits:
crops 33 px (sagittal) and 28 px (coronal) too high on average, turns of 8° against 3.4°, frame
nod 14° against 3.5°. A stage that trains the map, with masks on fastMRI (MRAnnotator reads its
sequences; the soft-tissue segmenter does not, fastMRI has no DESS or 3D series) or with the map
and frame heads of a trained joint model, had not been tried at this point; fastMRI bone masks
were added from the second round on ([bone masks](../../../modeling/external-data-and-distillation/external-data.md#bone-masks-on-fastmri)).

The three ConvNeXt runs lie within 0.0011 at ×4 on one fold and one seed each; the order A < B < C
was not established beyond that.

### Run B adopted for the ConvNeXt, and main at `288e50a` brought in (2026-10-13)

The user adopted run B's order (fastMRI only, then joint) as the ConvNeXt recipe, as the one that
keeps fastMRI in every step to the end. Main's three adoptions of the targets-and-inputs campaign
came in with the merge:

| Main's change | On the ConvNeXt |
|---|---|
| Average-pool shortcuts | ResNet-50 only (set by backbone name); ConvNeXt downsamples with 2×2 stride-2 convolutions that drop no pixel |
| OA detector beside the shared OA head | Reads the projected tokens and the knee geometry; applies unchanged |
| Mask decoder beside the labels | Adapted: stage widths 192, 384, 768 from the backbone's feature list, and its stride-8 grid at 3.5 px with an 8 px pitch (`<backbone>.layer2`, checked against the layers like the other two grids) in place of the ResNet's 0 px |

None of the three was measured on a ConvNeXt. The ConvNeXt checkpoints of this page (runs A, B, C,
stage 1) and S152's bundle predate the merge and load only with the code kept on the rented
machine at `/workspace/repo-convnext-20261013`. The mask decoder was removed again on 2026-10-14.
