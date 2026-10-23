# fastMRI joint training

Started 2026-10-12. Branch `fastmri-pretraining`, code uncommitted (`extra_training` option). One run so
far, on fold 3; nothing adopted.

## Setup

- **External data:** fastMRI knee DICOMs, 10,009 studies and 47,552 series, on the rented machine at
  `/workspace/data/fastmri` (Vast instance 55528957). JPEG 2000 lossless, read through the pinned
  `pylibjpeg` decoders. Preprocessing audited against the Kaggle path: decoded pixels identical,
  geometry and intensity statistics matching.
- **Pseudo-labels:** five ResNet-50 fold models of main's recipe at `0bd598f` (seed 1110484336; folds
  0 to 2 trained locally, 3 and 4 on the rented machine), ×4, τ 0.9, mirror TTA. One row per study
  and teacher in `/workspace/data/labels/prepared/fastmri-rn50-main-x4-tau09-tta/`; the scripts that
  made them are beside the predictions in
  `/workspace/predictions/fastmri/rn50-main-5fold-x4-tau09-tta/source/`. Mean spread between teachers
  0.018 to 0.029 per label.
- **Joint training:** every Kaggle batch is followed by a fastMRI batch, one optimizer step per pair,
  equal weight. The fold-k run reads teacher k's labels. fastMRI studies are drawn at random without
  repeats until all are used. L2-SP is applied on the Kaggle batch only; class weights come from the
  Kaggle rows only. Validation is the Kaggle fold alone.
- **Excluded from training:** 181 fastMRI studies (166 with a single series) whose knee geometry
  cannot be fitted. Their positions are all zero and `torch.hypot` in the routing weights has no
  finite gradient there, so one such study corrupts the weights. Listed in
  `excluded_from_training.csv`; the training table has 9,828 studies. Inference on them is fine.

## Fold 3, `rn50-joint-fastmri-fold-3` against `rn50-main-0bd598f-folds-3-4`

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

**S148, 0.962 public on 2026-10-12**, is the best public score so far: above the five-teacher
ensemble S140 (0.961) and every single ResNet-50 (S143 0.960, fold 1 with 0.25 teacher share). The
fold 3 control has no public score, so the fold's own share of that is unknown.

PF OA is the largest loss; it is also the label with the highest mean pseudo-label (0.48) and the
largest spread between teachers (0.029).

## Variants discussed, not run

The first three and six more are the nine seeds of the
[fastMRI joint training campaign](../../autoresearch/campaigns/fastmri-joint/INDEX.md), written,
validated and not started.

- Kaggle-only final epochs (for example the last three of ten), while the learning rate still moves
  the weights.
- L2-SP off in the joint run (`optimizer.anchor_alpha_early: 0`, `optimizer.anchor_alpha_late: 0`).
  L2-SP was adopted on the ResNet-18 for unseen sites and never tested on the ResNet-50; in the joint
  run its pull relative to the task gradient is already lower, since each step sums two batches.
- Class weights computed over Kaggle and fastMRI together.
- Children distilled from the joint teachers' held-out Kaggle predictions, keeping the current
  fastMRI labels.
- Mask predictions as extra fastMRI targets.

## Scanners in fastMRI against Kaggle

One DICOM header per study, read on 2026-10-13 (`/workspace/data/fastmri/study_headers.parquet` on
the rented machine); 9,828 training studies. Institution and station names are blank
(de-identified); 39 distinct device serial numbers.

| | fastMRI | Kaggle (4,407 studies) |
|---|---|---|
| Siemens | 81% | 44% |
| GE | 9% | 21% |
| Philips | 2% | 30% |
| Hitachi / Fujifilm | 5% | 0.5% |
| Toshiba / Canon | 0% | 5% |
| Unknown | 3% | 0% |
| 1.5 T | 69% | 57% |
| 3 T | 30% | 36% |

fastMRI's models are mostly older Siemens generations (Symphony 1,628, Skyra 1,423, Espree 1,309,
Aera 1,273, Verio 1,006, Sonata 567, Avanto 455), then Hitachi Echelon 387, GE Signa HDe 297 and
Excite 269, Philips Achieva 233. Kaggle's are newer and more mixed (Ingenia 741, Vida 447, Aera
400, Achieva dStream 327, Avanto fit 262, Signa Explorer 213, Signa Artist 208). The two share
Aera, Skyra, Signa Excite and Prisma. So fastMRI adds many scanners, but almost all of one vendor:
it widens the Siemens and Hitachi side and adds next to nothing for Philips, Toshiba or Canon.

## fastMRI stored uncompressed since 2026-10-13

The JPEG 2000 decoder (`pylibjpeg-openjpeg` 2.5.0) crashed a loader worker with a segmentation
fault in three of twelve campaign trainings, after 29 to 55 minutes each; the fault is inside
`openjpeg`'s `decode_pixel_data` and is sporadic, not tied to a file. The decoder was removed from
the pinned dependencies (Pillow's decoder gives identical pixels on 1,500 images but takes 13.8 ms
per image against 7.2 ms), and the 47,585 catalogued series on the rented machine were then
rewritten in place as uncompressed DICOM, keeping every identifier:

- 1,575,673 images converted in 27 minutes, 18 already uncompressed, 0 failed.
- 4,000 sampled images hashed before and after: 0 differ; none of 1,500 checked is still
  compressed.
- fastMRI takes 432 GB instead of 164 GB; 184 GB of the 1,200 GB disk are free.
- The compressed originals are gone from that machine; restoring them means downloading again.

Training no longer decodes fastMRI images, so the joint training is faster than the campaign's
reference (4,533 s): the `fastmri-joint` campaign's time limits no longer fit and it stays
stopped.

## ConvNeXt-Tiny on the joint setup, 2026-10-13

Backbone support was rebuilt on this branch (the earlier worktree was removed):
`convnext_tiny.in12k_ft_in1k` with ImageNet normalisation; token centres at 15.5 px with a 32 px
pitch for the deep read and 7.5 px with 16 px for the stride-16 read, measured on a 2,560 px input
as the centre of each token's receptive field and equal to the mapping (the ResNet-50's 0 px
origins were confirmed by the same method); the stride-8 feature augmentation after the second
stage; the L2-SP "early" group is the stem and first stage.

**Batch size and speed.** ConvNeXt pays 4.4 s the first time each batch size is seen and 120 ms
afterwards (11.7 s with `cudnn.benchmark` on; 214 ms flat with cuDNN off; the ResNet-50 has no
such cost). The model therefore pads the image count of every ConvNeXt backbone call to a
multiple of 16 with blank images and drops their outputs; ConvNeXt has no batch statistics, and
the padded outputs match the unpadded ones to 3e-5.

**Runs, fold 3, seed 1110484336, same pseudo-labels as the ResNet-50 joint run.**

| Run | Settings | ×1 | ×4 | Training |
|---|---|---:|---:|---:|
| `rn50-joint-fastmri-fold-3` (S148, 0.962) | ResNet-50 recipe | 0.926211 | 0.930745 | 75.5 min, compressed fastMRI |
| `convnext-tiny-joint-fastmri-fold-3` | ResNet recipe unchanged; stopped after epoch 0 (0.889449 against 0.908666) | | | |
| `convnext-tiny-tuned-joint-fastmri-fold-3` | backbone rate 1e-4 falling by 0.7 per stage toward the stem, late L2-SP 0.0001, stochastic depth 0.1 (`configs/convnext-tiny-tuned.yaml`) | 0.928074 | 0.933246 | 46.6 min, uncompressed fastMRI |

Tuned run, ×1 by epoch against the ResNet-50 joint run: −0.0212, −0.0056, −0.0025, +0.0039,
−0.0005, +0.0006, +0.0014, +0.0017, +0.0025, +0.0019. Its epoch 0 was level with the untuned run's
(0.887509 against 0.889449); the two were not compared beyond that epoch.

×4 by label against the ResNet-50 joint run: Fracture +0.0084, MCL +0.0076, Effusion +0.0062, ACL
+0.0055, PF OA +0.0044, Lateral Meniscus +0.0035, Synovitis +0.0017, Baker's +0.0015, Medial
Meniscus −0.0001, Contusion −0.0008, Medial OA −0.0026, Lateral OA −0.0054. ×4 soft macro
+0.0015, prediction 91 s, peak GPU memory 18.9 GB. An equal blend of the two models' logits
scores 0.933287 at ×4, the ConvNeXt's own score; their logits correlate at 0.98.

One run on a fold that has misranked two candidates against the public board today; not
submitted yet. Inference on a Kaggle T4 has not been timed for this backbone.

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
9,828 studies (`configs/convnext-tiny-tuned-fastmri-only.yaml`, run
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
and +0.0006 at epoch 9, for 56 more minutes of training. Run C follows.

### Run C: pretraining on fastMRI, then Kaggle only (2026-10-13)

`configs/convnext-tiny-tuned-kaggle-only.yaml` (the tuned recipe without `extra_training`) from
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
and frame heads of a trained joint model, has not been tried.

The three ConvNeXt runs lie within 0.0011 at ×4 on one fold and one seed each; the order A < B < C
is not established beyond that.

### Run B adopted for the ConvNeXt, and main at `288e50a` brought in (2026-10-13)

The user adopted run B's order (fastMRI only, then joint) as the ConvNeXt recipe, as the one that
keeps fastMRI in every step to the end. Main's three adoptions of the targets-and-inputs campaign
came in with the merge:

| Main's change | On the ConvNeXt |
|---|---|
| Average-pool shortcuts | ResNet-50 only (set by backbone name); ConvNeXt downsamples with 2×2 stride-2 convolutions that drop no pixel |
| OA detector beside the shared OA head | Reads the projected tokens and the knee geometry; applies unchanged |
| Mask decoder beside the labels | Adapted: stage widths 192, 384, 768 from the backbone's feature list, and its stride-8 grid at 3.5 px with an 8 px pitch (`<backbone>.layer2`, checked against the layers like the other two grids) in place of the ResNet's 0 px |

None of the three is measured on a ConvNeXt. Today's ConvNeXt checkpoints (runs A, B, C, stage 1)
and S152's bundle predate the merge and load only with the code kept on the rented machine at
`/workspace/repo-convnext-20261013`; the merged code needs new trainings. `configs/training.yaml`
is main's ResNet-50 recipe with `extra_training` and fold 3.

### Bone masks on fastMRI (2026-10-13)

fastMRI has no DESS or other 3D series (0 of 47,585 series), so the soft-tissue segmenter cannot
read it; MRAnnotator can. Poses are built from the bone map only (study frame, then crop turn,
shift and scale); the soft-tissue map only routes label reads.

- **Sample:** 1,230 of the 9,828 training studies, 12.5% as for Kaggle (550 of about 4,410),
  seed 20261013, allocated across scanner models by the square root of their study counts
  (Siemens 66% of the masks against 81% of the studies), one sagittal series each, the one with
  most images; 541 fat-suppressed, 689 not.
- **Chain, on the rented machine** (`~/.cache/knee-segmenters/fastmri1230`, segmenter environment
  rebuilt there with the local versions): conversion, MRAnnotator with `STEP=1.0` (14.3 minutes,
  0.7 s per knee), `segmenter_export_targets.py`, `segmenter_export_joint_targets.py` (4 mm
  closing, 135°, 2.5 mm) and `mask_measurements.py` of `scripts/adhoc/side_routed_session/`; the
  choice and conversion scripts (`fastmri_choose.py`, `fastmri_convert.py`) are beside the sample.
- **Against the Kaggle 550, medians:** joint 27.3 ml (29.7), kneecap joint 15.1 (18.3), femur 150.7
  (167.1), tibia 97.2 (109.7), kneecap 12.1 (12.9), fibula 7.8 (8.6); kneecap front 54.3 mm
  (55.9), back 38.7 (40.0), top 53.5 (54.6), bottom 21.1 (21.3), joint gap 6.1 (6.5 and 6.3).
  Two contact sheets (twelve scanners at random, and the smallest bones) show clean bones and
  joint classes on every scanner model looked at.
- **Dropped, 10 knees:** two with no fitted plateau and eight with femur under 80 ml or tibia
  under 45 ml (one is not a knee image, the others are partial bones).
- **Targets:** `data/labels/prepared/anatomy-mrannotator-fastmri1230-joints` (all 1,230) and the
  folder to train from, `anatomy-mrannotator-sample550-fastmri1230-joints` (550 Kaggle and 1,220
  fastMRI volumes linked, `sources.json` with the rule and the dropped knees). No config points
  at it yet and no run has used it.

## Main's recipe at `288e50a` on other folds (2026-10-13)

Meant as the five teachers for a new pseudo-label round: main's exact tree (`git archive`,
`/workspace/repo-main-288e50a` on the rented machine), `configs/training.yaml`, seed 1110484336,
Kaggle only. Reference per fold: the teacher of the current pseudo-labels (folds 0 to 2
`rn50-aligned-mixing-folds-0-2`, fold 3 `rn50-main-0bd598f-folds-3-4`, fold 4
`rn50-main-0bd598f-fold-4-workers-24`), same seed and split.

| Fold | Run | ×1 new / previous | Δ ×1 | ×4 new / previous | Δ ×4 |
|---|---|---|---:|---|---:|
| 0 | `rn50-main-288e50a-folds-0-4` | 0.934235 / 0.933184 | +0.0011 | 0.940416 / 0.940906 | −0.0005 |
| 1 | `rn50-main-288e50a-folds-0-4` | 0.923958 / 0.925323 | −0.0014 | 0.932858 / 0.933392 | −0.0005 |
| 2 | campaign run 27, copied as `rn50-main-288e50a-fold-2-campaign-run-27` (same source identity, seed and settings as main) | 0.930902 / 0.928460 | +0.0024 | 0.939634 / 0.937000 | +0.0026 |

Folds 3 and 4 were stopped by the user at fold 3's fourth epoch. Training takes 27.7 minutes a
fold on the rented machine; on the local one the new and the previous recipe both take 44.9.

×4 by label, new against previous (fold 0, fold 1, fold 2): PF OA −0.0035, −0.0025, −0.0029;
Lateral OA −0.0040, −0.0020, +0.0065; Medial OA +0.0001, +0.0017, +0.0014; MCL +0.0022, −0.0066,
+0.0108; Fracture −0.0026, +0.0009, +0.0041; Baker's −0.0034, +0.0004, +0.0036; Synovitis +0.0026,
−0.0016, +0.0008; Lateral Meniscus +0.0020, +0.0035, −0.0016; Contusion −0.0003, +0.0040, +0.0054.

The recipe was selected on fold 2 over 29 campaign runs, each adoption a step of about one repeat
distance there (0.0005, 0.0011, 0.0010 at ×4; a repeat moves 0.0010).

**Fold 0 without the OA detector** (`rn50-main-288e50a-no-oa-detector-fold-0`, the same tree with
the line adding the detector's correction to the OA logits removed, so every other weight starts
equal): ×1 0.933310 (−0.0009 against the run with it, +0.0001 against the previous recipe), ×4
0.940484 (+0.0001, −0.0004). ×4 by label against the run with the detector: Baker's +0.0034, MCL
+0.0027, Lateral OA +0.0012, PF OA +0.0010, Effusion +0.0009, ACL +0.0009, Medial Meniscus
+0.0002, Synovitis −0.0004, Medial OA −0.0012, Lateral Meniscus −0.0013, Contusion −0.0015,
Fracture −0.0051. Against the previous recipe PF OA is still −0.0025 and Lateral OA −0.0028, so
the detector is not what lowers them on this fold; the mask decoder and the average-pool
shortcuts are not separated.

**Fold 0 without the average-pool shortcuts** (`rn50-main-288e50a-no-avg-pool-shortcuts-fold-0`,
the same tree with the two lines that switch them on removed; mask decoder and OA detector kept):
×1 0.933821 (−0.0004 against the new recipe, +0.0006 against the previous), ×4 0.940342 (−0.0001,
−0.0006). ×4 by label against the new recipe: MCL +0.0022, Baker's +0.0013, ACL +0.0013, Lateral
Meniscus +0.0012, Contusion +0.0010, PF OA +0.0007, Synovitis +0.0004, Medial OA −0.0004, Medial
Meniscus −0.0011, Effusion −0.0018, Lateral OA −0.0027, Fracture −0.0029.

Fold 0 at ×4, four trainings: previous recipe 0.940906, new recipe 0.940416, without the OA
detector 0.940484, without the shortcuts 0.940342. The three runs from main's tree are within
0.00014 of each other and 0.0004 to 0.0006 under the previous teacher, and share its per-label
differences: PF OA −0.0035, −0.0025, −0.0028; Lateral OA −0.0040, −0.0028, −0.0067; Fracture
−0.0026, −0.0077, −0.0055; MCL +0.0022, +0.0049, +0.0044; Synovitis +0.0026, +0.0023, +0.0031.
What the three share and the previous teacher lacks: the mask decoder, the commits between the
teacher's code and `288e50a` other than the three adoptions, and the machine (the previous
teacher was trained elsewhere). Those are not separated; neither the OA detector nor the
shortcuts moves fold 0.

**Fold 0 without the mask decoder** (`rn50-main-288e50a-no-mask-decoder-fold-0`, the same tree
with the line adding the decoder's loss removed; OA detector and shortcuts kept): ×1 0.934069
(−0.0002 against the new recipe, +0.0009 against the previous), ×4 0.940614 (+0.0002, −0.0003).
×4 by label against the new recipe: Baker's +0.0046, Synovitis +0.0020, Lateral Meniscus +0.0018,
MCL +0.0014, Contusion +0.0002, Effusion 0.0000, ACL −0.0001, Medial Meniscus −0.0005, Fracture
−0.0011, PF OA −0.0014, Medial OA −0.0015, Lateral OA −0.0029.

**All five fold 0 trainings, ×1 / ×4:** previous teacher 0.933184 / 0.940906; new recipe 0.934235
/ 0.940416; without the OA detector 0.933310 / 0.940484; without the shortcuts 0.933821 /
0.940342; without the mask decoder 0.934069 / 0.940614. The four runs from main's tree span
0.0003 at ×4 and 0.0009 at ×1: no adoption changes fold 0 when removed. Against the previous
teacher all four keep PF OA (−0.0025 to −0.0048), Lateral OA (−0.0028 to −0.0069) and Fracture
(−0.0026 to −0.0077) lower and MCL (+0.0022 to +0.0049) and Synovitis (+0.0023 to +0.0047)
higher, so that pattern belongs to none of the three adoptions; what remains is the code between
`0bd598f` (with its uncommitted changes) and `288e50a` other than the adoptions, and the machine.

**Folds 3 and 4, trained afterwards** (`rn50-main-288e50a-folds-3-4`, main's tree unchanged):

| Fold | ×1 new / previous | Δ ×1 | ×4 new / previous | Δ ×4 | Training |
|---|---|---:|---|---:|---:|
| 3 | 0.925043 / 0.925655 | −0.0006 | 0.931805 / 0.931023 | +0.0008 | 27.4 min |
| 4 | 0.926211 / 0.926103 | +0.0001 | 0.933838 / 0.934064 | −0.0002 | 28.8 min |

×4 by label, fold 3: Synovitis +0.0036, Lateral Meniscus +0.0033, Baker's +0.0032, Effusion
+0.0017, MCL +0.0009, Lateral OA +0.0006, ACL 0.0000, PF OA −0.0002, Fracture −0.0004, Contusion
−0.0006, Medial Meniscus −0.0012, Medial OA −0.0016. Fold 4: Contusion +0.0033, Effusion +0.0031,
PF OA +0.0028, Fracture +0.0024, Lateral OA +0.0016, Medial Meniscus +0.0011, Baker's −0.0003,
Medial OA −0.0003, ACL −0.0011, Lateral Meniscus −0.0026, Synovitis −0.0064, MCL −0.0064.

**Five folds of main's recipe at `288e50a` against the previous teachers:** mean ×4 0.935710
against 0.935277 (+0.0004), mean ×1 0.928070 against 0.927745 (+0.0003). Without fold 2, where
the recipe was selected: ×4 −0.0001, ×1 −0.0002 over four folds. No label moves the same way on
all four fresh folds. The fold 3 run stopped earlier from the same code, seed and machine had
0.900969 and 0.911379 at its first two epochs against 0.901889 and 0.908213 here: two identical
launches differ by up to 0.003 within an epoch.

The five teachers of the new recipe: folds 0 and 1 in `rn50-main-288e50a-folds-0-4`, fold 2 in
`rn50-main-288e50a-fold-2-campaign-run-27`, folds 3 and 4 in `rn50-main-288e50a-folds-3-4`, all
on the rented machine with main's source identity.

## Second round: new teachers' labels, averaged, with bone masks on fastMRI (started 2026-10-13 night)

- **Pseudo-labels:** the five teachers of main's recipe at `288e50a`, ×4, τ 0.9, mirror TTA, 41
  minutes; `/workspace/predictions/fastmri/rn50-main-288e50a-5fold-x4-tau09-tta/` (scripts in
  `source/`) and `data/labels/prepared/fastmri-rn50-main-288e50a-x4-tau09-tta/`. Against the first
  round, same study and fold: correlation 0.978 to 0.995 per label, mean absolute difference 0.015
  to 0.026 (the spread between teachers inside one round is 0.018 to 0.030).
- **Averaged labels (user decision):** `training_table_averaged.parquet`, the mean of the five
  teachers per study repeated for every fold, 9,828 studies. Not held out: fold scores of runs on
  it are inflated through the four teachers that trained on the fold and are not comparable with
  earlier runs.
- **Class weights in a fastMRI-only stage (user decision):** from the fastMRI labels, each label's
  mean soft target as prevalence (`fixed_class_balance(..., soft_prevalence=True)`, the formula of
  the targets-and-inputs soft-prevalence candidate). Joint stages keep the Kaggle rows with the
  thresholded count.
- **Bone masks:** every run reads `anatomy-mrannotator-sample550-fastmri1230-joints`. In the
  fastMRI-only stage's first epoch the bone map reaches 0.90 agreement on 1.81 million mask tokens
  and the crops sit at +0.4 px (sagittal) and +2.7 px (coronal) with 3.7° turns, against 0.08, −33
  px, −28 px and 8.4° without fastMRI masks.
- **Chain on the rented machine** (`/workspace/outputs/logs/night_chain.sh`, state in
  `night_chain.state`), this worktree's merged code, seed 1110484336:
  1. `convnext-pretrain-fastmri-avg-labels-masks`: ConvNeXt-Tiny tuned recipe, fastMRI only, ten
     full passes, nominal fold 0;
  2. `rn50-joint-avg-labels-masks-folds-0-4`: main's ResNet-50 recipe, joint, five folds;
  3. `convnext-pretrained-joint-avg-labels-masks-folds-0-4`: ConvNeXt joint from the pretraining's
     last epoch, five folds;
  4. τ search (`tau_predict.py`, `tau_score.py` in `/workspace/outputs/logs/`): τ 0.70 to 1.00 by
     0.05 at ×4, ×5, ×6, no mirror, out-of-fold, per backbone and for their equal blend, results in
     `/workspace/outputs/analysis/tau-search/`.
- **τ and caching:** τ also moves the first look (frame and crop poses; up to 2 px at τ 0.7), so
  pooling cached tokens at another τ is off by up to 0.29 logits. Sharing only the first look's
  backbone pass between τ values is not exact either: it failed its check on ResNet-50 fold 0
  (0.0038 logits; per τ mostly under 0.0002 with single jumps of 0.0065 and 0.028, a discrete pose
  choice flipping), while two direct passes agree within 0.00005. It saved only ~16% of the
  backbone images (224 of ~1,200 per batch), so the search runs the direct path for every τ
  (`tau_predict.py`; the cached version kept as `tau_predict_cached.py`).

### Results (2026-10-14)

Fold scores are inflated by the averaged labels (four of the five teachers saw each validation
fold); compare runs only within this round.

| ×1 macro AUC | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
|---|---|---|---|---|---|---|
| ConvNeXt pretrained, then joint | 0.940925 | 0.932857 | 0.937863 | 0.933431 | 0.934253 | 0.935866 |
| ResNet-50 joint | 0.936159 | 0.928819 | 0.934051 | 0.928780 | 0.929416 | 0.931445 |

- ConvNeXt ahead on every fold, +0.0038 to +0.0048 (mean +0.0044). Training 47.6 to 50.1 min per
  fold for the ConvNeXt, 43.8 to 46.0 for the ResNet-50; pretraining 62.5 min, ×1 0.932150 on fold
  0 (its own fastMRI-only model). The ResNet-50's own ×4 per fold: 0.941205, 0.936327, 0.939919,
  0.934879, 0.935158 (mean 0.937498). No windowed-resampling fallback in any run.
- **τ and multiplier (ResNet-50, five folds, mean of folds):** flat. Best ×4 τ 0.85 0.937275
  against ×4 τ 0.9 0.937215 (+0.00006); the whole grid spans 0.0003; ×5 and ×6 never beat ×4.
  The same weights at ×4 τ 0.9 score 0.937498 in the runs' own prediction, 0.0003 apart from the
  search's (different image draws): the draw noise is as large as the whole τ effect.
- **Per fold** the best setting moves without a pattern shared with the model: ResNet-50 folds 0
  to 4 prefer τ 0.70, 1.00, 0.85, 0.70 (×4) and 0.90 (×6), gains over ×4 τ 0.9 of +0.00031,
  +0.00005, +0.00009, +0.00007, +0.00065. ConvNeXt folds 0 to 2 prefer ×4 τ 0.75, 1.00, 0.85
  (+0.00013, +0.00009, +0.00008); ×4 beats ×5 and ×6 by 0.0005 to 0.0008 on each. The leaning of
  τ follows the fold (fold 0 low, fold 1 high, fold 2 middle for both backbones). Decision: keep ×4
  τ 0.9.
- **Equal blend vs the ConvNeXt alone (×4 τ 0.9, folds 0 to 2):** 0.944084 vs 0.944282, 0.938865
  vs 0.938333, 0.942621 vs 0.942928; the blend does not beat the ConvNeXt alone here (−0.0002,
  +0.0005, −0.0003).
- The ConvNeXt τ predictions were stopped by the user after folds 0 to 2 (×4, ×5, ×6) and fold 3
  (×4); scores in `/workspace/outputs/analysis/tau-search/` (per-fold tables from
  `tau_per_fold.py` there). `last.ckpt` removed from the two five-fold runs; the pretraining's
  `last.ckpt` kept, it is the joint run's recorded starting weights.

### Distilled students on all Kaggle folds (2026-10-14)

Single students trained on all five Kaggle folds at 0.25 teacher share from the ten-model S154
ensemble, on the fastMRI mean-of-10 labels; ×4, τ 0.85, no TTA, batch 4, 8 workers. Fold-0
validation is inflated by training on that fold.

- **S156, the ConvNeXt-Tiny student: 0.964 public**, second of the three 0.964 in the
  score-sorted list (S154 first, S155 third).
- **S157, the ResNet-50 student (`rn50-distilled-student-full-data`, ×1 0.951254 on fold 0): 0.962
  public**, first of the 0.962 (above S148), below S152 (0.963). The ConvNeXt student is ahead
  by one rounding step. Two notebooks hung in `notebook_running` (24 and 7 min, against 1.5 to
  2.5 min for earlier single ResNet-50s at ×4) and were deleted, and one relaunch failed on the
  deleted slug; the fourth (`claverru/s157-resnet-50-distilled-student-x4-tau0-85-c`) ran in
  91 s.

## Third round: 15 silver teachers, fastMRI + OAI, pretraining then joint with a fade (2026-10-14 night to 2026-10-15)

- **Teachers:** five folds each of main's ResNet-50 (folds 0 to 3 on the local 3090), ConvNeXt-Tiny
  and MambaVision-T configs, Kaggle silver labels only. ×4 per fold: ConvNeXt 0.93616, 0.92928,
  0.93612, 0.93183, 0.93265; MambaVision 0.93485, 0.93030, 0.93516, 0.93021, 0.92979; ResNet-50
  fold 4 0.93314. Out of fold (×4), mean of folds: ResNet-50 0.935157, ConvNeXt 0.933210,
  MambaVision 0.932061; equal logit blend of the three 0.936992 (+0.0018 over ResNet-50, ahead on
  every fold by +0.0007 to +0.0024).
- **Labels:** the 15 teachers' probability mean at ×4, τ 0.85, mirror TTA on 9,828 fastMRI and
  9,437 OAI studies (`data/labels/prepared/silver15-avg-x4-tau085-tta`).
- **Masks:** bones on 1,946 fastMRI knees (`anatomy-mrannotator-sample550-fastmri1966-joints`),
  soft tissue on 1,819 OAI DESS knees with Kaggle's 263 (`soft-tissue-nnunet-knee-dess265-oai`).
- **Pretraining** (`configs/{family}-silver15-pretrain.yaml`, external only, fold 0 nominal,
  inflated through the teachers): best ×1 / ×4 ConvNeXt 0.93458 / 0.93636, MambaVision 0.93474 /
  0.93684, ResNet-50 0.93239 / 0.93511. Training-time meniscus overlap 0.66 to 0.67 against 0.59
  for a Kaggle-only ConvNeXt. Persistent DataLoader workers of three concurrent runs over the
  19k-study index filled the 251 GB host and killed one run at epoch 2; the runs resumed from
  their last epoch with `trainer.persistent_workers: false`.
- **Fade fix:** the quarter-cosine weight measured progress against `estimated_stepping_batches`,
  which counts both batches of a pair, so it would have ended at 0.71; it now uses the optimizer
  step count (deployed before the first joint run).
- **S158, `convnext-silver-fold-0` (no pretraining), ×4, τ 0.85, no TTA: 0.957 public**, first of
  three at 0.957 (S158 > S106 > S119). Baseline of the fine-tuning probe on ConvNeXt fold 0.
- **S159, `convnext-silver15-joint-fold-0` (pretraining, then joint with the fade), ×4, τ 0.85, no
  TTA: 0.961 public**, +0.004 over S158 on the same fold; fourth of five at 0.961 (S150 > S140 >
  S141 > S159 > S149). Local ×4 0.945443 against the teacher's 0.936161 (inflated through the
  teachers). Joint ×4 against the fold's silver teacher so far: ConvNeXt fold 1 0.939754 (+0.0105),
  ResNet-50 fold 0 0.944509 (+0.0054).

### Open questions for the final recipe (2026-10-15)

Local validation scores against silver labels, so it cannot judge choices that move the model
toward or away from silver (teacher share above all); only the public leaderboard can. Budget:
five submissions a day (reset 00:00 UTC), one kept every day for a big ensemble, so at most four
probes a day, each with about 0.001 of rounding and test noise.

Fold 0, ×4, τ 0.85, no TTA; local ×4 with pretraining is inflated through the teachers.

| Cell | Pretraining | Fine-tuning | Kaggle targets | Status | Local ×4 | Public |
|---|---|---|---|---|---|---|
| C1 ConvNeXt | No | Kaggle only | Silver | submitted, S158 | 0.936161 | 0.957 |
| C2 ConvNeXt | No | Kaggle only | Silver + teacher share | not trained | | |
| C3 ConvNeXt | No | Joint + fade | Silver | not trained | | |
| C4 ConvNeXt | No | Joint + fade | Silver + teacher share | not trained | | |
| C5 ConvNeXt | Yes | Kaggle only | Silver | submitted, S160 (trained on the local 3090, 4.7 min per epoch) | 0.945279 | 0.961, above S159 in the tie |
| C6 ConvNeXt | Yes | Kaggle only | Silver + teacher share | trained 2026-10-16 on the local 3090: S160 + share 0.75 of `teachers2-kaggle-x4-tau085-tta` (run `convnext-silver15-pretrain-share075-finetune-fold-0`); submitted, S167 | 0.944995 | **0.964, first in score-sorted order, above S163** |
| C7 ConvNeXt | Yes | Joint + fade | Silver | submitted, S159 | 0.945443 | 0.961 |
| C8 ConvNeXt | Yes | Joint + fade | Silver + teacher share | trained 2026-10-17 on the rented machine (`convnext-silver15-joint-share075-fold-0`, S159 config + share 0.75), submitted, S168 | 0.944662 | 0.964, below S167 in the tie (5th of eight 0.964s) |
| R1 ResNet-50 | No | Kaggle only | Silver | trained (`resnet50-silver-fold0`) | 0.939145 | |
| R3 ResNet-50 | No | Joint + fade | Silver | not trained | | |
| R7 ResNet-50 | Yes | Joint + fade | Silver | trained (`resnet50-silver15-joint-fold-0`) | 0.944509 | |

R2, R4, R5, R6 and R8 mirror the ConvNeXt cells and are not trained. MambaVision fold 0: silver
teacher ×4 0.93485, joint 0.946309, neither submitted. The teacher share is a dial (0.25, 0.5,
0.75, …), not a switch, so the grid cannot be covered.

Questions to resolve:

1. **Pretraining:** yes for ConvNeXt (C7 against C1, +0.004 public).
2. **Joint + fade or Kaggle-only fine-tuning after pretraining:** C5 against C7: both 0.961 public
   (S160 above S159 among six at 0.961); locally ×4 0.945279 against 0.945443, logit correlation
   0.992. Per label they split: silver-only minus joint is +0.0027 on bone + fluid (Fracture,
   Contusion, Effusion, Synovitis) and −0.0019 on menisci + OA, contrast +0.0046 (study bootstrap
   95% [+0.0010, +0.0092]). Equal blend +0.0004 over C7, routing joint on menisci/OA and silver-only
   elsewhere +0.0006 (chosen on the same fold). Hypothesis: OAI is an osteoarthritis cohort with
   noisy injury pseudo-labels, amplified by Kaggle-prevalence class weights; a per-label external
   weight could give both in one training.
3. **Teacher share on the Kaggle rows** (soft targets from the 3-family out-of-fold mean of the
   15 silver teachers): open; the student-distillation campaign's 0.25 was chosen on silver
   validation and S156's 0.964 confounds it with full data and other teachers. One point far
   from zero (for example 0.5) on the current best path, against it; default 0 (S159).
4. **ResNet-50 pretraining:** pretraining + joint (R7) or joint alone from RadImageNet (R3); it
   gains less from the external stage (+0.005 against +0.008 to +0.012).
5. **ResNet-50 as a student in the final ensemble:** good teacher (best single family out of fold,
   +0.0015 in pairs) but weakest student; out-of-fold ensembles of the 15 joint models with and
   without it, then the ensemble slots (10 ConvNeXt + MambaVision joint models one day, all 15
   the next).
6. **Final form:** fold models or full-data students, and the ensemble's members and weights.
7. **Per-label ensemble weights (specialists):** on the 15 silver teachers' out-of-fold predictions,
   weights fitted per label on four folds and scored on the fifth gain +0.0004 macro over equal
   weights (per fold −0.0003, +0.0002, +0.0003, +0.0004, +0.0012), against +0.0001 for one global
   weight set. Specialists (weights on all folds, ResNet-50 / ConvNeXt / MambaVision): ResNet-50 for
   Lateral OA (0.8/0.2/0.0), Fracture (0.8/0.0/0.2), Medial OA (0.7/0.2/0.1) and PF OA; MambaVision
   for Synovitis (0.3/0.1/0.6); ConvNeXt for MCL (0.2/0.6/0.2). Judged on silver; repeat on the joint
   models and confirm through an ensemble slot.

8. **OAI labels only where they carry signal (agreed 2026-10-15, joint recipe for tonight):** per
   source and label, the 15-teacher pseudo-labels show OAI (an osteoarthritis cohort) has almost no
   injuries (≥ 0.5 on 0.1% of studies for Fracture and MCL, 0.2% Contusion, 2.4% ACL) and
   uninformative fluid labels (confident, < 0.1 or > 0.9, on 0.1% for Effusion and 0% for
   Synovitis; Baker's 13% against 55% on fastMRI), while menisci and OA have rates close to
   Kaggle's. Recipe: fastMRI keeps all twelve labels; OAI keeps only Medial Meniscus, Lateral
   Meniscus, Medial OA, Lateral OA and PF OA (Baker's dropped with the fluid labels); OAI masks
   supervise the maps as before. Implementation: empty values for the dropped labels on the OAI
   rows of the external label table, and the external dataset marks empty labels unobserved (the
   masked loss already skips them); joint stage only, from the existing pretrainings.
   The existing pretrainings stay: C5 fine-tuned from the same pretraining was the better model on
   bone and fluid, so a Kaggle-only stage removes whatever the pretraining took from OAI, and
   re-pretraining three families plus redoing every joint run would cost most of a day. OAI's
   injury labels are mostly correct easy negatives; the fluid labels are the uninformative ones.
   Any future pretraining (for example a final round with new teachers) uses the same per-label
   rule from the start, more so if tonight's joint run shows bone and fluid recovering while
   menisci and OA keep their gain. Fallback when other ideas run out: redo the whole loop
   (pretraining, then joint) with OAI's bone and fluid labels dropped from the start.

Way through: climb from the current best (C7), one change per probe, keep a change only if it
beats the best, stop when the probe budget runs out; settled axes stay fixed (L2-SP as each
family's recipe in every stage). Hedge the rest through the final ensemble, mixing the trusted
recipe with competitive branches instead of choosing one.
- **15 joint models (2026-10-15 evening):** ×4 per fold ConvNeXt 0.945443, 0.939754, 0.944991,
  0.939364, 0.940230; MambaVision 0.946309, 0.937811, 0.946394, 0.939305, then fold 4; ResNet-50
  0.944509, 0.938544, 0.942666, 0.937400, then fold 4 (all inflated through the teachers). Out of
  fold, mean of folds: ConvNeXt 0.941956, MambaVision 0.941629, ResNet-50 0.940112, all fifteen at
  equal weights 0.943208, per-label family weights pulled half-way to equal 0.943400 (fitted on
  four folds, scored on the fifth; weights in
  `outputs/submission/label-weights/joint15-per-label-halfway.json`). Submitted as S161 (equal); S162
  (per-label) was cancelled by the user before scoring (bundle kept at
  `outputs/submission/exports/joint15-3fam-perlabel-x4-tau085-tta`); S161 is ×4, τ 0.85, mirror TTA, exported on the rented machine from this worktree's
  code (the per-label weights are an inference-only addition: `export.py --label-weights`,
  `models/label_weights.json`, a weighted per-label mean in the engine).

### Workflow from 2026-10-15 (user decision)

1. Local: search new architectures, kept only once shown to be good teachers.
2. Teachers: evaluate on silver and gold out-of-fold predictions (58 gold studies), per family and
   per combination, and build pseudo-labels only from the best teachers.
3. Students: train them on those labels and keep the best (ensemble contribution out of fold, then
   the leaderboard).
4. Recipe: pretraining or not, joint or not, teacher share on Kaggle rows, and the rest of the open
   questions above.

Teachers and students are judged separately. First reading on the silver teachers (out-of-fold
macro, silver / gold): ResNet-50 0.934632 / 0.938411, ConvNeXt 0.932376 / 0.937444, MambaVision
0.931554 / 0.928401 (gold 95% about ±0.02); ResNet-50 + ConvNeXt 0.936153 / 0.941863, all three
0.936719 / 0.940859. MambaVision is the weakest teacher on gold and the most valuable joint student
(+0.0010 to the other two). FocalNet's five silver folds and their fastMRI and OAI predictions are
queued on the rented machine for the next teacher reading.
- **S161 = 0.964 public**, a tie with S154–S156. Against the old round (Kaggle out of fold, silver
  / gold): teachers 0.935120 / 0.938638 old against 0.936719 / 0.940859 new; students S153/S154
  0.940316 / 0.947643 against S161's 0.942929 / 0.948511. The new students gain on every label on
  silver but lose Fracture (−0.0167) and Baker's (−0.0091) on gold, the labels whose OAI
  pseudo-labels are unreliable. Next (user decision): S161's exact configuration with the night4
  OAI-soft-tissue joint models as tomorrow's first submission.
- **S162, the 10 MambaVision + ConvNeXt joint models (no ResNet-50), same settings: 0.963 public**,
  one step below S161 (order: S154 > S156 > S161 > S155 at 0.964, then S152 > S162 at 0.963). Silver
  out of fold predicted +0.0001 for dropping ResNet-50; gold out of fold predicted −0.0021 (0.946379
  against 0.948511), the direction the board took.
- **OAI labels only for menisci and OA, 15 joint models (2026-10-16 night):** ×4 per fold within
  ±0.0007 of S161's members except MambaVision fold 2 (−0.0020). Out of fold, 15 equal: silver
  0.943143 against 0.943196, gold 0.949377 against 0.948511 (+0.0009, study bootstrap 95%
  [−0.0024, +0.0040]); on gold the injury labels recover (MCL +0.0091, Fracture +0.0042, Contusion
  +0.0040, Baker's +0.0018, PF OA +0.0064), Lateral OA falls (−0.0116, 11 positives); MambaVision
  alone rises on gold from 0.940918 to 0.944501. Submitted in S161's configuration as the first
  submission of 2026-10-16.
- **S163, the 15 OAI-soft-tissue joint models in S161's configuration: 0.964 public**, our top
  submission inside the tie (S163 > S154 > S156 > S161 > S155, all 0.964), above S161 as gold out
  of fold predicted.
- **S164, S163's members with per-label family weights chosen for the best gold out-of-fold AUC
  (in-sample gold +0.0046, silver −0.0007): 0.963 public**, one step below S163 (0.964); second of
  three at 0.963 (S152 > S164 > S162). Gold-tuned ensemble weights did not carry to the board; keep
  equal weights unless a weighting also holds on held-out silver.
- **Stacking on S163's members (session knee-abnormality-e9, gold only, leave-pair-out per
  label):** equal weights 0.949380; logistic regression on the label's 3 family logits 0.948893
  (−0.0005); TabPFN v2 on them 0.945299 (−0.0041); TabPFN on all 36 logits 0.934439; logistic
  regression on all 36 0.909443. Pooled leave-one-out AUC was biased against every stacker.
  Decision basis: equal weights for the final ensembles, no gold-fitted weights or stackers.
- **Kaggle-row teacher share against gold (same session; label table, not a student result):**
  gold agreement of the label table with ResNet-50 + ConvNeXt teachers' out-of-fold predictions:
  silver 0.9075, share 0.25 0.9379, 0.5 0.9422, 0.75 0.9473, teacher alone 0.9416 (0.75 picked
  among three on the same 58 studies, slightly optimistic). A per-label gold-fitted correction
  failed (0.9417, leave-pair-out). Bears on question 3: a teacher share of about 0.75.
- **C6, teacher share 0.75 after the pretraining (2026-10-16):** new Kaggle teacher table
  `teachers2-kaggle-x4-tau085-tta` (the 10 ResNet-50 + ConvNeXt silver teachers on their own validation
  folds, ×4, τ 0.85, mirror TTA, mean probability as a logit; each teacher run with its archived source
  because the teachers span three preprocessing identities). Logit r against the τ 1.0 out-of-fold mean
  0.9935 to 0.9988 per label, probabilities slightly higher (up to +0.024 on Contusion); gold agreement of
  the label table at share 0.75 0.9471 (τ 1.0: 0.9473). C6 = S160 + share 0.75 (only change; S160's
  archived source, seed 1110484336, 47 min, peak 18.9 GiB): ×1 0.94300 against S160's 0.94181, ×4
  0.944995 against S160 0.945279 (−0.0003) and S159 0.945443 (−0.0004). Per label against S160: Lateral
  Meniscus +0.0079, MCL +0.0053, Medial OA +0.0016, Baker's +0.0013, ACL +0.0007, Medial Meniscus
  −0.0013, PF OA −0.0018, Contusion −0.0018, Effusion −0.0020, Fracture −0.0030, Lateral OA −0.0035,
  Synovitis −0.0069. Fold 0's 12 gold studies: 0.9628, equal to S160 (S159 0.9643). Silver validation
  cannot judge a share (it moves the model away from silver); the board decides.

## Local runs to keep (main checkout `outputs/training`)

Pretrainings, including their `last.ckpt` (start weights of the joint runs): convnext-silver15-pretrain,
convnext-silver15-pretrain-resumed, mambavision-silver15-pretrain, mambavision-silver15-pretrain-resumed,
resnet50-silver15-pretrain, resnet50-silver15-pretrain-resumed, convnext-teachers2-pretrain,
focalnet-teachers2-pretrain, mambavision-teachers2-pretrain.

Silver teachers: resnet50-silver-fold0, resnet50-silver-fold1, resnet50-silver-fold2, resnet50-silver-fold3,
rn50-silver-fold-4, convnext-silver-fold-0, convnext-silver-fold-1, convnext-silver-only-imagenet-fold-2,
convnext-silver-fold-3, convnext-silver-fold-4, mamba-silver-fold-0, mamba-silver-fold-1, mamba-silver-fold-2,
mamba-silver-fold-3, mamba-silver-fold-4, focalnet-silver-fold-0, focalnet-silver-fold-1, focalnet-silver-fold-2,
focalnet-silver-fold-3, focalnet-silver-fold-4, effvit-b2-best-folds-0-1-3-4, efficientvit-sd02-fold-2.

Students: convnext-silver15-joint-oai-soft-fold-0 to 4, mambavision-silver15-joint-oai-soft-fold-0 to 4,
resnet50-silver15-joint-oai-soft-fold-0 to 4 (S163), convnext-silver15-joint-fold-0 to 4,
mambavision-silver15-joint-fold-0 to 4, resnet50-silver15-joint-fold-0 to 4 (S161),
resnet50-teachers2-joint-fold-0 to 4, focalnet-teachers2-joint-fold-0 to 4 and the later teachers2 joint runs.

- **S167 = C6 = 0.964 public, first in score-sorted order (above S163, the 15-model ensemble).**
  A single ConvNeXt fold-0 model, ×4, τ 0.85, no TTA: S160's recipe (0.961) with only the Kaggle
  targets changed to 0.25 silver + 0.75 teacher (ResNet-50 + ConvNeXt out of fold, ×4, τ 0.85,
  mirror TTA). +0.003 public for one change, while fold-0 ×4 said −0.0003 (validation inflated
  through the teachers): the local fold score cannot judge a teacher share; the label table's gold
  agreement (0.9075 → 0.9471) predicted it. Bears on questions 3 and 6: the share goes into every
  student, and an ensemble of share-0.75 students is the next candidate.
- **S166 = 0.963 public** (S163's 15 + the 5 round-2 FocalNet joint students, 20 equal members, ×4,
  τ 0.85, mirror TTA; shared CUDA-graph pool after a private-pool OOM on the T4): −0.001 against S163,
  first among the 0.963s. Out of fold it predicted silver +0.0003, gold −0.0026: the gold reading
  called the direction a second time (after S162), silver did not. Ensemble members are chosen by the
  gold out-of-fold reading (within 0.001 of the best silver).
- **When gold is trustworthy:** the readings that called S162 and S166 used all 58 gold studies,
  out of fold over five folds. One fold holds only about 12 gold studies, and one study moves the score by
  about 0.01, so a single-fold gold reading decides nothing (C6 tied S160 there at 0.9628 and still gained
  +0.003 public). For a teacher-share student, even the 58-study reading is somewhat inflated: the
  teachers that labelled its training rows trained on the fold that holds those gold studies. The
  label table's gold agreement (58 studies, 0.9075 → 0.9471) is what predicted S167.
- **How large a gold difference counts (S154 against S155):** adding the five ResNet-50s to the five
  ConvNeXt models read silver −0.0004, gold −0.0003 out of fold (S155 0.940901 / 0.947541, S154
  0.940478 / 0.947291), and the board tied at 0.964 with S154 sorted first. Gold differences of about
  0.002 to 0.003 (S162, S166) were followed by a full step on the board both times; differences under
  about 0.0005 end in a tie, with the score-sorted order not predictable from them.
- **S168 = C8 = 0.964 public, below S167 (C6) in score-sorted order** (S167 > S163 > S154 > S165 >
  S168 > S156 > S161 > S155). Same settings as S167 (single ConvNeXt fold 0, ×4, τ 0.85, no TTA); the
  only difference is the joint stage. Once the Kaggle rows carry the 0.75 teacher share, the joint stage
  adds nothing on the board, so the cheaper Kaggle-only fine-tune (C6) is the recipe of the next cycle:
  five folds of every family from its round-2 pretraining with share 0.75, ConvNeXt first.
- **S169 (round-2 ensemble):** round 1's ResNet-50 + round 2's ConvNeXt and MambaVision, 15 equal
  members, picked by gold out of fold within 0.001 of the best silver (silver 0.942741, gold 0.949980
  against S163's 0.943175 / 0.949380, a tie-sized +0.0006).
- **S169 = 0.964 public, first in score-sorted order** (S169 > S167 > S163 > S154 > S165 > S168 > S156 >
  S161 > S155): round 1's ResNet-50 + round 2's ConvNeXt and MambaVision, 15 equal members, ×4, τ 0.85,
  mirror TTA, same bundled code as S163. A tie with S163 sorted above it, as the tie-sized gold reading
  (+0.0006) suggested. The round-2 labels (ResNet-50 + ConvNeXt teachers, OAI rule from the pretraining)
  are at least as good as round 1's for ConvNeXt and MambaVision.
- **S170 = 0.965 public, new best** (the only 0.965): the five ConvNeXt folds of the share cycle
  (`convnext-share075-finetune-fold-0..4`: round-2 pretraining, Kaggle-only fine-tune with teacher share
  0.75), equal, ×4, τ 0.85, mirror TTA. One family of five share students beats every 0.964, including
  the 15-member ensembles S169 and S163 and the single share model S167: the share carries over to the
  folds of an ensemble. Next: ConvNeXt + MambaVision, then + ResNet-50, from the same cycle.

## Next generation (planned 2026-10-17)

The share children become the teachers of a third generation.

| Step | What | When (CEST) |
|---|---|---|
| 1 | Share cycle ends; out-of-fold selection over the five families; two submissions: best gold overall and best gold within 0.001 of the best silver | ~14:30 |
| 2 | The 25 share children predict their own validation folds at ×4, τ 0.85, mirror TTA (`/workspace/predictions/gen3-teachers/`) | ~14:30 to ~15:30 |
| 3 | Teachers = the families of whichever submission scores better | ~16:00 |
| 4 | New Kaggle teacher table (mean probability as a logit); gold agreement over the 0.05 share grid against the current table's 0.9471 at 0.75; continue only if higher; share at the new peak (0.75 unless a clear >0.001 difference) | ~16:00 |
| 5 | Five families × five folds, Kaggle-only fine-tune with the new table, from the same round-2 pretrainings | ~16:00 to ~05:00 Sunday |
| 6 | Out-of-fold selection, then a submission proposed to the user | Sunday morning |

The current table's share curve (gold agreement): 0.0 0.9075, 0.25 0.9370, 0.5 0.9437, 0.65 0.9456,
0.70 0.9463, **0.75 0.9471 (peak)**, 0.80 0.9469, 0.85 0.9458, 0.90 0.9454, 1.0 0.9428. Per-label peaks
range from 0.10 (ACL) to 1.0 (Contusion), not used: per-label values chosen on 58 studies overfit.
No new pretraining in this generation: better pretraining labels left the round-2 ConvNeXt students
level, and the joint stage adds nothing once the share is in (S168 below S167).
- **S171 = 0.964 public, below S170 (0.965)**, 9th of ten 0.964s: the share children of MambaVision +
  ResNet-50 (10 equal), picked as the best gold overall out of fold (gold 0.952163 against ConvNeXt
  alone's 0.949055, +0.0031; silver equal, 0.941281 against 0.941280). The first time a gold gain above
  0.002 did not carry to the board. The share inflates every child's gold reading through teachers that
  saw each fold's gold labels, so among share children gold is a weaker guide than it was for S162 and
  S166. S172 (+ EfficientViT) still scoring.
- **Selection error (2026-10-17):** S171 and S172 were picked on the share children's gold out-of-fold
  readings and both left out ConvNeXt, the only family the board had confirmed (S170 0.965). The gold
  readings of share students leak through teachers that saw each fold's gold labels, and the leak differs
  by family, so it reorders them. Rule from here: share students are judged on the board only; new
  ensembles start from the board's best (ConvNeXt × 5) and add a family only when a submission shows it
  helps; the next generation's teachers include ConvNeXt. A clean local reading needs cross-fitted
  teachers (no teacher has seen the gold labels of the fold it labels).
- **S172 = 0.964 public**, just above S171 (MambaVision + ResNet-50 + EfficientViT share children, 15
  equal). Neither share-children ensemble without ConvNeXt beats ConvNeXt × 5 (S170, 0.965).

## Third generation on new splits (2026-10-18)

- **S174 = 0.965 public, first among the 0.965s** (ConvNeXt mixed × 5, `convnext-gen3mixed-finetune-fold-0..4`);
  **S173 = 0.965, second** (ConvNeXt + ResNet-50 share children, 10 equal); S170 third.
- The gen-3 queue was stopped at 04:05 UTC by user order. Of the third generation only ConvNeXt mixed is kept;
  every other gen-3 run (ConvNeXt all5, the four other families' mixed and all5) was deleted locally and on the
  machine. Their fold readings on the old split, x1 validation macro AUC: MambaVision mixed 0.94130 0.93541 0.94126
  0.93626 0.93607; EfficientViT mixed 0.94014 0.93444 0.93999 0.93519 0.93592; FocalNet mixed 0.94129 0.93386
  0.93944 0.93452 0.93608; ResNet-50 all5 0.93755 0.93057 0.93501 (folds 0-2).
- Reason: every family trained on the same folds and seed leaves out the same studies, so the families add
  correlated members to ConvNeXt (share children agree at 0.99 logit correlation). The four other families are
  retrained on the mixed table, share 0.75, each on its own new split and seed, for ensemble diversity.
- Splits (scanner-stratified, `scripts/tools/validation.py prepare`, `data/folds/`), one per family:
  `all_5fold_scanner_v2_r1` ResNet-50 (split seed 1533777639), `_r2` MambaVision (3082927044), `_r3` EfficientViT
  (2128871193), `_r4` FocalNet (2597761050), `_r5` ConvNeXt (2423726674). Largest overlap of any fold of one split
  with a fold of another 0.23-0.26 (chance 0.20).
- One training seed per run (25 seeds), folds 0-4: ResNet-50 1203997460 643699413 3046000151 2761326745 3154334900;
  ConvNeXt 3311724702 765003022 4108126569 206399518 2150820031; MambaVision 2424155815 558766553 3500031755
  1497311460 3863367298; EfficientViT 1404199329 3988543562 2583390982 3384567468 3942274526; FocalNet 662050637
  246703576 2685161398 982647884 3735886651.
- `distilled()` in `data/labels.py` required the teacher table's fold to equal the student's fold, which only
  holds on the teacher's own split; it now requires a teacher value for every training study. The table stays
  out-of-fold on its own split; a student's validation studies were already seen by its teachers on the old
  split, so the board stays the only judge.
- Runs `<family>-gen3mixed-r<k>-finetune-fold-<f>`, queue `/root/cycle_gen3_resplit.sh` (tmux `gen3r`, started
  04:31 UTC): ResNet-50, ConvNeXt, MambaVision, EfficientViT, FocalNet. The old-split ConvNeXt mixed runs (S174)
  stay. 2026-10-18 ~10:40 CEST the user approved one submission: ConvNeXt r5 + ResNet-50 r1, 10 equal, x4 τ 0.85 TTA, sent to the Submission session.
- **NextViT-S as a sixth gen-3 family (2026-10-18, user decision):** the backbone-families-2 campaign's incumbent
  (fold 2 x4 0.938851, ties FocalNet SRF; third family over ResNet-50 + FocalNet +0.00093, logit correlation 0.958
  with the ResNet-50). Tree `/home/claudio/src/knee-abnormality-nextvit` on the machine (this worktree's code, main
  `318eba2` + the `distilled()` fix; sync-free venv). `configs/nextvit-teachers2-pretrain.yaml` and
  `configs/nextvit-gen3mixed-finetune.yaml` differ from `configs/nextvit.yaml` exactly as ConvNeXt's do from
  `configs/convnext.yaml`. Pretraining seed 4213577004 (old split fold 0, external only, 10 epochs); fine-tunes on
  `all_5fold_scanner_v2_r6` (split seed 1624582548), seeds 4081590439 3631603500 763392539 14540636 952220979.
  Chain `/root/cycle_nextvit.sh` (tmux `nextvit`) takes the next free slot of the three, real-batch preflight before
  each run. Approved for today: the 25-model gen-3 ensemble and the 30-model ensemble with NextViT.
- **S175 = 0.965 public, first among four 0.965s** (S175 > S174 > S173 > S170): ConvNeXt r5 + ResNet-50 r1, 10 equal.
  The board order (full-precision public score) reads as a 2x2, family by generation: gen 2 ConvNeXt S170 (4th),
  gen 2 ConvNeXt + ResNet-50 S173 (3rd), gen 3 ConvNeXt S174 (2nd), gen 3 ConvNeXt + ResNet-50 S175 (1st). Adding
  ResNet-50 helps in both generations and the newer generation helps with and without it; the generation step is the
  larger (S174 above S173). The new splits and seeds are not separated in this reading (S175 changed them too).

## Rebuild from gen 1 with a seed per run (planned 2026-10-18, not started)

The user's plan: every generation from gen 1 retrained with one seed per run; gen 1 keeps the shared split
`all_5fold_scanner_v2` (its gold out-of-fold readings are clean, so teachers can still be selected), gen 2 and gen 3
use one new split per family. The current pretrainings stay.

- Clean gen-1 reading (x4 out-of-fold, 4,407 studies, 58 gold): NextViT silver teachers (`nextvit-silver-fold-0,1,3,4`
  and the campaign's fold 2, one seed per fold) 0.9352 all / 0.9376 gold, the best single teacher on all studies;
  ResNet-50 0.9346 / 0.9384, ConvNeXt 0.9324 / 0.9374, FocalNet 0.9328 / 0.9308, MambaVision 0.9316 / 0.9284.
  Over all 63 tables of six families (EfficientViT from `effvit-b2-best-folds-0-1-3-4` + `efficientvit-sd02-fold-2`)
  the top six lie within 0.00015 on all studies and all contain ResNet-50 and NextViT (inclusion effects +0.0012 and
  +0.0009; MambaVision -0.0003, EfficientViT -0.0001); ConvNeXt + ResNet-50 + NextViT is first on gold (0.94302 against
  T2's 0.94161) and 0.00015 below the best on all studies. T2 + NextViT gains +0.0016 on all studies (bootstrap 80%
  [+0.0013, +0.0019]) and on every label.
- Gen 1 to retrain: ConvNeXt, ResNet-50, FocalNet, MambaVision and EfficientViT, 25 runs, main's configs (identical to
  the gen-0 teachers' recipe), one new seed each; NextViT's is done. The table is chosen afterwards from all 63
  combinations of the rebuilt teachers, since run-to-run variation can reorder the close tables (single-run reruns land
  within about ±0.0003; a five-run table is estimated near 0.00013; the bootstrap SD of a table difference on 4,407 studies
  is about 0.0002).
- Execution: one job list on the machine claimed by a dispatcher on each machine (3 slots there, 1 on the local 3090),
  main's code on both; pretrainings, the all-Kaggle pretraining predictions and exports stay on the machine.
- Chain (user's list, 2026-10-18): gen 1 (26 runs: 25 + NextViT fold 2 with a new seed, so 30 distinct seeds) →
  x4 τ0.85 TTA RSNA predictions of all 30 gen-1 fold models on their own folds → choose gen-1 teachers (63
  combinations) → their OAI + fastMRI predictions (machine only) → new external table → pretrain the six families
  (machine only) → gen 2 (6 × 5, share 0.75 on the new gen-1 table) → gen-2 RSNA predictions → mixed table (50/50) →
  gen 3 (6 × 5) → export and submit. Splits: gen 1 shared; gen 2 and gen 3 one new split per family **and per
  generation** (12 new splits, `r7`–`r12` for gen 2, `r13`–`r18` for gen 3), 30 new seeds per generation.
- Gen-1 teacher choice (user, 2026-10-18): a reasoned judgement over gold (clean at gen 1, proven reliable, 58 studies),
  silver (4,407 studies), the measured seed noise (old vs new gen-1 sets), ranking stability across both sets, per-label
  breadth, diversity and external-prediction cost; default ConvNeXt + ResNet-50 + NextViT when unclear. The chain runs
  unattended: defaults decide, no blocking questions.
- State 2026-10-18 ~23:00 CEST (user away, chain unattended): job list `/workspace/dispatch/jobs.tsv` (claims, done
  markers, dependencies), machine dispatcher `/root/dispatch_machine.sh` (tmux `machine`; after a restart
  `/root/dispatch_machine2.sh` via `/root/revive.sh`), local dispatcher in the scratchpad (tmux `rebuild`, logs
  `outputs/logs/rebuild/`), status page tmux `status`. Gen-2 teacher table: all six families unless one is clearly weakest
  both as a gen-1 teacher and in pretraining. ConvNeXt gen-1 seed reruns (folds 0-2, x1): single-run silver SD about
  0.0008, logit correlation 0.982-0.984; external x4 TTA old vs new seeds 0.975 (fastMRI) and 0.93-0.94 (OAI).
- **S177 = 0.965, last of five 0.965s** (S175 > S174 > S173 > S170 > S177): all six gen-3 families, 30 equal. More families sorted below ConvNeXt + ResNet-50 (S175); S176 (25, no NextViT) pending.
- **S176 = 0.964, first of the 0.964s** (five families, no NextViT). With S175 > S177 > S176: NextViT helps (0.964 -> 0.965), MambaVision + EfficientViT + FocalNet weaken ConvNeXt + ResNet-50; leading evidence for the final members and the gen-2 table.
- Seed noise at the table setting (2026-10-19, x4 τ0.85 TTA, own folds, shared split): ConvNeXt + ResNet-50 table with the
  old seeds (T2) 0.93654 silver / 0.94278 gold, with the new gen-1 seeds 0.93637 / 0.94509 (−0.00017 / +0.0023), logit
  correlation 0.988. Silver seed noise of a two-family table about 0.0001-0.0002; gold moves about 0.002 from the seed
  alone, so gold gaps below about 0.002-0.003 between tables are within seed noise. New single teachers: ResNet-50
  0.93445 / 0.94357, ConvNeXt 0.93322 / 0.94084.
- **Plan change after S175/S176/S177 (2026-10-19, reviewed with the user):** MambaVision, EfficientViT and FocalNet are
  dropped after gen 1 (their gen-3 members weaken the ensemble on the board, and they are the weakest gen-1 teachers;
  FocalNet lowers the four-family tables by 0.0022 on gold). Pretraining, gen 2 and gen 3 continue with ConvNeXt,
  ResNet-50 and NextViT only (3 pretrainings, 15 gen-2 runs, 15 gen-3 runs); the gen-2 half of the mixed table and the
  final ensemble use the same three families. Their gen-1 runs stay for the teacher analysis. Open option: a second gen-3
  run per fold for those families on its own split and seed (30 members).
- **Gen-1 teacher decision (2026-10-19 ~08:20 CEST): ConvNeXt + ResNet-50 + NextViT.** Full six-family reading at x4 τ0.85
  TTA (`outputs/logs/rebuild/teacher-selection-full.txt`): first on gold (0.94666) of 63 tables, 0.93807 on silver within
  about 0.0002 of the best; the gold runners-up with EfficientViT tie on gold (−0.0001 to +0.0002, inside the ~0.002 seed
  noise) but lose about 0.001 on silver in every bootstrap resample. Inclusion effects silver / gold: ResNet-50 +0.0010 /
  +0.0022, NextViT +0.0009 / +0.0009, ConvNeXt +0.0001 / +0.0008, EfficientViT −0.0001 / +0.0008, FocalNet +0.0002 /
  −0.0016, MambaVision −0.0001 / −0.0006. Tables: `data/labels/prepared/rebuild-t2-x4-tau085-tta` (Kaggle, 15 teachers)
  and `rebuild-gen1-x4-tau085-tta-oai-soft-tissue` (external, 19,265 studies).
- 2026-10-19 ~09:00 CEST (user): FocalNet, MambaVision and EfficientViT pretrainings added at the lowest machine priority (seeds 1698999170, 752222899, 1726674044) to keep the option of gen-2/gen-3 runs for them; decide after the gen-3 board result.
- **S178 (probe A) = 0.965, first of six 0.965s** (S178 > S175 > S174 > S173 > S170 > S177): old-cycle gen-3 ConvNeXt + ResNet-50 + NextViT, 15 equal. NextViT added to ConvNeXt + ResNet-50 beats S175: the three-family core is confirmed as members. Probe C (+ EfficientViT) pending.
- **S179 (probe C) = 0.965, 5th of seven 0.965s**, below probe A (S178): EfficientViT added to ConvNeXt + ResNet-50 + NextViT moves the ensemble down (gen 1 predicted a tie or slight loss). EfficientViT stays out of gen 2 and gen 3; its queued pretraining was removed.
- 2026-10-19 (user): two submissions approved after probe B: rebuild gen 2 (ConvNeXt + ResNet-50 + NextViT, 15) once gen 2 finishes, then rebuild gen 3 (same families, share 0.75) once gen 3 finishes; I notify the Submission session when each set of 15 is complete. MambaVision, if probe B wins, joins as a gen-3 member only (not a gen-2 teacher).
- **S180 (probe B) = 0.965, 7th of eight 0.965s**, below probe A: MambaVision added to the three moves the ensemble down further than EfficientViT. MambaVision and FocalNet pretrainings removed; the rebuild is ConvNeXt + ResNet-50 + NextViT only. Board family tests, all against S178: + EfficientViT 5th, + MambaVision 7th, + all others (S177) 8th.

## Silver labels v3 (started 2026-10-19)

- Why: the gen-1 error analysis (`outputs/logs/rebuild/gen1-error-analysis.md` in the main checkout) found most top errors are label problems shared by all three families. Six audit agents proposed per-label rules (`outputs/logs/rebuild/label-audit/{ligaments,menisci,osteoarthritis,fluid,bone,cross-cutting}.md`); a final-rules agent drafts the set and the drop-in `apply_v3_rules` code.
- Reading pass instead of regexes or JevK5: Claude subagents read all 4,407 reports for the rule questions. Resumable state and procedure in `outputs/labels-v3-reading/README.md` (main checkout); `status.py -v` lists pending batches. If the quota stops it, resume only the pending batches after the reset.
- Then: re-evaluate each rule on the new groups, build `rebuild_state_v3` (post-calibration overrides, silver rows only, gold untouched, folds of `all_5fold_scanner_v2` kept), rescore the six gen-1 families on v3 for the teacher choice, retrain gen 1 on v3 (new seeds), judge on gold (optimistic: some rules were written after reading gold reports) and silver, then the rest of the chain if it holds.
- Vast credit 2026-10-19 17:30 CEST: $91.34 at $1.97/h, runs out Wed ~15:30 CEST; the v3 chain needs ~$25 more, the deadline ~$75 more (user's call).
- 2026-10-19 evening: v3 built (labels-v3 worktree, uncommitted; build report `outputs/logs/rebuild/label-audit/final/v3-build.md`): gold cross-fitted macro AUC 0.880 → 0.906 (optimistic), 748 silver flips, 40 studies masked; verdict blend rejected (weight 0). Second soft pass v3.1: 62 silver cells confirmed wrong by the image ensemble, the reader verdict and a blind Opus review moved to 0.45/0.55 (`outputs/labels-v3-reading/pass2/v3_1_changes.csv`); label source `rebuild_state_v3_1`, manifest `configs/labels/rebuild-state-v3-1-label-sources.json` (copied into this tree and the machine's nextvit tree), split `all_5fold_scanner_v3_1` (v2 folds). Teacher choice on v3.1 silver unchanged (ConvNeXt + ResNet-50 + NextViT first on gold and silver).
- Queued 2026-10-19 19:40 CEST behind gen 3: v3.1 gen 1, runs `{convnext,resnet50,nextvit}-v31-gen1-fold-{0..4}` (configs/<family>.yaml + v3.1 overrides, new seeds in rebuild-seeds.tsv) and their rsna predictions. Decision rule at the end: continue the chain if gold ≥ old gen 1 (0.94666) − 0.002 and silver on v3.1 better.
- 2026-10-19 ~20:00 CEST, final label choice **v3** (not v3.1): board-order correlation of local silver scores over 26 re-scorable submissions (`outputs/labels-v3-reading/lb-correlation/results.md`): Spearman v2 0.20, v3 0.43, v3.1 0.41; v3 > v2 in 99.6% of bootstrap draws; v3.1 − v3 = −0.017 [−0.046, +0.005], lower in 81% of draws; v3.1 also mixes the v2-trained ensemble into gen-1 targets. A v3.2 built in parallel was worse. Held gen-1 jobs switched to v3 and released as `{convnext,resnet50,nextvit}-v3-gen1-fold-{0..4}` (same seeds), label source `rebuild_state_v3`, split `all_5fold_scanner_v3`, manifest `configs/labels/rebuild-state-v3-label-sources.json` (copied into this tree and the machine's nextvit tree).
- 2026-10-19 ~21:20 CEST final label review (other agent's v3.2 findings, `outputs/labels-v3-reading/pass3/`): Ambition X 69 big label/ensemble disagreements → 0 wrong (model errors on trauma knees); MCL low-grade wording 45 → 2 borderline; "intact8" corruption is decodable ("intact" = "0."), 425 size-dependent cells → 14 wrong, mostly sub-1 cm OA lesions that gold graders often call positive; old fracture + avulsion/osteochondral rule gap 5 positive cells (known, not fixed). v3 kept unchanged; gen-1 jobs released.
- v3 chain configs written 2026-10-19 21:15 (this tree + machine nextvit tree): `configs/{convnext,resnet50,nextvit}-v3-{pretrain,gen2,gen3}.yaml` = the v2-cycle rebuild configs with label manifest/source/split v3, external table `rebuild-v3-gen1-x4-tau085-tta-oai-soft-tissue`, teacher tables `rebuild-v3-t2-x4-tau085-tta` (gen 2) and `rebuild-v3-mixed-x4-tau085-tta` (gen 3). `teacher_share: 0.75` is a placeholder: the share is recalculated on the v3 gen-1 table with v3 cross-fitted gold values, and the user dropped the keep-0.75 rule (take the bootstrap-averaged optimum). Gen-2/gen-3 splits: `all_5fold_scanner_v3_r1..r6` (ConvNeXt/ResNet-50/NextViT gen 2 r1/r2/r3, gen 3 r4/r5/r6), picked from 300 candidates to minimise the worst fold-overlap χ² against each other and all 19 existing splits (worst 29.6 vs 45–66 for a random pick).
- 2026-10-19 board: S181 (rebuild gen 2 x15) 0.965, 7th of the 0.965s; S182 (rebuild gen 3 x15) 0.965 FIRST of ten 0.965s, above S178. Gen 3 adds over gen 2 under v2 labels.
- 2026-10-20 02:00 CEST v3 gen 1 complete. Three-family ensemble (x4 tau 0.85 TTA own fold): gold 0.95340 vs 0.94666 old (+0.0067, bootstrap 10/90% +0.0017/+0.0122, P 0.95); per family ConvNeXt +0.0107, ResNet-50 +0.0058, NextViT +0.0049; silver on v3 targets −0.0005 (seed noise), on v2 targets −0.0066 (expected). Gold per label: Synovitis +0.072, Lateral Meniscus +0.014, Fracture +0.013, Medial Meniscus +0.005, others within ±0.002 except Lateral OA −0.010 and Contusion −0.013. Gate passed. Gen-2 share recalculated on `rebuild-v3-t2-x4-tau085-tta` with v3 cross-fitted gold values (`outputs/logs/rebuild/share_curve_v3.py`): optimum 0.80 (0.9584; 0.75 = 0.9571; per-draw argmax 10/50/90% 0.70/0.80/0.90); `teacher_share: 0.80` written into the three v3 gen-2 configs on both machines and `v3-gen2-share` released. Gen-3 configs still hold the 0.75 placeholder until the mixed-table check.
- 2026-10-20 06:30 CEST: 15 v3 external predictions done (~55 min each), external table rebuild-v3-gen1-x4-tau085-tta-oai-soft-tissue built (19,265 studies, teacher min correlation with mean ≥ 0.935), 3 v3 pretrainings running (~6.5 h like the v2 cycle's, ETA ~13:10). Revised: gen 2 ~17:30, gen 3 ~23:00 Tuesday; Vast credit lasts to ~Wed 15:45.
  Correction: v3 pretraining trains on the 19,265 external studies only (extra_training alone: true; 2,409 batches of 8 per epoch, ~26–30 min), with the v3 gen-1 teachers' soft labels; the Kaggle validation fold is only scored, not trained on. Pace ~30 min/epoch → ETA ~11:30.
- 2026-10-20 ~09:15 CEST (user): gen 2 keeps teacher share 0.80; gen 3 also trains at 0.80 (gen-3 configs set on both machines). Gen-3 jobs are now gated only on the v3 mixed table (built automatically by v3chain), no separate gen-3 share check; the mixed-table curve is still computed for information when it exists.
- 2026-10-20 ~11:30 CEST (user): approved the v3 gen-2 submission (ConvNeXt + ResNet-50 + NextViT v3 gen 2, 15 members) as soon as gen 2 is trained; compare with S181 (v2 gen 2). Gen-1 v2/v3 board pair skipped (attribution only, no decision depends on it).
- 2026-10-20 15:17 CEST: v3 gen 2 complete (15 runs); export of the approved gen-2 submission sent to the Submission session (compare with S181). Mixed table and gen 3 follow automatically after the last gen-2 predictions.
- 2026-10-20 15:40 CEST: v3 gen-2 three-family ensemble vs v2 gen 2 (both partly leaky on gold): gold 0.95702 vs 0.95195 (+0.0051; Synovitis +0.051, Fracture +0.007, Contusion +0.007, PF OA −0.008); silver on v3 targets equal (−0.00002), on v2 targets −0.0064. Mixed table built 15:36; its share curve (information only): 0.75 0.9594, 0.78 peak 0.9595, 0.80 0.9593 — 0.80 within 0.0002 of the optimum. Gen 3 started 15:37 on all four slots at share 0.80.
- 2026-10-20 ~16:50 CEST board: S183 (v3 gen 2 x15) 0.965, 9th of eleven 0.965s, directly below S181 (v2 gen 2, 8th). The gold gain (+0.005 at gen 2) does not show on the public board; the two are within the same rounded score, tie order slightly favours v2.
- 2026-10-20 ~17:00 CEST (user): two tau probes on the S183 weights (v3 gen 2 x15) sent to the Submission session: tau 0.15 and tau 0.5 (S183 is tau 0.85). Today's slot use: S183 + these two = 3 of 5.
- 2026-10-20 ~19:00 CEST board: S184 (S183 weights, tau 0.15) 0.963, 0.002 below S183 (tau 0.85), last of five 0.963s. S185 (tau 0.5) scoring.
- 2026-10-20 ~19:10 CEST board: **S185 (S183 weights, tau 0.5) 0.967, new best, only 0.967**; tau curve on the same 15 v3 gen-2 models: 0.15 → 0.963, 0.5 → 0.967, 0.85 → 0.965. Today's submissions used: S183, S184, S185 (3 of 5).
- 2026-10-20 ~20:00 CEST (user, conditional on it discriminating the curve shape): S183 weights at tau 0.3 sent to the Submission session. Predictions at tau 0.3 by fit scale: decorrelation ~0.969, log tokens/log tau ~0.966, raw tokens ~0.964. Today's submissions: S183, S184, S185, tau 0.3 (4 of 5).
- 2026-10-20 ~20:05 CEST shutdown checklist for the rented machine (user: no more trainings planned; tau is set at run time in the Kaggle notebook, so tau probes reuse uploaded weights): (1) when v3 gen 3 is complete, Submission session uploads the 15 v3 gen-3 checkpoints to Kaggle (upload only) and confirms all candidate weight sets (v2 gen2/gen3 = S181/S182, v3 gen2 = S183, v3 gen3) are on Kaggle and loadable with runtime tau; (2) copy home: /workspace/predictions (rebuild, rebuild-external, tau015), /workspace/tau-mech, /workspace/outputs/logs, /workspace/dispatch (jobs, claims, done), v3 gen-3 and v2 gen-2/gen-3 checkpoints missing locally; (3) verify counts/sizes, report. Stop/destroy is the user's call.
- 2026-10-20 ~21:00 CEST machine ready to close: everything needed is local (main checkout + `outputs/rented-machine-archive-20261020`, 11 GB): all predictions incl. v3 gen-3 fold-4 OOF (run on the machine after the local PC crash), label tables, folds, export trees, the 87 candidate checkpoints (v2/v3 gen2/gen3, v3 gen1, 6 pretrainings), run metadata, scripts, job queue, tau-mech. Not copied: fastMRI/OAI images (no more external work planned; students train locally from the pretrained checkpoints on Kaggle studies), old non-candidate bundles (S170–S177 era, weights on Kaggle), machine caches. All four candidate sets are on Kaggle (v3 gen3 raw at tau 1.0; tau is baked at export and re-exported locally per probe). Stop/destroy is the user's call.
- 2026-10-20 ~21:40 CEST board: S186 (S183 weights, tau 0.3) 0.966. Four-point curve 0.15/0.3/0.5/0.85 → 0.963/0.966/0.967/0.965: log-tau parabola optimum 0.43 (10–90% 0.41–0.45), linear-tau 0.55, decorrelation (1/tau) scale misses beyond rounding and is rejected. γ (v2-vs-v3 same family r at tau 0.15 = 0.915) ≈ 0, so 30 v2+v3 models take the 15-model tau; gen 3 reacts to tau like gen 2, ~5–7% less (optimum ≤0.03–0.05 sharper). Full four-set mechanics (40 more models) running on the machine.
- 2026-10-20 ~22:45 CEST (user): last slot of the day = all 60 models (v2+v3 × gen2+gen3, three families × 5 folds), equal, x4 tau 0.5 mirror TTA, eager; sent to the Submission session. Rationale: tau 0.5 is the measured best point and loses ≤0.0001 under both the log-τ (opt 0.43) and linear-τ (opt 0.55) fits; same tau as S185 so the comparison isolates adding v2 and gen3 members. Subagent model (outputs/tau-diag/optimal-tau-method.md, judged partly naive: borrows the v3 gen2 board curve for all sets, local β, mixing never tested on the board) predicted 60 @0.4 0.9678 vs S185 0.9667.
- 2026-10-20 ~23:00 CEST (user: two 60-model submissions allowed at 02:00, goal = find the optimal tau): S188 = 60 @ tau 0.4, S189 = 60 @ tau 0.6, same members and code as S187 (60 @ 0.5), launched after the 02:00 reset. Reading by tie order among equal rounded scores: 0.4 best → optimum ≤0.45 (next 0.3–0.35); 0.5 best → 0.45–0.55 settled; 0.6 best → ≥0.55 (next 0.7).
- 2026-10-20 ~23:05 CEST (user): S188/S189 no longer fire at a fixed 02:05; gated on S187 (first 60-model run): S187 complete ≥0.967 → submit both; <0.967 or error/timeout → hold both and report.
- Gate refined (user): S187 = 0.966 → submit only S188 (60 @ 0.4; more models tolerate sharper tau, so a drop at 0.5 points sharper); ≤0.965 or failure → nothing.
- 2026-10-20 23:15 CEST (user): instance 55948670 DESTROYED after everything needed was copied home (outputs/rented-machine-archive-20261020 + main checkout). Credit left $82. No rented machine exists now.
- 2026-10-20 23:06 CEST: S187 submitted (60 @ effective tau 0.5; description shows the per-member scale 0.5882 for members baked at 0.85). Score expected ~04:00–05:00 CEST; gate knee-gate-staged decides S188/S189 after 02:05.
- 2026-10-20 ~23:40 CEST: S188 (claverru/s188-...-x60-tau0-4 v1) and S189 (claverru/s189-...-x60-tau0-6 v1) staged (visible runs done, submission.csv verified; Kaggle descriptions show the first member's export scale 0.4706/0.7059, effective tau 0.4/0.6). S187 visible run: peak GPU 9.7/14.6 GiB, ~21 s/study; score expected ~05:00–05:30 CEST, 9 h cap at 08:06 CEST.
- 2026-10-21 ~04:20 CEST board: **S187 (60 @ tau 0.5) 0.968, new best, only 0.968** (+0.001 over S185 = 15 v3 gen2 @ 0.5). Gate fired: S188 (60 @ 0.4) and S189 (60 @ 0.6) submitted 04:21 CEST; scores expected ~09:30–10:00 CEST.
- 2026-10-21 ~09:25 CEST board: S188 (60 @ tau 0.4) 0.968, sorts FIRST above S187 (60 @ 0.5, 0.968): the 60-model optimum is at 0.4 or sharper (tie order). S189 (60 @ 0.6) pending.
- 2026-10-21 ~09:45 CEST board: S189 (60 @ tau 0.6) 0.968, 3rd of three 0.968s. Tie order S188 (0.4) > S187 (0.5) > S189 (0.6), monotone: the 60-model optimum lies below ~0.45 (log midpoint of 0.4/0.5), either 0.4–0.45 or sharper; a tau 0.3 probe would locate it. No submissions pending; gate service exited.
- 2026-10-21 (rented 5090 knee-tau 57021023, destroyed ~13:10 CEST, $74.67 credit left; results in main `outputs/rented-machine-archive-20261021-knee-tau/` and `outputs/tau-diag/width/`): no local accuracy proxy reproduces the board tau order (v2/v3 silver, gold, k=1/3/extrapolated 15, scanner groups, study size: all peak at 0.85, board peaks 0.4–0.5). Attended-width proxy: measured training width (x1, tau 1) 109 tokens; x4 matches it at tau ≈ 0.51; board optimum sits slightly sharper (15: 0.43–0.5; 60: below ~0.45 by tie order S188 > S187 > S189). Width-matched tau vs x4 (15 v3 gen2 members, 120 studies): x4 0.35/0.40/0.45/0.50 → x6 0.326/0.371/0.417/0.463, x8 0.319/0.363/0.408/0.453. x6 has +28% tokens but only +16% width; x8 is "all slices" (+5% tokens over x6). Proposed 2x2 for today's 3 slots: x4 0.35, x8 0.36, x8 0.32 (anchors S188 = x4 0.40), pending the user's go.
- 2026-10-21 ~13:25 CEST (user): today's slots 3–4 = 60 members at x8, tau 0.39 (S190) and 0.33 (S191), bracketing the S188-width equivalent (x8 0.36) on both sides; slot 5 reserved for an adaptive third x8 probe after they score (~21:00–22:00). Earlier staged x8 0.36/0.30 bundles cancelled.
- 2026-10-21 ~13:45 CEST (user): staggered instead (crash risk shared by same-code x8 runs): one submission now, S190 = 60 at x8 tau 0.36 (same attended width as S188 x4 0.40; best single estimate and clean x8-vs-x4 test). Next: if S190 > S188 → x8 0.33, then 0.39/0.30; if below → x4 between 0.35 and 0.40. Bundles x8 0.39/0.33/0.30 kept on disk.
- 2026-10-21 ~14:00 CEST (user): S190 aims at the estimated optimum, not the S188-equivalent: x8 tau 0.35 (estimated x4 optimum 0.37–0.40 → x8 0.34–0.36, centre 0.35). The 0.36 launch stopped before submission.
- 2026-10-21 ~14:45 CEST (user, "silver bullet"): S190 = 60 members at x8 with a per-label (per attention channel) tau from the rule "each label attends c x its own training width (x1, tau 1)", c = 0.55. Basis: per-label attended width (width/v3g2*.parquet, eff|k per target) differs 2.8x across labels at one global tau (x4 tau 0.40: contusion 0.36x … MCL 0.99x of training width) because routing gives labels different token counts; locally the most-squeezed labels lose most when sharpened (Spearman -0.67, p 0.018). Channel taus: ACL 0.27, MCL 0.24, Effusion 0.41, Synovitis 0.35, Baker's 0.41, Contusion 0.44, Fracture 0.34, meniscus 0.305, oa 0.38 (PF OA prior row 0.35), routing_temperature 0.34. Weights-only per-row scaling, no code change. If it falls short, the remaining 2 slots return to global-tau x8. Per-study tau (x8 uneven slice gains) estimated ~1e-5 AUC and not implemented; local per-study rules never beat a single tau (no local signal for sharpening).
- 2026-10-21 ~14:30 CEST: S190 (x8 per-label tau, fp32 weights, bs2) FAILED in the hidden rerun after ~12 min — CUDA OOM almost certainly (worst training pair est. 14.2 of 14.6 GiB; my miss, no headroom). User: resubmit the same per-label x8 bundle with fp16 weight storage (resident weights 6.7 → 3.4 GiB; worst pair est. ~10.9 GiB) as S191, slot 4 of 5.
- 2026-10-21 14:55 CEST: S191 submitted (slot 4; slot 5 unused) = S190's per-label x8 bundle with the memory fix v2b: model.backbone.half() + requires_grad_(False) in checkpoint_model (export tree copy, bundled source 523f69f3-based; resident weights 6.7 → 3.2 GiB). Gates vs S190 visible: completed; GPU peak 7.76 / 5.18 GiB (was 10.77 / 8.19); max |diff| ≤ 1e-4 per label; 0.252 s/img → hidden ≈ 7–7.3 h, score ~22:00. Failed variants: full model.half() and Conv/Linear-only half both fail at model.py:1498 side_head (fp32 input in an autocast-off region). requires_grad_(False) alone did not lower memory (cache hypothesis rejected); the saving is the fp16 backbones. Worst hidden pair est. ~10 GiB.
- 2026-10-21 ~15:30 CEST: 12-clean-model own-fold test (every study is held out by exactly 12 of the 60 members; 858-study subset incl. 58 gold; all members re-predicted at tau 0.85/0.5/0.4, fp16 graphs, `outputs/tau-diag/sub12/`, analysis `outputs/tau-diag/sub12_analysis.{py,txt}`): macro AUC (v3 silver) 3 models 0.9461/0.9444/0.9423, 6 v3 0.9459/0.9442/0.9422, 6 v2 0.9464/0.9445/0.9427, 12 all 0.9468/0.9451/0.9434; 0.4−0.85 −0.0038 (3) → −0.0034 (12); v2 silver and gold same order. Averaging does not move the local optimum: the board's preference for sharper tau (stronger at 60) is not an ensemble-size effect and no local factor tested (labels, size, scanner, study size, per-study rules, k up to 12) reproduces it. Hidden test studies are smaller (~104 images/study at x4 from S187 runtime vs 123 in training). Tau can only be tuned on the board. **Correction 2026-10-22:** the "gold" numbers in this and the other 2026-10-21 entries were scored against the v3 silver table, not the radiologist labels; corrected results and the explanation are in [attention temperature](../validation/results/2026-10-22-attention-temperature.md).
- Decision tree after S191 (all x8, backbone fp16, bs2): S191 > S188 → per-label c 0.45 and 0.65 (bracket c); S191 ties 0.968 below S188 → global x8 tau 0.36 (separates x8 from per-label); S191 ≤ 0.967 → global x8 0.36 first, then a milder per-label spread. Historical support that the board rewards focused attention beyond local: docs/research/medial-lateral-transfer.md (attention constraint: local down, public up).
- 2026-10-21 ~17:00 CEST unseen-data tau test (user-approved): rented knee-oai (Vast 57066072, RTX 5090, $0.83/h; destroy is the user's call). OAI baseline knees with complete MOAKS forms (1,339; labels mapped for 9 targets: ACL complete tear, meniscal tear/maceration/root, compartment full-thickness cartilage loss ≥10% → OA, effusion-synovitis ≥moderate, Hoffa synovitis ≥moderate, popliteal cyst) from the OAI "MR Image Assessments" release (`data/external/oai-assessments/`), images via signed links from package 1228997 (same 4 series per knee as the pretraining layout). 15 v3 gen-1 models (never saw OAI/fastMRI), every knee through every model, x4, tau 1.0/0.85/0.6/0.45/0.3, mirror TTA, fp16 graphs, widths logged (`outputs/tau-diag/oai_tau.py`); local seen-data reference for the same models/taus on the 858-study subset (`outputs/tau-diag/gen1-local/`); analysis `outputs/tau-diag/oai_analysis.{py,txt}`. Existing tau-0.85 OAI predictions of these models vs MOAKS: 1 model 0.9126, 5 folds of one family 0.9156, 15 all 0.9182 (9-label macro).
- 2026-10-21 22:30 CEST (user): efficiency-board students distilled from S188, one per family (ConvNeXt, ResNet-50, NextViT), running unattended on the local 3090. Teacher table `data/labels/prepared/s188-x60-oof-x4-tau04-tta` = logit of the mean probability of the 12 S188 members that held each study out, re-predicted at S188's x4 tau 0.4 mirror TTA (predictions `outputs/predictions/s188-oof-x4-tau04-tta`, ~4 h; parity at tau 0.85 against `outputs/predictions/rebuild` passed). Students `configs/{family}-s188-student.yaml` = the v3-gen3 configs with that table, teacher share 1.0 (user: tau's board gain sits on cells where silver and gold disagree, so a silver share would undo it; gold studies keep gold targets), `train_on_validation`, initial weights `{family}-v3-pretrain/fold_0/last.ckpt`, seed 1110484336; preflight cap 22 GiB, sequential via `outputs/predictions/s188-oof-x4-tau04-tta/students.sh` (log `students/chain.log`). Precedent S156 (0.25 share, OOF table at the ensemble's own tau). Then: export from `inference-speed`, submit each, pick by efficiency.
- 2026-10-22 03:08 CEST: ConvNeXt student `convnext-s188-student` trained (best epoch 9, fold-0 x1 0.944085 / x4 0.946344, inflated by training on fold 0; 3,613 s, peak 20.3 GB); handed to the Inference Speed session for submission (user: submit only the ConvNeXt). ResNet-50 and NextViT students train next and are not to be submitted. First launch failed in the preflight (`scripts` not importable from its measuring subprocess; fixed with `PYTHONPATH=$TREE`).
- 2026-10-22 (Inference Speed session): **S193 = convnext-s188-student, x4 tau 0.5, no mirror TTA, accelerated (compiled CUDA-graph backbone, Kaggle-built compile cache, FP16 slim): 0.960 public, 4th of five 0.960s**; hidden scoring 544 s. Bundle `outputs/submission/exports/convnext-s188-student-x4-tau05-fast`. Reference single ConvNeXt students: S156 0.964 (share 0.25 on S154's 10 at tau 0.85, joint fastMRI, x4 tau 0.85), S167 0.964.
- 2026-10-22 05:10 CEST: all three S188 students trained (fold-0 scores inflated by training on fold 0): ConvNeXt epoch 9 x1 0.944085 / x4 0.946344 (3,613 s); ResNet-50 `resnet50-s188-student` epoch 9 x1 0.942194 / x4 0.944665 (3,110 s); NextViT `nextvit-s188-student` epoch 8 x1 0.943229 / x4 0.946361 (3,664 s). ResNet-50 and NextViT not submitted (user).
- 2026-10-22 05:24 CEST (user): **gen 4 on all data**, one model per family per label version, mimicking gen 2/3. Teacher tables (built with `outputs/logs/rebuild/build_kaggle_table.py`, specs `gen4-{v2,v3}-table-spec.json`): `rebuild-gen4-x4-tau085-tta` (v2) and `rebuild-v3-gen4-x4-tau085-tta` (v3) = logit of the equal (1/3 each) mean of the gen-1 teacher, gen-2 and gen-3 groups of that version, each group the family mean of own-fold x4 tau 0.85 TTA predictions (gen-1 groups as in the gen-3 mixed tables). Configs `configs/{family}-{rebuild,v3}-gen4.yaml` = gen-3 configs with that table, share 0.80 (v2 was 0.75 in gen 3), `train_on_validation`, fold 0; initial weights `{family}-{rebuild,v3}-pretrain`. Run ids `{family}-gen4` / `{family}-v3-gen4`; six new seeds in `outputs/logs/rebuild/gen4-seeds.tsv`, checked against the 186 seeds in training records and job lists. Sequential chain `outputs/logs/rebuild/gen4.sh` (log `gen4/chain.log`), ~1 h per model.
- 2026-10-22 ~10:00 CEST (user): three **v2v3 gen-4** models on the rented machine (Vast 57177970, RTX PRO 6000 Max-Q 96 GB, `ssh -i ~/.ssh/vast_knee -p 12651 root@185.17.198.197`), trained side by side in tmux sessions `{family}-v2v3-gen4` (launcher `/workspace/outputs/logs/run_one.sh`, state `/workspace/outputs/logs/gen4mix/state`). Teacher table `rebuild-v2v3-gen4-x4-tau085-tta` = equal mean of the six v2/v3 gen-1/2/3 groups (spec `outputs/logs/rebuild/gen4mix-table-spec.json`), share 1.0, v3 labels, all data, initial weights `{family}-rebuild-pretrain` (v2 pretraining: on Kaggle fold 0, never trained on, it beats the v3 pretraining on v2 targets by ~0.004 and on v3 targets by 0.0002–0.0007 for all three families; the 12 gold studies there favour v3 by +0.002–0.004). Seeds in `outputs/logs/rebuild/gen4mix-seeds.tsv`, checked against all 192 recorded seeds.
- 2026-10-22 11:35 CEST: gen 4 complete. Local (fold-0 x1 / x4, inflated by training on fold 0; v2 runs scored on v2 targets, v3 on v3): convnext-gen4 0.945970 / 0.947657, convnext-v3-gen4 0.949640 / 0.951167, resnet50-gen4 0.942347 / 0.944240, resnet50-v3-gen4 0.946678 / 0.948590, nextvit-gen4 0.948305 / 0.949971, nextvit-v3-gen4 0.950721 / 0.952660 (~55–65 min each on the 3090). Remote v2v3 (three side by side, ~1 h 35 min; copied to main's outputs/training without last.ckpt, hashes match): convnext-v2v3-gen4 0.946133 / 0.948109, resnet50-v2v3-gen4 0.944133 / 0.945919 (handed to the Inference Speed session for submission, user's order), nextvit-v2v3-gen4 0.946361 / 0.947807.
- 2026-10-22 (Inference Speed session): **S195 = resnet50-v2v3-gen4, x4 tau 0.85 (user's choice), no mirror TTA, accelerated (compiled CUDA-graph backbone, Kaggle-built compile cache, FP16 slim): 0.964 public, 2nd of thirteen 0.964s, above every other single model at 0.964 (S167 4th)**; submit-to-scored 483 s; exported on the rented machine (bundle `outputs/submission/exports/resnet50-v2v3-gen4-x4-tau085-fast`). Against S193 (ConvNeXt S188 student, tau 0.4 table, share 1.0, inference tau 0.5): 0.960.
