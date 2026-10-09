# Anatomy from pretrained segmenters: ideas

Started 2026-10-07 as a working list at the repository root, moved here on 2026-10-10. The
one-off scripts it names under `scripts/adhoc/side_routed_session/` were not kept.
Sources: the user's survey (`knee_mri_anatomy_pretrained_models.md`), two research reviews of the
same day, and the frame-and-targets campaign
([run notes](../../autoresearch/campaigns/frame-and-targets/run-notes.md)).

## Why

Every frame we built reads the knee's geometry from where label-trained heads look. Those spots
carry 4 to 6 mm of noise, move with training, and cannot give a bone's direction. A pretrained
segmenter gives the bones themselves.

## Rules we keep

- The segmenter runs **offline, on training and validation studies only**. Nothing from it goes
  into the loader or the Kaggle bundle; at prediction the model uses only what it learned.
- Models must be free and publicly accessible without institutional sign-off (host rulings in
  `docs/modeling/supervision.md`). Clear: TotalSegmentator `total_mr` (Apache-2.0, femur only).
  Passes the host's test: MRAnnotator (CC-BY-NC, public weights). Unchecked: the others.

### Models cleared by the user's licence and rules pass (2026-10-07)

In priority order: `aagatti/nnunet_knee` (femur, tibia, patella, cartilage, menisci); SA-Profile
(knee MRI landmarks); MRAnnotator (femur, tibia, patella, fibula; yes with a caveat); PYCAD Knee
(ACL, PCL, meniscus, cartilage); SKM-TEA models (cartilage, meniscus); kneeseg SKI10;
MultiPlanarUNet knee; Denizlab IWOAI; SAM-Med3D, MedSAM2, nnInteractive (promptable refinement);
TotalSegmentator `total_mr` (legal, less knee-specific); kneeseg OAI (weak).
Removed: SAT, RobustCartilageSegmentation, BiomedParse, TotalSegmentator
`appendicular_bones_mr`, DOSMA weights, LDSeg, KOA-Segmentator, AutoMedLabel, KneePreM.

### Benchmark so far (idea 0)

- `aagatti/nnunet_knee`, full resolution, 24 sagittal volumes of validation studies: on
  fluid-sensitive fat-suppressed series femur 11/11, tibia 11/11, patella 10/11, clean outlines;
  on T1/PD series 1/13 for each bone; menisci found in none. About 9 cm of femur and 6 cm of tibia
  visible. 45 s per volume at full resolution, about 10 s with the coarse model and a larger step.
- 4,150 of the 4,407 training studies (94%) have a fat-suppressed sagittal series; 257 have only a
  T1/PD sagittal.
- MRAnnotator's weights could not be downloaded from the command line (the Google Drive link
  returns "not found" without a browser session).
- Scripts and volumes: `/home/claudio/.cache/knee-segmenters/`; drawings:
  `outputs/analysis/frame-and-targets/segmenter-benchmark/`.

## What we compute

- **One sagittal series per study** (largest field, most slices), as a 3D volume. Every series of
  a study shares patient coordinates, and series move under 2.7 mm against each other in 99% of
  pairs, so one mask places the bones in all slices of the study.
- **Four structures:** femur, tibia, patella, fibula.
- **Kept:** one small volume per study in patient coordinates at 2 to 3 mm.
- **Stages:** about 30 studies (benchmark), the 874 validation knees (yardstick), all training
  studies (targets).

## Use cases (agreed with the user, 2026-10-07)

The shared ingredient is a per-image token map (femur, tibia, kneecap, none), projected from one
segmented volume per knee onto every image of the study through the shared patient coordinates.

1. **Which tokens each label should look at.** Regions defined by the bones (femur-tibia gap per
   side, kneecap-femur gap, inside the bones, around them), to check where the heads look today
   or to guide them in training. Ideas 1, 4, 7 below.
2. **Fitting the frame better.** Joint centre, joint level, plateau tilt and femur lean per knee
   as targets for the frame. Ideas 2, 3, 8, 11, 12.
3. **Distilling the anatomy.** A head predicts the token map during training and is dropped at
   prediction; works on whichever images have a map. Idea 10. First pick: cheapest, and uses 1,
   2 and 4 can read its output at prediction.
4. **The input.** Crops centred and turned by the real bones in training; view sizes and offsets
   from measured anatomy. Ideas 5, 9, 14.
5. **Measurements as extra targets.** Femur-tibia gap per side, cartilage volume per compartment,
   kneecap position, predicted alongside the labels. New.
6. **Which slices matter.** Where each slice cuts the knee, for bag sampling or pooling. Part of
   idea 11.
7. **Diagnosis and cleaning, no training.** Blanking out structures, series without a knee, left
   and right. Ideas 4, 13.

## Ideas, most promising first

Status: `todo`, `doing`, `done`, `dropped`.

### 0. Prerequisite: does a segmenter work on our data? — `doing`

Benchmark on about 30 studies across sites, vendors and sequences: femur found, tibia found,
plausible, failures, centimetres of shaft visible, seconds per series. Drawings on the six sample
knees (validation rows 532, 200, 370, 374, 45, 5). MRAnnotator first, TotalSegmentator `total_mr`
as the open-licence comparison.

### 1. Regions that are the anatomy (replaces the cylinder) — `todo`

A head predicts, per token, what it contains (femur, tibia, kneecap, the spaces between). Labels
read tokens by what they are: menisci and compartment OA in the joint space of their side, PF OA
between kneecap and femur, ACL in the notch, Contusion and Fracture inside the bones, Effusion and
Synovitis around them. The bend needs no angle: each bone carries its own regions. Medial and
lateral come from the fibula. The same head teaches the backbone anatomy.
Needs clean masks; tokens are 12 to 14 mm, so thin structures are located by the bones around
them.

### 2. Yardstick for what we have — `todo`

Against the masks, on the validation knees: the frame's centre, joint line, the two height cuts,
the learned tilt, the menisci tilt, the template angle, the kneecap spot, the knee side, and the
true bend per knee. Decides how wrong the current frame is and which of today's estimates were
right.

### 3. The two tilts (double cylinder), learned from measured angles — `todo`, fallback for 1

A small head predicts each knee's tibia and femur direction from the tokens, trained on the axes
fitted to the masks. The femur section follows it. Use if the masks are only good enough for an
axis.

### 4. Which structure each label depends on — `todo`

At validation, blank out one structure at a time and see which labels drop. A measured basis for
the regions of idea 1.

### 5. Coverage of each crop — `todo`

How much of each structure each 84, 93 and 120 mm view contains, per series type. The view table
set from measured anatomy.

### 6. Knee side from the fibula — `todo`

A truth for left and right and for medial and lateral; supervision for the side head, which today
relies on the scan header and a small guesser.

### 7. Attention guidance in training — `todo`

Label heads are nudged toward their structure during training only. Needs the meniscus,
cartilage and ligament models for the heads that read soft tissue.

### 8. Training with the true frame — `todo`

Routed heads train on the frame from the masks while a predictor learns it; at prediction the
predictor takes over.

### 9. Crops aligned by the anatomy — `todo`, after the frame is settled

Train on knees turned and centred by the true anatomy; at prediction use the model's own estimate
from a first look. Follows `crops-turned-to-the-frame` (×4 +0.0015, ×1 −0.0006, 0.952 public).

### 10. Segmentation as a side task or a pretraining stage — `todo`

A decoder trained on the masks alongside the labels and dropped at prediction, or a stage that
teaches the backbone to segment knees before the labels.

### 11. Landmarks, token positions, slice depth — `todo`, come almost free with 1

Joint centre, kneecap centre, tibial tuberosity, fibular head as regressed points; each token's
coordinates in the bone-defined frame; where each slice cuts the knee.

### 12. Constants from the population — `todo`

Loader anchors and offsets per series type, core and ring radii, cut heights, set once from
measured bone positions and sizes.

### 13. Cleaning rules — `todo`

Series where no knee is found point at wrong body parts or bad scans; rule-based checks designed
and validated against the masks (no trained model in the loader).

### 14. Anatomy-aware augmentation — `todo`

Shifts and rotations about the joint; mixing between series registered by the bones.

### 15. Soft-tissue models (second step) — `todo`

Menisci, cartilage, cruciates from `aagatti/nnunet_knee`, PYCAD or SKM-TEA: a better joint line
and regions for the meniscus, OA and ACL labels. Licence and training data to be checked first.

## Dropped

- **Anything needing the segmenter at prediction:** fixed geometry per knee from masks at test
  time, registration of test series by masks.
- **RadImageNet's 18 knee classes as locators:** pathology labels per key image; their maps are
  one blob at the joint centre for every class (`outputs/analysis/frame-and-targets/radimagenet-knee/`).
  The pretrained head's outputs do carry pathology (ACL 0.73 to 0.76, Contusion 0.80 to 0.82 with
  no training), which is a separate, small idea for the label heads.
- **Angles from pairs of attention spots**, including tendon spots: 15 to 19° of noise at the
  levers we have.

## Waiting in the stopped campaign

`femur-section-turned-to-a-running-template-2` and
`tibia-and-femur-sections-turned-to-running-templates-2`: the femur band's angle by matching its
tokens to a running template (0.76 agreement between the two series of one knee in a check without
training). Worth checking against the masks in idea 2 before running.

## Log

- 2026-10-07 18:40: benchmark of `aagatti/nnunet_knee` with the coarse model (6 s per volume) on
  34 fat-suppressed sagittal volumes: femur and tibia 34/34, patella 21/34, menisci 5/34; femur
  axis repeats to 1.6° between two series of one knee (median, 6 pairs), tibia axis to 9.1°.
  Keep only each bone's largest connected piece (stray pieces in muscle on row 532).
- 2026-10-07 18:50: dropped SA-Profile (only the kneecap groove of the femur; no tibia, no tilts)
  and idea 6 (knee side from the fibula): the user considers medial and lateral already right.
  MRAnnotator's only remaining use is the 6% of studies without a fat-suppressed sagittal.
- 2026-10-07 19:20: tibia direction from the plateau (a plane fitted to the top of the tibia
  mask) instead of the shaft: repeats to 1.4° in the sagittal plane and 0.9° in the coronal plane
  between two series of one knee (median, 5 pairs); the tibia shaft fit stays poor (4.8°, wrong on
  row 370 where the picture cuts the bone). Femur shaft 2.5°. A rule "the tibia is the piece
  right under the femur" fixes row 532 and rejects 3 of 34 volumes. On 26 knees: femur lean sd
  4.8°, plateau tilt sd 3.7°, uncorrelated (−0.18); bend (femur against the plateau's normal)
  mean −1.7°, sd 6.6°, range −12 to +16°, beyond 5° in 54% of knees; bend repeats to 1.9°.
  Scripts `bone_geometry.py`, `tibia_fit.py`, `bend.py` in `/home/claudio/.cache/knee-segmenters/`;
  drawings `bone-axes-first-fit.png`, `bone-axes-second-fit.png`.
- 2026-10-07 19:50: masks projected onto run 24's crops and 7 by 7 token grid for 26 knees
  (`scripts/adhoc/side_routed_session/token_anatomy_maps.py`, CPU only; `token-anatomy-maps.png`,
  `token-anatomy-shares.parquet`). The outlines land on the bones in all three planes, T1/PD
  series included, from the one fat-suppressed sagittal mask. Per sagittal or coronal image about
  10 tokens are mostly femur and 7 mostly tibia; half the tokens hold some bone, and in 70% of
  those one bone fills at least half the token. The kneecap is at the crop's edge or outside
  (0.3 to 0.7 tokens per image). The sagittal and coronal crops show almost no femur shaft.
- 2026-10-07 20:05: run 24's frame against the bones on the same 26 knees
  (`scripts/adhoc/side_routed_session/frame_against_bones.py`, `frame-against-bones.png`, `.csv`).
  Centre: the real joint centre is 1.2 mm lateral (sd 3.0) and 4.4 mm in front (sd 4.6) of the
  frame's. Heights: the lower cut is 11.0 mm above the joint level (sd 3.1, range 6.5 to 18.3),
  the upper cut 26.1 mm above (sd 2.5); neither is between the bones. Nod: frame +13.7° (sd 4.1),
  plateau +1.0° (sd 3.9), femur −0.2° (sd 4.7); frame minus plateau +12.7° (sd 6.1); correlation of
  the frame's nod with the plateau's −0.16 and with the femur's −0.06. Roll: frame sd 5.8°,
  plateau sd 3.2°, correlation +0.33.
- 2026-10-07: bone coverage of run 24's crops on the 26 knees, inside each crop's rectangle
  (`scripts/adhoc/side_routed_session/crop_bone_coverage.py`, `crop-bone-coverage.csv`). Sagittal
  84 mm (T1/PD): condyles 99%, plateau 97%, femur seen to 42 mm above the joint and tibia to 39 mm
  below (90 and 62 mm segmented), kneecap 52% inside on average and whole in 11% of knees.
  Sagittal 93 mm (fluid): femur 49 mm, tibia 42 mm, kneecap 74%, whole in 38%. Coronal 84 and
  93 mm: femur 42 and 48 mm, tibia 40 and 44 mm. Axial 120 mm: kneecap whole in 96 to 100%.
- 2026-10-07: a speed test with four segmenter processes at once ran WSL out of memory twice
  (about 28 processes). Keep one process (6 s per volume, about 14 GB); no parallel parts.
- 2026-10-07: single-process fast segmenter (`scripts/adhoc/side_routed_session/segmenter_fast.py`,
  run with the segmenter environment): reading in three threads, preparation and restoring on the
  GPU, the model's own sliding window. 0.56 s per volume on the 67 benchmark volumes (0.51 s of it
  the model), 4.0 GB of memory, 4 GB of GPU memory, read-ahead bounded to 12 volumes and a 14 GB
  memory guard. Overlap with the original pipeline's masks on the 34 fat-suppressed volumes
  (`segmenter_fast_agreement.py`): femur median 0.999 (lowest 0.983), tibia 0.998 (0.944),
  kneecap 0.993 (0.941). All 4,150 studies: about 40 minutes plus the series conversion.
- 2026-10-07: tried to go below that (`segmenter_fast_settings.sh`, 67 volumes each). Patches per
  call 1, 2, 4: 0.51, 0.50, 0.51 s per volume (4.3, 5.5, 7.9 GB of GPU memory): the GPU's compute
  is already full with one patch. Channels-last memory format: slower, 0.72 s. Coarser input
  breaks the masks: 1.3 times coarser gives overlap 0.964 femur, 0.919 tibia, 0.744 kneecap
  (lowest 0.33, 0.12, 0.06); 1.6 times gives 0.941, 0.835, 0.615. Kept: native scale, one patch
  per call, 0.51 s per volume. Next cost is reading: read the series with the repository's own
  DICOM reader and keep only a small bone volume, instead of converting to files.
- 2026-10-07: MRAnnotator. The README's visible weights link is dead; the address behind it works
  (Google Drive id `15gQuvmaDlbs7HlGpISWNqnnWYGp9-Xc4`, 2.03 GB, no login). Licence: CC-BY-NC,
  stated in the README and the paper (Radiology Advances 2024); weights public; only their
  dataset needs a usage agreement. Run on the 67 benchmark volumes in LAS orientation with a
  single-process script (`segmenter_mrannotator.py`, `segmenter_mrannotator_summary.py`,
  `mrannotator-samples.png`): femur, tibia, kneecap and fibula found in 34/34 fat-suppressed and
  33/33 T1/PD volumes, every knee usable (row 532 included), one connected piece per bone.
  Between the T1/PD and the fat-suppressed series of the same knee (28 pairs): plateau tilt
  differs by 0.4° at the median (90th percentile 1.4°), femur axis by 0.3° (1.0°). Overlap with
  the first model on fat-suppressed series: femur 0.87, tibia 0.85, kneecap 0.56 (its bones are
  smaller in volume, 157 against 195 ml of femur; its kneecap is complete where the first
  model's was partial). 3.3 s per volume (84 patches at 0.36 mm), 3.1 GB of memory, 5.3 GB of GPU.
  MRAnnotator replaces `aagatti/nnunet_knee` for the bones.
- 2026-10-07: MRAnnotator speed (`STEP` and `SKIP_AIR` settings of the script,
  `segmenter_compare.py`). No overlap between patches (`STEP=1.0`): 1.85 s per volume, 47 patches
  instead of 84; against the overlapped run the plateau tilt moves by 0.16° at the median (largest
  1.1°) and the femur axis by 0.09° (largest 3.1°); overlap femur 0.988, tibia 0.987, kneecap
  0.971, fibula 0.978 (lowest 0.92, 0.86, 0.65, 0.81). Skipping patches of air saves under two
  patches per volume (1.81 s) with identical masks. The cost is structural: the model works at
  0.36 mm in every direction, so our 3 to 4 mm slices are repeated about ten times.
- 2026-10-07: sample of 550 knees (400 training, 150 validation with the six sample knees; one
  sagittal series each, the one with most slices; `segmenter_convert_sample.py`, seed 20261007)
  segmented with `STEP=1.0`: 18.9 minutes, 2.06 s per knee, 5.3 GB of memory. Targets exported
  by `segmenter_export_targets.py` to `data/labels/prepared/anatomy-mrannotator-sample550/`
  (2.5 mm bone volume per knee, `knees.json` with joint centre, plateau normal, femur axis).
- 2026-10-07: copying each slice fewer times (`SQUASH`), against the `STEP=1.0` masks on the 67
  benchmark volumes. Half the copies: 1.3 s per volume, overlap femur 0.957, tibia 0.968, kneecap
  0.941, fibula 0.913 (lowest 0.83, 0.81, 0.73, 0.00); plateau tilt moves 0.32° at the median
  (largest 1.5°), femur axis 0.24° (90th percentile 1.4°, largest 13.9°). A third of the copies:
  0.7 s, overlap 0.856, 0.890, 0.863, 0.667; plateau 0.44°, femur axis 1.75° (90th percentile
  6.5°). Half the copies degrades mildly, a third clearly; kept at full copies.
- 2026-10-07: anatomy head built (uncommitted): `src/knee_abnormality/training/anatomy.py`
  (targets packed on the GPU, per-token shares from 16 points, linear head on the tokens before
  the position code), loss in `training/lightning.py`, `patient_position` and `token_steps`
  outputs in `modeling/model.py`, settings `paths.anatomy_targets` and
  `trainer.anatomy_loss_weight`.
- 2026-10-07: first run, `outputs/training/rn18-anatomy-head-sample550` (run 24's config, fold 1,
  seed 1110484336, anatomy weight 0.1, 400 supervised training knees; launched by
  `launch_anatomy_head.sh`). Against run 24: ×4 0.924947 against 0.925067 (−0.00012), ×1 0.916815
  against 0.917070 (−0.00026); sibling SD 0.0004; same 2 min 3 s per epoch. Anatomy on the 150
  validation knees at epoch 9: main class right for 94.0% of tokens, overlap femur 0.892, tibia
  0.883, kneecap 0.665, fibula 0.538, share error 0.071 (epoch 0: 0.845, 0.837, 0.414, 0.132);
  training agreement 93.3%. Labels at ×4 (yardstick in brackets): Lateral Meniscus +0.0042
  (0.0024), Medial OA +0.0017 (0.0008), Contusion +0.0017 (0.0013), Lateral OA +0.0015 (0.0034),
  Fracture +0.0011 (0.0044), ACL +0.0003, Baker's +0.0002, Medial Meniscus −0.0004, Effusion
  −0.0017 (0.0015), PF OA −0.0018 (0.0005), MCL −0.0037 (0.0029), Synovitis −0.0045 (0.0022).
  Lesson: the model learns femur and tibia per token from 400 knees at no cost in speed or macro
  score; the kneecap and fibula stay weak, the kneecap being outside most sagittal and coronal
  crops. The label heads do not use the prediction yet.
  A first launch lost its last two options to a wrapped paste and trained a plain repeat for one
  epoch (0.89281); its folder was deleted.
- 2026-10-07: joint plane from the predicted bone map, checked without training on the 150
  validation knees with the run above (`anatomy_joint_plane_check.py`,
  `anatomy-joint-plane-check.csv`). Plane fitted to femur share minus tibia share over bone
  tokens, refitted three times with weights near the plane (12 mm): level spread 1.7 mm with a
  constant offset (the plane sits 6.4 mm above the plateau's surface), forward tilt error sd 2.8°
  (real tilt sd 4.5°), sideways tilt error sd 2.0° (real sd 4.0°). Without the refit: 2.1 mm,
  3.3°, 3.3°. A separating plane (logistic): 1.8 mm, 3.2°, 2.5°.
- 2026-10-07: second run, `outputs/training/rn18-anatomical-frame-sample550`
  (`model.anatomical_frame`; `launch_anatomical_frame.sh`): a height head in the model predicts
  each token's height above the plateau, one plane per knee is fitted to the predictions, and the
  frame takes its height origin and tilt from that plane, with a joint row of half a token each
  side (learnable offsets) and the label gradient kept out of the plane. Drawings
  `anatomical-frame-3d.png`, `anatomical-frame-cylinders.png`.
  **The height target does not work as the plane's source.** At epoch 3 on the 150 validation
  knees (`anatomical_frame_knee_errors.py`): level spread 5.9 mm (7.6 mm from the loader's knee
  centre alone), forward tilt error sd 4.1° (real 4.5°), sideways 2.7° (real 2.9°); the height
  head's error is one shift per knee (sd 5.9 mm between knees, 3.8 mm inside a knee), so more
  tokens do not average it out. Reading: local appearance says what a token is, not how far it
  is from the joint. **Lesson for the next build: take the plane from the predicted bone map
  (1.7 mm, 2.8°, 2.0°), with the bone head inside the model, and drop the height head.**
  Final: ×4 0.923643 (−0.00142 against run 24), ×1 0.915436 (−0.00163).
- 2026-10-07: can the femur's and tibia's own tilts be known? Fitting a centre line through the
  femur tokens (`anatomy_bone_axes_check.py`) does not give the femur's direction: correlation
  with the real bend 0.04 to 0.33 from the model's map and 0.03 to 0.45 from the segmenter's own
  masks at token level; the crops show about 45 mm of femur, which is condyle. A linear probe on
  the trained model's bone-weighted mean features (`anatomy_tilt_probe.py`, 399 training knees,
  150 unseen): forward bend of femur against plateau correlation 0.70 (error sd 5.4° of 7.4°),
  plateau forward tilt 0.63, femur forward lean 0.53, sideways tilts 0.26 to 0.46.
- 2026-10-07: third run, `outputs/training/rn18-tiltable-frame-sample550`
  (`launch_tiltable_frame.sh`; height head removed). Bone head inside the model; joint plane
  fitted to predicted femur share minus tibia share over bone tokens with three refits near the
  plane; frame's height origin and tilt from that plane; joint row half a token each side
  (learnable); a bend head on bone-weighted pooled features (two bones by three planes) predicts
  the femur's forward lean, trained on the real shaft from the masks; the femur section above
  the upper cut is sheared forward by it. Checks before launch: `tiltable_frame_check.py`.
  Against run 24: ×4 0.924810 (−0.00026), ×1 0.916333 (−0.00074). On the validation knees at
  epoch 9: joint plane error 1.64 mm (1.85 at epoch 0), bend error 2.8° against 5.2° for always
  predicting no bend (5.1° at epoch 0), overlap femur 0.887, tibia 0.873, kneecap 0.640.
  Labels at ×4: MCL +0.0048, Contusion +0.0029, ACL +0.0010, Effusion +0.0009, Medial Meniscus
  +0.0003, Baker's +0.0003, Lateral Meniscus −0.0002, Lateral OA −0.0010, Medial OA −0.0012,
  Synovitis −0.0028, PF OA −0.0039 (yardstick 0.0005), Fracture −0.0042 (0.0044).
  Lesson: the plane from the bone map is accurate from the first epoch and the bend is learnable
  from the picture; the frame anchored on them scores level with run 24 overall, with PF OA the
  one clear loss.
- 2026-10-07: joint classes derived from the bone masks (`joint_class_candidates.py`,
  `joint_class_scale_free.py`; `joint-class-candidates.png`, `joint-class-scale-free.png`).
  Distance rules: within 4, 6, 8 mm of both bones give 1.2, 10.0, 20.3 ml; distances adding to
  12 mm or less 22 ml; straight between the bones 14.7 ml but 8 to 10 mm narrower. Direction
  rule (nearest femur and tibia surfaces on opposite sides): 10.9 ml at 150°, 31.1 ml at 120°,
  no millimetres, and it also fills the notch between the condyles. Scaling by the knee's own
  size is too noisy. Kneecap joint within 6 mm of kneecap and femur: 4 ml. Masks need the
  largest piece per bone kept (a stray tibia piece on row 374) and holes filled.
- 2026-10-07: seven-class targets, `data/labels/prepared/anatomy-mrannotator-sample550-joints/`
  (`segmenter_export_joint_targets.py`, 3 minutes on four workers at 1 mm): bones closed (largest
  piece, 4 mm closing by distance transforms, holes filled); joint = points with femur and tibia
  on opposite sides (135° or more), closed 4 mm, bones removed; kneecap joint the same between
  kneecap and femur. Medians: joint 29.7 ml, kneecap joint 18.3 ml. Checks
  `joint-class-closed-twice.png`, `anatomy-lookup-check-joints.png`,
  `joint-targets-pixel-view.png`: the joint is right and generous front and back; the kneecap
  class also covers the pouch above the kneecap and the gutters; classes are blocky from side to
  side (3 to 4 mm sagittal slices).
- 2026-10-07: fourth run, `outputs/training/rn18-anatomy-sections-sample550`
  (`launch_anatomy_sections.sh`; bend head and femur lean removed). Seven classes inside the
  model; sections by class (femur, joint and tibia shares, the rest placed by height against the
  joint plane); per-route section weights as parameters starting at 0.95 or 0.05; kneecap route
  also reads kneecap and kneecap-joint tokens. Against run 24: ×4 0.924694 (−0.00037, bootstrap
  95% interval −0.0019 to +0.0012), ×1 0.916203 (−0.00087, −0.0031 to +0.0012).
  Audit in `outputs/analysis/frame-and-targets/anatomy-sections-audit/` (`audit.txt`,
  `map-prediction.png`, `attention-per-label.png`, `route-weights.png`; scripts
  `audit_extract.py`, `audit_analyse.py`, `audit_visual.py`; run 24 extracted with its archived
  source):
  - Labels, paired bootstrap on 874 knees: Synovitis −0.0052 at ×4 (4% of draws above zero) and
    −0.0098 at ×1 (interval excludes zero); ACL −0.0062 at ×1 (excludes zero), −0.0018 at ×4;
    Medial OA +0.0051 at ×1 (excludes zero); PF OA −0.0026 at ×4 (15% above zero); the rest
    inside their intervals.
  - Map over the 150 validation knees: main class right 92.7%; overlap femur 0.897, tibia
    0.888, joint 0.741, kneecap 0.687, kneecap joint 0.633, fibula 0.568; joint tokens mistaken
    for femur 8%, tibia 7%, other 10%; plane error 1.58 mm.
  - Learned: the section weights did not move (all within 0.02 of their start); cuts +6.4 and
    −5.9 mm; core radius 9.3 mm against 14.8 in run 24, ring 21.9 against 26.3.
  - Attention by real anatomy: menisci and compartment OA moved from femur (0.34 to 0.39 of
    attention in run 24) to tibia and joint (femur 0.16 to 0.18, tibia 0.30 to 0.33, joint 0.28
    to 0.35) and became more focused (59 to 98 effective tokens against 89 to 135). ACL: femur
    0.63 to 0.40, joint 0.13 to 0.29, tibia 0.00 to 0.08. PF OA: femur 0.34 to 0.49, kneecap
    0.12 to 0.07, kneecap joint 0.15 to 0.13; its region grew (mean weight 0.25 to 0.31) and
    routing moves less of its attention (0.66 to 0.50). Synovitis: joint 0.05 to 0.11, kneecap
    and kneecap joint 0.14 to 0.09, region 0.30 to 0.34.
  - Sides consistent: fibula tokens weigh 0.95 in the lateral meniscus route and 0.00 in the
    medial one; no knee with an uncertain side; routing fallback never used.
  Lessons: sections by anatomy work for the joint labels; the front regions (kneecap,
  front_core) are a broad block of the front half of the knee and got broader, which is where
  PF OA and Synovitis lost; the section weights need a faster learning rate or a milder start to
  be learned at all; the core shrank and ACL lost femur attention.
- 2026-10-07: fifth run, `outputs/training/rn18-anatomy-front-sample550`
  (`launch_anatomy_front.sh`, `apply_front_regions_edits.py`): kneecap route = kneecap and
  kneecap-joint tokens (full at 30% share) plus the front block above the joint at a quarter;
  front_core above the joint row plus kneecap tokens; section preferences starting at 0.88 and
  0.12 and learning five times faster; core and ring fixed at 1.1 and 2.2 pitches. Against run
  24: ×4 0.924984 (−0.00008, interval −0.0016 to +0.0014), ×1 0.917087 (+0.00002, −0.0020 to
  +0.0018). Audit `audit-front.txt`. Labels: Medial OA +0.0062 at ×1 (interval excludes zero)
  and +0.0015 at ×4; Contusion +0.0030 at ×4 (94% of draws above zero); Baker's +0.0018 (87%);
  Lateral Meniscus +0.0039 at ×1 (88%); ACL back to level (0.0000 at ×4, −0.0012 at ×1); PF OA
  −0.0014 at ×4, +0.0005 at ×1, its attention on the kneecap joint 0.15 to 0.19; Synovitis
  −0.0068 at ×4 and −0.0110 at ×1, both intervals excluding zero. The section preferences now
  move (tear routes up to 0.4 from their start; label routes within 0.03).
  **Synovitis fell in every run that trains the anatomy map, including the first, where the
  frame was untouched** (×4: −0.0045, −0.0028, −0.0052, −0.0068), with its attention by anatomy
  now equal to run 24's. Reading: the cost comes from the anatomy loss shaping the backbone, not
  from the regions. Without Synovitis this run would be about +0.0005 at ×4 and +0.0009 at ×1.
  Next check: train the map without its gradient reaching the backbone.
- 2026-10-07: a map head on detached tokens (two layers, with Dice; `rn18-anatomy-reader-sample550`,
  stopped by the user's decision after one epoch): femur 0.52, tibia 0.44, joint 0.40, plane error
  4.6 mm against 0.85, 0.84, 0.68 and 1.8 mm with the gradient. The map's fast learning is the
  backbone adapting.
- 2026-10-07: sixth run, `outputs/training/rn18-anatomy-crops-sample550` (`launch_anatomy_crops.sh`;
  `apply_pose_crops_edits.py` on the crops-turned build of the frame-and-targets campaign): Dice
  added to the map's loss (gradient to the backbone, linear head) and posed crops: canvases with a
  17 mm margin for sagittal and coronal views, a first look without gradient in training and at
  prediction, each crop shifted so the joint centre lands at the average spot and turned so the
  joint is level (limits 12° and three quarters of the margin; training jitter ±3° and ±3 mm).
  Check with the true pose: `pose_crops_check.py`, `pose-crops-check.png`. Against run 24: ×4
  0.925885 (+0.00082, interval −0.0020 to +0.0035), ×1 0.917130 (+0.00006); **0.953 public as S122**,
  equal to run 24's S118. Training 1,863 s (a job ran beside it) and ×4 prediction 55 s against
  1,274 s and 33 s. Map: femur 0.905, tibia 0.899, joint 0.759, kneecap 0.735, kneecap joint 0.675,
  plane error 1.73 mm. Per target and per view: `target-analysis.txt`; audit `audit-crops.txt`;
  drawings `frame-prediction.png`, `resulting-crops.png`. Gains: Medial OA +0.0033 / +0.0112,
  Baker's +0.0041 / +0.0049, MCL +0.0065 / +0.0096, Lateral OA +0.0053 / +0.0041, Medial Meniscus
  +0.0021 / +0.0039. Losses: Lateral Meniscus −0.0046 / −0.0148, Synovitis −0.0056 / −0.0083,
  Contusion −0.0058 and Fracture −0.0057 at ×1, ACL −0.0030 at ×4.
  Flaws found in the audit: (1) the turned crops are cut bilinearly and keep 0.81 of the fine
  detail, bicubic keeps 0.98 (`posed_crop_sharpness.py`); (2) the shift across removes only a third
  of its spread (correlation 0.76 to 0.81 with the real one) while the shift up and down removes
  three quarters (0.97) (`pose_accuracy.py`); (3) the turn's correlation with the real one is 0.73
  to 0.80; (4) Synovitis pays for the map in every run; (5) the canvases covered the 95th
  percentile of the need (45 and 39 px), so they were not oversized; (6) the audit compared run
  24's attention with the anatomy looked up at the posed tokens' positions, corrected to its own
  positions (the figures moved by up to 0.05).
- 2026-10-07: campaign `autoresearch/campaigns/anatomy-masks/` created and validated, not
  started; reference and incumbent the sixth run; seven forced seeds.

- 2026-10-07 17:20: document written; benchmark (idea 0) started.
