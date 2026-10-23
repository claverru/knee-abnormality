# Anatomy from segmenters: uses and ideas

Statuses are those at the end of the working day, 2026-10-07; later work on these ideas is in the
[anatomy-masks campaign](../../../../autoresearch/campaigns/anatomy-masks/INDEX.md) and the head runs in
[anatomy head runs](anatomy-head-runs.md).

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
