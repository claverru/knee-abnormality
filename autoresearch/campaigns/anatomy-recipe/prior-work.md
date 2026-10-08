# Prior work for the anatomy-masks campaign

## This campaign's incumbent (2026-10-08)

`rn18-anatomy-map-centre-control`: the anatomy-masks reference with its two adopted seeds (posed
crops cut with bicubic interpolation; the joint centre across and front to back from the anatomy
map, with a fixed offset), the training speed-ups of 2026-10-08 (middle channel only from the
loaders, `channels_last`, memoised tokens and routing; the same weights give the same
predictions) and the two-knee pose fix (each series of a two-knee study takes its own knee's sign
in the pose). The same recipe before the speed-ups and the fix scored ×4 0.927083, ×1 0.917666
and 0.954 public as S125. Everything below this section describes the anatomy-masks reference
and its predecessors; what the anatomy-masks campaign's eighteen runs added is in
[its run notes](../anatomy-masks/run-notes.md), which also hold the yardsticks measured there:
three near-reruns of the incumbent gave ×1 of 0.917042, 0.917186 and 0.917892, and MCL's ×4 AUC
ranged from 0.9209 to 0.9368 across runs on 47 positives.


The summary the researcher checks proposals against. Every figure below was produced on
2026-10-07; the audits are under `outputs/analysis/frame-and-targets/anatomy-sections-audit/` and the drawings
under `outputs/analysis/frame-and-targets/segmenter-benchmark/`. The design the model started
from is in [side-routed shared heads](../../../docs/ongoing/side-routed-heads.md) and the two
earlier campaigns on it are [routing and shifts](../routing-and-shifts/INDEX.md) and
[frame and targets](../frame-and-targets/INDEX.md).

## The incumbent

`outputs/training/rn18-anatomy-crops-sample550`: fold 1 of `all_5fold_scanner_v2`, seed
1110484336, ResNet-18 (Refined RadImageNet), ten epochs. ×4 0.925885, ×1 0.917130:
+0.00082 and +0.00006 against run 24 (paired bootstrap intervals −0.0020 to +0.0035 and −0.0033 to
+0.0034), the first run of the line above run 24 at both multipliers and its best ×4 (reruns of
the old design ranged 0.9244 to 0.9251). Training 1,863 s and ×4 validation prediction 55 s,
against 1,274 s and 33 s for the runs without posed crops; part of the training time came from
a job run beside it for several epochs (about 2 min 40 s per epoch undisturbed against 2 min 7 s).

Per target against run 24 (×4 / ×1): Medial OA +0.0033 / +0.0112 (×1 interval excludes zero),
Baker's +0.0041 (interval excludes zero) / +0.0049, MCL +0.0065 / +0.0096, Lateral OA +0.0053 /
+0.0041, Medial Meniscus +0.0021 / +0.0039, Effusion +0.0014 / +0.0004, PF OA −0.0007 / +0.0023,
Contusion +0.0027 / −0.0058, Fracture −0.0018 / −0.0057, ACL −0.0030 / −0.0011, Synovitis −0.0056
/ −0.0083, Lateral Meniscus −0.0046 / −0.0148 (×1 interval excludes zero). Audit:
`outputs/analysis/frame-and-targets/anatomy-sections-audit/audit-crops.txt`.

Its design, all in `modeling/model.py`, `training/anatomy.py` and `training/lightning.py`:

- **The anatomy map.** One linear head on the tokens before the position code predicts seven
  shares per token: other, femur, tibia, kneecap, fibula, joint (between femur and tibia), kneecap
  joint (between kneecap and femur). Loss: cross-entropy against the target shares plus one minus
  the mean soft Dice over the classes present, weighted 0.1. The gradient reaches the backbone.
  Targets for 400 of about 3,500 training knees and 150 of 874 validation knees; knees without
  targets skip the loss.
- **The joint plane.** Per knee, a plane is fitted by least squares to (femur share minus tibia
  share) over the tokens with at least half bone, refitted three times with Gaussian weights of
  12 mm around the plane, on detached shares. It gives the frame's height origin and its forward
  and sideways tilt. The centre across and front to back, and the turn about the vertical, still
  come from the two meniscus attention spots.
- **Sections.** Femur, joint and tibia shares place a token in those sections; the rest of the
  token is placed by its height against two cuts half a token above and below the plane
  (learnable offsets). Each route has three section preferences, parameters that start at 0.88
  or 0.12 and move five times faster than a plain parameter.
- **Regions.** Eight sectors, a core and a ring of fixed radius (1.1 and 2.2 token pitches). The
  kneecap route reads kneecap and kneecap-joint tokens in full (from a 30% share) and the front
  block above the joint at a quarter; the front-and-core route reads above the joint row plus
  the kneecap tokens.
- **Crops.** Sagittal and coronal views arrive as canvases with a 17 mm margin (314 and 306 px).
  A first look without gradient, in evaluation mode, on the ×1 slices gives the pose; each crop
  is then cut so that the joint centre lands at a fixed spot (sagittal 7.4 mm across and 2.3 mm
  down from the crop centre, coronal −1.1 and 1.3 mm, the averages measured with the true plane)
  with the joint line level. Limits 12° and three quarters of the margin; training adds ±3° and
  ±3 mm at random per series. Axial views are untouched.

## Yardstick

Run 24, `outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors`: ×4 0.925067, ×1
0.917070, 0.953 public (S118; the incumbent scored the same 0.953 as S122); its ResNet-50: ×4 0.933209, ×1 0.922934, 0.957 public (S119).
Near-identical runs spread over 0.0007 at ×4 and 0.0010 at ×1. A paired bootstrap of two runs
over the 874 validation knees gives a 95% interval of about ±0.0015 on the macro difference.
Sibling standard deviation per target: Fracture 0.0044, Lateral OA 0.0034, MCL 0.0029, Lateral
Meniscus 0.0024, Synovitis 0.0022, Baker's 0.0015, Effusion 0.0015, ACL 0.0014, Contusion 0.0013,
Medial OA 0.0008, Medial Meniscus 0.0007, PF OA 0.0005; macro 0.0004.

## The targets

`data/labels/prepared/anatomy-mrannotator-sample550-joints/` (read through
`paths.anatomy_targets`):

- `volumes/<study>.npz`: one 2.5 mm volume per knee in patient coordinates with the seven classes
  and 255 outside the scan;
- `knees.json`: per knee the plateau's centre and normal and the femur's axis (patient
  coordinates);
- `measurements.json`: per knee the kneecap's extent relative to the plateau centre and the gap
  between femur and tibia on each side of the patient (kneecap front 55.9 mm (sd 5.2, n 550) and back 40.0 mm (sd 4.5, n 550) in front of the plateau centre, top 54.6 mm (sd 6.7, n 550) and bottom 21.3 mm (sd 5.7, n 550) above the plateau; gap on the patient's right 6.5 mm (sd 0.9, n 550) and left 6.3 mm (sd 0.9, n 550), the tenth percentile of the gap over each side);
- `classes.json`: the class order and how the joints were derived.

They come from MRAnnotator (nnU-Net, bones only: femur, tibia, kneecap, fibula) run on one
sagittal series per knee. On 67 benchmark volumes it found every bone on both fat-suppressed and
T1/PD series; between the two series of one knee the plateau tilt differs by 0.4° and the femur
axis by 0.3° at the median. Bones are closed (largest piece, 4 mm, holes filled); a joint is the
set of points with the two bones on opposite sides (135° or more), closed 4 mm. Known limits: the
classes are blocky from side to side (3 to 4 mm sagittal slices); the kneecap joint also covers
the pouch above the kneecap and the gutters beside the femur.

## The runs of 2026-10-07

All on run 24's recipe, fold and seed, ten epochs, against run 24.

| Run | What it added | ×4 | ×1 |
|---|---|---:|---:|
| `rn18-anatomy-head-sample550` | A five-class map as a side task, regions untouched | −0.00012 | −0.00026 |
| `rn18-anatomical-frame-sample550` | Frame anchored on a plane fitted to a height predicted per token | −0.00142 | −0.00163 |
| `rn18-tiltable-frame-sample550` | Plane fitted to the bone map; femur section leaning by a predicted bend | −0.00026 | −0.00074 |
| `rn18-anatomy-sections-sample550` | Seven classes, sections by class | −0.00037 | −0.00087 |
| `rn18-anatomy-front-sample550` | Kneecap-led front regions, section preferences that move, fixed core | −0.00008 | +0.00002 |
| `rn18-anatomy-crops-sample550` (incumbent) | Dice in the map's loss, posed crops | +0.00082 | +0.00006 |

What each one taught:

- **The map is learned quickly and generalises.** With the gradient reaching the backbone:
  femur and tibia overlap 0.85 after one epoch and 0.89 to 0.90 at the end, joint 0.74, kneecap
  0.69, kneecap joint 0.63, fibula 0.57 (soft Dice at token level on the 150 validation knees);
  main class right on 92.7% of tokens. With Dice in the loss the thin classes reach those values
  in the first epoch and end higher (the incumbent: femur 0.905, tibia 0.899, joint 0.759, kneecap 0.735, kneecap joint 0.675, plane error 1.73 mm).
- **The plane from the bone map is accurate, a height per token is not.** Bone map: 1.6 to 1.8
  mm from the first epoch. Height per token: 5.9 mm spread between knees and no tilt learned;
  local appearance says what a token is, not how far it is from the joint.
- **A bone's direction cannot be read from its tokens.** A line through the femur tokens
  correlates 0.03 to 0.45 with the real femur direction even on the segmenter's own masks: the
  crops show 42 to 49 mm of femur above the joint, which is condyle. A head trained on the real
  axis reached 2.8° of error against 5.2° for no bend.
- **Sections by anatomy move the joint labels onto the joint.** Menisci and compartment OA went
  from 0.34 to 0.39 of their attention on the femur to 0.16 to 0.18, with the tibia and the joint
  taking it, and became more focused. Medial OA gained (+0.0051 and +0.0062 at ×1, intervals
  excluding zero).
- **The front regions were a broad block.** The kneecap route was the front half of the knee
  above the tibia; with the cut at the true joint line it grew, and PF OA's attention left the
  kneecap for the femur. Led by the kneecap classes it recovered most of the loss.
- **A core learned from the labels shrank** from 14.8 to 9.3 mm and ACL lost 0.006 at ×1; fixed
  at its start it came back to level.
- **Section preferences at 0.95 and 0.05 never moved.** At 0.88 and 0.12 with a five-fold rate
  the tear routes moved by up to 0.4 and the label routes by about 0.03.

### The cost to Synovitis

Synovitis fell in every run that trains the map, including the first, where the regions were run
24's: ×4 −0.0045, −0.0028, −0.0052, −0.0068 (the last with intervals excluding zero at both
multipliers, −0.0110 at ×1), while its attention by anatomy in the last was equal to run 24's. The
loss comes with the anatomy loss shaping the backbone, not with the regions.
The incumbent, with Dice in the map's loss: −0.0056 at ×4 and −0.0083 at ×1.

A map head reading detached tokens (two layers, with Dice) was tried and stopped after one
epoch: femur 0.52, tibia 0.44, joint 0.40, plane error 4.6 mm, against 0.85, 0.84, 0.68 and 1.8
mm with the gradient. The fast learning is the backbone adapting, so stopping the gradient
entirely is not an option; a fraction of it has not been tried.

### What the posed crops did, by target

From `target-analysis.txt` and a sharpness measurement on 900 slices (same audit directory). Against the run
before it (`rn18-anatomy-front-sample550`, the same model without posed crops and without Dice),
in AUC ×10000 at ×4 / ×1 against run 24:

| Target | Before the crops | Incumbent | Reading |
|---|---:|---:|---|
| Medial OA | +15 / +62 | +33 / +112 | Gained in every map run, most with the crops |
| MCL | +22 / −23 | +65 / +96 | Gained with the crops |
| Baker's | +18 / +8 | +41 / +49 | Gained with the crops |
| Lateral OA | −16 / +20 | +53 / +41 | Gained with the crops |
| Medial Meniscus | +4 / +15 | +21 / +39 | Gained with the crops |
| Effusion | +12 / +1 | +14 / +4 | Level |
| PF OA | −14 / +5 | −7 / +23 | Level |
| ACL | 0 / −12 | −30 / −11 | Lower at ×4 |
| Contusion | +30 / +5 | +27 / −58 | Lost at ×1 with the crops |
| Fracture | −20 / −9 | −18 / −57 | Lost at ×1 with the crops |
| Synovitis | −68 / −110 | −56 / −83 | Down in every map run |
| Lateral Meniscus | +8 / +39 | −46 / −148 | Lost with the crops |

- **A turned crop is blurred by its bilinear cut.** Its fine detail (mean absolute Laplacian) is
  0.81 of the loader's crop on 900 slices, the same under 1° as at 6 to 12°; the same cut with
  bicubic interpolation keeps 0.98, and a crop that is only shifted keeps 1.00. In training the
  jitter turns every crop.
- **The pose is accurate up and down and weak across** (`pose-accuracy.txt`, against the pose the
  real joint asks for on 150 validation knees). Shift up and down: correlation 0.97, leaving 2.1
  (sagittal) and 1.9 mm (coronal) of 8.3 and 7.3 mm. Turn: correlation 0.73 and 0.80, leaving 2.9°
  and 2.0° of 4.2° and 2.9°. Shift across: correlation 0.76 and 0.81, leaving 3.1 and 2.9 mm of 4.7
  and 4.9 mm; it comes from the midpoint of two attention spots.
- **The map locates the centre better than the spots** (`centre-sources.txt`, spread of the error
  against the real plateau centre between knees): spots 3.0 mm across and 3.1 mm front to back;
  the centroid of femur and tibia tokens near the joint plane 2.2 and 1.7 mm; the joint class
  alone 2.3 and 1.7 mm; the loader's own centre 4.5 and 4.7 mm.
- **Attention left the posed views.** With the crops, nine of twelve targets moved attention
  from the sagittal views to the untouched axial ones (Contusion 0.21 to 0.28 on the fluid axial
  view, Fracture 0.24 to 0.32, Lateral Meniscus 0.07 to 0.11); the number of valid tokens per
  plane is unchanged, so this is the heads' choice.
- **The frame the model predicts** is drawn in `frame-prediction.png` and the crops it cuts in
  `resulting-crops.png`: after posing the joint is level and at the same place in every crop;
  the model's plane is within 0.1 to 7.4° of the real one on the six sample knees.

## Measured on the masks

- **The current frame against the bones (run 24, 26 knees):** centre within 1.2 mm across and
  4.4 mm front to back; lower cut 11 mm above the joint; forward tilt 13° off and uncorrelated
  with either bone.
- **The two tilts are real:** femur lean sd 4.8°, plateau tilt sd 3.7 to 4.5°, bend between them
  sd 6.6 to 7.4°, uncorrelated.
- **What the crops show (26 knees):** condyles and plateau 97 to 100% inside every view; femur
  seen to 42 to 49 mm above the joint and tibia to 39 to 44 mm below; the kneecap whole in 11%
  of the 84 mm T1/PD sagittal crops and 38% of the 93 mm fluid ones, and in 96 to 100% of the
  axial ones.
- **What the true pose asks of the crops (60 validation knees):** sagittal turn sd 4.3° (largest
  10.5°), coronal 2.9° (largest 12°); shift up and down sd 7.5 and 6.8 mm, across 4.5 and 4.2 mm;
  2% and 4% of series at the shift limit.
- **Cost of the crops:** training about 25% slower (first look and larger canvases); at ×4
  prediction the first look adds about a quarter of the model's work.

## Earlier results that bear on this campaign

| Mechanism | Setting | Macro delta | Source |
|---|---|---:|---|
| Crops turned by half the old frame's angle | on run 24, at prediction and in training | ×4 +0.0015, ×1 −0.0006; ACL +0.0055 | frame-and-targets, `crops-turned-to-the-frame` |
| Crops turned and centred on the old frame | same | −0.0004 | frame-and-targets |
| Random crop offsets in training, unposed | on the height-cuts model | −0.0012 to −0.0020 | routing-and-shifts run notes |
| Offset pattern averaged at prediction | 14 variants | 0 on average | routing-and-shifts run notes |
| Edema bone head in the model | on run 24 | ×4 −0.0006, ×1 −0.00005 | frame-and-targets |
| Femur section on a lean learned from labels | on run 24 | lean collapsed to one value | frame-and-targets |

## Reusable code

- The six runs above, each with `source_snapshot.zip` (unzip outside the candidate).
- `scripts/adhoc/side_routed_session/`: the builders of the prepared targets, in order:
  `segmenter_convert_sample.py`, `segmenter_mrannotator.py`, `segmenter_export_targets.py`,
  `segmenter_export_joint_targets.py`, `mask_measurements.py`.

## Open leads

- A fraction of the map's gradient (a tenth to a third) between the two tested extremes.
- Whether Dice costs Synovitis more than it gains elsewhere.
- The kneecap is cut by the sagittal crops; the posed crop could bring it inside.
- Turning a crop blurs it; whether the turn can be kept in one resampling, or is needed at all.
- The first look reads every ×1 slice though the pose is one thing per knee.
- The joint gap per side is a number about compartment OA the model is never told.
- 400 of 3,500 training knees carry targets; the map is already good, so the gain from more is
  unknown.

## Premise checks of the forced seeds (2026-10-08)

On the validation fold with the control's weights (output
`outputs/analysis/anatomy-masks-speed/recipe-seed-checks.json`).

- **Wider fluid crops show image, not padding:** a crop a ninth wider lies inside the acquired
  image for 99.7% of its area in coronal and 99.8% in sagittal; 1.7% and 0.7% of slices are
  under 95%.
- **Shift room:** with a 22.5 mm margin the pose asks for more than 11.7 mm in 14.6% of fluid
  coronal and 16.1% of fluid sagittal series, against 12.0% and 13.4% above today's 12.75 mm.
  The two field seeds therefore take a 24 mm margin, which leaves the zoomed view 12.8 mm.
- **Kneecap in the fluid sagittal crop, on the 550 mask knees with the incumbent's joint target:**
  the front reach goes from 51.6 to 56.7 mm and the kneecap's front is inside for 59% of knees
  against 20% (whole kneecap 36% against 8%). With the true-centre target of the offset seed the
  figures are 76% against 36%.
- **Femur's front:** the front tokens between 15.6 and 61.3 mm above the joint plane number 235
  per study, 18% of a study's tokens, and 57% are soft tissue; femur tokens among them are 60
  per study, 4.6%, read by the kneecap route at a mean weight of 0.31. The seed takes the femur
  tokens only.
- **Kneecap front from the first look, against the masks on 150 knees:** every knee has kneecap
  tokens. The share-weighted mean front coordinate correlates 0.67 with the mask's kneecap front
  and leaves a residual of 3.9 mm after its 11.9 mm offset (a constant leaves 5.3 mm); the
  largest front coordinate among kneecap tokens correlates 0.56. The move a knee needs is 7.4 mm
  on average, p90 14.3 mm, positive for 93% of knees, so the seed raises the fluid sagittal
  view's shift limit to 16.9 mm.
- **Centre offset:** +1.23 mm across and +2.07 mm front on the training knees (control's logs),
  +1.18 and +2.13 mm on 150 validation knees (frame comparison); the incumbent's crop targets
  differ from the true-centre ones by the same amount to within half a millimetre.

## Fluid fields from the extents (2026-10-08)

The two field seeds no longer use a zoom of 10/9, which was chosen because it maps 84 mm onto
93.3 mm. On the 550 mask knees with the incumbent's joint target, and on the validation fold with
the control's weights for the share of a posed crop inside the acquired image
(figures and `inside-image.json` in `outputs/analysis/anatomy-masks-field/fluid-fields/`):

| Field | mm/px | Coronal: bone edge + 5 mm inside | Sagittal: kneecap front inside | Whole kneecap | Coronal crop inside image | Sagittal crop inside image |
|---:|---:|---:|---:|---:|---:|---:|
| 93.3 | 0.417 | 45% | 20% | 8% | 99.96% | 99.98% |
| 103.7 | 0.463 | 87% | 59% | 36% | 99.85% | 99.92% |
| 108 | 0.482 | 95% | 73% | 52% | 99.75% | — |
| 112 | 0.500 | 99% | 83% | 65% | — | 99.73% |
| 120 | 0.536 | 100% | 97% | 87% | 99.06% | 99.21% |

- **Coronal 108 mm:** the field at which the bone edge plus 5 mm fits on the tighter side for 95%
  of knees; the 5 mm for the collateral ligaments is an assumption.
- **Sagittal 112 mm:** coverage rises all the way to 120 mm, so the extents set no value; 112 mm
  is 0.5 mm per pixel, the most common acquired resolution.
- **Margins:** 24.5 and 26.5 mm keep today's 12.75 mm of shift and 4.25 mm of slack; canvases
  about 27% and 35% larger in area than today's.
- **In the figures:** at 93.3 mm the collateral ligaments and the kneecap sit on the crop's edge
  or outside it; at 108 and 112 mm they are inside. Three of six coronal examples show a narrow
  black band at one side at 108 mm, where the acquisition ends.
