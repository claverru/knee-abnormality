# Studies 6 to 10: token grid, 225 and 257 px, canvas sizes

## 6. Token grid centred at prediction only

The ResNet's token centres are at pixels 0, 32, ..., 192 (confirmed from the incumbent's mapping:
origin 0, stride 32), so the grid is centred 15.5 px up and left of the crop's centre. Test: the
incumbent (`report-attribute-loss-off-2`), same weights, with the second-look crop window moved
16 px so the grid is centred on the knee; 874 validation studies. Predictions in
`outputs/analysis/centred-token-grid/`.

| Variant | ×4 | Δ | ×1 | Δ |
|---|---:|---:|---:|---:|
| Baseline (re-predicted) | 0.927185 | – | 0.917220 | – |
| Grid centred, sagittal and coronal | 0.926784 | −0.0004 | 0.915371 | −0.0018 |
| Grid centred, all views (axial with a 16 px empty band) | 0.925579 | −0.0016 | 0.914061 | −0.0032 |
| Moved 16 px the other way, sagittal and coronal | 0.926401 | −0.0008 | 0.917646 | +0.0004 |

- The re-predicted baseline is 0.0001 below the run's recorded scores (0.927291, 0.917342);
  predictions differ from the saved ones by 0.0007 on average, so 0.0001 is the floor here.
- **No gain from centring at prediction, and it says little:** the weights were trained with the
  off-centre grid, and a 16 px move in either direction costs about the same at ×4. Per label at
  ×4, centred: ACL +0.0031, MCL +0.0028, Medial OA +0.0021; PF OA −0.0032, Fracture −0.0040.
- The test of the fix is a training run with the grid centred. The left-right split was not
  computed.

## 7. 225 px (8×8 tokens, symmetric grid): two runs and an audit

Adopted by the user without a run after study 6, then trained. Same fold and seed, report
loss 0.

| Run | ×4 | ×1 (best epoch) | Training |
|---|---:|---:|---:|
| 224 px, `report-attribute-loss-off-2` | 0.927291 | 0.917342 | 1,352 s |
| 224 px, campaign control (report loss 0.1) | 0.926762 | 0.917074 | 1,391 s |
| 225 px, fields kept (`rn18-grid-225-report-loss-off`) | 0.924867 | 0.914928 | 1,401 s |
| 225 px, spacings kept (`rn18-grid-225-fixed-spacing-report-loss-off`) | see run | tracks the first 225 run within 0.0013 per epoch | – |

- 225 px against the 224 px report-loss-0 run at ×4: −0.0024 (paired bootstrap 95% −0.0046 to
  −0.0004). Per label: MCL −0.0122, Fracture −0.0072, Medial OA −0.0051 (the only one whose
  interval excludes zero), Baker's −0.0039, Synovitis −0.0036; ACL +0.0023, menisci +0.002.
- Speed: +0.7% training time, 53.9 s against 50.8 s at ×4, 4.2 GB. The 31% extra tokens are
  nearly free.
- Spacing is not the cause: the fixed-spacing run reproduces the first 225 run's curve.

**Audit (`outputs/analysis/audit-225/`): no implementation defect; not a fair test of symmetry.**

1. **The runs at the two sizes do not share augmentation draws.** The image-shaped noise draw
   advances the GPU random generator by an amount that depends on the image size, so every later
   draw differs. The 0.0005 rerun spread is between runs with identical draws and does not apply
   across sizes; the last recorded full-seed spread for a ResNet-18 recipe was 0.002 to 0.003.
   Fix: draw augmentation parameters from a generator separate from the image-shaped noise.
2. **Border tokens are valid about half the time in training and always at evaluation.** A token
   whose centre is on pixel 0 or on the last pixel fails the inside-the-image test under any
   rotation or shift. Valid in training: at 224 px the first row and column 60%, the last 94%; at
   225 px 56% and 54%. At evaluation all tokens are valid at both sizes. So the train/eval
   mismatch exists today for the first row and column, and 225 px extends it to the whole ring.
   Dropping the ring at prediction on the 225 checkpoint changes ×1 by +0.0003, so the ring does
   not hurt at inference; any loss is in the trained weights.
3. Side head slightly less reliable at evaluation at 225 px (validation loss 0.012 against
   0.0055); affects the 6% of studies with no metadata side.
4. `scripts/tools/check_side_routing.py:245` takes the pitch as field ÷ tokens, right only at
   7×7. Tool only.

Checked and correct at 225 px: backbone mapping and padding symmetry (flip-equivariance exact at
225, broken at 224), token pitch from the stride, crops (no half-pixel offset between turned and
unturned), augmentation inverse, anatomy targets, pooling, loader, run parity.

**With this backbone a symmetric grid always puts the first and last tokens on the border
pixels** (sizes 32k + 1). To test symmetry alone: 225 px with the ring always masked (inner 6×6),
or 8×8 with a half-pitch tolerance in the validity test so training matches evaluation; either
against a 224 px run on the same augmentation stream.

## 8. 257 px with the border tokens covered (inner 7×7, centred)

Rule added to the model for this run: a token whose centre is on the frame's
border pixel never counts. At 257 px with the spacings kept (fields 96.375, 107.08 and
137.68 mm) the backbone gives 9×9 tokens; covering the ring leaves 7×7 at pixels 32 to 224,
centred on the image centre (128), all fully inside the image, reading the same area as today's
crop with 16 px of real image around it. At 224 px the same rule would give 6×6.
`outputs/training/rn18-257-inner-7x7-report-loss-off`; same fold and seed, report loss 0.

| Run | ×4 | ×1 (best epoch) | Training | ×4 prediction |
|---|---:|---:|---:|---:|
| 257 px, inner 7×7 | 0.926763 | 0.917439 (epoch 10) | 1,615 s | 60.6 s |
| 224 px, `report-attribute-loss-off-2` | 0.927291 | 0.917342 | 1,352 s | 50.8 s |
| 224 px, campaign control | 0.926762 | 0.917074 | 1,391 s | 51.8 s |
| 225 px, 8×8 | 0.924867 | 0.914928 | 1,401 s | 53.9 s |

- **Against the like-for-like 224 px run:** ×4 −0.0005, ×1 +0.0001; training +19%, prediction
  +19%, 5.3 GB of GPU memory. Level, at a fifth more time.
- **Against the 225 px run:** ×4 +0.0019. Covering the border tokens removed what hurt there.
- **The curve is smooth** and tracks the 224 px references epoch by epoch (within 0.001 from
  epoch 2 on), with none of the 225 px runs' swings (dip at epoch 3, early peak at epoch 5).
  Figure: `outputs/analysis/centred-token-grid/validation-curves.png`.
- **Per label at ×4 against the like-for-like run:** up ACL +0.0074, Synovitis +0.0062, Medial
  Meniscus +0.0030, Lateral Meniscus +0.0026, PF OA +0.0022, Contusion +0.0017; down Fracture
  −0.0117, MCL −0.0061, Baker's −0.0053, Lateral OA −0.0037, Effusion −0.0018. The macro is
  level because the two smallest labels (Fracture 60 positives, MCL 47) fell while the fine-detail
  labels rose.
- **Rerun wobble seen here:** two launches of this run (one interrupted by the restart) differed
  by 0.0008 at epoch 1.
- **What remains at 257 px:** corner tokens can still fall off the image under strong
  augmentation (a 10° turn moves the corners about 30 px); nothing else known.
- **Memory:** available system memory fell to 7 to 12 GB during the run, with 0.5 GB of swap;
  tighter than the 224 px runs.


## 9. 257 px with all 9×9 tokens (border rule removed)

`outputs/training/rn18-257-full-9x9-report-loss-off`: the same 257 px setup as study 8 without
the border-token rule, so the ring of half-padded tokens counts.

| Run | Grid centred | Border tokens | ×4 | ×1 | Training |
|---|---|---|---:|---:|---:|
| 224 px, `report-attribute-loss-off-2` | no | first row and column | 0.927291 | 0.917342 | 1,352 s |
| 257 px, inner 7×7 | yes | none | 0.926763 | 0.917439 | 1,615 s |
| 257 px, full 9×9 | yes | whole ring | 0.926908 | 0.917344 | 1,651 s |
| 225 px, 8×8 | yes | whole ring | 0.924867 | 0.914928 | 1,401 s |

- **Neither centring the grid nor the border tokens changes the score.** The three 224 and
  257 px runs are within 0.0005 at ×4 and 0.0001 at ×1. The 9×9 run's curve is as smooth as the
  inner 7×7 run's.
- **This refutes the explanation given for 225 px in study 7** (border tokens valid half the
  time in training). What remains is the augmentation stream, which depends on the image size:
  both 225 px runs shared one stream, both 257 px runs another. Untested; a 225 px run with
  another seed would settle it.
- **Both 257 px runs move the same labels against 224 px at ×4** (they share a stream, so this
  is one observation, not two): up ACL +0.0074 / +0.0079, Synovitis +0.0062 / +0.0049, Medial
  Meniscus +0.0030 / +0.0038, PF OA +0.0022 / +0.0050, Contusion +0.0017 / +0.0027; down
  Fracture −0.0117 / −0.0069, MCL −0.0061 / −0.0073, Baker's −0.0053 / −0.0061, Lateral OA
  −0.0037 / −0.0033. The fields are 15% wider at 257 px, yet the edge labels (MCL, Baker's) fell.
- Cost of 257 px: +19 to 22% training time, +19 to 24% prediction time, 5.3 GB.

## 10. Canvas sizes that hold the pose (measured from the learned frame)

Asked by the user: canvases per plane-contrast, placed by rule in the loader, big enough for the
crop's shift, turn and scale and no bigger. From the incumbent's first-look frame on 3,517
training studies (94,250 images, `outputs/analysis/view-attention/images.parquet`); table in
`outputs/analysis/view-attention/canvas-sizes.csv`. For each image: the window the pose asks
for (shift before today's clamp, turn, scale = knee size clamped to 0.87 to 1.13, about the
joint target), plus the training jitter (3 mm, 3°); the canvas is what holds its four corners.
Axial is given shift and scale but no turn.

**What the pose asks for today.**

| View | Shift needed, median / 95% / 99% | Spread across / up-down (sd) | Turn, 95% / 99% | Clamped today |
|---|---|---|---|---:|
| Coronal T1/PD 84 mm | 6.0 / 13.9 / 16.3 mm | 3.8 / 6.9 mm | 6.6° / 8.9° | 11.2% |
| Coronal fluid 93 mm | 6.3 / 14.9 / 18.0 mm | 3.9 / 7.9 mm | 6.8° / 9.1° | 15.1% |
| Sagittal T1/PD 84 mm | 6.4 / 14.2 / 16.7 mm | 4.3 / 7.8 mm | 7.4° / 9.8° | 15.1% |
| Sagittal fluid 93 mm | 6.3 / 14.5 / 17.3 mm | 4.3 / 7.7 mm | 7.5° / 10.0° | 13.9% |
| Axial fluid 120 mm | 2.4 / 5.8 / 7.9 mm | 2.1 / 2.8 mm | not turned | – |
| Axial T1/PD 120 mm | 2.3 / 7.0 / 32.2 mm | 3.0 / 5.3 mm | not turned | – |

- The clamp rates reproduce the attention study's 11 to 15%, which checks the computation.
- **The canvases are not off-centre:** the median offset is about 1 mm in every view, so moving
  the loader's placement gains nothing.
- **The spread is up and down:** 7 to 8 mm against 4 mm across. The canvas is anchored on the
  knee across (from the series' intersection) and on the image up and down; the loader's own
  knee height is no better (the model moves its centre 8.0 mm sd up and down from it).
- Today's canvas holds the asked window for 69 to 77% of sagittal and coronal images.

**Canvases that hold 99% of the asked windows** (98% on both axes together):

| View | Today | Square | Pixels against today | Rectangle, width × height | Pixels against today |
|---|---|---|---:|---|---:|
| Coronal T1/PD 84 mm | 314 px (118 mm) | 370 px (138 mm) | 1.39 | 328 × 368 px (122 × 138 mm) | 1.22 |
| Coronal fluid 93 mm | 306 px (127 mm) | 364 px (151 mm) | 1.42 | 324 × 364 px (135 × 151 mm) | 1.26 |
| Sagittal T1/PD 84 mm | 314 px (118 mm) | 370 px (138 mm) | 1.39 | 336 × 370 px (126 × 138 mm) | 1.26 |
| Sagittal fluid 93 mm | 306 px (127 mm) | 364 px (152 mm) | 1.42 | 330 × 364 px (137 × 152 mm) | 1.28 |
| Axial fluid 120 mm | 224 px (120 mm) | 282 px (150 mm) | 1.58 | 278 × 278 px (148 × 149 mm) | 1.54 |
| Axial T1/PD 120 mm | 224 px (120 mm) | 386 px (206 mm) | 2.97 | 298 × 386 px | 2.29 |

Axial T1/PD has a tail of odd series (99% at 32 mm of shift); sizing it like axial fluid
(282 px) holds 95% of its images. Turning the axial crop as well (95% of knees within −13° to
+20°) would take its canvas to about 194 mm.

**Limits.** Sizes follow today's fields, targets and scale clamp, and one model's frame; a view
whose field or framed structures change needs the computation redone. The loader takes one
margin per view today, so rectangular canvases need a code change.
