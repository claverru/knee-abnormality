# Side-routed heads: manual runs after the campaign (2026-10-07)

**Neither random crop offsets in training (×4 0.925012, ×1 0.914850) nor a femur segment with its
own lean (×4 0.925202, ×1 0.916509) beat the incumbent (×4 0.925389, ×1 0.917411); neither was
adopted.** A sunflower crop pattern at prediction was no finding across fourteen variants.
Rotating the crops by half the angle of each knee's frame at prediction gained ×4 +0.0010,
+0.0007 and +0.0002 on three sets of weights, but was not submitted because the export refused
the checkpoint. Rule-based estimates of the knee's bend failed. All on fold 1, seed 1110484336,
after the side-routing campaign stopped. Design: [side-routed heads](../../../../modeling/side-routed-heads.md).

## Crop offsets without a locator (manual run)

Run `outputs/training/20261007-081431-side-routed-crop-offsets-3mm`, on the campaign's incumbent
(`height-cuts-from-the-spots-requeue`, ×4 0.925389, ×1 0.917411). The loader cut each 84 and
93 mm view with a 3 mm margin (uint8 canvases, as in the campaign's light port); the model cut the
224 px crop. Training: each slice's crop was offset at random, uniform in a disc of 3 mm, from its
own generator. Prediction: slice k of the T valid slices of a series sat at sunflower point k of
T (radius 3·sqrt((k+0.5)/T) mm, angle k·137.5°), a single slice at the anchor. The 120 mm axial
views did not move. The affine augmentation was unchanged. Build backup: scratchpad
`crop-offsets-build`.

- **×4 0.925012 (−0.0004), ×1 0.914850 (−0.0026): not adopted.** Guardrails passed, training
  1,263 s. Per label at ×4: Medial OA +0.0019, Lateral OA +0.0015, Synovitis −0.0038, MCL
  −0.0029, Baker's −0.0017, the rest within 0.0009.
- **The two halves, by predicting with the pattern and at the anchor**
  (`outputs/analysis/r/routing-and-shifts-frames-20261007/mcl-spatial-maps/crop-offsets-on-off-x{1,4}.npz`,
  `incumbent-pattern-on-off-x{1,4}.npz`):

  | Weights | Multiplier | Pattern | Anchor | Pattern's effect |
  |---|---|---:|---:|---:|
  | Trained with offsets | ×4 | 0.9250 | 0.9242 | +0.0008 |
  | Trained with offsets | ×1 | 0.9148 | 0.9154 | −0.0006 |
  | Incumbent, no retraining | ×4 | 0.9259 | 0.9253 | +0.0005 |
  | Incumbent, no retraining | ×1 | 0.9167 | 0.9174 | −0.0007 |

  Training with re-cut offsets cost 0.0012 at ×4 and 0.0020 at ×1 (anchor against the
  incumbent), in line with the learnable-views jitter runs 17 and 43 (−0.0008, −0.0009). The
  pattern at prediction helped at ×4 and hurt at ×1 on both sets of weights. On the incumbent at
  ×4: Lateral Meniscus +0.0040, Lateral OA +0.0032, MCL +0.0012, Medial OA +0.0012, Synovitis
  −0.0017, Fracture −0.0012, ACL −0.0011, PF OA −0.0010. The spot-shift locator's shift of the
  same morning had the same size and labels at ×4 (+0.0007).
- **First reading, refuted below.** A spread of crop positions over a series's slices looked like
  a prediction-time gain of 0.0005 to 0.0008 at ×4 that needs no training, concentrated on the
  lateral labels; random re-cut offsets in training do not help.
- **Pattern by plane, incumbent's weights** (`pattern-by-plane-*-x{1,4}.npz` in the same folder).
  Macro against the anchor at ×4: coronal only +0.0004 (0.925787), the other detail plane only
  +0.0002, both +0.0005. At ×1: −0.0001, −0.0004, −0.0007. Coronal only at ×4: Lateral Meniscus
  +0.0032, Lateral OA +0.0019, MCL +0.0014, ACL −0.0009, the rest within 0.0007; the Synovitis
  (−0.0012) and Fracture (−0.0018) losses come from the other plane. The two planes' effects add
  up, so restricting the pattern to coronal views keeps most of the gain at ×4 and removes the ×1
  cost, without gaining more than moving both.
- **What the pattern's gain is made of, incumbent's weights, coronal views, ×4**
  (`pattern-directions-coronal-x4.npz`, `pattern-radius-coronal-x4.npz`). Macro against the
  anchor (0.9253):

  | Variant | Macro | Variant | Macro |
  |---|---:|---|---:|
  | Sunflower 3 mm | +0.0004 | All slices 3 mm lateral | +0.0002 |
  | Sunflower 1.5 mm | +0.0002 | All slices 3 mm medial | −0.0003 |
  | Sunflower 4.5 mm | −0.0000 | All slices 3 mm up | −0.0005 |
  | Sunflower 6 mm | +0.0002 | All slices 3 mm down | −0.0003 |
  | Sunflower, across only | +0.0001 | Alternating ±3 mm across | −0.0001 |
  | Sunflower, down only | +0.0002 | Alternating ±3 mm down | −0.0002 |
  | | | Alternating ±6 mm across | −0.0001 |
  | | | Alternating ±6 mm down | −0.0000 |

  The fourteen variants average 0.0000 with a spread of about 0.00025, and the gain does not grow
  with the radius, so +0.0004 for the 3 mm sunflower is inside what any small displacement of the
  crops does. No direction carries it, a constant shift to any side does not help (no new anatomy
  is being reached), and two fixed positions per plane do not help. Per label the one repeat is
  Lateral Meniscus, above the anchor in 11 of 14 variants (mean +0.0011, largest +0.0032). Moving
  every coronal crop 3 mm medially costs Lateral OA 0.0043, the largest single effect: the
  lateral edge of the coronal crop matters to it, and moving 3 mm laterally does not add to it
  (−0.0002).
- **Lesson.** The prediction pattern is not a finding on the incumbent: the first reading was
  measured with one pattern on three sets of weights; across patterns on one set it is zero on
  average. It was kept out of the endgame list.
- **Where the pattern's effect on the lateral labels comes from** (incumbent's weights, ×4, the
  3 mm sunflower on both detail planes; `pattern-frame-swap-*.npz`,
  `pattern-content-or-position-*.npz`). Three interventions, each against the anchor:

  | Label | Pattern | Pattern crops, anchor's frame | Anchor crops, pattern's frame | Pattern pictures, anchor positions | Anchor pictures, pattern positions |
  |---|---:|---:|---:|---:|---:|
  | Lateral Meniscus | +0.0040 | +0.0038 | +0.0002 | +0.0036 | −0.0000 |
  | Lateral OA | +0.0032 | +0.0032 | −0.0005 | +0.0034 | −0.0005 |
  | MCL | +0.0012 | +0.0012 | −0.0004 | +0.0005 | +0.0001 |
  | Macro | +0.0005 | +0.0005 | −0.0001 | +0.0006 | −0.0001 |

  The effect is in the pictures the backbone sees. It is not the frame (which moves 0.5 mm and
  1.1° at the median between the two passes), and not the token positions used by the routing and
  the position code. In logits the pattern lowers scores slightly; for Lateral Meniscus the
  negatives in the top tenth of scores fall by 0.056, positives in the same score range by 0.025
  and the other negatives by 0.013, so confident false positives lose more than true positives.
  Lateral OA shows the same order with a smaller gap (0.064, 0.049). The frame-precision
  explanation proposed first is refuted by the frame swap.
- **Sensitivity to the crop position per label** (incumbent, ×4, coronal crops at the anchor and
  3 mm lateral, medial, up and down; spread of a knee's logit over the five, divided by the gap
  between the mean logit of positives and negatives; `pattern-directions-coronal-x4.npz`):
  Lateral Meniscus 0.029, MCL 0.024, Lateral OA 0.023, Medial OA 0.022, Medial Meniscus 0.022,
  then Fracture, Contusion and ACL at 0.014, Synovitis 0.013, Effusion 0.011, PF OA 0.010,
  Baker's 0.007. The five labels read on the coronal views by side are the sensitive ones, to the
  same degree within 1.3 times; the lateral ones are not a class apart. No relation to the number
  of positives (correlation −0.14), so the imbalance explanation was dropped. All four directions
  move a score by the same amount, except MCL (0.08 across against 0.05 up or down). Only the
  coronal views were moved, so the ranking also reflects how much each label reads from them.

## Crops cut by each knee's frame (prediction only, no training)

The user's idea: cut the crops following the frame the incumbent infers for each knee. Checked on
the incumbent's weights with the canvas loader (15 mm margin): a first pass at the anchor gives
the knee's frame; the 84 and 93 mm views are then re-cut and predicted again. Centring moves the
window by the frame centre's deviation from the fold mean, projected into the slice; rotation
turns the window in its plane so that the frame's side axis is level on coronal views and its up
axis is vertical on the other detail plane (clamped at 15°; 120 mm axial views untouched).
Scripts `frame_crops.py` and `frame_rotation_controls.py` in `scripts/adhoc/side_routed_session/`;
outputs in `outputs/analysis/r/routing-and-shifts-frames-20261007/frame-crops/`.

| Variant | ×4 (anchor 0.925344) | ×1 (anchor 0.917415) |
|---|---:|---:|
| Centred | −0.0004 | |
| Rotated | +0.0011 | −0.0002 |
| Centred and rotated | −0.0007 | |
| Rotated the wrong way (minus the angle) | −0.0007 | −0.0009 |
| Rotated by half the angle | +0.0010 | +0.0007 |
| Rotated, coronal views only | +0.0002 | +0.0002 |
| Rotated, other detail plane only | +0.0010 | −0.0004 |

- Rotation applied: median 3.6° on coronal (90th percentile 8.7°), 5.9° on the other plane
  (11.5°); the frame's axis is level in the re-cut crop to 0.00°. Centring moves the window 5 to
  7 mm at the median and 6% of slices reach the margin.
- The sign of the rotation matters: the frame's angle against its opposite is +0.0018 at ×4 and
  +0.0007 at ×1, with the same resampling in both, so this is alignment and not the picture
  disturbance measured above (spread 0.00025 over fourteen offset variants).
- Per label at ×4, full rotation: Fracture +0.0041, Lateral OA +0.0033, Lateral Meniscus
  +0.0031, MCL +0.0027, Contusion +0.0018, ACL and Baker's −0.0008. The wrong way costs ACL
  0.0048 and Fracture 0.0024.
- Half the angle is as good at ×4 and better at ×1, where it is the only variant above the anchor
  at both multipliers. The gain comes from the plane that shows the forward tilt, whose angle is
  the one still learned from the labels.
- Centring costs Fracture 0.0088 and PF OA 0.0037 and gains Medial Meniscus, Contusion, Effusion
  and ACL by about 0.002 each; it overrides the per-view offsets of the view table.
- All of this is with weights trained on unaligned crops (and ±10° of rotation augmentation).
- **The rotation on the two reference models** (worktree build: run 24's code with the canvas
  loader at a 6 mm margin and the rotation inside the model as a first look and a re-cut;
  `rotation-check-*-x4.npz`), ×4 against the same model at the anchor:

  | Weights | Anchor | Half angle | Full angle |
  |---|---:|---:|---:|
  | ResNet-18 run 24 | 0.925091 | +0.0007 | +0.0005 |
  | ResNet-50 on run 24's recipe (S119) | 0.933158 | +0.0002 | +0.0001 |

  Three sets of weights gain at ×4 with half the angle (+0.0010, +0.0007, +0.0002), the ResNet-50
  least. On all three Lateral Meniscus and Lateral OA gain 0.002 to 0.004; on the ResNet-50 ACL,
  Medial OA and PF OA each lose 0.0011. The anchor passes reproduce the recorded scores to
  0.00005.
- **Not submitted.** The export refused the ResNet-50 checkpoint with the new loader and model
  code: its `preprocessing_identity` differed from the executing sources, and the check says not
  to rewrite checkpoint identities. A model trained with this build would export normally.

## The knee's bend between femur and tibia (measurement attempt)

Asked while discussing a frame of three attached segments instead of one cylinder. Two
rule-based estimates of each bone's axis on the 874 validation knees, from the series at the full
120 mm field (`measure_bend.py`, `measure_bend_marrow.py` in `scripts/adhoc/side_routed_session/`;
outputs in `outputs/analysis/r/routing-and-shifts-frames-20261007/knee-bend/`):

- **Centre of the limb's outline per height.** Below the joint the centre drifts backwards by
  14° on average (SD 9.5°): that is the calf, not the tibia. Above the joint it is straight up to
  +50 mm and then bent by the end of the field. Not a bone axis.
- **Bright pixels well inside the limb on the T1/PD series.** They follow Hoffa's fat pad in
  front of the joint (+14 mm at the joint line) and the popliteal fat behind the femur (−15 mm
  above +40 mm); slopes have an SD of 21 to 35°. Not a bone axis either.
- Neither correlates with the frame's forward tilt (|r| ≤ 0.12). No number for the bend came out
  of this.
- **By eye**, on eighteen sagittal slices through a condyle (`sagittal-gallery.png`): the tibia
  stands close to vertical in nearly all of them, and the femur leans by anything from about 0°
  to 30° or more (rows 532 and 200 clearly bent; 370, 374 and 284 nearly straight). The 120 mm
  field shows about 60 mm of each bone, mostly condyle and plateau and little shaft.

## Femur segment with its own lean (manual run on the incumbent)

Run `outputs/training/20261007-101112-side-routed-femur-lean`. The frame above the upper cut was
laid out around an axis that leans forward or back from the centre of that cut:
`front' = front − max(height − upper cut, 0)·tan(lean)`. Lean per knee = `35°·tanh((start +
correction)/35°)`; start = atan of half the kneecap spot's forward offset from a shared learned
distance (1.57 token pitches at the start, the measured mean) over its rise above the cut, no
gradient through the spot; correction regressed per knee from the pooled frame tokens, zero at
the start, label gradient. Build backup: scratchpad `femur-lean-build`.

- Before training, on the incumbent's weights: start lean mean −0.3°, SD 9.8°; five knees bent by
  eye +2.5° on average, five straight ones −9.6°.
- **×4 0.925202 (−0.0002), ×1 0.916509 (−0.0009): not adopted.** Guardrails passed, 1,213 s.
- Per label at ×4: Effusion +0.0021, Synovitis +0.0021, Medial OA +0.0010, MCL −0.0024, Fracture
  −0.0021, Lateral Meniscus −0.0018, Contusion −0.0013, PF OA −0.0005, ACL −0.0002.
- **The lean went to one side for every knee.** Validation mean −2° after epoch 1, −30° after
  epoch 2, −25° at the end, with the mean size equal to the mean, so the start's spread between
  knees was overridden by the regressed correction; the shared distance stayed at 1.60. The whole
  frame's forward tilt ended at 15° against 6° for the incumbent, the core at 17.4 mm (16.3) and
  the ring at 28.2 mm (29.8).
- Drawing of the frame per label at epoch 7:
  `outputs/analysis/r/routing-and-shifts-frames-20261007/femur-lean-per-label-3d.png`.
- **Lesson.** A lean learned from the labels with a start from the spots does not keep the start:
  the labels reading the upper band (PF OA, Effusion, ACL and the top of MCL and Baker's) pull it
  one way and nothing pulls back. Effusion gains with it and PF OA does not.
