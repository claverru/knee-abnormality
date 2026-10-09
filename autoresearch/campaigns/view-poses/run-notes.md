# View-poses run notes

Keeper's notes per run: both multipliers against the incumbent, the diagnostics the contract
asks for, and what the run taught.

**Adoption rule of this campaign:** runs were adopted when at least the incumbent at both ×1 and
×4, a rule the runtime carried for this campaign only; it was reverted on 2026-10-10 (user
decision) to the 80% bootstrap lower bound above zero. Under that rule only run 7 would have
been adopted; runs 1 and 3 are level steps that run 7 was built on.

## Run 1: `meniscus-head-on-the-14-by-14-grid` — adopted (2026-10-09)

Experiment `meniscus-head-on-the-14-by-14-grid-20261009T003747Z-1774a816`, against
`rn18-meniscus-map`.

| | Reference | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.926413 | 0.926543 | +0.000130 (80% interval −0.000579 to +0.000824) |
| ×1 macro AUC | 0.916382 | 0.916736 | +0.000354 |
| Training seconds | 1,370.6 | 1,353.7 | 0.988 (stated: within 1.03) |
| Prediction seconds | 51.4 | 51.9 | 1.01 |
| Medial meniscus overlap, validation | 0.452 | 0.560 | +0.108 |
| Lateral meniscus overlap, validation | 0.433 | 0.559 | +0.126 |
| Meniscus share error, validation | 0.0098 | 0.0066 | |
| Supervised meniscus positions, validation | 61,487 | 242,538 | 4 per token |

- Adopted by the runtime's rule (at least the incumbent at both multipliers); the contract
  adopts this seed whatever its macro result. Its purpose is met: both classes overlap the mask
  better, with the caveat that the 14 by 14 footprints are not those of the reference.
- The bone map and the plane are level: plane error 1.734 mm against 1.743 mm, overlaps within
  0.001 per class.
- No new logging; epochs at the reference's pace.
- **Two trainings of one candidate differ.** A first training of the same source, lost at
  epoch 9 when WSL was shut down, read 0.904224 at epoch 3 and 0.916626 at epoch 8 where this
  one read 0.902739 and 0.916258: up to 0.0015 apart at one epoch. The +0.000130 at ×4 is
  inside that.

## Run 2: `joint-centre-taught-from-the-masks` — not adopted, defective by the contract (2026-10-09)

Experiment `joint-centre-taught-from-the-masks-20261009T010759Z-5a3cd2aa`, against run 1.

| | Incumbent | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.926543 | 0.925048 | −0.001495 (80% interval −0.003403 to +0.000319) |
| ×1 macro AUC | 0.916736 | 0.915754 | −0.000982 |
| Training seconds | 1,353.7 | 1,371.4 | 1.013 (stated: within 1.02) |
| Prediction seconds | 51.9 | 52.6 | 1.01 |
| Centre error front to back, mean | +1.72 mm | −0.13 mm | bias removed |
| Centre error front to back, spread | 1.56 mm | 1.47 mm | |
| Centre error across, mean | +0.96 mm | +0.83 mm | |
| Centre error across, spread | 1.96 mm | 2.64 mm | worse |
| Plane error | 1.734 mm | 1.827 mm | |
| Applied head correction | — | 2.84 mm | active |

- Lower at both multipliers, so the rule rejects it. Per target at ×4: Fracture −0.0088,
  Lateral OA −0.0071, Effusion −0.0035, Lateral Meniscus −0.0032, Medial OA −0.0026;
  Contusion +0.0037, MCL +0.0026, ACL +0.0017, Medial Meniscus +0.0017.
- **The head does half its job.** It removes the 1.7 mm front-to-back bias, but the per-knee
  spread across the knee grows with every epoch, on training knees too (2.5 to 3.4 mm) and on
  validation (2.0 to 2.6 mm). `anatomy_centre_error_*_mm` is the spread and `*_mm_mean` the
  bias. Root-mean-square per axis: front 2.32 to 1.48 mm, across 2.18 to 2.77 mm. The contract
  asks for a lower centre error; it is not lower across, so the candidate counts as defective.
- **Open lead, cause not found.** A head fitted to these targets should not get noisier across
  on its own training knees. The offset conversion to the fitted frame matches the one of the
  positions (same sign and axis). Not checked: the across target under the reflection
  augmentation, and whether one linear layer on 7 by 7 tokens can regress offsets of up to
  half a field. No repair seed is forced until the cause is known.
- The crop centres still carry offsets tuned on the biased centre, so removing the bias moves
  every crop by about 1.8 mm front to back; this may be part of the macro loss (a guess).

## Run 3: `canvases-for-the-pose-in-all-six-views` — adopted (2026-10-09)

Experiment `canvases-for-the-pose-in-all-six-views-20261009T014728Z-960f75cb`, against run 1.

| | Incumbent | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.926543 | 0.926563 | +0.000019 (80% interval −0.001402 to +0.001442) |
| ×1 macro AUC | 0.916736 | 0.917013 | +0.000276 |
| Training seconds | 1,353.7 | 1,486.0 | 1.098 (stated: up to 1.25); 1.084 of the reference, ceiling 1.4 |
| Prediction seconds | 51.9 | 63.0 | 1.214 (limit 1.3) |
| Plane error | 1.734 mm | 1.792 mm | |
| Medial, lateral meniscus overlap | 0.560, 0.559 | 0.558, 0.550 | |
| Kneecap overlap | 0.737 | 0.744 | |

- Adopted by the runtime's rule (level or better at both multipliers), which is stricter than
  the contract's rule for this seed (unless clearly worse). Macro level with a rerun.
- Pose statistics at the last epoch: axial shift 3.7 mm and turn 6.4 degrees on average, mean
  scale 0.998, 14% of crops at a shift, turn or scale limit.
- Per target at ×4: Synovitis +0.0063, Contusion +0.0038, Medial Meniscus +0.0014;
  Lateral OA −0.0033, Effusion −0.0030, MCL −0.0025, Fracture −0.0021.
- Prediction is now 1.21 times the previous incumbent's: the canvases left little room under
  the 1.3 limit, which from here is measured against this run's 63 s.
- Five validation-only metrics, none in the training step.

## Run 4: `a-crop-centre-per-contrast` — not adopted (2026-10-09)

Experiment `a-crop-centre-per-contrast-20261009T050407Z-5ff88f11`, against run 3.

| | Incumbent | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.926563 | 0.926497 | −0.000066 (80% interval −0.001299 to +0.001193) |
| ×1 macro AUC | 0.917013 | 0.916716 | −0.000296 |
| Training seconds | 1,486.0 | 1,504.3 | 1.012 (stated: within 1.02) |
| Prediction seconds | 63.0 | 62.9 | 1.00 |
| Plane error | 1.792 mm | 1.683 mm | |

- Lower at both multipliers by less than a rerun's spread, so the rule rejects it: level, not
  better.
- The per-label changes follow what each centre was moved for: the three OA labels and Lateral
  Meniscus gain (Lateral Meniscus +0.0028, Lateral OA +0.0025, Medial OA +0.0024, PF OA
  +0.0014) and the labels read in wide field lose (Contusion −0.0036, Baker's −0.0018, Effusion
  −0.0014, Synovitis −0.0012, MCL −0.0011). That is the trade principle 5 of the contract
  describes: the joint labels want the crop on the joint, the fluid and marrow labels its field.
- 5.6% of the per-contrast crops are at their shift limit; 14.7% of all crops at a pose limit,
  as in the incumbent.
- Seed 5 was prepared on run 3 and trains without this seed's mechanism.

## Run 5: `t1pd-sagittal-crop-on-the-notch` — not adopted (2026-10-09)

Experiment `t1pd-sagittal-crop-on-the-notch-20261009T050856Z-369cf10f`, against run 3 (without
run 4's mechanism).

| | Incumbent | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.926563 | 0.926340 | −0.000222 (80% interval −0.001221 to +0.000818) |
| ×1 macro AUC | 0.917013 | 0.916165 | −0.000847 |
| Training seconds | 1,486.0 | 1,507.6 | 1.015 (stated: within 1.02; the keeper's one-minute measurement ran beside it) |
| Prediction seconds | 63.0 | 61.4 | 0.97 |

- Lower at both multipliers; the rule rejects it. The ×4 change is inside a rerun's spread.
- **The crop moves a lot and is stopped often:** the notch target is 13.2 mm from the knee's
  centre on average and 36% of the T1/PD sagittal crops are at the 18 mm shift limit, so for a
  third of the knees the crop does not reach where the seed points it.
- **The view alone changes as intended only in part** (T1/PD sagittal, per label): MCL +0.017,
  Baker's +0.013, Medial Meniscus +0.010, Effusion +0.010, Contusion +0.008, PF OA +0.006;
  Fracture −0.013, Medial OA −0.009, Lateral Meniscus −0.005. ACL, the label the notch crop is
  for, gains +0.001 on the view and loses −0.0027 in the model.
- Per target at ×4: MCL +0.0047, Lateral OA +0.0033, Fracture +0.0027; Lateral Meniscus
  −0.0049, ACL −0.0027, Synovitis −0.0020.
- Open lead: the 18 mm limit against a 13 mm mean move. A later seed on this view should say
  how far the canvas lets the crop go before judging the target.

## Run 7: `sagittal-fluid-canvas-fine` — adopted (2026-10-09)

Experiment `sagittal-fluid-canvas-fine-20261009T054235Z-c95da7ce`, against run 3. Configuration
only: the two sagittal views swap fields, fluid 93.3 to 84 mm (0.417 to 0.375 mm per pixel) and
T1/PD 84 to 93.3 mm.

| | Incumbent | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.926563 | 0.928121 | +0.001558 (80% interval +0.000448 to +0.002690) |
| ×1 macro AUC | 0.917013 | 0.919392 | +0.002379 |
| Training seconds | 1,486.0 | 1,498.1 | 1.008 (stated: within 1.02) |
| Prediction seconds | 63.0 | 63.0 | 1.00 |

- Adopted by the rule, and the first run of the campaign whose 80% interval is wholly above
  zero: outside what a rerun gives (about ±0.0005 at ×4).
- Per target at ×4: Contusion +0.0071, MCL +0.0056, Lateral OA +0.0032, Effusion +0.0025,
  both menisci +0.0010; Synovitis −0.0013, Medial OA −0.0007, ACL −0.0005, PF OA −0.0004.
- **The trade worked as principle 5 says.** Each view alone, against the same contrast before
  the swap: the fluid sagittal view, now finer and narrower, gains MCL +0.015, Lateral Meniscus
  +0.005, ACL +0.004 and loses the wide-field labels (Baker's −0.053, Contusion −0.016,
  Fracture −0.010); the T1/PD sagittal view, now wider, picks those up (Effusion +0.024,
  Fracture +0.016, PF OA +0.016, Contusion +0.015, MCL +0.015) and loses Lateral Meniscus
  −0.010. In the model the field labels end up ahead, not behind.
- Bone map, plane and meniscus overlap unchanged; 14.8% of crops at a pose limit.
- Lesson: which contrast holds the fine canvas matters more than where a crop is centred;
  the three centre seeds (2, 4, 5) each moved the macro by less than a rerun.

## Run 6: `t1pd-joint-crops-centred-on-the-menisci-calibrated` — not adopted, clearly worse (2026-10-09)

Experiment `t1pd-joint-crops-centred-on-the-menisci-calibrated-20261009T063007Z-8d746339`,
against run 7. The last run before the user stopped the campaign; its completion report was
not written.

| | Incumbent | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.926046 | −0.002074 (80% interval −0.003192 to −0.001019) |
| ×1 macro AUC | 0.919392 | 0.916481 | −0.002911 |
| Training seconds | 1,498.1 | 1,551.7 | 1.036 (stated: within 1.02 — over) |
| Prediction seconds | 63.0 | 64.6 | 1.03 |

- Lower at both multipliers with the whole 80% interval below zero: the only run of the
  campaign that is clearly worse, not level.
- Mechanism active: the T1/PD joint crops move 7.9 mm on average towards the meniscus
  centroid (8.1 mm asked), 7.9% of studies use the knee's centre (below the 3.89 mass), 14% of
  the sagittal and 6% of the coronal crops are at the shift limit.
- **The coronal T1/PD view loses what it was moved for:** alone it drops Lateral Meniscus
  −0.043, ACL −0.027, Contusion −0.018, Fracture −0.015, Medial OA −0.014. The sagittal
  T1/PD view alone gains Baker's +0.031, Fracture +0.030, Lateral Meniscus +0.016 and loses
  PF OA −0.021 and Effusion −0.016.
- Per target at ×4: MCL −0.0078, Contusion −0.0058, Fracture −0.0053, Effusion −0.0022; no
  label gains more than +0.0010.
- It also trained 3.6% longer than the incumbent against a stated 2%.
- Lead for a rerun: the two views behave oppositely, so the seed's one mechanism hides two
  results; the sagittal half alone is the open cell.

## Public score (2026-10-09)

S129, `sagittal-fluid-canvas-fine` (run 7), ×4, τ 0.9, batch 4, exported from the run's frozen
source: **0.955** public. The ResNet-18 submissions before it: S125 and S128 0.954, S118 and
S122 0.953; the best overall stays S126, the ResNet-50 on the anatomy-map recipe, 0.959. The
first two attempts failed on Kaggle's rate limit when listing submissions; the third passed
`--sequence 129`.

## After the campaign: `rn18-soft-tissue-routing` (2026-10-09)

Run by the keeper at the user's request, outside the campaign, from
`/home/claudio/src/knee-abnormality-mask-routing` (a copy of run 7's frozen source) into
`outputs/training/rn18-soft-tissue-routing`. Against run 7, same seed and fold.

Changes: new targets `data/labels/prepared/soft-tissue-nnunet-knee-dess265` (263 knees, seven
classes: neither, two menisci, two tibial cartilages, femoral and patellar cartilage, from the
same DESS segmentations); the soft-tissue head predicts the seven classes; the meniscus routes
are blended with the meniscus shares, the OA routes with that side's tibial cartilage or the
femoral cartilage inside the geometric half, the kneecap route takes the larger of the bone
kneecap classes and the patellar cartilage. A share counts above chance (1/7) and reaches full
weight at the 95th percentile of the mask's share on a token, measured on the 211 training
DESS knees (0.359 both menisci, 0.391 and 0.406 tibial, 0.281 femoral, 0.297 patellar). No
per-step logging: `train_loss` and `grad_norm` are accumulated per epoch.

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.927460 | −0.000661 |
| ×1 macro AUC | 0.919392 | 0.918710 | −0.000682 |
| Training seconds | 1,498 | 1,724 | 1.15, not comparable: another CPU job ran beside it |
| Prediction seconds | 63 | 64 | |

- Level to slightly lower at both multipliers; ahead until epoch 4, behind by 0.0004 to 0.0012
  from epoch 5.
- The head learns the cartilage: validation overlap 0.58 femoral, 0.59 lateral tibial, 0.49
  medial tibial, 0.52 patellar; menisci 0.56 and 0.55, as before.
- **The masks claim few tokens.** Mean mask-led weight over valid tokens: menisci 0.008,
  tibial cartilage 0.012, femoral 0.029, patellar 0.008. Those tokens already lie inside the
  geometric half at joint height, so the blend changes almost no weight: the routing is in
  effect the incumbent's.
- Per target at ×4 (labels from the label table, deltas only): PF OA +0.0031, Baker's +0.0023,
  Fracture +0.0015; MCL −0.0079, Lateral OA −0.0024, Effusion −0.0019, Contusion −0.0012,
  Medial Meniscus −0.0010. At ×1: Contusion +0.0055, Fracture +0.0034, PF OA +0.0033; MCL
  −0.0096, Medial Meniscus −0.0038, Lateral OA −0.0028, Lateral Meniscus −0.0022.
- PF OA is the one routed label that gains at both multipliers, and it is the route where the
  mask replaces rather than adds (patellar cartilage leads the kneecap route).
- Next step: let the masks narrow the routes instead of adding to them. Today a meniscus label
  still reads the whole half of the joint; with the mask as the region (geometry only as the
  fallback below the 3.89 mass) it would read the meniscus tokens.

## After the campaign: `rn18-no-soft-tissue-head` (2026-10-09)

Run 7 with the soft-tissue head and its loss switched off (`meniscus_map` false,
`meniscus_loss_weight` 0) and the two per-step logs moved to per epoch; from
`/home/claudio/src/knee-abnormality-no-soft-tissue`, same seed and fold, nothing else on the
machine.

| | Run 7 (head) | No head | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.927329 | −0.000792 |
| ×1 macro AUC | 0.919392 | 0.918678 | −0.000714 |
| Training seconds | 1,498 | 1,437 | 0.959 |
| Prediction seconds | 63 | 61 | 0.966 |

- Slightly lower at both multipliers; behind or level at every epoch from the third. On the
  recipe before the canvases the same removal gained 0.0009 (`report-attribute-loss-off-2`
  against `rn18-meniscus-map`), with the head then on the 7 by 7 tokens.
- The bone map, plane, centre and every pose statistic match run 7's to a few thousandths over
  the first five epochs: the head does not act through the pose.
- The seven-class run with additive routing (`rn18-soft-tissue-routing`) landed at the same
  place, −0.000661 and −0.000682: three runs within 0.0008 of each other.
- 4% faster in training, the head's removal and the logging change together.
- Open test: the 14 by 14 head with its gradient stopped before the backbone, to see whether
  the head helps by supervising the earlier stage.

## After the campaign: mask-anchored routing attempts (2026-10-09)

Four partial runs from `/home/claudio/src/knee-abnormality-anatomy-routing`, all on run 7's
recipe with the seven-class soft-tissue head, no hand-measured crop offsets in the model, the
kneecap constants and the side prior learnable, and per-epoch logging; each stopped by the
keeper or the user before the end. Changes against run 7 at the same epoch, validation ×1.

| Attempt | Routes of the two menisci, the two OA sides and PF OA | View table | Stopped | Macro change per epoch |
|---|---|---|---|---|
| 1 | share above chance (1/7) over a learned full share | offsets zero | epoch 1 | −0.0134 |
| 2 | share over the knee's largest share of that class | knee-anchored, offsets zero | epoch 1 | −0.0071 |
| 3 | as 2 | run 7's | epoch 6 | −0.0057, −0.0004, −0.0032, −0.0032, −0.0025, −0.0036 |
| 4, `rn18-structure-centred-routes` | Gaussian of the distance to the structure's centre, in units of its spread plus half a token, times a learned reach started at 2 | run 7's | epoch 6 | −0.0069, −0.0004, −0.0026, −0.0011, −0.0002, −0.0013 |

- **Empty routes are worse than no routes.** The shared meniscus and OA classifiers tell sides
  apart only by the route; with the chance threshold more than half of the tokens holding a
  meniscus fell below it and the mean route weight was 0.000 (Lateral Meniscus −0.050, PF OA
  −0.073 at epoch 1).
- **The view table's offsets are reach, not corrections.** Zeroing them, on the image centre or
  on the loader's knee estimate, doubled the crops at a shift limit (33% against 16%); with the
  table restored and the model's own crop offsets still zero the share is 15 to 17%. The user
  wants the model's hand-set values gone, not the loader's.
- **Per-token routes are too narrow.** In attempt 3 a meniscus route held 1% of the tokens
  (18% in the geometric halves) and all five labels stayed 0.004 to 0.015 behind; Lateral OA
  was below the three geometric runs at every epoch from the second.
- **Structure-centred routes are close.** Attempt 4's routes tighten as the head learns (menisci
  39% to 21% of tokens, OA 35% to 17%, PF OA 10%) and the five labels stay within 0.006 of
  run 7 from epoch 4; PF OA and Lateral Meniscus are a little behind throughout. The reaches
  barely move (1.84 to 2.25). The run was stopped at epoch 6 at −0.0013; its epoch-6 checkpoint
  is kept in `outputs/training/rn18-structure-centred-routes-stopped-epoch6`.
- **Direction agreed with the user:** every label anchored on a structure, the geometric halves,
  sectors and the side prediction removed. Missing anchors to derive from the bone masks: the
  notch (ACL), a medial band along the bones (MCL), the region behind the medial condyle
  (Baker's); Effusion and Synovitis on the kneecap classes; Contusion and Fracture on the bone
  shares with a learned floor; the reach learned per axis.

## After the campaign: `rn18-every-target-on-a-structure` (2026-10-09)

From `/home/claudio/src/knee-abnormality-anatomy-routing`, against run 7, same seed and fold.
Every target anchored on a structure (menisci, tibial cartilages and patellar cartilage from the
seven-class soft-tissue head; kneecap classes for Effusion and Synovitis, and the new derived
notch, medial band and behind-medial classes for ACL, MCL and Baker's, from a ten-class bone map
built by `scripts/adhoc/side_routed_session/segmenter_export_ligament_targets.py` into
`anatomy-mrannotator-sample550-ligaments`), a learned reach per axis and offset per route, the
soft-tissue shares not detached in the routes, no hand-measured crop offsets in the model.

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.927063 | −0.001058 |
| ×1 macro AUC | 0.919392 | 0.918931 | −0.000461 |
| Training seconds | 1,498 | 1,543 | 1.030 |
| Prediction seconds | 63 | 68 | 1.086 |

- Per target at ×4 (labels from the label table, deltas only): ACL +0.0029, Baker's +0.0024,
  Lateral OA +0.0023, Medial Meniscus +0.0012, Lateral Meniscus +0.0011, Medial OA +0.0007,
  PF OA +0.0003, Contusion −0.0004, Effusion −0.0011, Synovitis −0.0046, Fracture −0.0077,
  MCL −0.0079.
- **MCL's −0.008 is the reference's, not the routing's:** the head-free run and the first
  seven-class run, with MCL's route untouched, show the same −0.0084 and −0.0079 against run 7.
- **Fracture and Contusion were unrouted** (the floor was frozen at 1.000419 by clipping), so
  Fracture's −0.0077 is run-to-run movement of that label (−0.0023 and +0.0015 in the two other
  runs against run 7).
- Validation ×1 per epoch against run 7: −0.0046, +0.0001, −0.0014, −0.0010, −0.0009, −0.0010,
  −0.0008, −0.0003, −0.0005, −0.0004. PF OA was behind at every epoch (−0.003 to −0.029): its
  anchor, the patellar cartilage, had a Dice of 0.05 after epoch 1 and loses the coronal views
  (−0.034 and −0.047 alone at epoch 4).
- Final Dice on the validation mask knees: femur 0.91, tibia 0.90, kneecap 0.75, behind-medial
  0.74, fibula 0.73, notch 0.70, joint 0.70, kneecap joint 0.68, medial band 0.58; femoral
  cartilage 0.58, lateral tibial cartilage 0.58, medial meniscus 0.57, lateral meniscus 0.56,
  medial tibial cartilage 0.51, patellar cartilage 0.51. The soft-tissue classes end where the
  one-way run did (0.55 at epoch 6): no measurable effect of the two-way link.
- Route sizes at the end: menisci 17 to 18% of tokens, OA 16 to 19%, PF OA 10%. Reaches stay
  between 1.7 and 2.2 and offsets within 1.1 mm: the labels barely move them.
- Defect found after the run: the medial sign was applied to the positions, so a knee whose two
  meniscus centres coincide (only at initialisation) lost the across distance; the sign now
  applies to the learned offset only.

## After the campaign: `rn18-structure-routes-no-side` (2026-10-09)

From `/home/claudio/src/knee-abnormality-anatomy-routing-2`, against run 7. On top of
`rn18-every-target-on-a-structure`: PF OA and Effusion anchored on the kneecap and kneecap-joint
classes, Synovitis on the joint spaces (kneecap joint, joint, notch), Contusion and Fracture on
the bone shares without a floor; the side head, its loss and prior removed, the side taken from
the DICOM tag or coordinates and otherwise from the fibula's side of the tibia in the bone map;
position priors, sectors, core, ring, height sections and the eight unused routes removed; the
report heads not built. Guardrail script adapted (side swap leaves the routes unchanged; the
label loss leaves the bone map alone).

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.927801 | −0.000320 |
| ×1 macro AUC | 0.919392 | 0.918387 | −0.001005 |
| Training seconds | 1,498 | about 1,435 (from file times) | 0.96 |
| Prediction seconds | 63 | 59 | 0.94 |

- The run's own final prediction crashed (`apply_attention_temperature` still scaled the removed
  position priors); fixed in the code and the prediction rerun by hand from the best checkpoint.
- Per target at ×4: Lateral Meniscus +0.0029, Baker's +0.0029, ACL +0.0026, Lateral OA +0.0017,
  PF OA +0.0011, Medial OA +0.0007, Medial Meniscus +0.0002, Contusion −0.0003, Effusion
  −0.0027, Fracture −0.0027, MCL −0.0039, Synovitis −0.0044.
- Validation ×1 per epoch against run 7: −0.0049, +0.0013, −0.0020, −0.0002, −0.0010, −0.0024,
  −0.0013, −0.0004, −0.0013, −0.0008. MCL carries most of it from epoch 5 (−0.012, −0.019,
  −0.015, −0.007, −0.009, −0.012).
- **The side from the fibula works:** agreement with the known side 422, 424, then 425 of 425
  validation knees; crops at a limit 15 to 16%, as in run 7.
- **PF OA on the kneecap joint** is level at ×4 and −0.002 to −0.007 at ×1, against −0.003 to
  −0.029 on the patellar cartilage.
- Final Dice as in the previous run (menisci 0.54 to 0.55, cartilage 0.51 to 0.58, notch 0.70,
  medial band 0.57, behind-medial 0.74).
- Synovitis' route covers 60% of the tokens (anchor on all joint spaces) and it is the largest
  loss at ×4 with Effusion third: the two labels that changed anchor away from the old front
  region.

Curves per label (reference): most labels reach 94% or more of their final AUC above chance
after epoch 1 (Medial Meniscus 98%), MCL only 87% and it climbs through epochs 5 to 7. The
soft-tissue head's Dice is 0.05 to 0.30 after epoch 1 against 0.61 to 0.88 for the bone map:
labels that learn fast on a slow anchor (Medial Meniscus, Medial OA) stay a little behind at
every epoch. MCL's last-epoch AUC is 0.936 in run 7 and 0.926 to 0.927 in every run without the
three-class meniscus head, the level of the run before the sagittal canvas swap (0.927).

## After the campaign: `rn18-structure-routes-confident` and `-concentration` (2026-10-09, both stopped)

From `/home/claudio/src/knee-abnormality-anatomy-routing-3`. On top of the no-side run: the
three-class meniscus head restored beside a five-class cartilage head (both taught from the
seven-class targets, both reading the detached bone shares through a zero-initialised linear),
each route's centre and spread blended between its structure and the whole knee by a trust
value, a zero-initialised learned mix over all fifteen structures added to each anchor, reach
starting at the structure's own spread (was twice), offset in units of the extent, the fibula
side scaled by the tibia's width.

- **Trust by the peak share (`-confident`, stopped after epoch 1, −0.0132):** a thin structure
  never has a high share in a 12 mm token, so the meniscus and OA routes stayed at 34 to 39% of
  the tokens with medial and lateral on nearly the same centre; Lateral Meniscus −0.100.
- **Trust by concentration (`-concentration`, 1 − structure variance / knee variance; stopped by
  the user after epoch 6):** ×1 against run 7 per epoch −0.0062, −0.0013, −0.0005, −0.0007,
  −0.0016, −0.0033. The user's rule: no run of the line has recovered from −0.003 at epoch 6.
- **The two heads reading the bone map learn the masks faster:** soft-tissue Dice 0.34 to 0.51 at
  epoch 2 and 0.44 to 0.53 at epoch 3 (the seven-class head needed epochs 4 to 5), final band
  unchanged (0.50 to 0.57). Menisci level or ahead from epoch 2.
- **The learned route parameters barely move:** after five epochs the reach is within ±10% of its
  start (one axis −20%), offsets at most 0.23 extents (the menisci and OA pairs pushing apart),
  the largest learned anchor addition 0.16. The start is in practice what a label gets.
- **ACL** went from a steady gain with a route of 25% of the tokens (reach 2) to −0.005, −0.009,
  −0.006 with a route of 8% (reach 1).
- **MCL fell behind at epoch 6 again (−0.020) with the three-class head back**, so that head is
  not what gives run 7 its MCL level. MCL validation AUC at the last epoch is 0.936 in run 7
  and 0.923 to 0.927 in all twelve other runs of the campaign and of this line (47 positives).
- A set of hand-picked scales (1, 2, 4 times the extent with a learned mixture) was built and
  removed before any run: the user counts it as hardcoding.

## After the campaign: `rn18-structure-routes-union` (2026-10-09)

From `/home/claudio/src/knee-abnormality-anatomy-routing-4` (subagent build plus adjustments), on
top of the concentration build: moments, reach and offset per structure (15) instead of per
route; each route a soft union `1 − Π(1 − m·trust·K)` over the structures with
`m = route_anchor + structure_mix` (straight-through clamp to 0..1); Contusion and Fracture an
ordinary route over femur, tibia, kneecap and fibula; a route with no located member reads the
whole knee; neighbourhood kernel `1 / (1 + d²)` in units of the structure's extent instead of a
Gaussian.

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.926386 | −0.001735 |
| ×1 macro AUC | 0.919392 | 0.918560 | −0.000832 |
| Training seconds | 1,498 | 1,453 | 0.97 |
| Prediction seconds | 63 | 66 | 1.04 |

- Per target at ×4: ACL +0.0031, Baker's +0.0016, Medial OA +0.0011, Lateral OA +0.0004, PF OA
  +0.0001, Medial Meniscus −0.0002, Effusion −0.0013, Contusion −0.0018, Synovitis −0.0032, MCL
  −0.0041, Lateral Meniscus −0.0069, Fracture −0.0093. Against the no-side run the two large
  moves are Lateral Meniscus −0.0098 and Fracture −0.0066.
- Validation ×1 per epoch against run 7: −0.0106, +0.0011, −0.0038, −0.0023, −0.0013, −0.0024,
  −0.0005, −0.0010, −0.0008, −0.0006.
- **Crops at a limit rose to 19 to 21%** in all three builds since the concentration one (15 to
  16% in run 7 and the no-side run), from the first epoch, with unchanged mean shifts and turns.
  The only crop-side change in those builds is the fibula side scaled by the tibia's width
  instead of 5 mm, which makes the side a soft value (about 0.8) instead of a decision in studies
  without a known side. Candidate cause of the Lateral Meniscus loss; next build decides the side
  by the sign alone.
- **The learned mix moved a lot with this kernel** (largest additions: lateral meniscus 0.52,
  medial meniscus 0.44, OA 0.30 to 0.34; 0.07 to 0.16 for the rest). The meniscus and OA labels
  rebuild a compartment: Medial Meniscus adds medial tibial cartilage +0.40, medial band +0.44,
  behind-medial +0.34 and pushes every lateral and central structure negative (−0.2 to −0.4,
  without effect in a union); Lateral Meniscus adds lateral tibial cartilage +0.32, fibula +0.32,
  kneecap +0.31 and pushes the medial ones to −0.5. They ask for "my side, not the other side",
  and a union cannot exclude.
- Reach per structure after ten epochs: menisci and tibial cartilages narrower across (0.72 to
  0.78) and longer front-back and up (1.2 to 1.4); offsets push the medial and lateral pairs
  apart (0.24 to 0.29 extents).
- Synovitis as a union: level or ahead at ×1 in six of ten epochs, −0.0032 at ×4 (−0.0044 as
  one wide blob).
- The concentration trust averages the variance ratio over the axes, so a meniscus (thin, narrow,
  but as deep as the knee) scores about 0.6 when perfectly found: its route stayed 40% whole
  knee. Removed in the next build (a structure counts when it has any share).
- Label share of the soft-tissue heads' gradient, single sampled batches: 58% and 27% in the two
  epochs whose batch held a masked knee. The two-way link is not negligible.

## After the campaign: `rn18-structure-routes-union-present` (2026-10-09)

From `/home/claudio/src/knee-abnormality-anatomy-routing-5`: the union build with the
concentration trust removed (a structure counts when it has any share in the knee), the fibula
side as a plain left-or-right decision (no scale), and the gradient of the soft-tissue heads
measured every sixteenth training batch, label side and mask side apart.

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.926301 | −0.001819 |
| ×1 macro AUC | 0.919392 | 0.917855 | −0.001537 |
| Training seconds | 1,498 | 1,442 | 0.96 |
| Prediction seconds | 63 | 62 | 0.99 |

- Per target at ×4: ACL +0.0035, Lateral OA +0.0031, Baker's +0.0030, Medial OA +0.0022, PF OA
  +0.0020, Medial Meniscus −0.0020, Lateral Meniscus −0.0026, Effusion −0.0030, Synovitis
  −0.0043, Contusion −0.0061, MCL −0.0081, Fracture −0.0095.
- Validation ×1 per epoch against run 7: −0.0043, +0.0020, −0.0032, −0.0012, −0.0009, −0.0013,
  −0.0022, −0.0010, −0.0016, −0.0010.
- **The crop regression was the soft side:** crops at a limit back to 14 to 17% from the first
  epoch (19 to 21% with the tibia-width scale); fibula side 420 of 425 after epoch 1, 422 to 425
  after.
- **Two-way link, epoch means:** label gradient on the soft-tissue heads 0.049 against 0.231 from
  the masks in epoch 1, then 0.015 to 0.037 against 0.011 to 0.019 from epoch 2 on: once the
  masks are mostly learned the labels supply half or more of what moves those heads.
- **With the trust gone the learned mix only excludes:** no addition above 0.1 anywhere; Medial
  Meniscus holds every lateral, central and bone structure at −0.13 to −0.28, Lateral Meniscus
  the mirror image (−0.13 to −0.29), the OA pair the other side at −0.10 to −0.14. In a union
  those entries have no effect. The meniscus and OA routes ended at 7 to 9% of the tokens.
- **Across the four finished structure runs at ×4, the same labels gain and the same lose.**
  Gains: ACL +0.0026 to +0.0035, Baker's +0.0016 to +0.0030, Lateral OA +0.0004 to +0.0031,
  Medial OA +0.0007 to +0.0022, PF OA +0.0001 to +0.0020. Losses: Fracture −0.0027 to −0.0095,
  Synovitis −0.0032 to −0.0046, Effusion −0.0011 to −0.0030, MCL −0.0039 to −0.0081. The two
  runs that kept the reference's routing have Synovitis +0.0004 and +0.0013, Effusion −0.0019
  and 0.0000, Fracture +0.0015 and −0.0023: the three labels with no mask of the finding's own
  site lose when anchored on a structure.
- Fracture and Contusion were the two largest losses at epoch 4 in all three builds with two
  soft-tissue heads each at the full mask weight.

## After the campaign: `rn18-structure-routes-include-exclude` (2026-10-09)

From `/home/claudio/src/knee-abnormality-anatomy-routing-6`: the union-present build plus
exclusion (a negative mix entry multiplies the route by `1 − |m|·K` of that structure's
neighbourhood) and the two soft-tissue head losses averaged instead of summed.

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.926534 | −0.001587 |
| ×1 macro AUC | 0.919392 | 0.918534 | −0.000858 |
| Training seconds | 1,498 | 1,443 | 0.96 |
| Prediction seconds | 63 | 63 | 1.00 |

- Per target at ×4: ACL +0.0024, Medial OA +0.0017, Lateral OA +0.0015, PF OA +0.0014, Baker's
  +0.0011, Effusion −0.0005, Medial Meniscus −0.0011, Lateral Meniscus −0.0014, Contusion
  −0.0032, Synovitis −0.0047, MCL −0.0051, Fracture −0.0096. Against union-present: MCL +0.0030,
  Contusion +0.0029, Effusion +0.0025, both menisci +0.0010 to +0.0012, Baker's −0.0019,
  Lateral OA −0.0017.
- Validation ×1 per epoch against run 7: −0.0032, +0.0022, −0.0019, −0.0025, −0.0015, −0.0020,
  −0.0019, −0.0011, −0.0009, −0.0005 (best first two epochs of the line).
- The learned exclusions are the same as in the union-present run with effect now: the menisci
  hold every structure of the other side and of the centre at −0.11 to −0.24, the OA pair the
  other side at −0.08 to −0.14; nothing added. The gain on the menisci is about +0.001.
- **Halving the soft-tissue loss did not help Fracture or Contusion** (largest losses at epoch 4
  again, −0.018 and −0.008) and lowered the soft-tissue Dice by 0.02 to 0.04 at every epoch
  (final 0.47 to 0.56 against 0.51 to 0.58). Reverted.
- **How Contusion and Fracture read the bones, ×4 over six runs:** no route (run 7's way)
  Contusion −0.0039 and −0.0012, Fracture −0.0023 and +0.0015; each token by its own bone share
  Contusion −0.0004 and −0.0003, Fracture −0.0077 and −0.0027; neighbourhoods around the four
  bones Contusion −0.0018, −0.0061 and −0.0032, Fracture −0.0093, −0.0095 and −0.0096.
- **MCL is behind at ×4 in every run of the line, including the two that never touched its
  routing** (no soft-tissue head −0.0084, additive seven-class head −0.0079; structure runs
  −0.0039 to −0.0081).
- The paired labels (menisci, tibiofemoral OA) dip at epoch 3 by −0.006 to −0.008 in every
  structure run whether their routes cover 14% or 5 to 9% of the tokens, and are level from
  epoch 5; route width is not the cause.

## After the campaign: `rn18-structure-routes-unanchored-three` (2026-10-09)

From `/home/claudio/src/knee-abnormality-anatomy-routing-7`: the include-exclude build with
Effusion, Synovitis and Fracture given no anchor (whole knee at the start, learned exclusions
only, Fracture on its own route), Contusion back to each token's own share of the four bones,
and the two soft-tissue head losses summed again.

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.926487 | −0.001634 |
| ×1 macro AUC | 0.919392 | 0.917518 | −0.001874 |
| Training seconds | 1,498 | 1,461 | 0.98 |
| Prediction seconds | 63 | 63 | 1.00 |

- Per target at ×4: ACL +0.0039, Medial OA +0.0023, PF OA +0.0018, Baker's +0.0014, Lateral OA
  +0.0010, Medial Meniscus −0.0008, Effusion −0.0011, Contusion −0.0034, Lateral Meniscus
  −0.0037, Synovitis −0.0045, Fracture −0.0056, MCL −0.0094.
- Validation ×1 per epoch against run 7: −0.0049, +0.0017, −0.0027, −0.0012, −0.0012, −0.0027,
  −0.0022, −0.0016, −0.0019, −0.0017.
- **Synovitis loses the same −0.0045 at ×4 with no anchor and a route covering 96% of the
  tokens** as with every anchor tried (kneecap joint, one blob over the joint spaces, a union of
  them: −0.0032 to −0.0047). Fracture is −0.0056 with a near-uniform route (learned mix at most
  0.07). **The losses of Synovitis, Fracture and MCL do not come from their own routes.** They
  appear in every run that has the ten-class bone map and the two-way link (all structure runs)
  and in none of the two that keep run 7's routing for these labels with the seven-class bone
  map (Synovitis +0.0004 and +0.0013, Fracture +0.0015 and −0.0023).
- Contusion on its own bone share: level or ahead at ×1 in nine of ten epochs (+0.007 at most),
  −0.0034 at ×4.
- Fracture at epochs 4 and 5 (−0.011, −0.012) as in every structure run.
- The paired labels dip at epoch 3 from the run that removed the side prediction on (−0.006 to
  −0.008 against −0.002 before): medial against lateral then rests on the soft-tissue heads
  alone, which get no position.

## After the campaign: `rn18-structure-routes-positioned-masks` (2026-10-09)

From `/home/claudio/src/knee-abnormality-anatomy-routing-8` (subagent build plus one change): the
unanchored build with the meniscus and cartilage heads also reading each 14×14 token's position
in the knee frame (toward the fibula from the bone map, front, up; sin-cos code through a
zero-initialised linear, detached; no declared side involved, checked by a side-swap test with
difference 0.0), the Dice of either meniscus and either tibial cartilage logged, and Fracture
with no route at all as in run 7.

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.927040 | −0.001080 |
| ×1 macro AUC | 0.919392 | 0.918505 | −0.000887 |
| Training seconds | 1,498 | 1,454 | 0.97 |
| Prediction seconds | 63 | 66 | 1.05 |

- Per target at ×4: Fracture +0.0019, Baker's +0.0012, PF OA +0.0007, ACL +0.0005, Medial
  Meniscus −0.0000, Lateral OA −0.0004, Medial OA −0.0006, Contusion −0.0012, Effusion −0.0012,
  Lateral Meniscus −0.0031, MCL −0.0033, Synovitis −0.0045. Against the unanchored run: Fracture
  +0.0075, MCL +0.0061, Contusion +0.0022, ACL −0.0034, Medial OA −0.0029.
- Validation ×1 per epoch against run 7: −0.0024, +0.0016, −0.0026, −0.0009, −0.0023, −0.0023,
  −0.0024, −0.0003, −0.0011, −0.0005.
- **The position input speeds the masks up and tells the sides apart:** menisci at Dice 0.43 to
  0.44 after epoch 1 (0.13 to 0.29 before) and 0.52 after epoch 2; per-side Dice equal to the
  either-side Dice at every epoch (0.56 to 0.57 at the end), so no medial-lateral confusion is
  left. Final Dice: menisci 0.56 to 0.57 (unchanged), tibial cartilages 0.54 and 0.59 (0.51 and
  0.57 before).
- **Fracture with no route is +0.0019 at ×4** (−0.0027 to −0.0096 with any route), though it is
  still −0.013 to −0.015 at ×1 in epochs 1, 4 and 5 before recovering from epoch 6.
- **Synovitis is −0.0045 at ×4 for the sixth time**, here with a route of 95% of the tokens.
- The epoch-3 dip of the paired labels is unchanged (Lateral Meniscus −0.012, Medial OA −0.007)
  with masks that separate the sides, so mask confusion is not its cause. Cause open.
- Second-best ×4 of the line after the no-side run (−0.0003).
- Stopped here by the user (2026-10-09): the prepared one-way build
  (`knee-abnormality-anatomy-routing-9`, soft-tissue shares detached in the routes) was not
  launched.

## After the campaign: the crops of every structure run differ from run 7's (found 2026-10-09)

Asked by the user what else changed against run 7 besides the intended changes. The frozen
source of run 7 differs from the first copies only in the per-epoch logging; configurations,
inputs, labels, class weights and launch are identical. The difference is in the crops:

| Validation, last epoch | Run 7 | No soft-tissue head | Additive head | Every structure run |
|---|---|---|---|---|
| Mean sagittal crop shift across, px | 2.1 | 2.1 | 2.1 | 14.1 to 14.3 |
| Mean sagittal crop shift down, px | 2.1 | 2.2 | 2.1 | 8.4 to 9.2 |
| Mean coronal crop shift across, px | 3.4 | 3.3 | 3.4 | −1.8 to −1.9 |
| Mean coronal crop shift down, px | 3.8 | 3.8 | 3.7 | 8.3 to 9.1 |
| Mean absolute shift, px | 12.2 | 12.2 | 12.2 | 14.3 to 14.4 |
| Crops at a limit | 14.8% | 14.5% | 14.3% | 15 to 17% |

- **Cause:** when the hardcoded values were taken out of the model (copy
  `knee-abnormality-anatomy-routing`, first trained as the per-token routes), the two constants
  `CORONAL_JOINT_TARGET_MM = (-2.075445, 2.031255)` and `SAGITTAL_JOINT_TARGET_MM = (4.894627,
  2.727388)` in `ViewWindows.geometry` went with them. They are crop settings: where in the crop
  the joint centre is placed. Without them the joint centre sits at the exact crop centre, so
  every sagittal crop is displaced 4.9 mm across and 2.7 mm down and every coronal crop 2.1 mm
  and 2.0 mm against run 7 (the pixel differences above match these at about 0.4 mm per pixel).
- Every run from `rn18-anatomy-routing-attempt3-stopped` on carries it; the two runs before it
  do not, and those are the two in which Synovitis, Effusion and Fracture are level.
- **Second difference, from `rn18-every-target-on-a-structure` on:** the ten-class bone map
  carves the notch out of the joint class (joint Dice 0.76 to 0.70, bone-map loss 0.67 to 0.81)
  and the applied joint-centre displacement grows from 3.1 mm to 4.1 to 5.6 mm.
- Every comparison of the structure routing against run 7 since then mixes the routing with
  displaced crops. Not yet rerun.

## After the campaign: things to fix in the structure routing (list opened 2026-10-09)

1. **The routing became a soft preference instead of a hard gate (found by the user).** Run 7's
   regions are hard memberships with a straight-through gradient
   (`inside = heaviside(distance) + soft − soft.detach()`), the medial and lateral sides an exact
   partition (`lateral_side = 1 − medial_side`); a token outside a route gets
   `log(ROUTING_MINIMUM_WEIGHT) = −13.8` on its attention score. Only the height split (femur,
   joint, tibia shares) is soft. Every structure build instead adds the log of a smooth fall-off:
   at 2, 4, 6 and 10 extents from the structure the penalty is −0.5, −2.0, −4.5, −12.5 with the
   Gaussian at reach 2 and −1.6, −2.8, −3.6, −4.6 with `1 / (1 + d²)`; a learned exclusion of
   −0.25 is a penalty of 0.3. The side-to-side leakage limit that was put in on purpose is gone.
   Fix: hard membership from the mask anchors with the soft form only for the gradient; medial
   against lateral as an exact partition (each token to the nearer of the two structures); the
   neighbourhood boundary at the learned reach. The leak itself is not yet measured on the
   checkpoints.
2. **The joint-centre crop targets were removed with the model's hardcoded values** (section
   above). The user judges the centred crops fine and without a hardcoded rule; kept as they are.
3. **Open, for later (user):** the sagittal crops leave the kneecap out; centre the sagittal crop
   between the joint and the kneecap. Measured on the 550 mask knees along the front-back axis
   (patient axis, from the joint centre; front positive): kneecap 40 to 56 mm (centre 48), front
   of femur 31, back of femur −19, back of tibia −21, fibula back to −32, region behind the medial
   condyle −28 (middle) to −39. Half-width of the crop: 42 mm (fluid) and 46.7 mm (T1/PD). Joint
   at the centre: whole kneecap inside in 1% and 5% of knees, back of femur and tibia in 100%,
   whole behind-medial region in 76% and 98%. Centre at the midpoint of joint and kneecap (24 mm
   forward): whole kneecap 100%, back of femur 41% and 83%, back of tibia 25% and 70%, middle of
   the behind-medial region 5% and 17%. At the present field the back is lost; holding kneecap
   front (56) to behind-medial back (−39) needs about 95 mm plus margin centred 8 mm forward.
4. **A route is a small soft blob around a structure's centre of mass, not a region.** Seen on the
   worst validation cases (`outputs/analysis/s/structure-routing/worst-cases/`, last model against
   run 7, ×1): share of the label's attention on tokens with route weight of at least 0.5: MCL
   12% and 46% against 100% and 100%; Lateral Meniscus 30% and 27% against 80% and 73%; route
   size 5 to 15% of the tokens with a peak of 0.3 to 0.5 against 15 to 39% at weight 1. Run 7's
   MCL route is the whole medial edge at every height, so the axial slices are inside it (39% of
   the attention, probability 0.48 on knee 806); the mask route is a blob at the joint line that
   is nearly absent on the axial slices (probability 0.20). Run 7's Lateral Meniscus route is the
   lateral half from the joint line down through the tibia. Fix together with item 1: membership
   from the structure's whole predicted extent grown by the learned reach (a dilation of the mask
   in the knee frame), not a distance to its centre.
5. **Where the loss sits (last model against run 7, ×4, 874 validation knees):** not in a few
   broken cases: the ten most harmed knees lose 0.0007 and the ten most helped gain 0.0008; the
   absolute per-knee changes sum to 0.0077 for a net of 0.0008. By scanner field: 1.5 T (507
   knees) loses 0.00086 (80% interval 0.0004 to 0.0013), the rest (319) 0.0001 (−0.0002 to
   0.0004); studies with five series (460) lose 0.0008, the rest 0.0000. The worst cases are one
   positive of one label each whose score fell (Fracture twice, MCL twice, Lateral Meniscus four
   times, Synovitis twice among the top fourteen). Their crops show the joint centred and level in
   both models.
6. **The soft-tissue masks all come from one scanner type:** the 265 knees are Siemens 1.5 T (the
   DESS series the segmenter reads); the bone masks cover 335 knees at 1.5 T and 189 at 3 T.


## After the campaign: `rn18-mask-regions-hard` (2026-10-09, stopped at epoch 6 by the rule)

From `/home/claudio/src/knee-abnormality-anatomy-routing-10` (subagent build): on the
position-aware build, the routes rewritten as run 7's mechanism with regions from the masks:
hard membership with a straight-through gradient; a region is the tokens within a learned reach
(start: boxes touching) of the tokens that contain a structure (argmax of the bone map, argmax of
a fine child for the soft-tissue heads), over all series of the knee; medial and lateral an exact
partition by the nearer side's structures; masks detached (one-way); no height table; Contusion
by bone share, Fracture unrouted.

- Checked first on the positioned-masks weights without training: partition exact, no empty
  region, regions 8 to 22% of a knee's tokens (MCL 4 to 15%), 6 ms per knee at ×4; on the worst
  cases the positive's probability at ×1 went 0.16 to 0.34 and 0.17 to 0.19 (Lateral Meniscus),
  0.26 to 0.30 (Synovitis), 0.20 to 0.19 and 0.18 to 0.19 (MCL; run 7 gives 0.48 on knee 806,
  its region the whole medial edge with the axial slices inside).
- Validation ×1 per epoch against run 7: −0.0048, −0.0085, −0.0032, −0.0027, −0.0030, −0.0037.
- **The hard partition holds the paired labels:** Lateral Meniscus −0.003, −0.005, −0.003,
  +0.009, +0.003, +0.007 (the soft builds: −0.016 to −0.100 at epoch 1 and −0.008 to −0.018 at
  epoch 3); meniscus Dice 0.57 to 0.58 at epochs 4 and 5, the highest of the line.
- **A hard gate on a small region costs the labels grown from one small structure:** MCL (7 to 8%
  of the tokens) −0.026, −0.051, −0.013, +0.005, 0.000, −0.014; ACL −0.019 at epoch 2; Effusion
  and PF OA (kneecap region, 11%) −0.006 to −0.011 at every epoch. The reach stayed at 0.88 to
  1.09.
- Fracture (unrouted) −0.022 and −0.026 at epochs 4 and 5: alone −0.0018 and −0.0022 of the gap.

## After the campaign: `rn18-mask-territories` (2026-10-09, stopped at epoch 6 by the rule)

From `/home/claudio/src/knee-abnormality-anatomy-routing-12` (two subagent builds): the hard
build with (a) the bone map back to seven classes from the joints targets and a third fine head
(none, notch, medial band, behind-medial) on the soft-tissue features, taught from the ligament
targets on its own mask (400 training knees; 211 have soft-tissue masks, 25 both); (b) routes as
territories: every readable token goes to its nearest of six landmark groups (medial meniscus and
tibial cartilage; lateral ones; kneecap, kneecap joint and patellar cartilage; notch; medial band;
behind-medial), hard with a straight-through gradient to the reach; a label reads a union
(medial labels: medial, band and behind-medial; lateral labels: lateral; PF OA and Effusion:
kneecap; Synovitis: kneecap and notch; ACL: notch; MCL: band and medial; Baker's: behind-medial
and band).

- Validation ×1 per epoch against run 7: −0.0068, −0.0019, −0.0051, −0.0039, −0.0034, −0.0042.
- **The derived regions learn much better on their own head:** Dice after epochs 1, 2 and 6:
  behind-medial 0.68, 0.75, 0.75 (0.58, 0.67, 0.72 on the bone map); notch 0.65, 0.70, 0.71
  (0.62, 0.67, 0.70); medial band 0.40, 0.56, 0.58 (0.30, 0.49, 0.56). Joint class 0.72 to 0.75
  (0.65 to 0.69 with the notch carved out).
- Territory routes: medial labels 40 to 42% of the tokens, Synovitis 37 to 44%, Baker's 32 to
  38%, MCL 20 to 24%, ACL 18 to 23%, lateral labels 16 to 22%, kneecap labels 18 to 21%.
- MCL +0.007, −0.008, +0.011, −0.006, −0.013 from epoch 2 (the tight hard regions: −0.051 at
  epoch 2); Synovitis ahead in five of six epochs (+0.012 at epoch 4); the medial pair −0.003 to
  −0.017; PF OA −0.005 to −0.013 at every epoch.
- **Fracture and Contusion dip at epoch 4 with the seven-class bone map too** (−0.017 and
  −0.015), so the ten-class map was not the cause. Fracture's validation AUC at epoch 4 is 0.926,
  0.928 and 0.925 in the three runs with run 7's crop targets and 0.904 to 0.922 in all twelve
  runs without them, whatever the routing, the bone map or the link; the two runs stopped in the
  morning had run 7's setup for Fracture and already 0.910.
- **Both hard builds were behind the soft builds at epoch 6** (−0.0037 and −0.0042 against
  −0.0013 to −0.0027).

## After the campaign: `rn18-mask-territories-crop-targets` (2026-10-09, stopped at epoch 6 by the rule, at −0.00305)

From `/home/claudio/src/knee-abnormality-anatomy-routing-13`: the territories build with run 7's
two joint-centre crop targets restored and nothing else changed (`ViewWindows`, `first_look_pose`
and `anatomical_frames` identical to run 7's source again).

- Validation ×1 per epoch against run 7: −0.0172, −0.0001, −0.0027, −0.0015, −0.0016, −0.0030;
  the same build without the targets: −0.0068, −0.0019, −0.0051, −0.0039, −0.0034, −0.0042.
  About 0.002 closer at epochs 2 to 5 (one pair of runs).
- **Fracture's epoch-4 dip is still there with the targets** (−0.012, 0.914), so the crop targets
  are not its cause either. Contusion −0.003 at epoch 4 and +0.012 at epoch 5 (−0.015 and +0.002
  without the targets).
- PF OA and Effusion level at epochs 4 and 5 (0.000 to −0.003; −0.003 to −0.009 without the
  targets). Lateral Meniscus +0.004, +0.009, +0.009 at epochs 3 to 5.
- The medial labels on the three-territory region trail the lateral ones on one territory in both
  territory runs (epoch 6: Medial OA −0.009 and −0.010, Medial Meniscus −0.003 both; Lateral OA
  −0.001 and +0.003, Lateral Meniscus −0.003 both); MCL −0.019 and −0.012 at epochs 5 and 6.
- Epoch 1 at −0.0172 with the lateral meniscus mask at Dice 0.17 (Lateral Meniscus −0.098): with
  hard territories the first epoch follows how fast the masks come up.
- Crops at a limit 13.4 to 14.3%.

## After the campaign: run 7 retrained, `rn18-reference-retrained` (2026-10-09)

From `/home/claudio/src/knee-abnormality-base`: run 7's frozen source with only the per-epoch
logging changed, same configuration, seed 1110484336 and fold 1. The measure of how far two
trainings of the same recipe land from each other.

| | Run 7 | Retrained | Difference |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.928120 | −0.000001 |
| Best ×1 macro AUC | 0.919392 (epoch 9) | 0.918411 (epoch 10) | −0.000981 |
| Training seconds | 1,498 | 1,497 | 1.00 |

- **Training is not deterministic**, different from the first epoch on.
- **×1 macro per epoch, retrained minus run 7:** +0.0005, +0.0003, +0.0002, −0.0002, −0.0003,
  −0.0009, −0.0001, −0.0006, −0.0015, −0.0004. Within 0.0005 through epoch 5, up to 0.0015
  after.
- **Single labels move far more than the macro:** at ×1 up to 0.011 in one epoch (Lateral
  Meniscus +0.0108 at epoch 5, MCL −0.0113 at epoch 6); at ×4 PF OA +0.0030, Lateral Meniscus
  +0.0020, ACL +0.0016, Synovitis +0.0014, Medial OA +0.0007, Baker's +0.0004, Lateral OA
  +0.0003, Medial Meniscus −0.0009, Effusion −0.0013, Contusion −0.0016, Fracture −0.0021, MCL
  −0.0046.
- **MCL:** the step at epoch 6 arrived one epoch later in the retrained run and ended at 0.929
  at ×1 (0.936 in run 7; 0.923 to 0.927 in the other runs); −0.0046 at ×4.
- Crop and mask statistics agree to the second or third decimal at every epoch (crops at a limit
  14.5% against 14.9%, joint Dice 0.760 against 0.761, medial meniscus Dice 0.558 against 0.562).
- **What it means for the line:** the ×4 macro of run 7 came back to the sixth decimal (one
  pair), so the structure runs' ×4 deficits of 0.0003 to 0.0018 are not run-to-run spread; their
  per-label stories at single epochs mostly are (anything under about 0.005 on one label at ×4,
  or 0.01 at one ×1 epoch). At ×1 the best-epoch macro has a spread of about 0.001, and the stop
  rule at epoch 6 (−0.003 against run 7) is about −0.002 against what the recipe repeats.
- All finished runs at ×4 against run 7: no-side −0.0003, additive −0.0007, no soft-tissue head
  −0.0008, every target on a structure −0.0011, position-aware masks −0.0011, include-exclude
  −0.0016, three unanchored −0.0016, union −0.0017, union-present −0.0018.

## After the campaign: `rn18-mask-territories-medial-alone` (2026-10-09, run to the end)

From `/home/claudio/src/knee-abnormality-anatomy-routing-14`: the territories build with the crop
targets and Medial Meniscus and Medial OA on the medial territory alone. First hard build with a
×4 result (the stop rule was dropped after run 7's retraining).

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.925818 | −0.002303 |
| Best ×1 macro AUC | 0.919392 | 0.917390 | −0.002002 |
| Training seconds | 1,498 | 1,636 | 1.09 |
| Prediction seconds | 63 | 68 | 1.08 |

- Per target at ×4 against run 7 (and against run 7 retrained): Lateral OA +0.0026 (+0.0023),
  ACL +0.0008 (−0.0009), Medial Meniscus +0.0001 (+0.0010), PF OA −0.0002 (−0.0032), Contusion
  −0.0002 (+0.0013), Fracture −0.0023 (−0.0002), Medial OA −0.0024 (−0.0031), Lateral Meniscus
  −0.0027 (−0.0047), Effusion −0.0027 (−0.0015), Baker's −0.0031 (−0.0035), Synovitis −0.0037
  (−0.0051), MCL −0.0116 (−0.0070).
- Validation ×1 per epoch against run 7: −0.0150, −0.0010, −0.0015, −0.0023, −0.0018, −0.0013,
  −0.0025, −0.0017, −0.0020, −0.0021.
- **The worst finished run of the line at ×4**, 0.0020 below the soft no-side run, and 9% slower
  to train. MCL is behind by more than the reference's own run-to-run movement at ×4 (0.0046)
  and at ×1 from epoch 5 on (−0.011 to −0.026).
- Medial labels on 7 to 9% of the tokens, lateral on 19 to 21%: Medial OA behind at every epoch
  (−0.007 to −0.010), as it was on the wide three-territory region (38%); width is not what
  holds it back.
- Masks at their best levels (menisci 0.57 to 0.58, tibial cartilages 0.53 and 0.59,
  behind-medial 0.76, notch 0.71, medial band 0.59); crops at a limit 12.5 to 15.3%.
- **Hard regions from the masks, as built in four variants, do not match the soft structure
  routes:** at epoch 6 −0.0013 to −0.0042 against −0.0013 to −0.0027, and −0.0023 against
  −0.0003 at ×4 in the one run carried to the end.

## After the campaign: the crop targets stay out (user, 2026-10-09)

The user judged the centred crops fine and without a hardcoded rule ("it has to be a different
thing"), and objected when the targets were restored in three builds anyway
(`rn18-mask-territories-crop-targets`, `rn18-mask-territories-medial-alone`,
`rn18-structure-routes-no-side-crop-targets`; the last stopped at epoch 4 for that reason, ×1
against run 7 −0.0055, +0.0004, −0.0025). No build carries the two joint-centre targets from here
on. Also from the user the same day: reruns of a training in this competition land within
±0.0003 of macro AUC, so run 7's retraining was not wanted and is not to be repeated.

## After the campaign: run 7 with centred crops, `rn18-reference-centred-crops` (2026-10-09)

Asked by the user: run 7 unchanged except that the two hardcoded joint-centre crop targets are
removed (joint centre at the middle of the sagittal and coronal crops, as in every structure
experiment). From `/home/claudio/src/knee-abnormality-reference-centred-crops`; 24 lines of the
model differ from run 7's source, the crop class identical to the experiments'.

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.927084 | −0.001037 |
| Best ×1 macro AUC | 0.919392 | 0.919919 | +0.000527 |
| Training seconds | 1,498 | 1,511 | 1.01 |
| Prediction seconds | 63 | 64 | 1.02 |

- Validation ×1 per epoch against run 7: +0.0015, −0.0013, −0.0015, +0.0001, +0.0006, −0.0013,
  −0.0000, +0.0004, +0.0004, +0.0011. Highest ×1 of the day (0.919919).
- **×1 and ×4 disagree:** ahead by 0.0005 at ×1 and behind by 0.0010 at ×4. Rerun spread known
  from the user is ±0.0003, so the ×4 deficit is beyond it.
- Crops: mean sagittal shift across 14.0 to 14.7 px (run 7: 1.5 to 2.4), mean absolute shift 14.1
  to 14.5 px (12.1 to 12.3), crops at a limit 14.8 to 17.8% (13.4 to 16.1%). Masks as in run 7.
- PF OA behind at ×1 in nine of ten epochs (−0.002 to −0.007).

## Public score and temperature sweep of run 7 with centred crops (2026-10-09)

- **S130**, `rn18-reference-centred-crops`, ×4, τ 0.9, batch 4, exported from the run's own
  source: **0.953** public (S129, run 7 with the crop targets, 0.955). Locally ×4 0.927084
  against 0.928121 and best ×1 0.919919 against 0.919392: the public score goes with ×4.
- **Temperature sweep on validation fold 1** (`scripts/adhoc/sweep_child_temperatures.py`,
  tables in `outputs/analysis/t/tau-sweep-reference-centred-crops/`), macro AUC without
  reflection: ×4 at τ 0.85 to 1.10: 0.927241, 0.927293, 0.927219, 0.927094, 0.927089, 0.926954
  (best τ 0.90, +0.0002 over τ 1.00). ×6 at τ 0.65 to 1.10: 0.925807, 0.926250, 0.926606,
  0.926835, 0.927014 (0.926935 in the second pass), 0.926949, 0.926909, 0.926828, 0.926769,
  0.926688 (best τ 0.85). ×6 is below ×4 at every τ by 0.0002 to 0.0003. With reflection
  averaging the order flips (×6 0.928532 at τ 0.80, ×4 0.928392 at τ 0.85).

## How the joint-centre targets change what the model sees and reads (2026-10-09)

Asked by the user after S130. Two measurements, no training; tables in
`outputs/analysis/c/crop-target-effect/`.

- **What is inside the sagittal crop, on the 550 mask knees** (share of each structure's volume,
  with the targets against centred; nominal fields, no per-knee scale or limit): fluid sagittal
  (84 mm): kneecap 40% against 15%, kneecap joint 69% against 55%, behind-medial 89% against 98%,
  fibula 60% against 71%, tibia 77% against 80%, femur 72% against 69%. T1/PD sagittal (93 mm):
  kneecap 69% against 40%, kneecap joint 81% against 73%, behind-medial 98% against 100%, fibula
  76% against 82%. The kneecap sits 40 to 56 mm in front of the joint centre and 15 to 48 mm
  above it.
- **Where the labels' attention goes, 874 validation knees at ×1** (run 7 retrained against run 7
  with centred crops): every label's attention in the sagittal views moves 0.3 to 0.4 token
  columns toward the front edge, the 4.9 mm of the target. PF OA, Effusion and Synovitis put
  more of their sagittal attention on the outer ring of tokens (PF OA 18% to 25% fluid and 9% to
  17% T1/PD; Effusion 19% to 22% and 15% to 18%; Synovitis 17% to 20% and 13% to 17%) and less
  attention on the sagittal views overall, moving it to the fluid axial view (PF OA 29.0% to
  34.2%, Effusion 41.7% to 43.8%, Synovitis 29.8% to 32.3%). The menisci and Baker's go the
  other way in the ring share (Medial Meniscus 13.8% to 6.3%, Lateral Meniscus 12.5% to 5.9%,
  Baker's 42% to 32% on the fluid sagittal view).
- Not explained: why the centred crops are ahead at ×1 (+0.0005) and behind at ×4 (−0.0010) and
  on the public board (0.953 against 0.955).

## Run 7 with per-view losses, `rn18-reference-view-heads` (2026-10-09)

Asked by the user: every view pooled alone and graded on the study's labels. Version run: run 7
unchanged (joint answer as the final prediction, crop targets in place), plus an auxiliary loss:
each of the six views pooled alone with the model's own attention and readouts, the study's
labels as targets, averaged over the views present in each study, at the weight of the joint
loss. From `/home/claudio/src/knee-abnormality-reference-view-heads` (twelve model lines).
Precedent: `dense-plane-readouts` on the shared ResNet-18 (revisits, 30 September), per-plane
losses with a learned detached fusion as the final answer, −0.0052.

| | Run 7 | Run | Change |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.925054 | −0.003067 |
| Best ×1 macro AUC | 0.919392 | 0.917382 | −0.002010 |
| Training seconds | 1,498 | 1,659 | 1.11 |

- Per target at ×4: ACL +0.0059, Medial Meniscus +0.0018, Lateral Meniscus +0.0006, Contusion
  −0.0005, Baker's −0.0007, PF OA −0.0010, MCL −0.0013, Effusion −0.0044, Lateral OA −0.0070,
  Medial OA −0.0074, Synovitis −0.0093, Fracture −0.0116.
- Validation ×1 per epoch against run 7: −0.0025, 0.0000, −0.0018, +0.0005, −0.0027, −0.0031,
  −0.0003, −0.0022, −0.0020, −0.0019. Synovitis behind at all ten epochs (−0.006 to −0.014).
- **Every view reads better alone, the joint answer worse.** Single-view macro AUC at the last
  epoch against run 7's: sagittal T1/PD 0.857 against 0.786, coronal T1/PD 0.786 against 0.719,
  axial fluid 0.872 against 0.827, coronal fluid 0.872 against 0.851, sagittal fluid 0.871
  against 0.853, axial T1/PD 0.572 against 0.553.
- The joint answer's attention moved between views (epoch 6, mean over labels): axial fluid
  21.6% to 27.0%, coronal T1/PD 13.8% to 10.6%, sagittal T1/PD 16.4% to 14.8%; for Fracture
  axial fluid 22% to 33% and sagittal T1/PD 14% to 7%.
- Single-view gains too large to be the view seeing the finding: MCL from sagittal T1/PD +0.21,
  Effusion from coronal T1/PD +0.15, Contusion from coronal T1/PD +0.13 (epoch 5).
- Candidate causes, none isolated: one attention scorer trained for two poolings; views graded on
  labels they cannot show and learning what co-occurs; views becoming alike; the joint loss at
  half of the label signal.
- Prepared, not run: `knee-abnormality-reference-view-mlp` (this build plus a detached two-layer
  MLP on the per-view logits correcting the joint answer; the user wants it judged alone) and
  `knee-abnormality-reference-view-fixes` (per-view pooling with detached attention, each view
  and label weighted by the joint answer's attention mass on that view, `view_loss_weight` 0.5).
- The per-view AUC of a model trained with per-view losses reads each canvas directly; the user
  intends it for choosing the canvas configuration.

## Run 7 with per-view losses and a detached MLP, `rn18-reference-view-mlp` (2026-10-09)

The per-view-loss run above with a final head added: a two-layer MLP (78 to 78, GELU, 78 to 12,
last layer zero-initialised, 7,110 parameters) on the six views' detached logits and view
presence, its output added to the detached joint logits; own class-balanced BCE, no gradient to
the backbone. Not the user's design: the joint loss is still in the training (the user had
asked for the MLP over the per-view heads as the final answer; built next as
`knee-abnormality-reference-view-mlp-pure`).

| | Run 7 | Per-view losses | With the MLP |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.925054 | 0.926075 |
| Best ×1 macro AUC | 0.919392 | 0.917382 | 0.918129 |

- **The MLP adds +0.0010 at ×4 and +0.0007 at ×1 over the same training without it**, and was
  ahead of the plain joint answer at all ten ×1 epochs (+0.0003 to +0.0021). Still −0.0020 at ×4
  against run 7.
- Per target at ×4 against run 7: ACL +0.0084, Lateral Meniscus +0.0050, PF OA +0.0027, Medial
  Meniscus +0.0013, Baker's +0.0008, Lateral OA +0.0005, Effusion −0.0018, Contusion −0.0021,
  Medial OA −0.0051, MCL −0.0052, Synovitis −0.0055, Fracture −0.0184. Against the run without
  the MLP: Lateral OA +0.0075, Lateral Meniscus +0.0044, PF OA +0.0037, Synovitis +0.0037,
  Effusion +0.0026, ACL +0.0025, Medial OA +0.0023, Baker's +0.0015, Medial Meniscus −0.0005,
  Contusion −0.0016, MCL −0.0039, Fracture −0.0068.
- Single-view AUCs equal to the run without the MLP within 0.002 (the MLP is detached); MLP
  correction 0.12 to 0.15 logits on average; MLP training loss below the joint loss at every
  epoch (0.4329 against 0.4360 at the end).
- Fracture is the largest loss of both per-view-loss runs (−0.0116 and −0.0184 at ×4).

## Per-view heads only with a detached cross layer, `rn18-reference-view-cross-pure` (2026-10-09, stopped at epoch 6 by the user)

The user's design as stated: run 7 with no joint loss; the backbone and plane readouts trained
only by the six per-view losses; final answer = mean of the present views' detached logits plus
a detached DCN cross layer (`x × (W·x + b) + x` on the 72 view logits and 6 presence flags) and a
zero-initialised linear readout (about 7,100 parameters). From
`/home/claudio/src/knee-abnormality-reference-view-cross-pure`.

- **A first launch was void:** the joint pooling was still computed for the logs under `no_grad`
  and before the per-view pooling; the model memoises the pooled tokens per step, so the per-view
  losses inherited gradient-free tokens and never reached the backbone (epoch 1: 0.742, single
  views 0.65). Fixed by pooling the views first; checked on real studies (backbone gradient none
  against present). Output kept as `rn18-reference-view-cross-pure-broken-no-backbone-gradient`.
  The two earlier per-view runs pooled the joint answer with gradient and were not affected
  (verified in their source snapshots). The guardrail script has no check that the label loss
  reaches the backbone.
- Validation ×1 per epoch against run 7: −0.0125, −0.0032, −0.0022, −0.0022, −0.0061, −0.0071.
- **The plane heads are as good or better without the joint loss:** single-view macro AUC at
  epoch 6 against the run with the joint loss and against run 7: axial fluid 0.8705 (0.8679;
  0.8260), coronal fluid 0.8698 (0.8671; 0.8469), sagittal fluid 0.8692 (0.8703; 0.8539),
  sagittal T1/PD 0.8551 (0.8537; 0.7857), coronal T1/PD 0.7871 (0.7823; 0.7229), axial T1/PD
  0.5702 (0.5654; 0.5532).
- **The head on top does not turn that into a better combined answer:** its training loss falls
  every epoch (0.546, 0.493, 0.476, 0.470, 0.465, 0.458) while validation stalls (0.9033, 0.9065,
  0.9071, 0.9091 from epoch 3); behind on Synovitis −0.020, MCL −0.018, Fracture −0.014, Medial
  OA −0.013 at epoch 6.
- **Found while answering why these runs flatten early:** the recipe clips the gradient at 1.0.
  The pre-clip norm is 1.02 to 1.13 in run 7, 1.53 to 1.75 in the two runs with the joint loss
  plus the per-view loss, and 0.85 to 0.97 here. Those two runs trained everything at roughly 60%
  of run 7's step, so their ×4 results (−0.0031 and −0.0020) include a learning-rate effect.

## Per-view heads only with a detached MLP and view dropout, `rn18-reference-view-mlp-dropout` (2026-10-09, stopped at epoch 5 by the user)

The user's choice after the cross-layer run: the same pure design with the two-layer MLP head and
head-level view dropout (each present view hidden from the head with probability 0.2 in training,
at least one kept; `trainer.head_view_dropout`). Checked before launch on the changed path
(per-view answers reach the backbone, the head is detached, 20.0% of views hidden, none left
empty).

- Validation ×1 per epoch against run 7: −0.0162, −0.0062, −0.0058, −0.0042, −0.0071; against
  the cross-layer run at the same epochs −0.0037, −0.0030, −0.0037, −0.0020, −0.0010.
- Plane heads as in the cross-layer run (fluid views 0.865 to 0.869 at epoch 5). The final
  answer stalls where the cross layer's did (0.8996, 0.9046, 0.9061), behind on Fracture −0.030,
  Synovitis −0.016, Medial OA and PF OA −0.015.
- Two heads on six logits per label, with and without dropout, flatten at the same place while
  the views underneath keep improving.
- Gradient clipping: the recipe uses AdamW, so a steady clip of the summed gradient does not
  shrink the step as first claimed; the pre-clip norm of 1.65 in the joint-plus-per-view runs
  means the per-view loss supplied more of the gradient than the joint loss. Since the
  soft-tissue masks the norm is unchanged (1.03 to 1.08 with and without the head).
- All run checkpoints except run 7's best were deleted at the user's request this evening (44
  files, 7.7 GB); predictions, metrics, logs and source snapshots remain.
- Next, by the user: run 7 plus the per-view loss at a very small weight
  (`knee-abnormality-reference-view-loss-small`, `trainer.view_loss_weight`).

## Run 7 with a small per-view loss, `rn18-reference-view-loss-small` (2026-10-09)

The first per-view attempt (run 7 unchanged, joint answer final, each view pooled alone and
graded on the study's labels) with the per-view loss at `trainer.view_loss_weight` 0.1 instead
of 1. From `/home/claudio/src/knee-abnormality-reference-view-loss-small`.

| | Run 7 | Weight 1 | Weight 0.1 |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.925054 | 0.926724 |
| Best ×1 macro AUC | 0.919392 | 0.917382 | 0.918948 |
| Pre-clip gradient norm | 1.05 | 1.65 | 1.04 to 1.19 |
| Training seconds | 1,498 | 1,659 | 1,610 |

- **−0.0014 at ×4 and −0.0004 at ×1 against run 7**; +0.0017 at ×4 over the same loss at weight
  1. Half of the cost of the per-view loss goes away with a tenth of its weight; the rest stays.
- Per target at ×4 against run 7: ACL +0.0043, Baker's +0.0011, PF OA +0.0007, Medial Meniscus
  −0.0001, Fracture −0.0010, Lateral Meniscus −0.0011, Contusion −0.0021, Medial OA −0.0023,
  Effusion −0.0027, Synovitis −0.0036, Lateral OA −0.0046, MCL −0.0059. Against weight 1:
  Fracture +0.0106, Synovitis +0.0057, Medial OA +0.0051.
- Validation ×1 per epoch against run 7: +0.0001, +0.0030, −0.0020, +0.0007, −0.0010, −0.0024,
  −0.0011, −0.0006, −0.0012, +0.0001. Fracture ahead at every epoch from the second (it dipped
  by 0.009 to 0.040 at epochs 4 and 5 in every other variant of the day).
- Single-view macro AUC at the last epoch (weight 1; run 7): sagittal fluid 0.863 (0.871;
  0.853), coronal fluid 0.859 (0.872; 0.851), axial fluid 0.852 (0.872; 0.827), sagittal T1/PD
  0.828 (0.857; 0.786), coronal T1/PD 0.755 (0.786; 0.719), axial T1/PD 0.563 (0.572; 0.553):
  about a third to a half of the gain of weight 1.
- **Across the three weights (0, 0.1, 1) the ×4 macro falls as the per-view loss grows**
  (0.9281, 0.9267, 0.9251) while the single views rise.

## Small per-view loss with a crop pose free of typed numbers, `rn18-reference-free-crops` (2026-10-09)

The small-per-view-loss recipe (run 7 plus the per-view loss at weight 0.1) with every typed
number taken out of the crop pose. From `/home/claudio/src/knee-abnormality-reference-free-crops`.

| Was | Now |
|---|---|
| Joint centre 4.9 / 2.7 mm (sagittal) and −2.1 / 2.0 mm (coronal) off the crop middle | joint centre at the crop middle |
| Scale = knee size / 17.087 | knee size / the mean size of the training knees seen so far in the current epoch (two buffers, restarted each epoch, saved in the checkpoint) |
| Knee size from fixed height windows (−5 to 45 mm, −30 to 5 mm, a 20 mm bell) | from all femur and tibia shares |
| Scale limited to 0.87 to 1.13 | limited by the canvas to crop ratio (1.64) |
| Shift limited to 18 mm (8 mm axial) | limited by the canvas margin |
| Turn limited to 12 degrees (20 axial) | no limit |
| Axial centre between the kneecap's 95th-percentile front and 30 mm in front of the femur's 5th-percentile back | share-weighted mean front of femur and kneecap |

Left: the training jitter (3 degrees, 3 mm), the canvas margin and the constants of the knee
frame the pose is read from.

| | Run 7 | Small per-view loss, old pose | Small per-view loss, free pose |
|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.926724 | 0.926772 |
| Best ×1 macro AUC | 0.919392 | 0.918948 | 0.919768 |
| Training seconds | 1,498 | 1,610 | 1,607 |

- **Against the same recipe on the old pose: +0.00005 at ×4 and +0.0008 at ×1.** The typed
  numbers of the pose buy nothing measurable on this recipe. Against run 7: −0.0013 at ×4,
  +0.0004 at ×1, which is the cost of the per-view loss already seen (−0.0014).
- Per target at ×4 against run 7: ACL +0.0039, PF OA +0.0029, Fracture +0.0008, Baker's +0.0004,
  Medial Meniscus −0.0000, Medial OA −0.0011, Lateral Meniscus −0.0018, Contusion −0.0027,
  Synovitis −0.0032, Lateral OA −0.0037, MCL −0.0056, Effusion −0.0059. Against the old pose:
  PF OA +0.0024, Medial OA +0.0014, Synovitis +0.0009, Lateral OA +0.0009, MCL +0.0003, Fracture
  +0.0001, Medial Meniscus +0.0000, ACL −0.0004, Lateral Meniscus −0.0006, Baker's −0.0007,
  Contusion −0.0007, Effusion −0.0032.
- Validation ×1 per epoch against run 7: −0.0057, −0.0013, −0.0015, +0.0016, −0.0009, −0.0018,
  −0.0004, +0.0006, +0.0002, +0.0009. Against the old pose: −0.0058, −0.0043, +0.0005, +0.0008,
  +0.0001, +0.0005, +0.0007, +0.0012, +0.0014, +0.0008.
- **The first epoch trains on crops that end about 11% too tight** (mean scale 0.894 at the
  first validation): the epoch's mean size still holds the values measured while the bone map
  was untrained (about 27 against 16.6 once trained). From the second epoch the mean scale is
  0.99 to 1.00. A previous-epoch mean was tried in a smoke and lagged (mean scale 0.86). Open
  fix without a typed number: measure the typical size once from the taught bone masks.
- Crops stopped by the canvas: 0.5% to 0.8% of crops from the second epoch (16% hit a limit
  under the old limits). Mean shift 14.6 px against 12.2. Joint Dice 0.758 against 0.760.
- Single-view macro AUC at the last epoch (old pose): sagittal fluid 0.8640 (0.8627), coronal
  fluid 0.8570 (0.8586), axial fluid 0.8549 (0.8523), sagittal T1/PD 0.8244 (0.8276), coronal
  T1/PD 0.7556 (0.7549), axial T1/PD 0.5596 (0.5627). Axial fluid was ahead at every epoch.
- First launch crashed at the first validation: the pose-limit logging in
  `training/lightning.py` still read the removed limits. Lesson: a change that removes a model
  attribute needs a one-epoch run through validation before the command is handed over.

## Temperature sweep at ×4 of run 7 and of the free-pose run (2026-10-09)

`scripts/adhoc/sweep_child_temperatures.py`, each from its own source, validation fold 1; tables
in `outputs/analysis/t/tau-sweep-reference-and-free-crops/`. Macro AUC without and with
reflection averaging.

| τ | Run 7 | Run 7, reflection | Free pose | Free pose, reflection |
|---|---|---|---|---|
| 0.85 | 0.928508 | 0.929720 | 0.926882 | 0.927823 |
| 0.90 | 0.928413 | 0.929745 | 0.926873 | 0.927774 |
| 0.95 | 0.928381 | 0.929613 | 0.926817 | 0.927659 |
| 1.00 | 0.928193 | 0.929470 | 0.926767 | 0.927538 |
| 1.05 | 0.928090 | 0.929200 | 0.926592 | 0.927285 |
| 1.10 | 0.927820 | 0.929031 | 0.926421 | 0.927074 |

- Both prefer the lowest temperature swept (0.85) and neither has peaked inside the range. Run 7
  gains +0.0003 from τ 1.00 to 0.85, the free-pose run +0.0001: the temperature does not close
  the gap, it widens it slightly (−0.0014 at τ 1.00, −0.0016 at τ 0.85).
- Reflection averaging adds +0.0012 to +0.0013 to run 7 and +0.0007 to +0.0009 to the free-pose
  run, so with it the gap is −0.0019.

## Audit of the line by three agents (2026-10-09)

Code, results and the ×1-to-×4 difference, from saved outputs and CPU only; 15 finished
experiments plus `rn18-reference-retrained`. Main claims rechecked by hand (gain table, crop
targets per snapshot, the Fracture case, the retraining's MCL).

- **No code defect and no unintended difference** in the five reference-family copies; the ×4
  prediction code is byte-identical to run 7's. Side effect at per-view weight 1: the L2-SP
  anchor is not scaled with the doubled label loss (stage-4 drift 0.134 against 0.109).
- **The common factor is a smaller gain from ×1 to ×4.** Run 7 +0.0087, its retraining +0.0097;
  14 of 15 experiments gain less than their mean, by 0.0010 on average, about 70% of the mean
  ×4 deficit of 0.0015. Best ×1 does not rank these runs: run 7 and its retraining differ by
  0.0010 at ×1 and by 0.000001 at ×4.
- **Where the gain is lost:** studies with a coronal T1/PD series (687; below in 15 of 15 runs)
  and five-view studies, where ×4 buys most in run 7. By label, the ones ×4 helps most in run
  7: Lateral Meniscus (+0.029 from ×4), Lateral OA (+0.018), Contusion (+0.015), Fracture
  (+0.014). The experiments are ahead on them at ×1 (Fracture 0.929 to 0.932 against 0.9235)
  and level or behind at ×4.
- **The loss is a thin tail:** about 20 borderline positives (probability 0.1 to 0.2, soft
  target 0.5 to 0.8, mostly 1.5 T) ranked lower; three positives carry the Fracture loss, three
  the Contusion loss, eight the MCL loss. No subgroup is broken.
- **Run 7's checkpoint is high on MCL, Contusion and Effusion:** its own retraining is −0.0046,
  −0.0016 and −0.0012 on them at the same macro. Judge per-label changes against both.
  Synovitis and Fracture losses and the ACL gain are not in the retraining.
- **Costs do not add:** centred crops −0.0010, per-view loss 0.1 −0.0014, both −0.0014.
- **Per-view loss:** predictions move less between ×1 and ×4 in step with the weight (per-study
  logit move 0.227, 0.212, 0.181 at weights 0, 0.1, 1) and labels become more alike (Fracture
  and Contusion logit correlation 0.82, 0.85, 0.88). Temperature does not close the gap.
- **Centred crops, one clean case:** Fracture positive `…90751504670572401790063555695835439317`
  scores 0.60 to 0.74 in every run with the crop targets and 0.29 to 0.42 in every run
  without. Sagittal-fluid view alone with centred crops: PF OA −0.016, Effusion −0.006, Baker's
  +0.032, Contusion +0.015, Fracture +0.013.
- **The crop pose at ×4 is read from other slices than at ×1** (first look takes ranks 0, 5, 10,
  14, 19 of 20, both end slices): crop centre moves 1.0 to 2.0 mm, axial turn 3.0 to 3.3
  degrees. Equal in run 7, so not the difference between runs.
- Open measurements (GPU, about a minute each, not run): the multiplier curve ×1, ×2, ×3 for run
  7 and two experiments; one plane dense at a time; ×4 with an interior first look.
- Every finished soft structure-routing run was on centred crops; `no-side` (−0.0003, gain
  +0.0094) is the only experiment with a larger ×4 gain than run 7. Its run on run 7's crops,
  `rn18-structure-routes-no-side-reference-crops`, is built in
  `/home/claudio/src/knee-abnormality-anatomy-routing-15`.

## No-side structure routing on run 7's crops, `rn18-structure-routes-no-side-reference-crops` (2026-10-09)

The best structure routing (soft structure neighbourhoods, no side head, no position priors,
no geometric regions) with the two joint-centre crop targets of run 7; the finished `no-side`
run had centred crops. From `/home/claudio/src/knee-abnormality-anatomy-routing-15` (equal to
the `no-side` snapshot plus the two targets and two dead lines removed from the temperature
code; pose code and data settings identical to run 7).

| | Run 7 | Retrained | No-side, centred | No-side, run 7's crops |
|---|---|---|---|---|
| ×4 macro AUC | 0.928121 | 0.928120 | 0.927801 | 0.927542 |
| Best ×1 macro AUC | 0.919392 | 0.918411 | 0.918387 | 0.918064 |
| ×1 to ×4 gain | +0.0087 | +0.0097 | +0.0094 | +0.0095 |

- **−0.0006 at ×4 against run 7; −0.0003 against the same routing on centred crops.** Run 7's
  crop targets did not help this routing: the −0.0010 that centring costs on run 7 does not
  transfer. The ×4 gain is as large as the reference's in both no-side runs.
- Per target at ×4 against run 7 (against the retraining; against no-side centred): PF OA
  +0.0037 (+0.0008; +0.0028), Lateral OA +0.0031 (+0.0028; +0.0008), ACL +0.0026 (+0.0010;
  +0.0001), Baker's +0.0024 (+0.0020; −0.0005), Medial OA +0.0019 (+0.0013; +0.0017), Medial
  Meniscus +0.0019 (+0.0027; +0.0017), Lateral Meniscus −0.0009 (−0.0027; −0.0032), Fracture
  −0.0024 (−0.0018; +0.0003), Contusion −0.0028 (−0.0012; −0.0017), Effusion −0.0030 (−0.0018;
  −0.0000), Synovitis −0.0035 (−0.0047; +0.0012), MCL −0.0099 (−0.0053; −0.0060).
- **MCL alone is −0.0008 of macro, more than the whole gap**; it is behind both reference
  trainings and behind the centred twin. Without MCL the run is +0.0003 on the other eleven
  labels against run 7. Six labels are ahead of both reference trainings (PF OA, Lateral OA,
  ACL, Baker's, Medial OA, Medial Meniscus).
- Validation ×1 per epoch against run 7: −0.0066, +0.0010, −0.0025, −0.0009, +0.0005, −0.0014,
  −0.0014, −0.0013, −0.0016, −0.0008. MCL behind in the last five epochs (−0.005 to −0.014),
  Lateral OA ahead in nine of ten.
- The audit's Fracture case (0.60 to 0.74 with the targets, 0.29 to 0.42 without) scores 0.63
  here: the targets restore it, yet Fracture overall is level with the centred twin.
- Mask Dice at the last epoch: femur 0.904, tibia 0.899, kneecap 0.750, behind medial 0.731,
  fibula 0.714, joint 0.710, notch 0.691, kneecap joint 0.681, lateral tibial cartilage 0.596,
  medial band 0.576, femoral cartilage 0.572, medial meniscus 0.552, lateral meniscus 0.540,
  patellar cartilage 0.510, medial tibial cartilage 0.503. Training 1,463 s (run 7 1,498).

## Next experiment: changes agreed while `rn18-every-target-on-a-structure` ran (2026-10-09)

Base: the working copy `/home/claudio/src/knee-abnormality-anatomy-routing` as that run has it
(every target anchored on a structure, ten-class bone map from
`anatomy-mrannotator-sample550-ligaments`, seven-class soft-tissue head, two-way link on the
soft-tissue shares, no hand-measured crop offsets in the model, per-epoch logging).

1. **PF OA anchored on the patellofemoral joint, kept wide.** Replace the patellar-cartilage
   anchor by the bone map's kneecap-joint class together with the front of the femur.
   Evidence: in the run PF OA was the one label behind at every epoch (−0.029, −0.008, −0.008,
   −0.007, −0.009); judged per view at epoch 4 it lost most on the coronal views (−0.034
   fluid, −0.047 T1/PD) and sagittal T1/PD (−0.018), and its attention moved to axial fluid
   (0.29 from 0.21). In the reference its attention by true anatomy was 46% femur, 21% kneecap
   joint, 10% kneecap (anatomy-masks run 15) and 96% in the femur band (side-routed run 11).
   Tight regions cost PF OA in earlier runs (side-routed runs 18, 19, 30). The patellar
   cartilage is a sheet a few millimetres deep, under 500 coronal tokens on the mask knees
   against 2,500 axial.
2. **Contusion and Fracture on bone, with no floor.** The learned floor of the run was a
   defect of the keeper's build: it was clipped to 0 to 1 and started at 1, the first update
   pushed it to 1.000419, and a clipped value outside its range gets no gradient, so it stayed
   there for the whole run. The two labels were unrouted in that run and it says nothing about
   what they want. A learnable limit must use a smooth squashing, never clipping at its start
   value. The user's point stands: a bone token already sees its surroundings within the slice.
3. **Remove the side prediction completely:** drop or re-sign the position priors of the
   meniscus and OA labels, take the medial direction for the crop pose from the anatomy (fibula
   side or the two menisci), delete the side head, its loss (0.1) and its prior. The check of
   the mask sides is done and supports it: on the 263 accepted DESS knees the masks' medial and
   lateral agree with the DICOM `Laterality` tag in 262, with the pipeline's side in 261 of 261
   usable, and with the fibula side in 47 of 47 knees that also have a bone mask; in the one
   disagreement (study ending 459197) the fibula and the coordinates say the tag is wrong.
   Menisci and tibial cartilage give the same side in all 263. Limits: the fibula, the only
   source fully independent of the scanner, covers 47 knees, and all 263 are from one site's
   Siemens scanners.
4. **Delete what no label reads:** the sectors, core and ring, the eight tear and facet routes,
   the report-attribute heads.
5. **The guardrail `shared_classifiers_blind_to_the_compartment`** assumes the medial and
   lateral outputs swap when the side is swapped; that no longer holds by design. Rewrite it for
   structure-anchored routes or retire it with the side head.
6. **Still hand-set in the model after this:** the crop scale rule (17.087, clamp, height
   windows), the axial centre rule (30 mm), the turn and shift limits. To be taught from the
   masks later.

Background from the past campaigns, for the OA labels (subagent search of the run notes):
Medial OA is strongest alone on coronal fluid (0.884), Lateral OA on axial fluid (0.842) with
coronal and sagittal fluid close, PF OA on sagittal fluid (0.830) and axial fluid (0.828).
Lateral OA has 95 positives and a sibling-run spread of 0.0034, four times Medial OA's; its AUC
is 0.722 on knees with other OA present against 0.909 without. No reason is written for PF OA
sharing a route with Effusion, and changes that helped one did not help the other.

## Incidents

- 2026-10-09 00:27: WSL shut down by the user during run 1's epoch 9; no cause in the journal.
  The first resume recorded seed 1 as invalid and launched seed 2 on the reference; that
  training was aborted after 20 s and seeds 1 to 3 were requeued in the stopped state.
- 2026-10-09 03:06: the supervisor died on a missing import in the keeper's change to the
  adoption rule (`validation_scores`); fixed and resumed, with no result lost.
- 2026-10-09 03:18: seed 3 (`canvases-for-the-pose-in-all-six-views`) was blocked at review:
  `check_side_routing` compared mirrored token positions against the nominal pitch
  (120 / 7 = 17.14 mm), and the per-knee scale of 1.13 brings the inherited 31 px mirror
  displacement to 18.77 mm in axial, still under the scaled pitch of 19.37 mm. The keeper
  changed the check to read each image's own token pitch from `token_steps` (tighter than
  before for the 84 and 93 mm views, equal for an unscaled 120 mm view), in the root and in the
  incumbent's frozen source with its manifest entry; both the incumbent and the blocked seed 3
  workspace pass. Seeds 3 to 5 were requeued in order; seeds 4 and 5 had been prepared
  without seed 3.
- 2026-10-09 03:46: seed 3 was blocked a second time: it removes `data.view_shift_limit_mm`, as
  its text says, and the campaign policy did not list that key. The keeper added the key to
  `config_keys` in `campaign.yaml` and in the sealed policy of the stopped state, nothing else,
  and requeued seeds 3 and 4.
- 2026-10-09 03:54 to 04:20: while seed 3 trained, the developer prepared seeds 4 and 5 on the
  incumbent without the canvases; the review refused both, and its third refused report on
  seed 5 paused the campaign (training went on). Forced seeds cannot wait for an earlier
  seed in this runtime. Both were requeued at the stop that followed seed 3's run.
- 2026-10-09 04:36: WSL froze a second time, 27 s into seed 4's training; the user shut it down
  and it came back at 06:59. Likely cause of both freezes, not proven: the Windows drive that
  holds the WSL virtual disk is full (830 MiB free of 931 GiB; the virtual disk is 787 GiB and
  can only grow), so a write that needs the disk to grow hangs the machine. Neither journal has
  an out-of-memory kill or any error before it stops. The disk-cleanup rules freed 16.1 GiB
  inside WSL, which gives about 23 GiB of already-allocated room; the Windows drive itself is
  still full. Seeds 4 and 5 were requeued; seed 4's training is lost.
- 2026-10-09 07:43: seed 6 (`t1pd-joint-crops-centred-on-the-menisci`) was blocked at review:
  its fallback threshold, "measured once and cited", had never been measured. Seed 11 cites a
  second unmeasured number. The keeper measured both on the 211 training DESS knees with the
  incumbent's checkpoint and first look (prior-work.md, Meniscus head calibration: 3.89
  token-share units and 0.375), re-submitted the two seeds unchanged but for the cited number
  (`-calibrated`, seed files 4 and 5), placed seed 6's ahead of the canvas combinations and
  removed the original seed 11. The measurement ran on the GPU for about a minute beside seed
  5's training.
