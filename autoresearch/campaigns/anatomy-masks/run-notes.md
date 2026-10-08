# Anatomy masks: keeper's run notes

Written by the keeper after each result: both multipliers against the incumbent, the per-target
table, the map and crop diagnostics, repairs and adoption decisions.

## Started 2026-10-08 00:01 CEST (Codex)

Reference `rn18-anatomy-crops-sample550` (×4 0.925885, ×1 0.917130, training 1,863 s, ×4
prediction 55 s). Budget 0.5 to 1.2 of the reference's training time (932 to 2,236 s). Seven
forced seeds; the first two are adopted whatever their result (contract, Adoption).

- **Repair before the first run:** the first start (2026-10-07 23:53) paused at calibration,
  "Baseline data/preprocessing identity differs from historical control". The worktree's
  `modeling/model.py` carried another session's edit of 23:43 (the bicubic cut and the map
  centre, for a prediction on the existing weights), so the source hash was not the reference's.
  The edit is kept as a patch and a full copy in `outputs/analysis/anatomy-masks-prestart/`,
  `model.py` was restored from the reference's `source_snapshot.zip`, the first start's state
  moved to `outputs/autoresearch/anatomy-masks-aborted-start-20261007`, and the campaign started
  again; calibration passed.
- **Seed added by the user (00:16):** `the-first-look-on-the-whole-canvas-at-224-px`
  (`first-look-ideas.yaml`), forced, eighth in the queue.

## 1. `turned-crops-cut-with-bicubic-interpolation`: adopted

Accepted by the runtime's own verdict, which the contract's exception would have forced anyway.

| | Reference | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.925885 | 0.927039 | +0.001154 (95% +0.000016 to +0.002433) |
| ×1 | 0.917130 | 0.917771 | +0.000641 |
| Training | 1,863 s | 1,674 s | 0.90 |
| ×4 prediction | 55.2 s | 52.7 s | 0.95 |

Per target at ×4 (delta, paired 95% interval):

| Target | Delta | Interval |
|---|---:|---|
| Fracture | +0.0054 | +0.0009 to +0.0119 |
| Contusion | +0.0033 | −0.0001 to +0.0076 |
| Medial OA | +0.0021 | −0.0005 to +0.0050 |
| Lateral Meniscus | +0.0019 | −0.0024 to +0.0066 |
| ACL | +0.0012 | −0.0017 to +0.0042 |
| MCL | +0.0011 | −0.0040 to +0.0071 |
| Medial Meniscus | +0.0004 | −0.0021 to +0.0036 |
| Baker's | +0.0003 | −0.0013 to +0.0021 |
| Synovitis | +0.0002 | −0.0030 to +0.0033 |
| PF OA | −0.0002 | −0.0027 to +0.0021 |
| Effusion | −0.0003 | −0.0018 to +0.0012 |
| Lateral OA | −0.0014 | −0.0059 to +0.0029 |

- **Map and crops, last epoch on validation, unchanged:** plane error 1.72 mm (1.73); overlap
  femur 0.904, tibia 0.898, joint 0.756, kneecap 0.733, kneecap joint 0.669; turn spread
  sagittal 3.57° and coronal 3.41°; mean shift 11.8 px. Every study had turned crops.
- **Reading:** the three targets the seed named as losing fine detail (Lateral Meniscus,
  Contusion, Fracture) all rose, Fracture with an interval above zero; one run, and only
  Fracture's interval excludes zero. The seed's own test (Lateral Meniscus up by one sibling SD,
  0.0024, at ×1) is not read here: the ×1 per-target table was not computed.
- **Timing:** the reference's 1,863 s included a job beside it, so the 0.90 is not a speed-up.

## 2. `joint-centre-from-the-map-instead-of-the-spots`: adopted by the contract

The runtime's verdict was reject (80% interval −0.000810 to +0.000896 includes zero); adopted by
hand under the contract's exception for the second forced seed. Valid run, activation and budget
checks passed.

| | Incumbent (seed 1) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927039 | 0.927083 | +0.000044 (95% −0.001247 to +0.001333) |
| ×1 | 0.917771 | 0.917666 | −0.000105 |
| Training | 1,674 s | 1,687 s | 1.01 |
| ×4 prediction | 52.7 s | 53.1 s | 1.01 |

Per target at ×4 (delta, paired 95% interval), none with an interval that excludes zero:

| Target | Delta | Interval |
|---|---:|---|
| Lateral OA | +0.0024 | −0.0023 to +0.0074 |
| Lateral Meniscus | +0.0013 | −0.0042 to +0.0061 |
| MCL | +0.0009 | −0.0057 to +0.0077 |
| ACL | +0.0008 | −0.0045 to +0.0063 |
| Effusion | +0.0004 | −0.0018 to +0.0025 |
| PF OA | −0.0000 | −0.0036 to +0.0036 |
| Synovitis | −0.0000 | −0.0040 to +0.0039 |
| Baker's | −0.0001 | −0.0017 to +0.0015 |
| Contusion | −0.0009 | −0.0036 to +0.0015 |
| Fracture | −0.0012 | −0.0070 to +0.0046 |
| Medial OA | −0.0013 | −0.0046 to +0.0019 |
| Medial Meniscus | −0.0017 | −0.0039 to +0.0005 |

- **Centre, last epoch on validation:** the map's centre was applied on every knee and moved the
  joint centre 3.1 mm on average from the spots' one. Against the plateau centre its error has a
  spread of 1.9 mm across and 1.6 mm front to back, with means of 0.9 and 1.7 mm left after the
  fixed offset.
- **Map and crops:** plane error 1.76 mm (1.72); overlap joint 0.758, kneecap 0.734; turn spread
  sagittal 3.58° and coronal 3.38°; mean shift 11.8 px (11.8). Shift spread per axis, logged for
  the first time: coronal 9.3 px across and 17.6 px down, sagittal 10.5 and 18.5 px.
- **Reading:** level with the incumbent at both multipliers, one run. The 1.7 mm mean error
  front to back says the fixed offset measured on one batch does not hold on the validation
  knees; a lead for a later seed is to remeasure it on more knees.
- **Lesson for the keeper:** the stop before the end leaves the run awaiting the orchestrator's
  decision, and an adoption is refused until that decision exists. On resume the decision came
  at once and the third seed's training launched on the incumbent without the second seed. It
  was cancelled at epoch 2 (`stop --abort`, recorded invalid), the adoption applied, and the
  seed requeued as `a-tenth-of-the-map-gradient-to-the-backbone-2`
  (`gradient-requeue-ideas.yaml`), second in the queue behind the Dice seed, which the developer
  had already taken.

## 3. `a-tenth-of-the-map-gradient-to-the-backbone`: invalid, requeued

Cancelled at epoch 2 because it was built on the incumbent without the second seed (see above).
No result. Its requeue runs fifth.

## 4. `the-map-loss-without-its-dice-term`: not adopted

Below the incumbent at both multipliers.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.926748 | −0.000335 (95% −0.001612 to +0.000928) |
| ×1 | 0.917666 | 0.917182 | −0.000484 |
| Training | 1,687 s | 1,679 s | 1.00 |
| ×4 prediction | 53.1 s | 53.0 s | 1.00 |

Per target at ×4 (delta, paired 95% interval), none with an interval that excludes zero:

| Target | Delta | Interval |
|---|---:|---|
| Fracture | +0.0050 | −0.0019 to +0.0141 |
| PF OA | +0.0009 | −0.0020 to +0.0036 |
| Synovitis | +0.0008 | −0.0028 to +0.0045 |
| Baker's | +0.0004 | −0.0010 to +0.0021 |
| Lateral OA | +0.0002 | −0.0029 to +0.0034 |
| Effusion | −0.0002 | −0.0018 to +0.0015 |
| Medial Meniscus | −0.0003 | −0.0046 to +0.0027 |
| Medial OA | −0.0008 | −0.0031 to +0.0013 |
| ACL | −0.0015 | −0.0039 to +0.0010 |
| Lateral Meniscus | −0.0022 | −0.0071 to +0.0024 |
| Contusion | −0.0023 | −0.0078 to +0.0011 |
| MCL | −0.0040 | −0.0096 to +0.0013 |

- **Map, last epoch on validation:** without Dice the overlaps fall, most on the small classes:
  kneecap 0.692 (0.734), kneecap joint 0.643 (0.669), joint 0.747 (0.758), femur 0.897 (0.904),
  tibia 0.893 (0.897). The plane error improves, 1.59 mm (1.76); the centre's mean error front
  to back is 1.9 mm (1.7).
- **The three targets that pay for the map:** Synovitis +0.0008, PF OA +0.0009, Effusion
  −0.0002, all under one sibling SD (Synovitis 0.0022). The seed's question, whether Dice costs
  Synovitis more than it gains elsewhere, gets no sign of a cost from this run.
- **Reading:** one run, macro inside the ±0.0015 rerun band at both multipliers. Dice stays. The
  one thing the run gives is that cross-entropy alone fits the plane better (−0.16 mm) with a
  worse kneecap, so a lighter Dice weight is an open step between the two.

## 5. `a-tenth-of-the-map-gradient-to-the-backbone-2`: not adopted

Level at ×4, below at ×1.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.927078 | −0.000004 (95% −0.001608 to +0.001684) |
| ×1 | 0.917666 | 0.916223 | −0.001443 |
| Training | 1,687 s | 1,692 s | 1.00 |
| ×4 prediction | 53.1 s | 53.3 s | 1.00 |

Per target at ×4 (delta, paired 95% interval; * interval excludes zero):

| Target | Delta | Interval |
|---|---:|---|
| Contusion | +0.0057 * | +0.0005 to +0.0135 |
| MCL | +0.0041 | −0.0058 to +0.0152 |
| ACL | +0.0036 * | +0.0002 to +0.0072 |
| Medial Meniscus | +0.0028 * | +0.0001 to +0.0053 |
| Medial OA | −0.0002 | −0.0033 to +0.0027 |
| Effusion | −0.0004 | −0.0030 to +0.0023 |
| Synovitis | −0.0009 | −0.0062 to +0.0045 |
| Fracture | −0.0014 | −0.0105 to +0.0081 |
| Lateral OA | −0.0022 | −0.0072 to +0.0029 |
| PF OA | −0.0027 | −0.0076 to +0.0020 |
| Baker's | −0.0036 * | −0.0070 to −0.0005 |
| Lateral Meniscus | −0.0049 | −0.0102 to +0.0003 |

- **Map, last epoch on validation:** much worse with a tenth of the gradient: femur 0.852
  (0.904), tibia 0.845 (0.897), joint 0.694 (0.758), kneecap 0.619 (0.734), kneecap joint 0.592
  (0.669); plane error 1.95 mm (1.76); centre's mean error front to back 1.9 mm (1.7).
- **The three targets that pay for the map:** Synovitis −0.0009, PF OA −0.0027, Effusion −0.0004.
  The seed expected Synovitis up by one sibling SD (0.0022); it did not rise. Its test fails.
- **Reading:** one run. The targets split instead of moving together: ACL, Medial Meniscus and
  Contusion up with intervals above zero, Baker's down with an interval below zero, Lateral
  Meniscus down 0.0049 at the edge of its interval. The map lost 5 to 11 points of overlap, and
  the sections, the plane and the centre are read from that map, so the losses on routed targets
  (PF OA, Lateral OA, Lateral Meniscus) fit a worse map rather than a freer backbone. A lead:
  keep the full gradient for the map and give the damped copy only to the label tokens, or a
  third instead of a tenth.

## 6. `the-first-look-on-a-third-of-the-slices`: not adopted

Below the incumbent at both multipliers, and not cheaper at prediction.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.925555 | −0.001528 (95% −0.003220 to +0.000246; 80% −0.002672 to −0.000356) |
| ×1 | 0.917666 | 0.917029 | −0.000637 |
| Training | 1,687 s | 1,639 s | 0.97 |
| ×4 prediction | 53.1 s | 53.3 s | 1.00 |

Per target at ×4 (delta, paired 95% interval; * interval excludes zero):

| Target | Delta | Interval |
|---|---:|---|
| MCL | +0.0081 | −0.0002 to +0.0177 |
| ACL | +0.0022 | −0.0013 to +0.0064 |
| Baker's | +0.0005 | −0.0021 to +0.0035 |
| Medial OA | −0.0001 | −0.0042 to +0.0036 |
| PF OA | −0.0010 | −0.0071 to +0.0047 |
| Effusion | −0.0017 | −0.0048 to +0.0013 |
| Synovitis | −0.0018 | −0.0083 to +0.0041 |
| Medial Meniscus | −0.0024 | −0.0060 to +0.0011 |
| Contusion | −0.0048 | −0.0157 to +0.0032 |
| Lateral OA | −0.0050 | −0.0113 to +0.0007 |
| Fracture | −0.0061 * | −0.0116 to −0.0009 |
| Lateral Meniscus | −0.0061 | −0.0152 to +0.0022 |

- **First look:** 10.6 slices read per knee in training and 14.9 left out (0.42 of them, not a
  third: a view of five or six slices keeps two).
- **Pose, last epoch on validation:** the pose got noisier. Turn spread sagittal 4.19° (3.58)
  and coronal 5.38° (3.38); mean shift 16.3 px (11.8). The plane error logged against the
  targets is unchanged, 1.74 mm (1.76), and so are the centre's errors (1.99 mm across, 1.61 mm
  front to back) and the map's overlaps (joint 0.754, kneecap 0.724).
- **Cost:** training 0.97 of the incumbent, where the seed expected 0.9; ×4 prediction did not
  get faster at all.
- **Reading:** one run; the 80% interval at ×4 lies below zero. Two slices per series are too
  few for the turn: the coronal turn's spread grew by 2° while the plane error stayed, so the
  logged plane error is not measuring the pose the crops receive. Two things to check before any
  retake: why prediction time did not fall (whether the ×4 pass takes the reduced first look),
  and where the logged plane error is read from. A retake that keeps every slice of the
  sagittal and coronal series and thins only the axial ones would keep the turn's inputs whole.

## 7. `fluid-sagittal-crops-moved-forward-for-the-kneecap`: not adopted

Below the incumbent at both multipliers.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.926591 | −0.000491 (95% −0.001605 to +0.000652) |
| ×1 | 0.917666 | 0.916476 | −0.001190 |
| Training | 1,687 s | 1,699 s | 1.01 |
| ×4 prediction | 53.1 s | 53.4 s | 1.01 |

Per target at ×4 (delta, paired 95% interval; * interval excludes zero):

| Target | Delta | Interval |
|---|---:|---|
| MCL | +0.0023 | −0.0019 to +0.0066 |
| Medial Meniscus | +0.0013 | −0.0008 to +0.0036 |
| ACL | +0.0012 | −0.0005 to +0.0030 |
| PF OA | +0.0010 | −0.0029 to +0.0046 |
| Effusion | −0.0003 | −0.0023 to +0.0019 |
| Lateral OA | −0.0003 | −0.0036 to +0.0032 |
| Lateral Meniscus | −0.0003 | −0.0061 to +0.0055 |
| Baker's | −0.0013 | −0.0039 to +0.0011 |
| Medial OA | −0.0019 | −0.0041 to +0.0004 |
| Fracture | −0.0019 | −0.0095 to +0.0046 |
| Contusion | −0.0021 | −0.0075 to +0.0022 |
| Synovitis | −0.0036 * | −0.0074 to −0.0001 |

- **The move, last epoch on validation:** the fluid sagittal crops moved 23.0 px across on
  average (sd 8.1), 10.5 mm (sd 1.4) from the incumbent's placement, and **38% of the fluid
  sagittal series sat at the shift limit** (2 to 4% before). The limit is three quarters of the
  17 mm margin, 12.75 mm, and the seed adds 8.6 mm to shifts that already had a spread of 4 mm.
- **Kneecap in the crop:** kneecap tokens 2.0% and kneecap-joint tokens 2.6% of the fluid
  sagittal tokens per image; the incumbent does not log these, so the seed's comparison could
  not be made. Map and plane unchanged (plane error 1.78 mm, kneecap overlap 0.733).
- **The seed's test:** PF OA +0.0010 and Effusion −0.0003, both under one sibling SD; Synovitis
  −0.0036 with an interval just below zero. Not shown wrong by its own rule (PF OA and Effusion
  not both below), not shown right either.
- **Reading:** one run. The move was clipped for more than a third of the series, so the crop
  the seed describes was not what many knees received. The lead is the same move with a margin
  that holds it: `data.view_shift_limit_mm` is one of the two config keys the campaign may
  change, and 17 mm leaves 12.75 mm where this placement needs 8.6 mm plus the pose's own shift
  (sd 4 to 7 mm).
- **Earlier attempt:** the first candidate (`…002848Z-ac858b1a`) was sent back at review and
  rebuilt; only the second trained.

## 8. `joint-gap-per-side-as-a-target`: not adopted

Below the incumbent at both multipliers.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.926501 | −0.000582 (95% −0.001957 to +0.000715) |
| ×1 | 0.917666 | 0.916789 | −0.000877 |
| Training | 1,687 s | 1,698 s | 1.01 |
| ×4 prediction | 53.1 s | 54.4 s | 1.02 |

Per target at ×4 (delta, paired 95% interval; * interval excludes zero):

| Target | Delta | Interval |
|---|---:|---|
| Medial Meniscus | +0.0020 | −0.0003 to +0.0042 |
| Lateral Meniscus | +0.0016 | −0.0029 to +0.0068 |
| MCL | +0.0015 | −0.0055 to +0.0086 |
| Lateral OA | +0.0006 | −0.0044 to +0.0057 |
| PF OA | +0.0005 | −0.0030 to +0.0039 |
| Baker's | −0.0004 | −0.0026 to +0.0019 |
| Effusion | −0.0009 | −0.0030 to +0.0013 |
| ACL | −0.0012 | −0.0049 to +0.0021 |
| Contusion | −0.0017 | −0.0058 to +0.0015 |
| Medial OA | −0.0020 | −0.0047 to +0.0005 |
| Fracture | −0.0032 | −0.0108 to +0.0049 |
| Synovitis | −0.0037 * | −0.0076 to −0.0000 |

- **The gap head learned something, last epoch on validation:** mean absolute error 0.58 mm
  against 0.72 mm for predicting the training mean (medial 0.53 against 0.67, lateral 0.62
  against 0.77), on 154 supervised gaps; 368 gaps supervised in training. A fifth below the
  baseline: the gap varies by well under a millimetre between knees, and the head recovers
  little of that.
- **The seed's targets:** Medial OA −0.0020 and Lateral OA +0.0006, neither beyond one sibling
  SD in either direction, so its test (both OA targets below by more than one SD) does not
  reject it, and nothing supports it. Map and plane unchanged (plane error 1.78 mm).
- **Synovitis** fell 0.0037 with an interval touching zero, the second run in a row where a
  new auxiliary reading costs it (the kneecap crop seed: −0.0036).
- **Reading:** one run, macro inside the rerun band. The gap in mm is a narrow target (baseline
  error 0.72 mm on a token pitch of several mm); a lead is the gap relative to the knee's own
  other side, or narrowing as a class, instead of millimetres.

## 9. `the-first-look-on-the-whole-canvas-at-224-px` (user's seed): not adopted

Level at ×4, below at ×1.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.926969 | −0.000114 (95% −0.001495 to +0.001255) |
| ×1 | 0.917666 | 0.917010 | −0.000656 |
| Training | 1,687 s | 1,718 s | 1.02 |
| ×4 prediction | 53.1 s | 52.7 s | 0.99 |

Per target at ×4 (delta, paired 95% interval; * interval excludes zero):

| Target | Delta | Interval |
|---|---:|---|
| MCL | +0.0024 | −0.0050 to +0.0092 |
| ACL | +0.0018 | −0.0010 to +0.0048 |
| Medial Meniscus | +0.0012 | −0.0013 to +0.0036 |
| Medial OA | +0.0010 | −0.0017 to +0.0039 |
| PF OA | +0.0006 | −0.0027 to +0.0040 |
| Contusion | +0.0006 | −0.0043 to +0.0058 |
| Fracture | +0.0003 | −0.0080 to +0.0087 |
| Baker's | +0.0001 | −0.0016 to +0.0018 |
| Lateral OA | +0.0001 | −0.0037 to +0.0038 |
| Effusion | −0.0007 | −0.0023 to +0.0010 |
| Synovitis | −0.0030 | −0.0068 to +0.0007 |
| Lateral Meniscus | −0.0056 * | −0.0111 to −0.0007 |

- **The look, last epoch on validation:** 0.547 mm per pixel sagittal and 0.550 coronal (axial
  0.536); 27% of the first look's tokens lie in the margin the central crop never showed.
- **The pose it gives is almost the incumbent's:** plane error 1.73 mm (1.76); centre moved
  3.3 mm from the spots (3.1), error spread 1.97 mm across and 1.56 mm front to back (1.94,
  1.56); turn spread sagittal 3.73° (3.58) and coronal 3.69° (3.38); mean shift 12.1 px (11.8).
  The map, trained only on the posed crops at 0.375 and 0.417 mm per pixel, reads the resized
  canvas as well as the central crop, without ever being trained at that scale.
- **Shift limit, logged here for the first time:** 13% of sagittal and 11% of coronal series at
  the limit on validation. The 2 to 4% quoted in prior-work.md was for the true pose; the
  model's pose hits the limit three times as often.
- **Kneecap in the first look:** 1.0% of sagittal first-look tokens are kneecap and 1.7%
  kneecap joint.
- **Reading:** one run, a starting point. The wider look changed neither the pose nor the map,
  so on this run the central crop was not what limited the pose. Ways on: (1) the limit is the
  binding thing, 11 to 13% of series: the two field seeds grow the margin to 22.5 mm and will
  show the share at the limit with more canvas; (2) train the map on the resized canvas as well,
  so the look is read at the scale it is shown; (3) use what only the wide look sees, the
  kneecap's place in front of the joint, to set the sagittal crop's forward shift per knee
  instead of the fixed 8.6 mm of the kneecap seed. Lateral Meniscus fell 0.0056 with an interval
  below zero and no change in pose to explain it; with twelve targets and one run that needs a
  second reading before it is taken as an effect.

## 10. `sagittal-crops-a-ninth-wider` (user's seed): not adopted

Below the incumbent at both multipliers.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.926466 | −0.000617 (95% −0.001972 to +0.000785) |
| ×1 | 0.917666 | 0.917169 | −0.000497 |
| Training | 1,687 s | 1,800 s | 1.07 |
| ×4 prediction | 53.1 s | 57.0 s | 1.07 |

Per target at ×4 (delta, paired 95% interval; * interval excludes zero):

| Target | Delta | Interval |
|---|---:|---|
| Baker's | +0.0024 * | +0.0001 to +0.0054 |
| Fracture | +0.0017 | −0.0029 to +0.0067 |
| PF OA | +0.0017 | −0.0019 to +0.0052 |
| Medial Meniscus | +0.0011 | −0.0022 to +0.0038 |
| Contusion | +0.0007 | −0.0026 to +0.0051 |
| Synovitis | +0.0004 | −0.0041 to +0.0055 |
| Medial OA | +0.0004 | −0.0024 to +0.0031 |
| ACL | −0.0012 | −0.0060 to +0.0034 |
| Effusion | −0.0017 | −0.0043 to +0.0005 |
| Lateral OA | −0.0023 | −0.0063 to +0.0018 |
| Lateral Meniscus | −0.0029 | −0.0107 to +0.0040 |
| MCL | −0.0076 * | −0.0154 to −0.0006 |

- **The crop, on validation:** zoom 1.111 applied; mean sagittal field 98.4 mm (the two series
  types at 93.3 and 103.7), coronal 89.2 mm and axial 120.0 mm unchanged.
- **Kneecap in the sagittal crops:** 1.8% kneecap and 2.4% kneecap-joint tokens per image. The
  kneecap seed, which moved the fluid crop forward at the old field, logged 2.0% and 2.6% for
  the fluid series alone; the incumbent logs neither, so the gain in kneecap from the zoom is
  not measured by this run.
- **Pose and limit:** turns and shifts as the incumbent (sagittal turn spread 3.56°, shift
  spread 7.1 mm down and 4.1 mm across); 15% of sagittal series at the shift limit, against 13%
  in the whole-canvas run with the 17 mm margin. The larger margin was spent on the wider crop,
  as designed, so the share at the limit did not fall.
- **Map:** slightly lower at the coarser pixel: kneecap 0.725 (0.734), kneecap joint 0.663
  (0.669), joint 0.753 (0.758); plane error 1.77 mm (1.76).
- **The seed's targets:** the ones the wider field was for moved its way: Baker's +0.0024 with
  an interval above zero (the back of the joint enters), PF OA +0.0017. Effusion −0.0017. The
  ones that read fine detail fell: Lateral Meniscus −0.0029, ACL −0.0012, neither beyond its
  interval. Its rule (PF OA, Effusion and Baker's all below, or Lateral Meniscus and ACL both
  below by more than one sibling SD) does not reject it.
- **MCL −0.0076** with an interval below zero, on 47 positives, from a change to the sagittal
  crops, which do not show the MCL: read as noise or as attention moving between views until a
  second run says otherwise.
- **Cost:** 1.07 at both training and prediction, from the larger canvases.
- **Reading:** one run, macro inside the rerun band. The split is what the seed predicted in
  direction: the periphery gains, the detail pays. Ways on: widen only the fluid sagittal series
  (Baker's, PF OA and Effusion are read there) and leave T1/PD at 84 mm for the menisci and ACL;
  and log the kneecap token share on the incumbent so the next crop seed has its comparison.

## 11. `coronal-crops-a-ninth-wider` (user's seed): not adopted

Above the incumbent at ×4, below at ×1; the contract needs both.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.927236 | +0.000153 (95% −0.001246 to +0.001546) |
| ×1 | 0.917666 | 0.917294 | −0.000372 |
| Training | 1,687 s | 1,796 s | 1.06 |
| ×4 prediction | 53.1 s | 55.8 s | 1.05 |

Per target at ×4 (delta, paired 95% interval; * interval excludes zero):

| Target | Delta | Interval |
|---|---:|---|
| Medial Meniscus | +0.0017 | −0.0004 to +0.0041 |
| MCL | +0.0015 | −0.0039 to +0.0066 |
| Synovitis | +0.0015 | −0.0019 to +0.0052 |
| Contusion | +0.0013 | −0.0046 to +0.0065 |
| Effusion | +0.0005 | −0.0013 to +0.0023 |
| Lateral OA | −0.0001 | −0.0039 to +0.0039 |
| Medial OA | −0.0002 | −0.0034 to +0.0033 |
| ACL | −0.0002 | −0.0050 to +0.0042 |
| PF OA | −0.0003 | −0.0033 to +0.0027 |
| Fracture | −0.0005 | −0.0106 to +0.0088 |
| Lateral Meniscus | −0.0008 | −0.0056 to +0.0042 |
| Baker's | −0.0027 * | −0.0054 to −0.0003 |

- **The crop, on validation:** zoom 1.111 applied; mean coronal field 99.1 mm (93.3 and 103.7),
  sagittal 88.6 mm and axial 120.0 mm unchanged.
- **Pose and limit:** as the incumbent (coronal turn spread 3.32°, shift spread 6.8 mm down and
  3.8 mm across); 13% of coronal series at the shift limit. Map unchanged (plane error 1.77 mm,
  kneecap 0.736).
- **The seed's targets:** MCL +0.0015 and Contusion +0.0013, the sides of the joint, both under
  one sibling SD; the menisci did not pay for the coarser pixel (Medial +0.0017, Lateral
  −0.0008), unlike in the sagittal run. Its rule (MCL not above, or both menisci below by more
  than one SD) does not reject it.
- **Baker's −0.0027** with an interval below zero; in the sagittal run it was +0.0024 with an
  interval above zero. The two wider-field runs moved it in opposite directions by the same
  amount, which is what two draws of one noisy target look like, so neither is read as an effect.
- **Cost:** 1.06 training, 1.05 prediction.
- **Reading:** one run, level with the incumbent (macro inside the rerun band at both
  multipliers), and the better of the two field seeds: nothing fell beyond its interval except
  Baker's, and the fine-detail targets held. Ways on: the coronal field on the fluid series
  only (the 103.7 mm crop is where the bone plus 5 mm goes from 55% to 92% of knees), which
  also halves the extra cost; or coronal wider together with the fluid sagittal wider, the two
  halves that did not hurt the menisci.

## 12. `posed-crops-keep-their-subpixel-shifts` (researcher's first idea): not adopted

Below the incumbent at both multipliers. The crop's shift and the training jitter are no longer
rounded to whole canvas pixels before the bicubic cut.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.926197 | −0.000885 (95% −0.001841 to +0.000163; 80% −0.001523 to −0.000215) |
| ×1 | 0.917666 | 0.917042 | −0.000624 |
| Training | 1,687 s | 1,700 s | 1.01 |
| ×4 prediction | 53.1 s | 52.8 s | 0.99 |

Per target at ×4 (delta, paired 95% interval; * interval excludes zero):

| Target | Delta | Interval |
|---|---:|---|
| Fracture | +0.0024 | −0.0036 to +0.0094 |
| Medial Meniscus | +0.0009 | −0.0004 to +0.0023 |
| Baker's | +0.0006 | −0.0014 to +0.0027 |
| Medial OA | −0.0003 | −0.0023 to +0.0017 |
| Contusion | −0.0003 | −0.0029 to +0.0017 |
| Effusion | −0.0005 | −0.0020 to +0.0011 |
| Synovitis | −0.0006 | −0.0034 to +0.0023 |
| Lateral OA | −0.0009 | −0.0044 to +0.0027 |
| ACL | −0.0011 | −0.0044 to +0.0016 |
| Lateral Meniscus | −0.0012 | −0.0046 to +0.0018 |
| PF OA | −0.0017 | −0.0040 to +0.0004 |
| MCL | −0.0078 * | −0.0147 to −0.0019 |

- **The change, on validation:** every crop now has a fractional shift; it differs from the
  rounded one by 0.38 px, 0.15 mm, on average. Pose and map as the incumbent (plane error
  1.73 mm, turn spreads 3.62° and 3.36°).
- **New diagnostic:** the angle left between the crop and the joint plane after posing is 2.35°
  on average in sagittal crops and 1.53° in coronal ones (spread 2.8° and 1.9°), with a mean of
  +1.0° sagittal. The posed crops are not level to within the plane's own error; part is the
  12° limit and part the pose read before the crop.
- **MCL across this campaign's runs** (×4 AUC, 47 positives): 0.9287 incumbent, then 0.9247,
  0.9329, 0.9368, 0.9310, 0.9302, 0.9311, 0.9211, 0.9302, 0.9209. A spread of about 0.005 from
  run to run, so the two drops of 0.008 with intervals below zero (this run and the wider
  sagittal crops) sit at the low end of that spread, about 1.5 spreads from the incumbent. The
  paired interval inside one run understates the run-to-run noise of this target.
- **Reading:** one run; without MCL the macro delta would be −0.0003. A shift of 0.15 mm is
  below what any of these targets resolves, so the run is closer to a rerun of the incumbent
  with a different crop grid than to a test of a mechanism, and it reads as one: ×4 −0.0009,
  ×1 −0.0006. That is a second yardstick for what a rerun gives here.

## 13. `soft-bone-shares-fit-the-joint-plane` (researcher): not adopted

Above the incumbent at ×4, below at ×1. The joint plane is fitted with every token weighted by
its bone share instead of only the tokens above half bone.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.927219 | +0.000136 (95% −0.001172 to +0.001399) |
| ×1 | 0.917666 | 0.916565 | −0.001101 |
| Training | 1,687 s | 1,691 s | 1.00 |
| ×4 prediction | 53.1 s | 53.0 s | 1.00 |

Per target at ×4 (delta, paired 95% interval), none with an interval that excludes zero:

| Target | Delta | Interval |
|---|---:|---|
| Fracture | +0.0036 | −0.0016 to +0.0098 |
| MCL | +0.0031 | −0.0059 to +0.0117 |
| Medial Meniscus | +0.0009 | −0.0012 to +0.0031 |
| ACL | +0.0006 | −0.0027 to +0.0044 |
| Effusion | +0.0005 | −0.0017 to +0.0027 |
| PF OA | +0.0002 | −0.0033 to +0.0037 |
| Synovitis | +0.0001 | −0.0035 to +0.0037 |
| Contusion | −0.0002 | −0.0029 to +0.0024 |
| Baker's | −0.0003 | −0.0020 to +0.0015 |
| Medial OA | −0.0009 | −0.0040 to +0.0021 |
| Lateral OA | −0.0023 | −0.0064 to +0.0019 |
| Lateral Meniscus | −0.0036 | −0.0084 to +0.0019 |

- **The plane got better, last epoch on validation:** error 1.60 mm against 1.76, the largest
  gain in the plane of any run so far with the map unchanged (joint 0.758, kneecap 0.736); the
  first look's own plane error, logged for the first time, is 1.62 mm. 22% of the plane's
  weight now comes from tokens under half bone that the incumbent discarded.
- **The crops turned less:** turn spread sagittal 3.34° (3.58) and coronal 3.19° (3.38), with
  the same shifts. A steadier plane gives steadier turns.
- **Reading:** one run, level at ×4 and 0.0011 down at ×1, inside the rerun band. The mechanism
  did what it said (plane −0.16 mm) and the score did not follow, as with the Dice seed, where
  the plane also improved by 0.16 mm without a gain. Two runs now say that a plane better by a
  sixth of a millimetre is not what the targets are waiting for; the sections and crops built
  on the plane are coarser than that (a token is several millimetres).
- **Queue after this run:** `map-gradient-projected-away-from-task-conflict` training;
  `patellar-routing-uses-unsaturated-shares`, `joint-centre-with-equal-view-exposure`,
  `sparse-first-look-at-interior-depth-quantiles`, `regions-integrate-the-token-footprint`.

## 14. `map-gradient-projected-away-from-task-conflict` (researcher): not adopted; the mechanism removed nothing

Level at ×4, below at ×1. Per knee, the part of the map's gradient at the token features that
opposes the label loss's gradient was to be removed.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.927176 | +0.000094 (95% −0.000834 to +0.001079) |
| ×1 | 0.917666 | 0.917186 | −0.000480 |
| Training | 1,687 s | 1,696 s | 1.01 |
| ×4 prediction | 53.1 s | 53.4 s | 1.01 |

No target moved by more than 0.0023 and no interval excludes zero (ACL +0.0012, Contusion
+0.0013, MCL −0.0023; Synovitis −0.0001, PF OA +0.0001, Effusion −0.0001).

- **What the run measured:** the two gradients are orthogonal. The map's gradient opposes the
  label gradient in 48 to 53% of covered knees in every epoch, a coin toss, and removing the
  opposing part keeps 0.99999 of the map's gradient (removed norm 2e-6 to 9e-6 of the scale).
  The projection was applied and changed nothing.
- **Map and plane identical to the incumbent's** (plane error 1.77 mm, overlaps to the third
  decimal, anatomy loss 0.677 against 0.678).
- **Reading:** this is a rerun of the incumbent in all but name, and it gives the rerun
  yardstick on the current recipe: ×4 +0.0001, ×1 −0.0005, largest target move 0.0023 (MCL).
  With the subpixel run (×4 −0.0009, ×1 −0.0006), two near-reruns put the incumbent's ×1 of
  0.917666 at the top of what its own recipe gives, about 0.0005 above them. Every run of this
  campaign since the second adoption is below the incumbent at ×1 by 0.0004 to 0.0014, so part
  of that common ×1 deficit is the incumbent's own draw, and the two-multiplier rule is asking
  candidates to beat a favourable draw.
- **Closed by measurement:** the map does not cost Synovitis through opposing gradients at the
  token features. The tenth-of-the-gradient seed lost the map without freeing Synovitis either.
  What the map costs, it costs through the shared capacity or through the sections and crops
  read from it, not through a gradient that can be projected out.

## 15. `patellar-routing-uses-unsaturated-shares` (researcher): not adopted; best ×4 of the campaign

Above the incumbent at ×4, below at ×1. The kneecap route reads kneecap and kneecap-joint tokens
by their share itself instead of the share divided by 0.3 and capped at one.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.927415 | +0.000332 (95% −0.000555 to +0.001315; 80% −0.000268 to +0.000979) |
| ×1 | 0.917666 | 0.917366 | −0.000300 |
| Training | 1,687 s | 1,713 s | 1.02 |
| ×4 prediction | 53.1 s | 53.6 s | 1.01 |

Per target at ×4 (delta, paired 95% interval), none with an interval that excludes zero:

| Target | Delta | Interval |
|---|---:|---|
| Fracture | +0.0023 | −0.0028 to +0.0082 |
| Contusion | +0.0022 | −0.0009 to +0.0061 |
| Medial Meniscus | +0.0014 | −0.0001 to +0.0029 |
| ACL | +0.0008 | −0.0017 to +0.0037 |
| Lateral OA | +0.0006 | −0.0022 to +0.0034 |
| Medial OA | +0.0005 | −0.0015 to +0.0024 |
| Baker's | +0.0003 | −0.0006 to +0.0012 |
| PF OA | +0.0003 | −0.0023 to +0.0028 |
| Synovitis | +0.0000 | −0.0026 to +0.0029 |
| Lateral Meniscus | −0.0002 | −0.0035 to +0.0032 |
| Effusion | −0.0013 | −0.0032 to +0.0005 |
| MCL | −0.0030 | −0.0074 to +0.0012 |

- **What the routes read, last epoch on validation** (share of each route's attention by true
  class, on the covered tokens): PF OA 21% kneecap joint, 10% kneecap, 46% femur, 19% other;
  Effusion 13% kneecap joint, 3% kneecap, 44% femur, 36% other; Synovitis 9% kneecap joint, 3%
  kneecap, 44% femur, 36% other. Even with the route led by the kneecap, nearly half of what
  these three targets attend to is femur.
- **The premise held on the live model:** 15% of the tokens the old rule promoted to full
  membership had a true kneecap share under 0.3 (23% at the first epoch), and 3.7% of tokens
  were saturated.
- **The seed's targets** did not move: PF OA +0.0003, Synovitis 0.0000, Effusion −0.0013. The
  macro's +0.0003 comes from Fracture, Contusion and Medial Meniscus, which the kneecap route
  does not serve.
- **Reading:** one run, level with the incumbent at both multipliers and the closest candidate
  to adoption so far: ×4 best of the campaign, ×1 0.0003 under, which is above both near-reruns
  of the incumbent (0.917042 and 0.917186). By the rule it is not adopted. The attention table
  is the lead: the PF OA, Effusion and Synovitis routes spend 44 to 46% on femur tokens, so the
  next step on this line is what "front of the femur" should be to them (the trochlea and the
  pouch are there), not the kneecap share's shape.

## 16. `joint-centre-with-equal-view-exposure` (researcher): adopted by the contract

At least the incumbent at both multipliers and within the prediction limit; the runtime's
verdict was reject (80% interval −0.000444 to +0.000837), so the keeper requested the adoption
by hand at 08:06 under contract.md, Adoption. It applies when the training then running
(`sparse-first-look-at-interior-depth-quantiles`, built on the old incumbent) ends. Each view's
tokens are divided by the view's valid-token count before the bone-weighted centre is taken.

| | Incumbent (seed 2) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.927262 | +0.000180 (95% −0.000749 to +0.001206) |
| ×1 | 0.917666 | 0.917892 | +0.000226 |
| Training | 1,687 s | 1,800 s | 1.07 |
| ×4 prediction | 53.1 s | 53.3 s | 1.00 |

Per target at ×4, none with an interval that excludes zero: Contusion +0.0019, MCL +0.0013,
Baker's +0.0010, Medial OA +0.0005, Fracture +0.0001, ACL +0.0000, Synovitis +0.0000, Lateral
OA −0.0002, Medial Meniscus −0.0003, Effusion −0.0003, Lateral Meniscus −0.0007, PF OA −0.0012.

- **How much the mechanism moved:** the joint centre by 0.06 mm on average (spread 0.06 mm in
  each direction), and the crops by 0.03 to 0.04 mm (0.07 to 0.11 px). The views were already
  near equal: each holds 19 to 24% of a knee's valid tokens, so dividing by the count changes
  little. The centre's errors are the incumbent's (0.95 mm and 1.76 mm mean, 1.95 and 1.56 mm
  spread).
- **View weights in the centre, logged for the first time:** the two axial views carry 11 and
  12% of the centre's weight with 21 and 24% of the tokens; the sagittal and coronal views carry
  20 to 27% each. The first look's centre is worse than the final pass's: mean error 1.22 mm
  across and 2.10 mm front to back, against 0.95 and 1.76 mm.
- **Reading:** this is a third near-rerun of the incumbent (a 0.06 mm change), and it landed
  above it at both multipliers where the other two landed below at ×1: ×1 of 0.917042, 0.917186
  and 0.917892 for three runs that differ from the incumbent by less than a tenth of a
  millimetre of geometry. That spread, 0.0009, is the yardstick for one run at ×1 on this
  recipe. The adoption follows the rule as written; it records a draw of the same recipe, not a
  gain from equal exposure, and the reference for later runs is now 0.0002 higher at each
  multiplier for the same reason.
- **Lead from the logs:** the centre's mean error front to back, 1.76 mm (2.10 mm in the first
  look), is the same in every view and every run since the second seed. It is a fixed offset
  that was measured on one batch and can be remeasured on the training knees.

## 17. `sparse-first-look-at-interior-depth-quantiles` (researcher): not adopted

Built on and compared with the incumbent before the last adoption (seed 2). Above it at ×4,
below at ×1; against the new incumbent it is level at ×4 (0.927259 against 0.927262) and 0.0017
below at ×1. The reduced first look takes its two slices per series from the interior of the
stack instead of the two end slices.

| | Seed 2 (its reference) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.927259 | +0.000177 (95% −0.001154 to +0.001435) |
| ×1 | 0.917666 | 0.916231 | −0.001435 |
| Training | 1,687 s | 1,632 s | 0.97 |
| ×4 prediction | 53.1 s | 50.9 s | 0.96 |

Per target at ×4, none with an interval that excludes zero: ACL +0.0019, Synovitis +0.0018,
Medial Meniscus +0.0018, Fracture +0.0015, Baker's +0.0009, Lateral OA +0.0007, Effusion
+0.0005, MCL −0.0000, PF OA −0.0005, Medial OA −0.0005, Contusion −0.0027, Lateral Meniscus
−0.0032.

- **The look:** 10.8 slices per knee read and 14.7 left out, as in the forced seed; 8.7 of the
  slices per knee differ from the end-slice choice.
- **Pose from the sparse look, last epoch on validation:** the first look's plane error is
  1.91 mm and its centre errors spread 2.55 mm across and 2.08 mm front to back, against 1.62,
  1.95 and 1.70 mm for the full first look (runs 13 and 16). Turn spread sagittal 3.87° and
  coronal 3.62° (3.58, 3.38); with the end slices (run 6) it was 4.19° and 5.38°. Interior
  slices recover most of the turn's steadiness, most of all in coronal, but not all of it.
- **Cost:** this time the saving shows at prediction, 0.96, and training 0.97.
- **Reading:** one run. Against run 6, the same reduction with end slices, it is 0.0017 better
  at ×4 and 0.0008 worse at ×1; the pose figures say the interior choice is the better sparse
  look. The ×1 deficit of 0.0014 is larger than the three near-reruns' (0.0005, 0.0006 and
  −0.0002), so a noisier pose costs most where each series is read once. A 4% saving at
  prediction does not pay for that; the first look's cost is better attacked by reading it at
  lower resolution than by reading fewer slices.
- **Adoption applied at 08:33:** `joint-centre-with-equal-view-exposure` is the incumbent
  (×4 0.927262, ×1 0.917892). The idea then in the developer's hands,
  `patellar-regions-sum-disjoint-anatomy-shares`, went back to be rebuilt on it.

## Where the training time goes (read-only investigation, 2026-10-08 08:55)

By a subagent from the frozen sources, `timing.json` and the progress bars of the training jobs
(read at one-second resolution); nothing was run on the GPU, and the
savings below are estimates unless marked measured.

| Part of an epoch | Unposed design | Seed 2 | Incumbent |
|---|---:|---:|---:|
| Training loop, 442 steps | 101 s | 139 s | 147 to 148 s |
| Validation, 110 steps | 15 to 16 s | 24 s | 26 to 27 s |
| First batch and between phases | 5 s | 7 s | 7 s |
| Epoch | about 119 s | about 167 s | about 178 s |

The unposed column is `frame-and-targets/frame-read-from-the-spots`, not identical code.

- **The last adoption's 11 s per epoch are its diagnostics** (measured): metric columns went from
  353 to 631 in training, each logging call forces a GPU sync, and one of them solves the pose
  again every step. Prediction, which logs nothing, did not change.
- **The posed crops' 38 s of training and 8 s of validation are mostly canvas size:** about 25 s
  from the 314 px canvases against 224 px, about 6 s from the first look's backbone pass and
  about 7 s from its heads, the bicubic cut and the jitter.
- **Changes that leave the output identical, by estimated saving per epoch:** drop the exposure
  diagnostics (11 s, measured); send only the middle channel of each canvas to the GPU, since the
  model reads only that one of the three (up to about 15 s, a guess); reduce the other 350
  per-step diagnostics once per epoch (8 to 14 s); cut crops only for valid slots (2 to 4 s);
  compute tokens and routing once for validation's seven poolings (1 to 3 s).
- **Not determined:** whether training waits on the GPU, the main thread or the loader. One
  hundred steps timed as loader wait against step time would settle which of the above matters.

## Public scores (2026-10-08)

Both at ×4, τ 0.9, batch 4, exported from each run's frozen source.

| Submission | Run | ×4 fold 1 | ×1 fold 1 | Public |
|---|---|---:|---:|---:|
| S118 | run 24, no posed crops | 0.925067 | 0.917070 | 0.953 |
| S122 | campaign reference, posed crops | 0.925885 | 0.917130 | 0.953 |
| S123 | reference weights, bicubic cut and map centre at prediction only | — | — | 0.953 |
| S125 | seed 2: bicubic cut and map centre, trained | 0.927083 | 0.917666 | 0.954 |
| S124 | equal view exposure | 0.927262 | 0.917892 | 0.954 |
| S119 | ResNet-50 of run 24, no anatomy map | 0.933209 | 0.922934 | 0.957 |
| S126 | ResNet-50 on the adopted recipe (`rn50-anatomy-map-centre`) | 0.933265 | 0.924650 | 0.959 |

- S125 is +0.001 public over the reference and over the same two changes applied only at
  prediction (S123), with +0.0012 at ×4 on fold 1. One submission at three decimals; the two
  forced seeds are the campaign's only step that shows on both.
- S124 took over two hours to be scored by Kaggle and came back at 0.954, the same as S125.
- S126 is the best submission so far: +0.002 public over the same backbone without the anatomy
  map, with ×4 on fold 1 level (+0.00006) and ×1 +0.0017. One submission at three decimals.

## 18. `patellar-regions-sum-disjoint-anatomy-shares` (researcher): trained and compared, not decided

The last run, finished after the user's stop; its comparison exists but the orchestrator's
decision was never taken, so the runtime lists it as awaiting a decision. Built on the last
incumbent (equal view exposure). Below it at both multipliers, so not adoptable by the rule.

| | Incumbent (equal view exposure) | Run | Delta |
|---|---:|---:|---:|
| ×4 | 0.927262 | 0.926840 | −0.000423 (80% −0.001273 to +0.000437) |
| ×1 | 0.917892 | 0.917475 | −0.000417 |
| Training | 1,800 s | 1,877 s | 1.04 |
| ×4 prediction | 53.3 s | 53.6 s | 1.01 |

One run, inside the rerun band at both multipliers; per-target figures were not extracted. Its
training is another 77 s longer than the incumbent's, the same kind of cost as the incumbent's
own (per-step diagnostics, see the speed section above).

## Stopped 2026-10-08 09:14 CEST

The user asked for a graceful stop at 08:44; the supervisor ended when the last training and its
comparison finished. Final incumbent `joint-centre-with-equal-view-exposure` (×4 0.927262, ×1
0.917892), which the user doubts and the keeper recommended withdrawing to seed 2 (×4 0.927083,
×1 0.917666, 0.954 public as S125); no decision was given. Still queued and never run:
`joint-plane-balances-soft-femur-and-tibia-mass` (built), `regions-integrate-the-token-footprint`,
`joint-map-covariance-sets-the-across-axis`.

## After the stop: seed 2 adopted, training sped up, two-knee pose fixed (2026-10-08)

- **Adoption by the user:** `joint-centre-from-the-map-instead-of-the-spots` is the recipe; the
  equal-view-exposure adoption is withdrawn. The worktree's source was set to that run's frozen
  source (same source hash) and the adoption requested in the runtime; it applies when the
  campaign is next resumed.
- **Speed:** training 3.18 to 4.04 steps per second, an epoch 167 to about 140 s, with the same
  predictions from the same weights. What changed and how it was checked is in
  [Training runtime and memory](../../../docs/modeling/training-runtime-and-memory.md#posed-crop-recipe-where-a-step-goes-and-what-was-removed-2026-10-08).
  The source hash changed, so the recipe needs one full training before its checkpoints load.
- **Two-knee studies:** the pose took one side sign per study, the majority of its tokens, so in
  a study with both knees the minority knee's series had the coronal turn and the across shift
  mirrored. Each series of a two-knee study now uses its own knee's sign, as the token positions
  already did. On the validation fold with seed 2's weights, 872 of 874 studies are unchanged
  and the two that moved (by up to 0.028 and 0.003 in a probability) are the fold's two two-knee
  studies. Still pooled: one joint plane and one centre are fitted over both knees of such a
  study.
- **Audit, three read-only reviews (data path, model, training loop):** no leakage, label, loss,
  schedule, metric or selection defect. Checks that were run: the fold's macro AUC recomputed
  from the saved predictions equals the recorded one for all ten epochs and the final ×4 file;
  three optimizer groups cover all 101 parameters once; the schedule and the anchor penalty
  recompute to the logged values; folds do not overlap. Known and left as they are: contrast
  mixing on unaligned grids (aligned twice in fixes-and-removals, lower both times). Added: a
  per-epoch count of loader failures (`preparation_failures`), which training never read
  before. Small and not changed: `route_section_shift` is weight-decayed while the other frame
  parameters are not; a batch without anatomy or side targets still moves those heads through
  Adam's momentum; the best checkpoint is chosen on ×1 where epochs 8 and 9 differ by 0.00003.

## Contrast mixing under posed crops (measured 2026-10-08)

Contrast mixing blends the two contrasts of a plane pixel for pixel (known, and left so after the
aligned donor lost twice in fixes-and-removals). How far apart the blended pixels are in the
patient, in the slice plane, on 40 training batches with seed 2's weights
(output
`outputs/analysis/anatomy-masks-speed/contrast-mixing-registration.json`):

| Crops | Plane | Centre, median (p90) | Corners, median (p90) |
|---|---|---:|---:|
| Loader crops | coronal | 0.16 mm (1.19) | 6.60 mm (6.68) |
| Loader crops | sagittal | 0.18 mm (0.48) | 6.60 mm (6.67) |
| Posed | coronal | 0.23 mm (0.40) | 6.59 mm (6.62) |
| Posed | sagittal | 0.22 mm (0.38) | 6.60 mm (6.65) |
| Posed with jitter | coronal | 3.10 mm (5.49) | 7.42 mm (8.54) |
| Posed with jitter | sagittal | 3.01 mm (5.24) | 7.35 mm (8.42) |
| Any | axial | 0.44 mm (12.67) | 0.45 mm (12.73) |

- The pose itself does not misalign the pair: both crops are placed on the same joint centre, and
  the centre's tail gets shorter (p90 1.19 to 0.40 mm in coronal). The 6.6 mm at the corners is
  the 84 against 93.3 mm fields, as before.
- The training jitter does: it is drawn per series, so the pair is 3 mm apart at the centre
  where it was 0.2 mm. A jitter drawn once per study and plane would keep the augmentation and
  return the centre to 0.2 mm.
- Axial pairs, never posed, are 12.7 mm apart or more in a tenth of the slices (the two axial
  series are each cropped about their own image centre). This predates the campaign.

## Control of the adopted recipe (2026-10-08)

`outputs/training/rn18-anatomy-map-centre-control`: seed 2's recipe with the speed-ups and the
two-knee pose fix, same fold and seed.

| | Seed 2 | Control | Delta |
|---|---:|---:|---:|
| ×4 | 0.927083 | 0.926762 | −0.000321 |
| ×1 (best epoch) | 0.917666 | 0.917074 | −0.000592 |
| Training | 1,687 s | 1,391 s | 0.82 |
| ×4 prediction | 53.1 s | 51.8 s | 0.98 |

×1 per epoch against seed 2: +0.0010, −0.0003, −0.0001, +0.0000, −0.0012, −0.0006, +0.0002,
−0.0007, −0.0006, −0.0008. The control ends below seed 2 at both multipliers, by amounts the
campaign's near-reruns of seed 2 also showed (×1 0.917042 and 0.917186); with the same weights
the two codes give the same predictions, so the difference comes from the training trajectory.
No true same-seed rerun exists on this recipe to say how much of it is chance; the user declined
one. The control is the reference of the
[anatomy-recipe campaign](../anatomy-recipe/INDEX.md) and its eight forced seeds.

## Equal view exposure against seed 2, in depth (2026-10-08)

By a subagent, each run loaded with its own frozen source, ×1, 874 validation knees
(tables in
`outputs/analysis/anatomy-masks-frame-compare/`). Verdict: a rerun of seed 2 in effect.

| What differs | Centre, mean (mm) | Plane normal, mean | Crop shift, mean (mm) | Prediction, mean | Macro |
|---|---:|---:|---:|---:|---:|
| Same code and weights, twice | 0 | 0° | 0 | 0.00002 | +0.000006 |
| The mechanism, same weights | 0.063 | 0° | 0.04 | 0.0011 | +0.00004 to +0.00009 |
| The weights, same code | 0.26 | 0.73° | 0.21 | 0.0113 | −0.00001 to +0.00005 |

- The mechanism leaves the plane and the crop turn untouched and moves the centre 0.06 mm; the
  two runs' different weights move the frame four times more and the predictions ten times more.
- Against the targets (150 validation knees) the mechanism makes the centre 0.009 to 0.010 mm
  worse (intervals above zero), not better; plane error and map agreement are the weights'.
- Equal view exposure is below seed 2 at ×1 in seven of ten epochs; its edge exists at epochs 1,
  9 and 10 only. Re-predicting the same checkpoints at batch 4 instead of 8 moves each ×1 macro
  by 0.00005 to 0.00009 and shrinks the edge from 0.00023 to 0.00009.
- The centre's fixed offset, +1.2 mm across and +2.1 mm front on these knees, is fifty times the
  mechanism's effect; it is the eighth seed of the anatomy-recipe campaign.
- Validation is deterministic: the same code and weights twice give the same frame to the last
  digit.
