# Learnable-views prior work

What was measured while the learnable build was made, all on 2026-10-02 in the
`view-optimisation` worktree. The full record, with the checks against main and every per-label
table, is the [view-optimisation notes](build-notes.md); the hand
search the views are compared with is the
[smart-crops campaign](../smart-crops/view-effects-by-target.md).

## Where the learnable recipe stands

Fold 0, seed 1533697213, one run each. ×4 is the adoption metric; ×1 is the last epoch's
validation.

| Run | What it is | ×4 | ×1 | Training, s | Public |
|---|---|---:|---:|---:|---:|
| `views-budget-shift-centre` | **Reference**: views, label bias and budget learned from the 120 mm centre crop | 0.932489 | 0.9273 | 1,096 | 0.947 (S94) |
| `views-budget-jitter-centre-v2` | The same without the budget shift, before middle-slice loading | 0.932704 | 0.9273 | 1,521 | |
| `fluid-detail-93-replacement` | Hand recipe on main (84 and 93 mm, knee anchors) | 0.932880 | 0.9293 | 1,134 | 0.950 (S93) |
| `view-frozen-control` | Hand recipe on this build, views frozen | 0.932627 | 0.9290 | 1,419 | |
| `rn18-field-120-control` | 120 mm centre crop on main, the learned runs' starting point | 0.930666 | 0.9275 | 1,149 | 0.947 (S92) |
| `frozen-learned-views-run4` | The reference's final views frozen from the start, 4 mm and ±5% jitter, no label bias, budget 6/6/5/5/5/5 | 0.930609 | 0.9262 | 1,008 | |

- **The learned runs tie the hand recipe at ×4 and trail it at ×1.** From ×1 to ×4 they gain
  +0.0052 and +0.0054; the three fixed-view runs gain +0.0032 to +0.0036.
- **The public score followed ×1.** S94 scored 0.947 against 0.950 for S93, the level of the
  120 mm recipe (S92). One submission; a fault in the Kaggle prediction path of this build has
  not been ruled out (the parity check is recorded in the notes).
- **The final views alone do not carry the learned run's score.** Frozen from the start with
  the hand recipe's jitter, they score 0.930609: −0.0019 against the run they came from, level
  with the 120 mm centre crop, and −0.0020 against the hand views frozen on the same build. At
  ×1 they score 0.9262, so the narrow jitter did not close the ×1 gap. What the learned run adds
  over its own final views is therefore in how it trains (the wide jitter, the moving views,
  the label bias, the random slice counts), not in where the views end. Which of these is not
  separated; seeds 1 to 3 each remove one.
- **Per label, the reference against the hand recipe on main (×4):** ACL +0.0041, lateral OA
  +0.0036, medial meniscus +0.0026, PF OA +0.0021; lateral meniscus −0.0051, synovitis −0.0050,
  effusion −0.0028, fracture −0.0027. PF OA and lateral OA up and synovitis down are shared with
  `views-budget-jitter-centre-v2`.

## Where the views end

Widths in mm (start 120). The hand recipe is 120 on axial, 84 on coronal and sagittal T1/PD, 93
on coronal and sagittal fluid.

| View | Reference | `views-budget-jitter-centre-v2` | `view-learnable-centre-sharp` |
|---|---:|---:|---:|
| Axial T1/PD | 93 | 108 | 129 |
| Axial fluid | 135 | 121 | 119 |
| Coronal T1/PD | 91 | 107 | 117 |
| Coronal fluid | 120 | 117 | 112 |
| Sagittal T1/PD | 122 | 105 | 84 |
| Sagittal fluid | 112 | 108 | 109 |

- **Learned view values do not reproduce between runs.** The three runs differ by up to 38 mm
  on one view; the runs also differ in build details, so this is not a pure repeat.
- **Shared between the reference and the run before it:** axial and coronal T1/PD end narrower
  than their fluid views, and axial fluid ends widest.
- **Timing of the movement (reference):** most of each width's movement happens in epochs 0 to
  3, the widths wander by a few millimetres until epoch 6 and stop around epoch 8, when the
  cosine schedule has decayed. Axial T1/PD: 108, 102, 93, 93, 98, 95, 92, 92, 93, 93. Sagittal
  T1/PD: 119, 134, 138, 132, 126, 122, 122, 121, 122, 122. Sagittal fluid: 116, 108, 108, 104,
  103, 103, 110, 112, 112, 112.
- **Shifts** stay within about 6 mm of the image centre in the reference (largest: sagittal
  T1/PD 5.4 mm backward and 4.1 mm down). The hand recipe's 10 mm downward shift and knee
  anchors were not found from the centre crop.
- **Budgets** stay within a quarter of a slice of 6/6/5/5/5/5 in both runs of the second
  version: 5.79, 5.95, 5.13, 5.13, 4.87, 5.14 in the reference.
- **Label bias** stays small: largest about ±0.10 (axial fluid for Baker's cyst +0.10 in every
  run, coronal fluid for MCL +0.09, axial T1/PD for ACL −0.09).

## Precedent table

Deltas are ×4 macro AUC against the hand recipe on main (0.932880). Every row is one run.

| Mechanism | Setting | Run | Macro delta | Notes |
|---|---|---|---:|---|
| Views and label bias learned, view rate 0.0006 | 448 px canvas | `view-learnable-centre` (deleted) | not scored | Views moved about 1 mm per epoch: the rate was too low |
| The same at view rate 0.006 | blurred build | `view-learnable-centre-lr6e-3` | −0.0013 | Sagittal T1/PD ran to the 140 mm bound; the build lost fine detail |
| The same after the detail fix | 384 px canvas, bicubic | `view-learnable-centre-sharp` | −0.0009 | Sagittal T1/PD found 84 mm and the forward shift |
| Budget alone, learned on the hand views | first version | `budget-learnable-recipe` | −0.0018 | Three flaws, below |
| Views, bias and budget together | first version | `views-budget-learnable-centre` | −0.0013 | The same flaws |
| The same, second version, 8 mm and ±10% placement jitter | | `views-budget-jitter-centre-v2` | −0.0002 | First tie |
| The same with the budget shift and middle-slice loading | the recipe | `views-budget-shift-centre` | −0.0004 | Reference; main's training time |
| The reference's final views frozen | 4 mm jitter, no bias, fixed budget | `frozen-learned-views-run4` | −0.0023 | Level with the 120 mm centre crop |

## Flaws found and fixed

These are not directions. An idea that reintroduces one is rejected at review.

- **Lost detail.** Cutting windows with bilinear sampling from a coarse canvas kept 64–75% of
  the fine-detail energy of main's input. Fixed with bicubic sampling, a low-pass matched to the
  spacing ratio and the 384 px canvas at 0.375 mm (95–101%).
- **Intensity.** Windows took the canvas-wide quantiles, so their contrast differed from main's.
  Fixed by rescaling each window from its own quantiles per series and view.
- **Budget loophole.** The first version let most studies load 27 slices and down-weighted the
  extra ones, so "more slices" was confounded with the budget. Fixed by holding the study total
  and rounding at random.
- **Absent types.** A type absent from a study pushed its own budget down. Fixed by computing
  the shares, and their gradient, over the types present in the study.
- **Step time.** Three channels were shipped where one is read, and per-step logging cost about
  a tenth of a second. Fixed: middle-slice loading and per-epoch logging; a step now takes
  0.235 s against 0.239 s on main.

## Two views on one series type

The user's named direction for this campaign (2026-10-02). Since the schema merge a series type
may list several views; each selects its own slices of the type, so a second view costs no
extra slices. No run has used it. What smart-crops measured with hand-placed views added to the
same slices ([view effects](../smart-crops/view-effects-by-target.md)):

- **Sagittal fluid has no single good width.** Zoomed to 93 mm: macro +0.0005 and +0.0007,
  Baker's cyst −0.0090 and −0.0047, MCL −0.0025 to −0.0062; 100 mm is no compromise.
- **Added views in every label's pool tie** (twelve runs), with costs on Baker's cyst, the
  menisci and contusion.
- **A routed second view works mechanically:** a 120 mm sagittal fluid view sent to Baker's cyst
  alone gave it +0.0023 and tied on macro (−0.0003) at 26% more time.
- **Build check:** with two views on coronal T1/PD the loader and model run on CPU, the study
  total holds at 26 or 32, each view draws its own slices and receives budget and view
  gradients.

## Not built

- The spread of a type's slices along the stack as a learned quantity; it needs its own
  estimator.
- The jitter draws used as direct probes of the loss in place of the gradient through the
  resampling.
- Reproducible view values across runs.

## Ensembling

Outside this campaign (endgame), recorded so it is not proposed here: a learned run averaged
with the hand recipe gains +0.0011 to +0.0013 over the better member at ×4 on fold 0, against
+0.0003 for two builds of the same views
([notes](build-notes.md#second-version-of-the-learned-sampling-and-the-view-jitter-2026-10-02)).
