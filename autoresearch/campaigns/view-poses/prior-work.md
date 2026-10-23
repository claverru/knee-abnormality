# View-poses campaign: prior work

**Skeleton.** The measurements below were taken on 2026-10-08 and are held in full, with
tables and limits, in the [view pose studies](../../../docs/research/geometry-and-views/view-pose-studies/INDEX.md); they move here
when the campaign is written out. Figures and tables are under `outputs/analysis/`. The one-off
scripts were lost in a WSL restart; outputs remain.

## The incumbent

`report-attribute-loss-off-2`, the anatomy-recipe campaign's only adoption: the anatomy-map
recipe (seven-class bone map on 550 mask knees, joint plane and centre read from it, posed
sagittal and coronal crops, sections and routes) with the report-attribute loss at 0. ×4
0.927291, ×1 0.917342, 1,352 s. Its seven runs are in
[the anatomy-recipe run notes](../anatomy-recipe/run-notes.md).

## What the crop changes of 2026-10-08 gave

| Change | ×4 against its incumbent |
|---|---:|
| Fluid coronal crop at 108 mm for every knee | −0.0017 |
| Fluid sagittal crop at 112 mm for every knee | −0.0009 (PF OA +0.0032) |
| Fluid sagittal crop moved forward per knee | −0.0002 (PF OA +0.0015) |
| One scale per knee, sagittal and coronal | +0.0003 at both multipliers, mean scale 1.000 |
| 257 px, centred grid, with or without border tokens | −0.0005 and −0.0004 |

## Knee size

Size varies by 8% (femur width 73.7 mm, sd 6.3); the scanner's field does not normalise it. The
first look estimates it from the bone map: r 0.94 against the masks, residual 2.1 mm
(`outputs/analysis/k/knee-size-crops/`).

## What each view is read for

From the radiology literature and inspection of 102 uncropped training studies
(`outputs/analysis/t/targets-by-view/`; not a radiologist's reading).

| View | Wide need | Fine need |
|---|---|---|
| Sagittal fluid | effusion, pouch, Baker's, marrow (about 180 mm of height) | menisci, cartilage, ACL in a 60 to 90 mm box at the joint line |
| Coronal fluid | both collaterals, marrow | menisci, tibiofemoral cartilage (about 110 mm high holds both) |
| Axial fluid | whole limb cross-section for cyst and pouch (130 to 150 mm) | kneecap cartilage in a 70 × 40 mm box |
| Sagittal, coronal T1/PD | – | joint line: fracture line, bone spurs, meniscal signal |
| Axial T1/PD | no target depends on it; present in 19% of studies | – |

The "T1/PD" slot is not one contrast (sagittal: T2 in 45 of 102, T1 in 31, PD in 26).

## Each view alone

How well each view alone ranks each label in the incumbent (validation AUC of the per-view
readout at the last epoch; best view per label in bold), against the pixel size it is shown at.

| Label | Sag fluid | Sag T1/PD | Cor fluid | Cor T1/PD | Ax fluid | Ax T1/PD |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.885 | **0.891** | 0.851 | 0.810 | 0.759 | 0.572 |
| MCL | 0.727 | 0.575 | 0.871 | 0.693 | **0.924** | 0.579 |
| Medial Meniscus | **0.917** | 0.867 | 0.913 | 0.818 | 0.707 | 0.535 |
| Lateral Meniscus | **0.843** | 0.828 | 0.829 | 0.735 | 0.698 | 0.516 |
| Medial OA | 0.863 | 0.849 | **0.884** | 0.829 | 0.810 | 0.607 |
| Lateral OA | 0.835 | 0.836 | 0.837 | 0.735 | **0.842** | 0.572 |
| PF OA | **0.830** | 0.804 | 0.765 | 0.665 | 0.828 | 0.573 |
| Effusion | 0.898 | 0.802 | 0.856 | 0.645 | **0.903** | 0.520 |
| Synovitis | 0.824 | 0.816 | 0.808 | 0.648 | **0.830** | 0.599 |
| Baker's | 0.908 | 0.712 | 0.809 | 0.604 | **0.952** | 0.560 |
| Contusion | 0.872 | 0.747 | **0.903** | 0.693 | 0.862 | 0.479 |
| Fracture | 0.868 | 0.624 | **0.906** | 0.788 | 0.878 | 0.488 |
| Pixel size | 0.417 mm | 0.375 mm | 0.417 mm | 0.375 mm | 0.536 mm | 0.536 mm |

- The T1/PD views are the most zoomed and the best single view for ACL only (by 0.006); coronal
  T1/PD is best for nothing and axial T1/PD is near chance.
- Axial fluid is the best single view for five labels and level with the best for PF OA, at the
  coarsest pixel.
- The table shows what each view gives with today's crops; a view shown coarsely may look weaker
  than it could be.

**How the fields were chosen** ([smart-crops campaign](../../../docs/validation/results/crops-and-views/2026-10-02-smart-crops-campaign.md)):
in the order they were tried. T1/PD to 84 mm first (+0.0013), then sagittal and coronal fluid to
93 mm (+0.0010). Fluid was never tried tighter than 93 mm (sagittal at 100 mm: −0.0002). Axial
was tried wider only (130 mm 0.0000, 140 mm −0.0006), and zoomed once, as a second 84 mm view
on the kneecap joint taking half the axial slices, rule-placed and before the anatomy map:
−0.0013 ([learnable-views run 44](../../../docs/validation/results/crops-and-views/2026-10-03-learnable-views-campaign.md)).

## Errors against crop coverage

On 871 validation studies the crop's in-plane field does not explain the errors. Two effects
appear, both about which slices are read: Effusion and Synovitis fall by about 0.07 when the
axial slices stop below the pouch, and MCL by 0.12 when the sagittal slices stop short of the
medial edge (47 positives). Nothing measured explains the errors of PF OA, Lateral Meniscus or
Lateral OA (`outputs/analysis/e/errors-vs-coverage/`).

## Soft-tissue segmenter

`aagatti/nnunet_knee` (MIT weights, trained on DESS) on our volumes
(`outputs/analysis/s/soft-tissue-segmenters/`):

| Series | Menisci | Cartilage |
|---|---|---|
| DESS (10 volumes) | 10 of 10 pass | 8 to 10 of 10 pass; good by eye |
| Sagittal 3D PD fat-suppressed (10) | 3 to 5 of 10 | 9 to 10 of 10; cartilage only |
| 2D series (20) | 0 to 1 of 20 | about half the true volume; not usable |

No released weights exist for ligaments, effusion, the pouch, cysts or tendons.

## Our 3D series

413 training studies hold a fluid-sensitive 3D series: 265 DESS (one Siemens scanner, axial
0.6 mm, 86 mm slab from about 26 mm below to 65 mm above the joint line) and 148 sagittal 3D PD
(two scanners). They are already model inputs. A mask made on the 3D series lines up with the
2D series of the same study to about 1 mm on bones (median surface distance 0.6 to 1.1 mm,
best rigid shift 1.1 to 1.4 mm). Their labels differ from the rest (ACL and lateral meniscus
tears half as frequent; effusion 42% in DESS studies).

Masks for the 265 DESS studies: `~/.cache/knee-segmenters/dess-menisci/`, made with
`scripts.tools.meniscus_targets`.

## The reference and where its predicted structures sit

`rn18-meniscus-map` finished at ×4 0.926413 and ×1 0.916382 (training 1,371 s), against 0.927291
and 0.917342 for `report-attribute-loss-off-2`: −0.0009 and −0.0010 with the meniscus head.
Final validation overlaps: femur 0.904, tibia 0.898, joint line 0.758, kneecap 0.738, kneecap
joint 0.670, medial meniscus 0.452, lateral meniscus 0.433; plane error 1.74 mm; centre error
1.75 mm front to back and 0.98 mm across.

One pass of it over the 874 validation studies (`outputs/analysis/m/meniscus-map-preview/`:
`structures-in-crops.png`, `example-maps.png`, `structures-in-crops.csv`,
`structure-centroids-in-knee-frame.csv`), in knee-frame directions:

| View | What the predicted map shows |
|---|---|
| Sagittal fluid, 93 mm | **56% of the kneecap's mass is on the front edge line** of the crop; femur and tibia end 26 mm short of the back edge; the tibia fills the lower third |
| Sagittal T1/PD, 84 mm | **78% of the kneecap's mass on the front edge line** |
| Coronal, both | bones symmetric, ending 9 to 14 mm short of each side; the menisci on the token row just below the centre |
| Axial, both | **25 to 27% of the kneecap's mass on the front edge line**; the femur ends 37 mm short of the back edge; the bones' middle is 7 mm forward of the crop centre |

Centroids in the knee's frame (mean, spread across 786 studies):

| Structure | Medial | Front | Up |
|---|---|---|---|
| Kneecap | −10.4 mm (4.7) | +43.9 mm (4.1) | +32.0 mm (4.4) |
| Medial meniscus | +21.2 mm (4.9) | −4.5 mm (3.7) | −1.8 mm (1.1) |
| Lateral meniscus | −23.8 mm (3.8) | −10.8 mm (3.9) | −2.3 mm (1.1) |

- The kneecap's middle is 44 mm in front of the knee's centre and the sagittal crops are 42 and
  47 mm to a side: the kneecap is on the edge in every sagittal view.
- The menisci sit 8 mm behind the knee's centre on average and 2 mm below it, with a height
  spread of 1.1 mm across studies: a steady anchor for the joint line.
- At 7 by 7 the menisci reach a 15% share on the middle slice in about two knees of five in the
  drawn examples; the head finds them weakly.
- Positions are read on the token lattice, so percentiles step by a token (12 to 17 mm).

## Meniscus head calibration

Measured by the keeper on 2026-10-09 for the two seeds that cite a number "measured once"
([values](meniscus-calibration.json)). Cohort: the 211 studies of fold 1's training side that
have a DESS meniscus mask; no validation study. Model: the selected checkpoint of the incumbent
`canvases-for-the-pose-in-all-six-views-20261009T014728Z-960f75cb`. Pass: the validation-style
dataset at ×1 and the model's own first look (`view_windows.first_look_batch` then
`_encode_crops`: central 224 px crops at scale 1, evaluation mode, FP16), so the numbers are
those production sees in the first look.

- **Meniscus mass of a study:** the sum, over the first-look 7 by 7 tokens that are
  `visual_valid` and `routable`, of the predicted medial plus lateral share in
  `meniscus_shares` (the mean of the four 14 by 14 predictions). In token-share units: 5th
  percentile **3.89**, median 8.29, 95th percentile 13.15 (1st 2.88), over about 1,159 valid
  tokens per study. The sum over the 14 by 14 grid is four times these.
- **Share of a token holding a whole horn:** the mask's share of one meniscus on a 7 by 7
  first-look token (mean of its four 14 by 14 target footprints, all four inside the mask
  volume), over the tokens where it is above zero: 95th percentile **0.375** for the medial
  (5,159 tokens) and **0.375** for the lateral meniscus (4,663 tokens); medians 0.094 and
  0.109. The predicted share on those same tokens has a 95th percentile of 0.517 and 0.506.

## Canvas sizes

From the incumbent's first-look frame on 3,517 training studies
(`outputs/analysis/v/view-attention/canvas-sizes.csv`): the canvas that holds the window the pose
asks for (shift, turn, scale 0.87 to 1.13, training jitter) in 99% of images.

| View | Today | Square | Rectangle, width × height |
|---|---|---|---|
| Coronal T1/PD 84 mm | 314 px | 370 px | 328 × 368 px |
| Coronal fluid 93 mm | 306 px | 364 px | 324 × 364 px |
| Sagittal T1/PD 84 mm | 314 px | 370 px | 336 × 370 px |
| Sagittal fluid 93 mm | 306 px | 364 px | 330 × 364 px |
| Axial, 120 mm, not turned | 224 px | 282 px | 278 × 278 px |

- 11 to 15% of sagittal and coronal crops are clamped by today's canvas.
- The canvases are not off-centre (median offset about 1 mm); the knee wanders 7 to 8 mm up and
  down against 4 mm across, and the loader's own knee height is no better.
- Axial, turned as well: about 194 mm of canvas for a 120 mm field, 167 mm for 105 mm, 145 mm
  for 90 mm. A 90 mm turned crop fits the canvas of the unturned 120 mm one.

## Token grid

Tokens are centred at pixels 0 to 192 of the 224 px crop, 15.5 px off the crop centre; the first
row and column are valid 60% of the time in training and always at prediction. Neither matters
to the score ([token grid alignment](../../../docs/validation/results/crops-and-views/2026-10-08-token-grid-alignment.md)).
