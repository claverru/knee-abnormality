# Side-routed heads: hard-cut regions and the spot frame, runs 18 to 23 (2026-10-06)

**Hard cuts with the soft gradient on the eighteen regions fell to ×4 0.923121 (run 18) and climbed
run by run to 0.924358 (run 22) as the core went back to the side labels, the regions were widened
to the all-token attention, roll and turn were shared and the kneecap labels were freed from the
upper cut.** The frame hung on the two meniscus attention spots (run 23) scored 0.924251. The
attention-shape analysis found eight sectors centred on the axes with a core and an outer ring at
three heights, which became run 24. Fold 1 of `all_5fold_scanner_v2`, seed 1110484336, ResNet-18.
Design: [side-routed heads](../../../../modeling/side-routed-heads.md).

## Run 18 build: hard cuts with the soft gradient, knee-level term on every routed target

Decided with the user on 2026-10-06 (the per-knee frame settled for every run; the reports stayed
routed). On run 12's build, ACL in the core as in the agreed table:

- **Assignment.** Hard steps at every cut (midline, front cut, the two height cuts, the core
  radius), with the gradient of a sigmoid of the signed distance over the token box's extent along
  the cut's normal. This was run 7's mechanism in place of the overlap shares; the three sizes
  learned through it.
- **Knee-level term on every routed target.** One learned number per channel, zero at the start:
  the target's score was its region's evidence plus that share of the same head's evidence pooled
  over the whole knee. New for ACL, MCL, Effusion, Synovitis and Baker's; Contusion and Fracture
  were unrouted and had none. It was meant to let ACL keep the core as its region and still read
  the bruises on the lateral side.

All nine guardrails passed. First step on four studies: routing weights were 0 or 1 on
side-certain knees; the knee-level numbers of the seven routed channels and the three sizes
received gradient.

Run `outputs/training/20261006-135237-side-routed-regions-soft-gradient-knee-level` (run 18):

- ×4 macro 0.923121 against 0.924845 for run 7 (−0.0017) and 0.923687 for run 12 (−0.0006); ×1
  0.915100. The lowest of the eighteen-region runs.
- Against run 12 (the same build with overlap shares and the knee-level term on the shared heads
  only): MCL +0.0059, Fracture +0.0033, Contusion +0.0019, Lateral Meniscus −0.0001, ACL −0.0008,
  Effusion −0.0008, Medial OA −0.0010, Medial Meniscus −0.0024, PF OA −0.0025, Lateral OA
  −0.0029, Synovitis −0.0032, Baker's −0.0043. The five shared-head labels lost 0.0089 summed;
  Medial Meniscus was 0.0033 below run 7, against a sibling SD of 0.0007.
- **The sizes hardly moved under the soft gradient:** upper cut −0.2 mm, lower cut −28.8 mm, core
  radius 16.3 mm (start 0, −27.4, 13.7). With overlap shares they went to +4.9, −35.2 and 15.0.
  Nod 4.6° against 20.1°. So the joint band stayed 28 mm thick.
- **Every knee-level number learned negative:** ACL −0.027, MCL −0.008, Effusion −0.039,
  Synovitis −0.064, Baker's −0.033, meniscus −0.096, OA −0.021. The term acted as a contrast
  (region minus knee), not as a wider view; it did not make ACL read the lateral bruises.
  Contusion and Fracture still rose against run 12 (+0.0019, +0.0033), within their noise.
- Report fit on the validation set 0.7204 against 0.7194 in run 12: unchanged.

**Drawings** (`outputs/analysis/s/side-routing-loop-20261006/side-routed-regions-soft-gradient-knee-level/`):

- `attention-*.png`: each label looked where it should. Medial and Lateral Meniscus on the joint
  line of their side, ACL on the notch, MCL on the medial edge, PF OA on the kneecap and trochlea,
  Effusion and Synovitis on the fluid above the kneecap, Baker's on the cyst behind the medial
  side. A meniscus put 20% to 43% of its attention on one slice, on two to four tokens.
- `regions-*.png`: the core was the problem. It was a cylinder of 16 mm mean radius (22 mm in
  study 45) around the inferred up axis; on the axial slices it covered the notch and the inner
  half of both condyles and plateaus, which is where the meniscal roots and inner posterior horns
  are. With hard cuts the menisci, MCL and Baker's lost those tokens entirely; with overlap shares
  (runs 12 to 16) the tokens on the core's edge still counted in part, and Medial Meniscus stayed
  within 0.0008 of run 7, even with a 29 mm band (run 15).

Decision for run 19: the side labels keep their half of the core (Medial and Lateral Meniscus,
MCL, Baker's and the tear-location targets no longer exclude it). The core then defined ACL's
region and was added to the kneecap part for PF OA and Effusion; it was taken away from no one.

## Run 19: the side labels keep their half of the core

Run `outputs/training/20261006-141739-side-routed-regions-core-kept-by-sides`: run 18 with the core
no longer taken from the menisci, MCL, Baker's and the tear-location targets.

- ×4 macro 0.923473 against 0.923121 for run 18 (+0.0004) and 0.924845 for run 7 (−0.0014); ×1
  0.915175.
- Against run 18: MCL +0.0039, Fracture +0.0039, Medial Meniscus +0.0018, Contusion +0.0015,
  Medial OA +0.0009, Effusion +0.0007, Lateral OA +0.0006, ACL −0.0004, PF OA −0.0012, Baker's
  −0.0013, Synovitis −0.0022, Lateral Meniscus −0.0041.
- Against run 7, beyond their sibling SD: Synovitis −0.0073 (SD 0.0022), PF OA −0.0030 (0.0005),
  Baker's −0.0036 (0.0015), Lateral Meniscus −0.0025 (0.0024); MCL +0.0052 (0.0029).
- Sizes as in run 18: upper cut +0.4 mm, lower cut −27.9 mm, core radius 15.3 mm, nod 4.7°.

**The frame head, measured on this model** (200 validation knees):

- Angles, mean (SD, range): nod 4.5° (4.4, −9 to 20), roll −7.1° (5.4, −19 to 9), turn −4.7°
  (5.8, −21 to 6); 34% of knees had more than 10° of roll and 23% more than 10° of turn. Shift SD
  2.9, 3.3 and 4.9 mm. Sizes: lower cut −27.9 mm (SD 4.1, −36 to −19), upper cut and radius
  nearly constant.
- Both parts contributed: regression alone gave SD 3.5°, 4.1°, 5.2°; landmarks alone 3.4°, 3.0°,
  2.5°.
- The inferred turn did not follow the header rotation of the knee (correlation 0.06).
- **The frame was a consistent function of the knee, not noise:** inferred from a knee and from
  its mirror image (100 side-certain knees), nod, turn, the front and up shifts and the three
  sizes correlated 0.97 to 0.99. Roll (0.86, difference SD 3.0°) and the medial shift (0.75,
  2.1 mm), the two that move the midline, were the least reproducible.
- Drawing `regions-021.png`: a knee that stands upright in the coronal image got a roll of
  −13.8°; the midline and the core leaned across the lateral condyle and the height cuts sloped
  across the joint. So a consistent frame is not always an anatomical one.

**Reading of runs 18 and 19.** With hard cuts, a label loses whatever its region leaves out, and
nothing pulls a region wider (the sizes did not move under the soft gradient). The regions of the
table came from each label's three highest-attention tokens, which is too tight for that:
Synovitis kept 69% of its all-token attention in the femur band, the menisci about half in the
joint band, Baker's 91% without the tibia band.

## Run 20: the wider region table

Run `outputs/training/20261006-144444-side-routed-regions-wider-table`: run 19 with the region
table from the all-token attention (menisci on the femur and joint bands of their side, MCL and
Baker's at all heights, ACL on the core and the lateral quadrants above the tibia band, Synovitis
unrouted; OA, PF OA and Effusion unchanged).

- ×4 macro 0.923483 against 0.923473 for run 19, 0.924119 for run 11 and 0.924845 for run 7; ×1
  0.915464.
- Against run 19: Synovitis +0.0056, PF OA +0.0029, ACL +0.0027, Medial Meniscus +0.0015,
  Effusion +0.0013, Lateral OA +0.0012, Baker's +0.0010, Lateral Meniscus +0.0009, Contusion
  −0.0013, Medial OA −0.0020, Fracture −0.0034, MCL −0.0102. The labels whose regions were
  widened all recovered; the macro did not move because MCL fell by 0.010 (it ranged from 0.9096
  to 0.9236 over the day's runs, 47 positives).
- Against run 11 (octants with the same per-knee frame head): Contusion −0.0036, Synovitis
  −0.0036, Baker's −0.0034, Medial OA −0.0016, PF OA −0.0016, MCL +0.0038, the rest within
  ±0.0014.
- Frame: nod −3.6°, |roll| 8.3°, upper cut −1.7 mm, lower cut −27.5 mm, core radius 12.5 mm.
- Family means of the ×4 macro: octants (runs 5, 6, 7, 8, 10, 11) 0.92444; eighteen regions with
  overlap (12 to 16) 0.92401; eighteen regions with hard cuts (18, 19, 20) 0.92336. With a
  run-to-run SD of 0.0003 to 0.0004 the last family was about three SDs below the first.
- Drawings (`outputs/analysis/s/side-routing-loop-20261006/side-routed-regions-wider-table/`): in
  study 5 the head moved the centre 8 mm down, onto the joint line, which is anatomically right;
  and it turned the frame −12.9° and rolled it −9.6°, so on the axial slices the midline no longer
  ran along the notch and on the coronal slices the midline and the core leaned across a knee
  that stands upright. The start frame (headers) looked better aligned than the inferred one in
  turn and roll.

## Run 21 build: per-knee centre, nod and sizes; roll and turn shared

Agreed with the user on 2026-10-06 after the drawings of runs 19 and 20. On run 20's build, the
frame head still inferred a frame for every knee, but only the parts a knee really differs in: the
centre (three shifts), the nod (how bent the knee is) and the three sizes. Roll and turn were one
learned value each for all knees, as in run 7. The landmarks gave the nod only: the tilt of the
front and up landmarks in the front-up plane. Zero start unchanged; all nine guardrails passed.

Run `outputs/training/20261006-151245-side-routed-regions-nod-per-knee` (run 21):

- ×4 macro 0.923947 against 0.923483 for run 20 (+0.0005), 0.924119 for run 11 and 0.924845 for
  run 7; ×1 0.915177. The best of the hard-cut eighteen-region runs.
- Against run 20: Synovitis +0.0027, Baker's +0.0024, MCL +0.0023, Lateral Meniscus +0.0014,
  Lateral OA +0.0014, Medial OA +0.0012, Medial Meniscus +0.0001, Effusion −0.0003, Contusion
  −0.0004, PF OA −0.0015, ACL −0.0017, Fracture −0.0020. Against run 7 the routed five summed to
  −0.0006; Contusion −0.0029 and Fracture −0.0053 remained.
- Frame over the validation knees: shared roll −0.9° and turn −1.9°; nod −15.3° (every knee
  negative); centre shift up −3.6 mm; upper cut −0.2 mm, lower cut −30.8 mm, core radius 13.5 mm.
- Drawings (`outputs/analysis/s/side-routing-loop-20261006/side-routed-regions-nod-per-knee/`):
  the midline was upright on the coronal slices and ran along the notch on the axial ones. On the
  sagittal slices the height cuts sloped down towards the back of the knee, following the
  backward slope of the tibial plateau, so the joint band lay along the plateau. The same slope
  lifted the upper cut at the front, which took the lower kneecap out of the PF OA and Effusion
  region; in the octant runs the nod went the other way (+10°) for exactly that label.
- So one height cut cannot both follow the plateau and reach under the kneecap, which led to
  run 22.

## Run 22: the kneecap labels read the front above the lower cut

Run `outputs/training/20261006-153716-side-routed-regions-kneecap-above-lower-cut`: run 21 with PF
OA and Effusion on the front quadrants above the lower cut (femur and joint bands) plus the core
of the femur band, so their region no longer depended on the upper cut's slope.

- ×4 macro 0.924358 against 0.923947 for run 21 (+0.0004) and 0.924845 for run 7 (−0.0005); ×1
  0.916091. The hard-cut region runs climbed 0.923121, 0.923473, 0.923483, 0.923947, 0.924358.
- Against run 21: Fracture +0.0043, PF OA +0.0030, Lateral OA +0.0019, Medial Meniscus +0.0007,
  Effusion +0.0005, Medial OA −0.0009, Baker's −0.0009, Contusion −0.0011, Lateral Meniscus
  −0.0019. PF OA moved as predicted. Against run 7 the routed five were +0.0019 summed; Contusion
  −0.0040.
- Frame: nod −14.1°, centre shift up −3.5 mm; lower cut −16.5 mm (it moved from −30.8), upper cut
  −5.4 mm, core radius 17.8 mm.

**Where the heads look in knees with and without the finding** (this model, 400 validation
knees, attention-weighted position of each label's ten highest-attention tokens, geometric frame,
mm; `outputs/analysis/s/side-routing-loop-20261006/spots-side-routed-regions-kneecap-above-lower-cut.json`):

| Label | Positive knees: mean (SD) | Negative knees: mean (SD) |
|---|---|---|
| Medial Meniscus | 26.5, −5.3, −5.0 (4.9, 5.8, 6.8) | 25.0, −7.1, −5.9 (4.1, 5.0, 5.6) |
| Lateral Meniscus | −29.1, −3.9, −6.7 (5.8, 5.3, 6.8) | −26.3, −4.7, −5.9 (4.3, 5.2, 6.4) |
| ACL | −5.8, −4.3, −1.7 (4.5, 4.5, 5.1) | −5.4, −1.4, −0.3 (4.2, 3.9, 5.3) |
| PF OA | −2.5, 36.4, 13.1 (5.3, 9.8, 7.9) | −1.4, 31.6, 13.3 (4.4, 8.4, 8.1) |
| Medial OA | 21.6, −1.6, −18.8 (4.3, 5.2, 8.3) | 21.6, −3.9, −14.7 (5.4, 4.8, 8.6) |
| Lateral OA | −23.9, −3.6, −15.5 (6.5, 6.0, 8.2) | −20.4, −5.0, −12.8 (4.6, 5.3, 8.1) |

The heads look at the same place whether the finding is there or not, and the ten tokens held 18%
to 36% of the attention in both groups. From the two meniscus spots: distance 52.8 mm (SD 6.5),
midpoint −0.6, −5.5, −5.8 (SD 3.2, 4.7, 5.9), turn of the line between them −2.0° (SD 5.4), roll
0.6° (SD 4.8). A spot's SD of 4 to 6 mm mixes real differences between knees with the token grid
(12 to 17 mm), so the turn and roll from two spots carry an uncertainty of a few degrees.

## Run 23 build: the frame hung on the two meniscus spots, eight sections per level

Agreed with the user on 2026-10-06 ("learn everything" again, tied to structures, and eight
sections). On run 22's build:

- **Frame from the heads' own attention.** For each knee, the shared meniscus head's attention
  over the medial half and over the lateral half of the geometric frame gave two spots (centroids
  weighted by the squared attention, no gradient). The centre was their midpoint plus one shared
  learned shift; the side axis was the line between them, which fixed turn and roll per knee. The
  nod and the three sizes stayed regressed per knee as in run 21. The free regression of the
  centre and the centre landmark were removed.
- **Eight sections per level.** The two diagonals split each quadrant into a 45° section next to
  the side axis and one next to the front-back axis. Every diagnosis label kept its union. The
  tear-location targets became three per side: anterior horn (anterior section), body (the two
  sections next to the side axis) and posterior horn (posterior section), in the joint band.
- Guardrails: the frame no longer started at the geometric one (untrained heads give spots near
  each half's centroid: first step on four studies, centre within 7 mm and axes within 3°), so
  "learned geometry starts at zero" checked the shared rotation and the prior only; the mirror
  check compared the geometric position, since a frame read from the image differs between a knee
  and its mirror image; the side-flip check mirrored the frame with the side. All nine passed.

## The shape read from the heads' attention (2026-10-06)

Asked by the user: extract where the heads look and find the division of the knee that matches
it. Scripts in `scripts/adhoc/side_routed_session/` (`extract_attention.py`,
`analyse_attention.py`); outputs in `outputs/analysis/s/side-routing-attention-shape-20261006/`
(`attention.npz`, `heads-1..4.png`, `emergent-{8,12,18}-{geometric,spots}.png`, `shapes.json`).

- **Source.** Run 11's model on the 874 validation knees with a fitted centre: the 35 report
  question heads (v3.1, each over the whole knee) and ACL, Synovitis, Contusion and Fracture were
  free; the other eight labels were routed in that model and kept apart. Per knee and head, the
  ten most attended tokens with their attention, on knees where the finding is present.
- **Score.** The knee in 6 mm cells; a division was scored by how much of what separates the
  heads it keeps (mutual information between section and head, as a share of the value at single
  cells). Cuts, core radius and sector orientation were placed where they scored best.

| Division | Sections | Free heads | Diagnosis labels | Best cuts (mm from the geometric centre) |
|---|---:|---:|---:|---|
| Octants | 8 | 43.0% | 46.4% | |
| Quadrants × 3 heights | 12 | 46.4% | 51.4% | 0, +18 |
| Quadrants × 4 heights | 16 | 49.1% | 54.5% | −24, 0, +12 |
| Quadrants + core × 3 heights | 15 | 50.9% | 55.2% | core 12 |
| 6 sectors × 3 heights | 18 | 56.4% | 60.1% | −24, +12; sectors centred on the axes |
| 8 sectors × 3 heights | 24 | 57.7% | 61.9% | −24, +12; sectors centred on the axes |
| 12 sectors × 3 heights | 36 | 61.2% | 65.1% | −24, +12 |
| 8 sectors + core × 3 heights | 27 | 61.4% | 65.0% | core 12 |
| 8 sectors + core + outer ring × 3 heights | 51 | 64.4% | 68.5% | core 12, ring from 30 |
| Emergent (neighbouring cells grouped by who looks there) | 8 / 12 / 18 / 24 | 55.2 / 59.4 / 64.4 / 66.3% | 57.7 / 62.6 / 66.9 / 68.4% | |

- Going round the axis matters most (4 to 8 sectors: +11 points), then the core (+4 to +5) and
  the outer ring (+3); a third height adds +3. Eight emergent sections keep 55% where the octants
  keep 43%.
- The best sectors are centred on the axes (a medial sector, an anterior one, and so on), not
  bounded by them: the midline and the front cut run through the kneecap sector and the medial
  sector.
- Best height cuts for the sector divisions: 24 mm below and 12 mm above the geometric centre,
  about 10 mm below and 26 mm above the joint line.
- Centring every knee on its own two free meniscus spots (distance 50.2 mm, SD 5.7; midpoint SD
  3.0, 4.3, 6.2 mm) raises the sector divisions by 1.5 to 2.5 points and the eight emergent
  sections from 55.2% to 58.7%.
- **What emerges with eight sections:** a central column (the notch); the inner medial
  compartment; an outer medial shell that also wraps the back; the inner lateral compartment; the
  outer lateral and lower part; the front upper region (kneecap and pouch) with a small centre on
  the kneecap; an anterolateral piece at the joint. With twelve: the tibia below, the
  posteromedial corner and the kneecap split further.
- **Each label is a compact blob about 15 to 20 mm across** (`heads-1.png`): ACL at the centre;
  MCL at 35 to 45 mm medial; the menisci at 20 to 35 mm on their side, just behind the side axis
  and 5 to 15 mm below the centre; the OA sides at about 20 mm, 5 to 15 mm below; PF OA within
  10 mm of the midline, 25 to 45 mm in front, 10 to 25 mm up; Effusion and Synovitis in front and
  20 to 50 mm up; Baker's 20 to 40 mm medial, 25 to 35 mm behind; Contusion and Fracture spread,
  more lateral.

## Run 23 result

Run `outputs/training/20261006-160557-side-routed-meniscus-spot-frame-eight-sections`:

- ×4 macro 0.924251 against 0.924358 for run 22 (−0.0001) and 0.924845 for run 7 (−0.0006); ×1
  0.916249 against 0.916091. Level with run 22.
- Against run 22: Lateral Meniscus +0.0015, Medial OA +0.0014, ACL +0.0011, Contusion +0.0008,
  Baker's +0.0006, MCL +0.0004, PF OA +0.0003, Effusion −0.0002, Medial Meniscus −0.0014,
  Synovitis −0.0014, Fracture −0.0017, Lateral OA −0.0027. Against run 7: PF OA +0.0018, Effusion
  +0.0012, Lateral OA +0.0011; Contusion −0.0032, Fracture −0.0027, MCL −0.0021.
- Frame from the two meniscus spots, over the validation knees at epoch 8: centre 7.4 mm down and
  2.4 mm back, roll 0.0° (4.0° in absolute value), turn −2.5° (4.9°), nod −17.8°; upper cut
  −4.8 mm, lower cut −13.5 mm, core radius 16.6 mm.
- Drawings (`outputs/analysis/s/side-routing-loop-20261006/side-routed-meniscus-spot-frame-eight-sections/`):
  the midline stayed upright on the coronal slices and close to the notch on the axial ones
  (study 21: roll −3.2°, turn −5.9°, centre 8.8 mm down, on the joint line). The joint band was
  thin (9 mm) and sloped 18°.
- Training took longer than usual because the attention extraction shared the GPU.
- **Logging.** Runs 12 to 23 logged about 70 more values than run 7 (607 against 530), most of
  them on every training and validation step through Lightning's logger. From run 24 on, every
  diagnostic value was accumulated in plain tensors and written once per epoch
  (`_log_epoch_metric`, `_flush_epoch_metrics`); only `train_loss` and the gradient norms stayed
  per step. The values in the log are the same weighted means as before.
