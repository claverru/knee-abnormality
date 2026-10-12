# Soft-tissue reads campaign: prior work

Everything here was measured on fold 1 of `all_5fold_scanner_v2`, seed 1110484336, one run per
design, at ×4 unless said. One run is weak evidence: each figure carries its number of
positives. "References" means the two trainings of the recipe before the cell read:
`sagittal-fluid-canvas-fine` (×4 0.928121) and its retraining `rn18-reference-retrained`
(0.928120). Their scores on a label correlate 0.994 to 0.997 (Spearman, the five soft-tissue
labels) and differ by up to 0.0046 on a label.

## The masks

| Mask set | Classes | Knees | Source | Prepared set |
|---|---|---|---|---|
| Bones | femur, tibia, kneecap, fibula, joint zone, kneecap joint | 550 | MRAnnotator on 2D sagittal T1/PD | `anatomy-mrannotator-sample550-joints` |
| Menisci and cartilage | medial and lateral meniscus, femoral, medial tibial, lateral tibial and patellar cartilage | 263 | `aagatti/nnunet_knee` on 3D DESS | `soft-tissue-nnunet-knee-dess265` |

- All soft-tissue knees come from one scanner model (MAGNETOM Avanto fit, 1.5 T): 211 in
  training, 52 in the campaign fold's validation. The masks are 1 mm volumes in patient
  coordinates and supervise every series of the study.
- No masks exist for the ligaments, effusion, the pouch, Baker's cysts, tendons or the fat pad.
  The segmenter fails on routine 2D series (0 to 5 of 20 menisci pass).
- Mask-derived measurements do not separate the labels: on the 263 mask knees no volume or
  distance reaches 0.71 AUC for any of the five labels (medial meniscus volume against Medial
  Meniscus 0.703), and most of that follows knee size.

## The reference: `rn18-soft-tissue-added-read-bone-routes` (run 2)

×4 0.928909, ×1 0.918680, 1,599 s of training. +0.00079 against both references (above the
recipe in 91% of resamples, 5 to 95% interval −0.0002 to +0.0017).

- **A, the 4th-stage read.** All twelve labels read the 7 by 7 tokens with the recipe's regions,
  position prior and knee-level term.
- **B, the 3rd-stage cell read.** Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA and
  PF OA add a logit read from the 14 by 14 cells of the third stage
  (`KneeAbnormalityModel._soft_tissue_pool`). Its tokens are the projected third-stage features
  plus the plane and contrast embeddings: no position, no 4th-stage context. One softmax per
  label over all the open cells of the study, across views. One attention and readout channel is
  shared by the two menisci and one by the three OA labels. The readout starts at zero and has no
  bias.
- **The routing of B** (`_soft_tissue_routing_weights`), all from the detached seven-class map:
  - Menisci: cells with at least 1/32 of the knee's peak share of the meniscus, grown by one
    cell, weight the larger of the share and its 3 by 3 mean, kept only if nearer to the own
    meniscus centre than to the other.
  - OA: each cell belongs to the nearest of three compartment centres (medial tibial cartilage
    plus medial meniscus, the lateral pair, patellar cartilage); weight the larger of a
    cartilage rim (own cartilage plus the cell's femoral cartilage, scaled per knee, capped at 1,
    one 3 by 3 mean) and the bone map's joint zone (kneecap-joint class for PF OA).
  - Below one cell of total weight a label is unrouted.
- **Bone routing.** Contusion and Fracture are routed on the 4th stage by each token's bone-map
  share of femur, tibia, kneecap and fibula.
- **One loss on A + B.**

Logged at the selected epoch (validation): effective open cells 44 and 53 for the menisci, 196
to 303 for the OA labels; every study routed; map overlap 0.54 and 0.55 for the menisci, 0.48 to
0.58 for the cartilages; mean absolute B logit 0.09 to 0.16.

What B is worth there, same forward pass with and without it, 874 studies:

| Label | A alone | A + B | Change | B alone | Spread of A's logit | Spread of B's |
|---|---|---|---|---|---|---|
| Medial Meniscus | 0.9454 | 0.9462 | +0.0008 | 0.844 | 1.73 | 0.082 |
| Lateral Meniscus | 0.9104 | 0.9122 | +0.0018 | 0.794 | 1.36 | 0.075 |
| Medial OA | 0.9275 | 0.9282 | +0.0007 | 0.808 | 1.33 | 0.064 |
| Lateral OA | 0.9104 | 0.9126 | +0.0022 | 0.807 | 1.06 | 0.064 |
| PF OA | 0.9027 | 0.9032 | +0.0005 | 0.719 | 1.24 | 0.063 |

## The other runs of the line

| Run | Design | ×4 against the references | Source |
|---|---|---|---|
| 1, `rn18-soft-tissue-cells` | The five labels read only B | −0.0044 | `outputs/training/rn18-soft-tissue-cells` |
| 2, the reference | A + B, one loss, bone routing | +0.0008 | `outputs/training/rn18-soft-tissue-added-read-bone-routes` |
| 3, `rn18-soft-tissue-read-loss-wide-regions` | Run 2, B also has its own loss, wider regions | −0.0011 | `outputs/training/rn18-soft-tissue-read-loss-wide-regions` |
| 4, `rn18-soft-tissue-wide-regions` | Run 2 with the wider regions only | +0.000002 | `outputs/training/rn18-soft-tissue-wide-regions` |

- **Wider regions** (runs 3 and 4, [the change](wide-regions.diff)): the OA rim is the larger of
  one and two 3 by 3 mean passes; a meniscus label also opens the cartilage rim of its own
  compartment. Open cells rise from about 50 to about 300 per meniscus label.
- **Run 3: a loss on B's own logit swaps the reads.** B became the reader (0.891 to 0.946
  alone) and A shrank to a correction (0.73 to 0.83 alone, logit spread 0.25 to 0.39).
- **Run 4: B's gain is paid by A.** B adds +0.0016 on average over the five labels (+0.0012 in
  run 2), but A alone ends 0.0018 lower than in run 2, so the sum is level with the references.
- Across runs 2, 3 and 4 the more B can do, the less A keeps: the two share the work of one loss.
- Fracture is ahead of both references in runs 2, 3 and 4 (+0.0030 to +0.0048); Contusion is
  mixed.
- Earlier, on the 7 by 7 tokens: routing the labels by structure was level at best (−0.0003;
  MCL −0.0099 when read from a bone-derived band), soft-tissue shares added to the geometric
  regions gave −0.0007, hard regions from the masks −0.0023. A token is 12 mm and 6% meniscus
  where it is called a meniscus token; an open cell is 23 to 31% meniscus.

## What B sees that A does not (2026-10-10, from the saved ×4 predictions)

Run 1's reader is B trained alone, so its results by subgroup show what a cell read is good and
bad at. Changes are against the mean of the two references; positives of the subgroup against
all negatives of the label.

- **Different information.** Run 1's scores correlate 0.90 to 0.96 with a reference on the five
  labels, where the two references correlate 0.994 to 0.997. Blending run 1 into the retraining
  at logit weight 0.1 to 0.3 gives +0.0003 on the mean of the five (weight chosen on the
  validation labels).
- **B wins where the map is good.** On the mask scanner's 52 validation studies run 1 is ahead
  on the three labels with enough positives: Medial OA +0.023 (12 positives), PF OA +0.037 (14),
  Medial Meniscus +0.003 (19); mean +0.021, above zero in 98% of resamples (5 to 95%: +0.004 to
  +0.040). Run 3, whose reader is also B, gives +0.015 (99%). On the other 822 studies run 1 is
  behind on all five (−0.004 to −0.028). Two candidate causes, not separated: the map is better
  on the scanner it was taught on, or that scanner's 3D DESS series suits a fine read.
- **The map's grid is its ceiling.** Overlap rose from 0.45 on the 7 by 7 tokens to 0.55 on the
  14 by 14 cells, and on the tokens it was flat from epoch 5 with training equal to validation.
- **Partial cartilage damage is where B loses.** PF OA: partial low grade −0.031 (42
  positives), partial high grade −0.028 (33), focal extent −0.062 (15), full thickness −0.0015
  (109). Lateral OA: partial high grade −0.093 (8), full thickness −0.002 (35). In the reference
  the bone zone holds 60 to 67% of an OA label's routing weight and receives 35 to 57% of its
  attention, in the same softmax as the cartilage.
- **B's result depends on where the tear is.** Lateral anterior horn +0.052 (10 positives; run 3
  +0.055), lateral horizontal tears +0.019 (25; run 3 +0.020), medial body +0.010 (16), medial
  radial +0.004 (26); medial posterior horn and body −0.009 (107), lateral posterior −0.028 (21).
- **PF OA is B's weakest label**: 0.719 alone in the reference against 0.807 and 0.808 for the
  other two OA labels that share its reader, the smallest gain from B (+0.0005; +0.0003 in run
  4), −0.010 in run 1.
- **What B alone cannot see, and A supplies:** disease in the other compartments (Lateral OA
  −0.030 with another OA present, 79 positives, +0.015 with none, 16) and displaced or truncated
  menisci (medial displaced −0.009, 40; lateral displaced or truncated −0.021, 23).

## Precedents

| Mechanism | Setting | Run | Macro change at ×4 | Source |
|---|---|---|---|---|
| The five labels read only the cells | Recipe | `rn18-soft-tissue-cells` | −0.0044 | This document |
| Cell read added to the 4th-stage read, bone routing | Recipe | `rn18-soft-tissue-added-read-bone-routes` | +0.0008 | This document |
| Own loss on the cell read, wider regions | Run 2 | `rn18-soft-tissue-read-loss-wide-regions` | −0.0019 against run 2 | This document |
| Wider regions | Run 2 | `rn18-soft-tissue-wide-regions` | −0.0008 against run 2 | This document |
| Meniscus map as a side task, nothing reading it | Anatomy-map recipe | `rn18-meniscus-map` | −0.0009 | [View poses](../view-poses/prior-work.md) |
| Meniscus head moved from 7 by 7 to 14 by 14 | View poses | `meniscus-head-on-the-14-by-14-grid` | +0.0001, overlap +0.11 and +0.13 | [View-poses run notes](../view-poses/run-notes.md) |
| Labels routed by structure on the 7 by 7 tokens | Recipe | `rn18-structure-routes-no-side` | −0.0003 | [Structure-routing runs](../view-poses/structure-routing-runs.yaml) |
| Soft-tissue shares added to the geometric regions | Recipe | `rn18-soft-tissue-routing` | −0.0007 | [Structure-routing runs](../view-poses/structure-routing-runs.yaml) |
| Crops centred on the menisci, T1/PD | View poses | `t1pd-joint-crops-centred-on-the-menisci-calibrated` | −0.0021 | [View-poses run notes](../view-poses/run-notes.md) |

## Dropped before the start, and why

- **Masks predicted by the model for every study** (the user, 2026-10-10): not built.
- **B reading the other compartments, or reaching displaced menisci:** once B is a correction on
  A, A already carries that evidence.
- **A cell read for MCL:** MCL differs by 0.0046 between the two references, so a result could
  not be read.
- **Mask measurements as features:** see the masks section.

## Open leads for the researcher

- Finer cells for PF OA (second stage, 28 by 28), after the pooling and reader seeds.
- B scaled per study by the map's sharpness: sharpness per scanner is not measured; a seed must
  log it.
- The map's loss weight (0.1 today): no evidence that it limits the map.
- What separates the mask scanner's gain: the map's quality or the 3D series.
