# View pose: study notes

**Scripts are gone:** the one-off scripts of studies 1 to 7 lived in the session's scratch folder
under `/tmp`, which a WSL restart on 2026-10-08 wiped. Their outputs, tables and figures under
`outputs/analysis/` are intact; where a section names a script (including the fast segmenter
runner `fast_nnunet_knee.py`), it has to be rewritten to be run again.

Collection of the studies behind "a crop per knee and per plane-contrast", written at the
repository root on 2026-10-08 and moved here on 2026-10-10. The one-off scripts it names under
`scripts/adhoc/meniscus_masks/` were not kept. One section per study, written when its
agent returns.

**Terms.** *Pose*: a crop's position, rotation and scale. *Canvas*: the fixed image the loader
cuts before the model sees anything; it only has to be large enough and sharp enough. The crop is
made from the first look's 3D anatomy map of the knee, inside the canvas.

**Agreed so far (user, 2026-10-08).**

- All three planes get a pose (shift, turn, scale), per plane-contrast, not only sagittal and
  coronal.
- No hand-measured correction constants on the centre, plane or pose; improve the learning.
- The label loss must not choose the crop (it zoomed out in the past).
- The current model's attention is biased by the crops it was trained on; it cannot decide what
  each view should frame.
- Masks are needed on a subset of knees only, as today; a segmenter has to be good on some
  studies, with a way to tell which.
- Idea (user): if the soft-tissue segmenter only works on 3D series, make the masks on the 3D
  series in our data (836 series in 659 training studies: 268 Siemens DESS, about 147 studies
  with a sagittal 3D PD fat-suppressed series), supervise the map on those series' tokens only,
  and let the model generalise to 2D series. They are already model inputs (DESS in the axial
  fluid slot, 3D PD fat-suppressed in the sagittal fluid slot). Study 3 tests the segmenters on
  2D and on 3D series to tell which case holds. Transferring a 3D-made mask to the 2D series of
  the same study is an optional extra, checked in study 3b.

**Outside models: the rule is about the model, not its training data (user, 2026-10-08).** A
segmenter qualifies by its weights' own licence and free access. Checked: `aagatti/nnunet_knee`
weights MIT (model card; femur, tibia, kneecap, cartilage, menisci; trained on OAI-ZIB DESS);
SKM-TEA weights Apache-2.0 and ungated (`huggingface.co/arjundd/skm-tea-models` metadata; they
take qDESS echoes as input, so they suit our 3D DESS series). Regions inferred from the bone
masks are enough for the crops and the routing; real soft-tissue masks would only sharpen the
map's extra signal.

| Study | Status | Outputs |
|---|---|---|
| 1. Knee size and per-knee scale | back | `outputs/analysis/knee-size-crops/` |
| 2. Attention per view in the knee frame | back | `outputs/analysis/view-attention/` |
| 3. Soft-tissue segmenters | back | `outputs/analysis/soft-tissue-segmenters/` |
| 3b. Census of our 3D series; alignment of 3D-made masks with 2D series | back | `outputs/analysis/soft-tissue-segmenters/3d-series/` |
| 4. Which targets are seen in which plane-contrast | back | `outputs/analysis/targets-by-view/` |
| 5. Do the errors sit where the crop cut the finding? | back | `outputs/analysis/errors-vs-coverage/` |
| 8. Meniscus masks from DESS | built | `data/labels/prepared/meniscus-nnunet-knee-dess265/` |

## 1. Knee size and per-knee scale (back)

Masks on 550 knees; model estimates on the 150 fold-1 validation knees with the incumbent
(`report-attribute-loss-off-2`). Scripts in the session scratchpad (`knee-size/`).

**Knee size varies and the scanner does not normalise it.**

| Measure (550 knees) | Mean | sd | 5th to 95th |
|---|---:|---:|---|
| Femur width | 73.7 mm | 6.3 | 65.0 to 83.9 |
| Tibia width | 67.5 mm | 6.1 | 57.7 to 77.5 |
| Femur depth, front to back | 56.3 mm | 5.0 | 49.6 to 65.0 |

- Width and depth correlate 0.76; femur and tibia width 0.91. Size itself varies by 8.0%.
- Acquired field against knee size: r 0.15, 2% of the size variance. Native pixels average 0.34
  to 0.37 mm (95th percentile 0.56 to 0.66).

**How much of today's crop the anatomy fills** (mean, 5th to 95th):

| View | Measure | Fill | Knees over the field |
|---|---|---|---:|
| Coronal 84 mm | bone width | 0.87 (0.76 to 0.99) | 3.8% |
| Coronal 93.3 mm | bone width | 0.78 (0.68 to 0.89) | 0% |
| Sagittal 84 mm | femur back to kneecap front | 0.93 (0.81 to 1.05) | 20.2% |
| Sagittal 93.3 mm | femur back to kneecap front | 0.84 (0.73 to 0.94) | 0.7% |
| Axial 120 mm | femur width | 0.62 | 0% |
| Axial 120 mm | femur back to kneecap front | 0.65 | 0% |

**The model can size the knee from its first look.**

- Estimator: root of (share-weighted spread of femur and tibia tokens across) × (spread of femur
  tokens front to back), in the fitted frame. Against the masks: r 0.939, residual 2.1 mm (2.8%).
- Using two planes per direction is better than one (width from axial + coronal r 0.910; coronal
  alone 0.848).
- One shared size number leaves the fill varying by 2.7% (coronal) and 2.6% (sagittal); each
  plane's own measure gives 1.6% and 0%. Separate per-plane scales from the model gave no gain
  over the shared one (3.7% / 3.7% against 3.6% / 3.8%).

**Scaling equalises the framing; it does not widen it** (150 knees, share of series with the
structure fully inside):

| View | Structure | Today | Per-knee scale | Wider for all |
|---|---|---:|---:|---:|
| Coronal fluid | collateral region (bone edge + 10 mm) | 34.8% | 17.4% | 93.5% |
| Coronal 84 mm | femoral condyles | 91.6% | 98.5% | 100% |
| Sagittal fluid | whole kneecap | 26.0% | 15.3% | 88.7% |
| Sagittal 84 mm | whole kneecap | 7.8% | 1.1% | 56.4% |

Small knees lose the spare margin they have today; large knees gain (femoral condyles inside the
84 mm coronal crop, largest third: 73% to 95%).

**Axial: the problem is position, not size** (198 series, 150 knees).

- The bones sit 14.8 mm forward of the crop centre (sd 5.9); 6.8 mm of margin in front of the
  kneecap (sd 7.2; its front is cut in 13% of series), 36.4 mm behind the condyles.
- Kneecap front plus 25 mm behind the condyles inside: today 84.3%, resize only 84.3%, centre
  only 91.9%, centre and resize 98.5%. The turn was not measured.
- Centring on the bones themselves pushes the back of the knee out (12.6% inside).
- Axial has no canvas margin today.

**Run on it** (anatomy-recipe run 7, `crop-size-per-knee-from-the-first-look`): one shared scale
for the four sagittal and coronal views, axial untouched. ×4 0.927608 against 0.927291
(+0.0003; 80% −0.0009 to +0.0015), ×1 0.917680 against 0.917342. Mean applied scale 1.000,
spread 0.067, 5.1% of knees at the clamp. Up: MCL +0.0085, Synovitis +0.0025, Medial OA +0.0018.
Down: Fracture −0.0041, Lateral OA −0.0032, Lateral Meniscus −0.0028. It used a measured
reference (17.087) and clamp (0.87 to 1.13), which the agreed direction replaces by boxes read
from the map.

**Limits.** One checkpoint; the reference came from validation knees; coverage was measured in
the plane of the middle slice; the collateral and popliteal regions are proxies (bone edge plus a
margin), since the masks have bones only.

Drawings: `crops-coronal-fluid.png`, `crops-sagittal-fluid.png`, `crops-axial.png`,
`size-summary.png`.

## 2. Attention per view in the knee frame (back)

One pass of the incumbent over all training-fold studies: 3,517 kept (5 two-knee and 11
uncertain-side studies excluded), 94,250 images, 4.6 million tokens. Positions in the first
look's knee frame. **Bias, agreed with the user:** the model was trained on today's crops and
attends only inside them, and its attention includes the hand-built routing regions and the
learned position prior. This study describes the current model; it cannot decide what a view
should frame. Its unbiased parts are marked.

**Planes differ; the two contrasts of a plane read the same region.**

- Sagittal: the joint line and the front-upper quadrant (kneecap, pouch); almost nothing below
  −20 mm.
- Coronal: two blobs on the joint line at 20 to 35 mm medial and lateral; the full crop width
  but only about 55 mm of its height.
- Axial: a medial-and-front horseshoe on slices 5 to 35 mm above the joint.
- Fluid against T1/PD of the same plane: centroid moves at most 2.7 mm, map overlap 0.83 to
  0.90. Contrast changes how much a view is used (fluid gets 1.1 to 3.7 times the share), not
  where.

**Share of label attention per view:**

| View | Mean | Highest labels |
|---|---:|---|
| Axial fluid | 0.266 | Baker's 0.62, Effusion 0.46, Synovitis 0.35 |
| Coronal fluid | 0.229 | Lateral Meniscus 0.35, Medial Meniscus 0.35, MCL 0.30 |
| Sagittal fluid | 0.226 | PF OA 0.30, ACL 0.30, Medial Meniscus 0.27 |
| Sagittal T1/PD | 0.146 | ACL 0.24, PF OA 0.22 |
| Coronal T1/PD | 0.109 | Medial OA 0.17 |
| Axial T1/PD | 0.023 (0.116 when present) | present in only 694 studies |

**Facts about today's crops (not biased by attention).**

- **Token lattice offset:** the ResNet's tokens are centred at pixels 0, 32, ..., 192, so what is
  read sits 15.5 px up and left of the crop centre (5.8 mm at 84 mm, 6.5 at 93.3, 8.3 at 120).
  On the coronal medial edge the outer token is on the border for one image side and 31.5 px
  inside for the other, so left and right knees get different medial context (MCL: 58 to 61% of
  its coronal attention on that line on one side, 21 to 22% on the other).
- **Shift limit:** 11 to 15% of sagittal and coronal images are clamped at 12.75 mm; none reach
  the 12° turn limit.
- **Axial is unposed:** its centre varies by 3.9 / 4.9 mm (fluid) and 6.1 / 9.5 mm (T1/PD) in the
  knee frame, and it is turned 5.1 to 5.3° (sd 6°) against the knee's axes.
- **Slices:** middle slices are over-read (the coronal middle slice takes 38% of attention for
  19 to 20% of tokens); the lowest axial fluid slice takes 4% for 16%.
- **Canvas margin** for turn 12°, shift 12.75 mm and a 17% size allowance: 28.5 to 30 mm
  (sagittal), 30 to 32 mm (coronal), 34.5 mm (axial) with the canvas centred on the joint; 62 to
  69 mm and 40 to 42 mm with today's canvas centring, because reaching the joint centre already
  takes up to 23 to 25 mm.

**Sides where the view probably needs more than today's field** (attention at or above uniform
on the outer token line; only Baker's stays above the routing-corrected null):

| View | Side | Labels |
|---|---|---|
| Sagittal, both | back | Baker's 0.45 to 0.48 |
| Sagittal fluid | up | Effusion 0.14 |
| Coronal, both | medial | Baker's 0.41 to 0.45, MCL 0.40 to 0.43, Medial Meniscus 0.20 to 0.25 |
| Coronal, both | lateral | Lateral Meniscus 0.18 to 0.22 |
| Coronal fluid | up | Effusion 0.15 |
| Axial, both | medial | MCL 0.18, Baker's 0.14 to 0.17 |

The axial front edge shows no pile-up (2% of attention), though attention cannot see a kneecap
already outside the crop. Most edges are spare: the outer ring holds 10 to 22% of attention
against 49% if uniform.

**Size.** The read extent grows with knee size but less than proportionally (log-log slope 0.28
to 0.50 across and front, 0.06 to 0.24 up), capped by the fixed field.

**Boxes it proposed** (2.5 to 97.5% of pooled attention plus one token each side, squared; for
reference only, given the bias): sagittal about the same size moved up 10 to 11 mm; coronal 1.13
times wider moved up 7 mm; axial the same size moved 6 to 7 mm forward and 3 to 4 mm medial. The
lattice offset accounts for all of the coronal and axial shift and half of the sagittal one.

Drawings: `attention-six-views.png`, `attention-labels-<view>.png` (six), `edge-pileup.png`.
Tables: `t1` to `t6` CSVs, `proposed-boxes.csv`; per-token table `tokens.parquet`.

## 4. Which targets are seen in which plane-contrast (back)

Two sources independent of our model: radiology literature, and visual inspection of wide,
uncropped contact sheets of 102 training studies (8 positives per target, 6 negatives; fold 1
excluded), read by agents with the report as a guide. **Not a radiologist's reading.** Calls are
reliable for conspicuous findings (fluid, cysts, marrow oedema, gross fractures) and weak for
fine ones: medial-meniscus calls were at chance against negatives and PF OA was never clear.

**Target by view** (inspection counts clear/subtle/not/cannot-tell are in
`synthesis-target-view.csv`):

| Target | Sag fluid | Sag T1/PD | Cor fluid | Cor T1/PD | Ax fluid | Ax T1/PD |
|---|---|---|---|---|---|---|
| ACL | **primary** | useful | useful | little | useful | little |
| MCL | little | none | **primary** | useful | useful | little |
| Medial Meniscus | **primary** | **primary** | useful | useful | little | none |
| Lateral Meniscus | **primary** | useful | useful | little | little | none |
| Medial OA | useful | useful | **primary** | useful | little | none |
| Lateral OA | **primary** | little | **primary** | useful | little | none |
| PF OA | useful | little | none | none | **primary** | little |
| Effusion | **primary** | useful | useful | little | **primary** | little |
| Synovitis | **primary** | little | useful | none | **primary** | none |
| Baker's | **primary** | useful | useful | little | **primary** | little |
| Contusion | **primary** | little | **primary** | useful | **primary** | little |
| Fracture | useful | useful | **primary** | **primary** | useful | useful |

Medial Meniscus rests on literature alone. Where the sources disagree: axial fluid showed ACL as
clearly as sagittal; coronal fluid showed the largest effusion area and the most conspicuous
synovitis, which the literature does not use; literature rates short-TE coronal T1/PD useful for
the lateral meniscus and inspection saw nothing there.

**What each view must contain** (distances from the joint line and plateau centre; landmarks
from 400 mask knees: kneecap top 55 mm above the joint line, 95th percentile 66; kneecap front
56 mm ahead of centre; posterior condyles 23 mm behind; bone width 73 mm):

| View | Wide need | Fine need | Conflict |
|---|---|---|---|
| Sag fluid | pouch to 80 to 125 mm above the joint line; cyst from 45 mm above to 35 mm below, 30 mm behind the condyles; bone 65 mm above to 50 to 60 mm below | meniscus and cartilage within ±10 mm of the joint line, 45 mm front to back; ACL on 1 to 2 slices within 7 mm of mid; 1 mm tear line, 2 to 3 mm cartilage | **yes:** about 180 mm of height against a 60 to 90 mm box |
| Sag T1/PD | adds little | 1 mm line at the joint | no |
| Cor fluid | skin to skin for both collaterals; MCL from 40 to 50 mm above to 60 to 70 mm below the joint line | joint space ±25 mm; 2 to 3 mm ligament and cartilage | less: about 110 mm high, centred slightly above the joint line, holds both; pouch and cyst are in the front and back slices, not in a larger field |
| Cor T1/PD | tibia to 60 mm below the joint line | 1 to 2 mm fracture line, 3 to 5 mm osteophytes | no |
| Ax fluid | whole limb cross-section for cyst and pouch, 130 to 150 mm | 70 × 40 mm box at the kneecap, 3 mm cartilage band, in 7 to 12 slices | **yes** |
| Ax T1/PD | – | fracture line, fat-fluid level | serves no target alone |

**Views.**

- Irreplaceable: sagittal fluid (ACL, menisci, pouch height), coronal fluid (MCL, the
  tibiofemoral joint), axial fluid (kneecap cartilage, the Baker's neck).
- Near-redundant with their fluid partner: sagittal and coronal T1/PD. They add the fracture
  line and osteophytes and, by literature only, short-TE meniscal signal.
- Axial T1/PD: never above "little"; present in 19% of training studies; the least consistent
  content (3D gradient echoes, a two-leg volume, a kneecap cartilage series). No target depends
  on it.
- **The "T1/PD" slot is not one contrast.** Sagittal held T2 in 45 of 102, T1 in 31, PD in 26;
  coronal T1 in 88; axial T1 in 36 of 46. Most clear fluid or cyst calls in sagittal T1/PD came
  from T2 or gradient-echo series.

**Scanned coverage** (mask-based over 120 to 433 series per view, or inspection of 102 studies):

| Need | Sagittal | Coronal | Axial |
|---|---|---|---|
| ≥ 30 mm above the kneecap top | 96 to 98% | 93 to 95% | 37% fluid, 22% T1/PD |
| ≥ 40 mm above the kneecap top | 78 to 83% | 73 to 74% | 16% / 6% |
| ≥ 30 mm behind the posterior condyles | 100% | 17 to 19% | 98 to 99% |
| ≥ 60 mm below the joint line (MCL's tibial end) | 70 to 79% | 62 to 68% | 3 to 5% |
| 10 mm beyond the bone edge, medial or lateral | 67 to 73% | 100% | 99 to 100% |
| Front of the kneecap inside | 100% | 62 to 71% | 99% |

The pouch is the most-cut region: median 45 mm of sagittal field above the kneecap top against
pouches reaching 25 to 70 mm above it; whole in the field in 46 of 102 studies. Coronal stacks
ended inside the Baker's cyst in 3 of 8. Axial stacks start a median 18 to 24 mm above the
kneecap.

**Per target, where and how big** (inspection, n 8 each): effusion bulk is off the midline in 5
of 8, 3 to 14 mm thick; Baker's 45 to 110 mm long, 5 to 40 mm medial of mid; contusions 20 to
35 mm patches, at the condyle periphery in 5 of 8, so on the 1 to 4 outermost sagittal slices;
MCL injury in the femoral half, 40 to 45 mm medial of the notch; ACL in a box 0 to 30 mm above
the joint line; fracture line seen in 2 of 8, to 55 mm below the joint line.

**Limits.** Sheets show 12 of about 30 slices per series; sizes are ±5 mm; tear lines,
cartilage depth and non-displaced fracture lines were not resolved, so those definitions were
not tested; some figures in the literature summary came from search summaries, not opened pages.
Sources are linked in the agent's report; ESSR 2024 protocol, Radiology Assistant, Radiopaedia,
MOAKS.

## 5. Do the errors sit where the crop cut the finding? (back)

Asked by the user: the labels are caught well, so how do we know the cut regions cost anything?
871 fold-1 validation studies, the incumbent's saved ×4 predictions, geometry from one ×1 pass.
For each label, AUC in the third of studies with the least field around its region against the
third with the most; rank of positives against field (ρ+). 70 label-covariate pairs, not
corrected: 7 AUC intervals exclude zero against 3.5 expected by chance.

**Answer: the crops' in-plane field does not explain the errors.** No label shows a clear
dependence on how much field the crop leaves around its region. Two effects appear, and both are
about which slices are read, not how wide the crop is.

| Label (positives) | In-plane field | What does show |
|---|---|---|
| Effusion (244) | cannot tell; vanishes once axial reach is controlled | **axial upper reach:** AUC 0.872 / 0.933 / 0.948 by third, −0.076 (95% −0.128 to −0.023), ρ+ 0.22 |
| Synovitis (114) | cannot tell | **axial upper reach:** 0.848 / 0.883 / 0.914, −0.067 (−0.145 to −0.001), ρ+ 0.29 |
| MCL (47) | cannot tell | **sagittal medial reach:** 0.860 / 0.970 / 0.976, −0.116 (−0.190 to −0.042), ρ+ 0.51; missed 13 of 18 in the least third. Fragile at 47 positives |
| Baker's (114) | no (largest −0.015) | – |
| Medial Meniscus (339) | no; more coronal field goes with lower positive scores (ρ+ −0.24) | – |
| Lateral Meniscus (135) | no (+0.059, opposite sign) | sagittal lateral reach −0.053, interval includes zero |
| Medial OA, Lateral OA | no (−0.025, 0.000) | – |
| PF OA (290) | no; whole kneecap inside or not: −0.045 (−0.097 to +0.024) | – |
| ACL (131) | no in AUC | small knees lose sensitivity: 30 of 43 missed positives in the smallest third (OR 3.5, p 0.002) |
| Contusion, Fracture | no | – |

- **Axial upper reach:** the highest axial slice read sits a median 1.5 mm above the kneecap top;
  in the least third it is below −2.7 mm, so the pouch is not in any axial slice read. The effect
  survives controlling for knee size and in-plane field, and is about half the size of lacking
  the sagittal fluid series (−0.134 for Effusion).
- **In-plane field is knee size in disguise:** it correlates −0.91 to −0.94 with the size
  estimate and spans only ±4 mm, so this sample cannot separate a field effect from a size
  effect, and neither shows clearly.
- **The weak labels:** Synovitis is partly explained by axial reach. Nothing measured explains
  the errors of PF OA, Lateral Meniscus or Lateral OA.
- **The shift clamp** (177 of 871 studies) changes no label's AUC.

**Limits.** Landmarks are constants times the size estimate, not per-study measurements, so only
the ordering of studies is used. Slice reach is measured at ×1, scores at ×4 (the ×1 scores
agree: Effusion −0.070, MCL −0.126). Few positives for MCL, Fracture, Contusion and Lateral OA.

## 3b. Our 3D series: census and alignment with 2D series (back)

836 3D series in 659 of 4,407 training studies (headers from the 2026-09-17 census); alignment
on 14 studies (7 DESS, 7 sagittal 3D PD fat-suppressed, 5 scanner models) with the bone
segmenter.

**How many studies could supply a mask from a 3D series: 413 with a fluid-sensitive one.**

| Family | Series / studies | Plane | Slice (mm) | Pixel (mm) | Slab (mm) | Scanners |
|---|---|---|---|---|---:|---|
| DESS | 268 / 265 | axial | 0.6 | 0.25 | 86 | 98.9% one Siemens Avanto fit 1.5T, German reports |
| 3D PD fat-suppressed | 148 / 148 | sagittal | 0.4 to 0.89 | 0.28 to 0.89 | 128 | Philips Achieva 57%, Siemens Skyra 3T 42% |
| 3D T1/T2* gradient echo | 216 / 214 | sagittal, axial | 1.0 | 0.31 | varies | four scanner groups |
| Zero-TE (GE) | 130 / 88 | sagittal | 1.0 | 0.39 | 120 | 94% one GE Architect 3T |
| Reformats of the 3D PD series | 50 / 50 | axial | 0.9 | 0.28 | 167 | Philips Achieva |

- **Concentration:** the 413 come from three scanner and site groups. There is no institution
  tag; "site" is report language plus scanner model.
- **Every 3D series holds one knee** (the 320 mm DESS field is 124 mm of body at the median).
- **They are already model inputs:** no filter uses the acquisition type; they are loaded as
  ordinary views in their plane and contrast slot (3.06% of training triplets).
- **Their 2D companions:** DESS studies have sagittal T1/PD (264 of 265), coronal fluid (265)
  and axial T1/PD (263), but a 2D sagittal fluid series in only 20. The 3D PD studies have
  sagittal T1/PD, coronal T1/PD and axial fluid, and a 2D sagittal or coronal fluid series in 1.
- **Folds:** DESS 53 / 52 / 40 / 60 / 60; 3D PD 30 / 33 / 26 / 36 / 23. 59 of the 413 are among
  the 550 knees with bone masks.
- **Their labels differ from the rest** (percent positive):

| Group | n | ACL | Lateral Meniscus | Medial Meniscus | PF OA | Effusion | Synovitis | Contusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All training | 4,407 | 14.9 | 15.4 | 38.8 | 33.1 | 27.8 | 12.9 | 10.5 |
| DESS | 265 | 8.3 | 7.2 | 29.1 | 32.5 | 41.9 | 17.0 | 15.8 |
| 3D PD fat-suppressed | 148 | 7.4 | 14.2 | 36.5 | 43.2 | 7.4 | 6.1 | 6.8 |

**Coverage of the 3D slabs.**

- DESS (43 studies): the slab is centred on the kneecap, 65 mm above the joint line and 26 mm
  below at the median. Whole kneecap inside in 43 of 43; the cruciate band in 42 of 43; under
  20 mm of tibia below the joint line in 7 of 43. It covers 42 to 57% of a 2D series' voxels.
- Sagittal 3D PD (30 masks): femur, tibia, kneecap and the joint band inside in 30 of 30, with
  10 to 29 mm beyond the bones on each side.
- Menisci and cruciates were placed from bone landmarks, not segmented.

**Optional extra: a 3D-made mask on the same study's 2D series.**

| 3D type | 2D plane | Dice femur / tibia / kneecap | Surface distance, median (mm) | Best rigid shift, median / max (mm) |
|---|---|---|---|---|
| DESS | sagittal | 0.92 / 0.93 / 0.84 | 0.7 to 1.1 | 1.4 / 2.25 |
| DESS | coronal | 0.94 / 0.88 / 0.82 | 1.0 to 2.0 | 1.1 / 2.1 |
| 3D PD | sagittal | 0.94 / 0.93 / 0.90 | 0.6 to 0.9 | 1.1 / 2.5 |
| 3D PD | coronal | 0.86 / 0.81 / 0.79 | 0.9 to 1.8 | 2.1 / 3.7 |

- Good enough for 12 to 17 mm tokens; marginal for menisci and cartilage, which need 1 to 2 mm,
  the size of the shift between series. About half the error is motion between series.
- **The bone segmenter is unreliable on 2D coronal series** (shafts cut into boxes, one kneecap
  misplaced by 30 mm on the sheets), so the coronal figures understate the 3D mask.

**Limits.** Translation only; soft-tissue alignment not measured; 8 of 14 comparisons used
sagittal T1/PD because these studies lack 2D sagittal fluid; 3D volumes were halved in-plane to
about 0.5 mm before segmentation (8.8 s per volume, 9 GB).

## 3. Soft-tissue segmenters (back)

**`aagatti/nnunet_knee` is good on our 3D DESS series and not on routine 2D series.** The user's
fallback case holds: the mask model works on 3D series only.

**Why it failed before: contrast, not a pipeline error.** It was trained on DESS (OAI-ZIB).
Ruled out by test: configuration and label ids, axis order, the pipeline itself (its own test
image gives menisci of 2.3 and 1.6 ml), slice thickness (averaging that image to 3.2 mm slices
changes little), resampling our 2D volumes to 0.8 mm slices, and test-time mirroring.

**Results on our volumes** (found / passing a label-free acceptance test: volume in range, one
main piece, in place against the bone masks, correct side):

| Model, series (n) | Medial meniscus | Lateral meniscus | Femoral cartilage | Medial tibial | Lateral tibial | Kneecap cartilage | s per volume |
|---|---|---|---|---|---|---|---|
| nnunet_knee, DESS (10) | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 9 | 10 / 8 | 19 slow, 2.0 to 2.5 fast |
| nnunet_knee, sagittal 3D PD FS (10) | 7 / 5 | 9 / 3 | 10 / 10 | 9 / 9 | 10 / 9 | 10 / 10 | 27 |
| nnunet_knee, 2D FS sagittal (20) | 3 / 0 | 4 / 1 | 20 / 8 | 7 / 3 | 12 / 5 | 17 / 8 | 22 |
| nnunet_knee, 2D T1/PD sagittal (10) | 0 | 0 | 0 | 0 | 0 | 0 | 22 |
| nnunet_knee, 2D coronal (10) | 0 / 0 | 1 / 0 | 9 / 3 | 5 / 4 | 5 / 1 | 5 / 0 | 2 |
| SKM-TEA echo-1 U-Net, DESS (10) | 10 / 1 | 10 / 0 | 10 / 5 | 10 / 7 | 10 / 4 | 10 / 5 | 2.7 |
| SKM-TEA echo-1 U-Net, 3D PD FS (10) | 10 / 4 | 10 / 4 | 10 / 0 | 10 / 4 | 10 / 2 | 5 / 0 | 3.0 |

- DESS by eye: the passing masks are good (menisci as horn triangles and peripheral wedges,
  cartilage as a layer on each surface; median menisci 2.6 and 2.2 ml, femoral cartilage
  12.5 ml). With n 10, 10 of 10 only bounds the true rate above about 74%.
- 3D PD FS: cartilage good, menisci undersized and unstable (fast and slow runs agree only to
  Dice 0.65 to 0.80 on thin classes).
- 2D fat-suppressed sagittal: cartilage sits on the bones but covers about half the true volume.
- SKM-TEA: fragmented from slice to slice on our DESS; worse than nnunet_knee everywhere.
- The DESS acceptance test used bones from the same study's 2D sagittal T1/PD series, so it
  rests on alignment through patient coordinates (about 1 to 2 mm, study 3b).

**What exists and what does not.**

| Class | Provider | Usable knees |
|---|---|---|
| Menisci, medial and lateral | nnunet_knee on DESS | about 240 to 264 of the 264 DESS studies |
| Femoral, tibial, kneecap cartilage | nnunet_knee on DESS; cartilage only from sagittal 3D PD FS | DESS as above, plus about 130 of 147 |
| ACL, PCL, MCL, LCL, effusion, pouch, Baker's cyst, tendons, fat pad | **no released weights found** | 0 |

All DESS comes from one scanner and protocol (Siemens Avanto fit 1.5T).

**Other candidates** (weights' licence, per the rule): SKM-TEA Apache-2.0 (run); kneeseg SKI10
MIT (random forest, not run); MedSAM2 CC-BY-SA, promptable (not run); DOSMA, KneeXNet-2.5D,
CartiMorph licence unclear (not run); TotalSegmentator MR sub-tasks need a licence key; Denizlab
IWOAI and rocaseg AGPL or research-only. Not checked: MONAI model zoo, MRSegmentator, SAM-Med3D.

**Fast runner:** `fast_nnunet_knee.py` in the session scratchpad (`soft-tissue/`): one model
load, threaded reading, crop to foreground, no window overlap, no mirroring, mixed precision,
argmax from interpolated logits on the GPU. 2.46 s per DESS volume (5 GB): 264 DESS in about
11 minutes. Against the standard pipeline on DESS: Dice 0.96 to 0.98 on menisci and cartilage,
acceptance rates unchanged. Restoring by nearest label dropped thin classes to 0.85 to 0.93 and
was discarded. Not validated on 3D PD FS.

Drawings: `dess_full.png`, `dess_zoom.png`, `pd3d_full.png`, `native_full.png`,
`fast_coronal_s10.png`, `skm_dess_crop_U-Net_E1.png`.

## 6. Token grid centred at prediction only (done in session)

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

Rule added to the model for this run (uncommitted): a token whose centre is on the frame's
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

**State of the worktree after this run (uncommitted):** `configs/training.yaml` at 257 px with the
scaled fields and report loss 0; `configs/refined-resnet50.yaml` at 224 px with report loss 0;
the border-token rule in `modeling/spatial_tokens.py` (`interior_anchors`) and `modeling/model.py`.

## 9. 257 px with all 9×9 tokens (launched by the user, border rule removed)

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

## 8. Meniscus masks from DESS: built (2026-10-08)

Decided with the user: `aagatti/nnunet_knee` on the DESS series, menisci only (cartilage dropped),
kept as a second target set beside the bone masks, supervised on every series of the study
through the shared patient coordinates. Planned head: its own small head with its own loss on its
own knees, on the backbone's layer 3. Training code built on 2026-10-08, uncommitted (below).

- Tool: `scripts/tools/meniscus_targets.py` (`convert`, `segment`, `export`); builders in
  `scripts/adhoc/meniscus_masks/`; working folder `~/.cache/knee-segmenters/dess-menisci/`.
- Convert: 265 of 265 studies (the DESS series with most slices), 0.5 mm in-plane, cropped to the
  body with a 10 mm margin, turned to the axis order of the model's test image (checked by eye on
  five knees); 71 s, 3.1 GB.
- Segment: full-resolution model, fold 1, no mirroring; 265 volumes in 1,567 s (5.9 s each),
  2.7 GB of GPU memory.
- Export: `data/labels/prepared/meniscus-nnunet-knee-dess265/`, 1 mm volumes in patient
  coordinates (0 neither, 1 medial, 2 lateral, 255 outside the DESS volume), `knees.json` with the
  test per knee. **263 of 265 accepted**; per fold 53 / 52 / 40 / 59 / 59. Rejected: one medial
  meniscus in pieces (main piece 0.60), one knee at the scanner's centre so its side could not be
  checked. No knee had its sides swapped.
- Test (label-free, thresholds chosen by hand): volume 0.8 to 7 ml, main piece at least 0.8 of the
  class, median distance to the mask's own tibia at most 8 mm and femur at most 10 mm, medial
  meniscus nearer the body's midline than the lateral one. Measured: medial 2.30 ml (1.11 to
  4.81), lateral 2.10 ml (1.20 to 4.15), distance to the tibia 3.6 and 4.2 mm, menisci 50 mm
  apart. Nothing came near a threshold except the two rejected knees.
- Check on 2D series: `outputs/analysis/soft-tissue-segmenters/meniscus-targets-on-2d-series.png`
  (five knees, coronal fluid and two sagittal slices each): the targets land on the menisci of the
  2D series, on the right sides.
- 1 mm serves any grid down to 4 mm cells (21×21 at 84 mm): the last stage and layer 3, not
  layer 2. 794 MB on the GPU for the 263 volumes.

**Training code (uncommitted, no run yet).** On in `configs/training.yaml`
(`model.meniscus_map`, `paths.meniscus_targets`, `trainer.meniscus_loss_weight` 0.1; weight
copied from the bone map). A layer-3 version (14×14 cells) was built first and removed the same
day: the user wants the head on the 7×7 tokens and inside the model, to build on its output.

- `modeling/model.py`: with `model.meniscus_map`, a linear head with three outputs on the same
  7×7 tokens the bone head reads (before the position code); its scores are returned as
  `encoded["meniscus_scores"]`. It starts at zero: created inside the usual forked random state it
  would otherwise start equal to the first three rows of the bone head. Nothing reads it yet.
- `training/lightning.py`: the same loss form as the bone map (cross-entropy on the shares plus
  Dice), on the tokens of knees that have a meniscus volume and lie inside it, every series of the
  study. Logged as `meniscus_*` and `val_meniscus_*`. The switch and the weight must be set
  together.
- `training/anatomy.py`: the target reader and the loss take the class list and the number of
  sample points per token side: 4 for the bones as before, 8 for the menisci.
- CPU check on four fold-1 validation studies, untrained weights: 970 to 1,260 supervised tokens
  per DESS knee and none on the study without a mask; 45 to 80 tokens per knee hold some meniscus,
  13 to 41 more than a tenth, largest share 0.44 to 0.64, mean share of a meniscus token 0.057.
  The label logits are identical with and without the head (difference 0.0).
  Against a dense 24×24 lookup, the share error on meniscus tokens is 0.015 with 4×4 points
  (largest 0.13) and 0.005 with 8×8 (largest 0.05).

**First run: `outputs/training/rn18-meniscus-map`** (2026-10-08; `configs/training.yaml` with the
meniscus map on, fold 1, seed 1110484336; launched by the user). Reference: the incumbent
`report-attribute-loss-off-2`, same fold, seed and recipe without the meniscus map. One pair of
runs.

| | Run | Reference | Δ |
|---|---:|---:|---:|
| ×4 | 0.926413 | 0.927291 | −0.00088 (paired bootstrap 95% −0.0020 to +0.0002) |
| ×1, best epoch | 0.916382 (epoch 9) | 0.917342 (epoch 9) | −0.00096 |

- Training 1,371 s against 1,352 s; ×4 prediction 51.4 s; peak GPU memory 4.8 GB.
- Per label at ×4: ACL +0.0003, Medial OA −0.0002, Baker's −0.0002, Lateral Meniscus −0.0003,
  PF OA −0.0004, Medial Meniscus −0.0007, Lateral OA −0.0008, Synovitis −0.0011, Fracture −0.0011,
  Contusion −0.0012, Effusion −0.0015, MCL −0.0032. Family intervals (meniscus, OA, other eight)
  all include zero. Predictions moved little: mean absolute change 0.0098, rank correlation per
  label 0.989 to 0.998.
- ×1 per epoch against the reference: −0.0012, +0.0001, −0.0007, −0.0002, −0.0013, −0.0005,
  −0.0002, −0.0011, −0.0010, −0.0009. Contusion was below the reference from epoch 4 to 10 and
  Fracture in five of the last six epochs at ×1.
- Meniscus map, validation overlap (52 DESS knees), medial / lateral per epoch: 0.083 / 0.096,
  0.256 / 0.350, 0.355 / 0.391, 0.406 / 0.439, 0.429 / 0.426, 0.430 / 0.412, 0.449 / 0.420,
  0.446 / 0.425, 0.437 / 0.432, 0.452 / 0.433. Flat from epoch 5; training overlap ends at 0.439
  and 0.434, so it is not overfitting, it is the ceiling of this head on this grid.
- Bone map unchanged: final validation overlap femur 0.904, tibia 0.898, kneecap 0.738, fibula
  0.697, joint 0.758, kneecap joint 0.670, each within 0.004 of the reference; plane error 1.74 mm
  against 1.78 mm.
- 3D page: `outputs/analysis/rn18-meniscus-map-3d/learned-map-3d.html` (and a preview PNG), made
  by `scripts/adhoc/meniscus_masks/learned_map_3d.py rn18-meniscus-map rn18-meniscus-map-3d`: the
  four knees of the earlier `anatomy-masks-frame-3d` page, none with a DESS series, plus two DESS
  validation knees with the segmenter's masks. Four passes per knee (102 to 128 slices), token
  shares placed on a 3 mm grid and smoothed with a 5 mm Gaussian before taking each voxel's main
  class; a first version with one pass and no smoothing drew ragged bones. On all six knees the meniscus map sits at the joint level, medial on the
  medial side and lateral on the lateral side, in arcs seen from above; fused shares reach 0.18 to
  0.68 (weakest: the lateral meniscus of …259018 and the medial one of DESS knee …753617), so the map transfers to studies without a DESS series.
- Comparison: `outputs/analysis/rn18-meniscus-map-vs-incumbent/comparison.json`.
- Lesson: the model learns where the menisci are from 211 one-scanner knees and applies it to
  2D-only studies, with the bone map untouched; as a side task alone it does not move the labels
  (level to slightly below, inside the interval). Nothing reads the map yet.
