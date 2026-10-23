# Study 2: attention per view in the knee frame

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
