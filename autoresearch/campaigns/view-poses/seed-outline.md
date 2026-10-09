# View-poses seed outline

The plan behind the seeds. Eleven are written, in [three seed files](seed-ideas.yaml) because
a file holds five.

| # | Seed | Stage | Touches the canvases |
|---|---|---|---|
| 1 | `meniscus-head-on-the-14-by-14-grid` | what the poses are built on | no |
| 2 | `joint-centre-taught-from-the-masks` | what the poses are built on | no |
| 3 | `canvases-for-the-pose-in-all-six-views` | room for the pose | yes: 368 px, with scale and the axial pose; fields unchanged |
| 4 | `a-crop-centre-per-contrast` | where each crop looks | no |
| 5 | `t1pd-sagittal-crop-on-the-notch` | where each crop looks | no |
| 6 | `t1pd-joint-crops-centred-on-the-menisci` | where each crop looks | no |
| 7 | `sagittal-fluid-canvas-fine` (A) | canvas combinations | field only |
| 8 | `coronal-fluid-canvas-fine` (B) | canvas combinations | field only |
| 9 | `axial-fluid-canvas-fine` (C) | canvas combinations | field only |
| 10 | `both-fluid-joint-canvases-fine` (E) | canvas combinations | field only |
| 11 | `meniscus-labels-read-by-meniscus-share` | routing | no |

**Order (user, 2026-10-08):** the meniscus head and the taught centre first; then the canvases,
so the crops have room; then where each crop looks; then the canvas combinations; then routing.
The rest of this document is the reasoning, and what becomes a seed once its premise is checked;
the checks use data on disk and no training.

**Dropped after the reference's predicted maps were read** (`rn18-meniscus-map` on 874
validation studies; [prior work](prior-work.md#the-reference-and-where-its-predicted-structures-sit)):

- *Fluid sagittal crop reaching the pouch:* placing the crop's top 25 mm above the kneecap means
  moving it up by about 33 mm, more than any shift limit (12.75 mm today, 18 mm after the canvas
  seed). The pouch needs a wider field: combination C has sagittal fluid at 120 mm.
- *Fluid sagittal crop keeping the back of the knee:* the bones already end 26 mm short of the
  crop's back edge and cysts reach about 30 mm behind them, so the crop holds nearly all of that
  zone; moving back gains about 4 mm and loses the kneecap.

## Checks to run first

| Check | Question | Decides |
|---|---|---|
| Token shares | How much of a 12 mm token a meniscus or a cartilage layer fills, and in how many tokens per image | Whether seeds 1 and 2 are learnable at 7×7 or need the finer grid first |
| Real against inferred | How far the true menisci and cartilage are from the bone-derived "joint" and "kneecap joint" classes, at token scale | Whether the DESS masks add anything over regions inferred from bones on all 550 knees |
| Measurements against labels | How well cartilage volume ranks the OA labels and meniscus size or extrusion the meniscus labels, on the 265 studies | Whether seed 8 exists |
| Acceptance | The label-free acceptance test on all 265 DESS masks | How many usable knees each structure has |
| Fields from extents | The extent of the joint line's structures and of the kneecap joint in each plane, on the masks, in millimetres and in knee units | The fields of combinations A to D |
| Box from the first look | How well a box around predicted bone and joint shares at 12 mm tokens matches the mask's box, per view | Whether detail crops can be placed before the soft-tissue head exists |

## What the poses are built on, and where they look

The reference is `rn18-meniscus-map`, with the recipe's crops and canvases.

- **Seed 1, the meniscus head on 14 by 14.** At 7 by 7 the head overlaps the mask at 0.36 to
  0.39 against 0.89 for bones; it reads the previous stage's features plus the deep tokens.
  Adopted whatever its macro result.
- **Seed 4, a crop centre per plane-contrast** (settled by the user: one crop per
  plane-contrast). Fluid sagittal: from the back of the condyles to the kneecap's front and
  from the plateau to the kneecap's top, forward and up from today. T1/PD sagittal and T1/PD
  coronal: the joint centre. Fluid coronal: the bones from 30 mm below to 45 mm above the joint
  plane. The four hand-measured offsets go. Adopted when at least the incumbent at both multipliers.

**Seeds that move one view's centre**, each against the centre that view has in the incumbent:

| Seed | View | Where it looks | Rests on |
|---|---|---|---|
| On the notch | sagittal T1/PD | 15 mm above the joint plane | That view is the best single view for ACL only; the ACL is read from the joint line to 30 mm above it |
| On the menisci | sagittal and coronal T1/PD | the centroid of the predicted menisci | The menisci sit 8 mm behind the knee's centre with a height spread of 1.1 mm; needs the 14 by 14 head |

**The knee's centre itself** (`joint-centre-taught-from-the-masks`): bone tokens vote for the
mask's plateau centre, replacing the centroid of bone mass, which is 1.6 to 1.8 mm off front to
back and has no target of its own. It comes before the centres built on it.

**Proposed and not written: which slices are read.** Effusion and Synovitis fall by about 0.07
where the axial slices stop below the pouch, and MCL by 0.12 where the sagittal slices stop
short of the medial edge. But axial stacks begin a median 18 to 24 mm above the kneecap's top,
so part of this is what the scan contains, which no sampling rule recovers. Whether the loss is
sampling or coverage has to be separated before a seed is written: compare each study's reach
with the reach its stack allows.

What today's crops look at, for reference: the joint centre plus 4.9 mm forward and 2.7 mm up
(sagittal) and −2.1 and 2.0 mm (coronal). On the masks: kneecap front 56 mm ahead of the joint
centre, its top 55 mm above the joint line, back of the condyles 23 mm behind, bone edges 37 mm
to each side; crop half-width 47 mm.

The centre seeds follow the canvas seed, so their crops have 18 mm of shift; each logs the
share of crops at the limit, since a target the canvas would not let the crop reach is not a
wrong target.

Not a seed: centring axial on the bones (it pushes the back of the knee out, and axial is the
best view for Baker's cyst); centring the token grid on the target (the 257 px runs showed no
effect).

## Room for the pose

`canvases-for-the-pose-in-all-six-views`, third in the order and adopted unless clearly worse;
the centre seeds and every combination are built on it. Fields stay at 84, 93 and 120 mm.

| Part | What it is | Rests on |
|---|---|---|
| Canvases | 368 px square for every view (72 px of margin a side); axial gets a margin for the first time | Canvas measurement on 3,517 studies; 11 to 15% of crops clamped today |
| Shift limits | 18 mm for sagittal and coronal, 8 mm for axial | 99th percentile of the shift asked for |
| Scale | One size per knee from the first look, on all six views, clamped 0.87 to 1.13 | Anatomy-recipe run 7 on four views: +0.0003 at both multipliers, size estimate r 0.94 |
| Axial pose | Centred front to back on the box from 30 mm behind the condyles to the kneecap's front; turned to the knee's axes, up to 20° | A quarter of the kneecap's mass on the axial crop's front edge with 37 mm spare behind; 95% of knees turned −13° to +20° |

It reports the data-wait share of the step, the peak memory and the training time, since the
canvases are the open cost.

## Canvas combinations

The crops are smart in every run; what is fixed before training, and so has to be tried in
combinations, is the canvas of each plane-contrast: how many millimetres it holds and at what
pixel size. The pixel size sets how far that view can zoom, and the extent sets how far its crop
can move. One combination per run, each on the canvas seed and against the incumbent.

**Canvas for a nominal field** (holds shift, turn, scale 0.87 to 1.13 and jitter in 99% of
images; [prior work](prior-work.md#canvas-sizes)):

| Nominal field | Pixel size at 224 px | Canvas, sagittal or coronal | Canvas, axial turned |
|---:|---:|---|---|
| 84 mm | 0.375 mm | 138 mm, 370 px | about 137 mm, 366 px |
| 93.3 mm | 0.417 mm | 151 mm, 364 px | – |
| 90 mm | 0.402 mm | – | about 145 mm, 361 px |
| 120 mm | 0.536 mm | – | about 194 mm, 362 px |

Every canvas comes out at about 365 px a side: a tighter field needs fewer millimetres of
canvas at a finer pixel. So a combination changes what each canvas shows, not how many pixels
the loader carries.

**A combination is a trade, not a set of independent zooms** (user, 2026-10-08). Zooming one
view takes field away from the labels that needed it there, so another view has to keep or gain
that field. The rule every combination must meet: **each label keeps at least one of its strong
views at the field it needs.** From the single-view table and the image study:

| Need | Labels | Strong views |
|---|---|---|
| Fine, at the joint line (2 to 3 mm structures) | menisci, tibiofemoral OA, ACL | sagittal fluid, coronal fluid, sagittal T1/PD |
| Fine, at the kneecap joint | PF OA | axial fluid, sagittal fluid |
| Wide (fluid, cysts, pouch, back of the knee) | Effusion, Synovitis, Baker's | axial fluid, sagittal fluid |
| Wide to the edges (collaterals, marrow, fracture lines) | MCL, Contusion, Fracture | axial fluid, coronal fluid |

| # | Combination | Sag fluid | Sag T1/PD | Cor fluid | Cor T1/PD | Ax fluid | Ax T1/PD | The trade |
|---|---|---:|---:|---:|---:|---:|---:|---|
| base | Today's fields, smart crops | 93 | 84 | 93 | 84 | 120 | 120 | – |
| A | Sagittal fluid fine, axial keeps the field | 84 | 93 | 93 | 84 | 120 | 120 | Menisci, ACL and PF OA gain detail in sagittal; the pouch and Baker's rely on axial; sagittal T1/PD gives back the field |
| B | Coronal fluid fine, axial keeps the edges | 93 | 84 | 84 | 93 | 120 | 120 | Menisci and tibiofemoral OA gain detail in coronal; MCL and marrow at the edges rely on axial; coronal T1/PD gives back the field |
| C | Axial fluid fine, sagittal fluid wide | 120 | 84 | 93 | 84 | 93 | 120 | PF OA gains detail in axial; effusion, the pouch and Baker's move to a wide sagittal fluid; MCL keeps coronal |
| E | Both fluid joint views fine, axial keeps the field | 84 | 93 | 84 | 93 | 120 | 120 | A and B together: all joint-line detail on fluid; every wide need on axial at 120 mm |

Numbers are the nominal field of each view's canvas in millimetres.

**Dropped on 2026-10-08, with the reason:**

- *Axial fine with the field given back in coronal fluid:* a zoomed axial view stops showing the
  pouch and the back of the knee, and coronal series hardly contain either (30 mm behind the
  condyles inside in under 20% of studies; the pouch cut in half the effusion cases inspected).
  C makes the same zoom and returns the field in sagittal fluid.
- *Axial T1/PD's slices given to a second axial fluid canvas:* 20% of studies have that series
  (694 of 3,517), and it needs two canvases for one view.
- *Axial wider than 120 mm:* tried twice, 130 mm 0.0000 and 140 mm −0.0006.

**What the earlier runs say about these trades** (2026-10-02, fixed crops, before the anatomy
map): sagittal fluid back at 120 mm −0.0007 and coronal fluid back at 120 mm −0.0004 against
both at 93 mm; axial at 130 mm 0.0000 and at 140 mm −0.0006; "axial plus sagittal fluid at
120 mm scores Baker's 0.949, the zoomed fluid tokens 0.878".

The axial fluid field in C is 93.3 mm: the kneecap joint's 70 × 40 mm box with one token of
context on each side, at the pixel size the fluid views already use.

*Open:* whether 120 mm is the right wide field, from the extent of the pouch and the back of
the knee.

## Planned seeds after the combinations, in order

| # | Seed | Stage | Rests on | Premise still to check |
|---|---|---|---|---|
| 2 | Cartilage in the same head (femoral, tibial per side, kneecap) | structures | Cartilage 8 to 10 of 10 on DESS | token shares; acceptance |
| 6 | Detail crop on the joint line, sagittal and coronal fluid | detail | Wide against fine conflict in the image study; weak labels unexplained by field | box from the first look; slices' source |
| 7 | Detail crop on the kneecap joint, axial fluid | detail | Kneecap cartilage in a 70 × 40 mm box; axial at 0.54 mm per pixel | as seed 6 |
| 8 | OA labels routed by cartilage share | routing | The kneecap class leading the PF OA route | a cartilage head |
| 9 | Measurement targets from the masks | targets | – | measurements against labels |

Not in this campaign, though measured today: slice selection from the anatomy (Effusion and
Synovitis −0.07 when the axial slices stop below the pouch; MCL −0.12 when the sagittal slices
stop short of the medial edge). It needs no masks and stands as a seed of its own.
