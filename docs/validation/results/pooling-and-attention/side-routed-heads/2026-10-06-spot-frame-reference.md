# Side-routed heads: the spot-frame reference, runs 24 to 27 (2026-10-06)

**Run 24, the meniscus-spot frame with eight sectors, a core and an outer ring at three heights,
scored ×4 0.925067 and ×1 0.917070, the line's best, and 0.953 public (S118); it became the
reference from 2026-10-06 19:20 and the adopted recipe.** Four anchors (run 25) and ACL at all
heights (run 26) scored lower; the tear and OA detectors over the regions (run 27) were stopped.
ACL and Medial Meniscus sat below the old recipe from run 1, not because of any region. Fold 1 of
`all_5fold_scanner_v2`, seed 1110484336, ResNet-18. Design:
[side-routed heads](../../../../modeling/side-routed-heads.md).

## Run 24 build: the spot frame with the ring-and-sector division

Agreed with the user on 2026-10-06. On run 23's build (frame on the two meniscus spots, routed v4
reports, hard assignment with the soft gradient, knee-level term on every routed target, per-epoch
logging):

- **Division.** Around the frame's up axis: eight 45° sectors centred on the axes (medial,
  anteromedial, anterior, anterolateral, lateral, posterolateral, posterior, posteromedial), a
  core cylinder and an outer ring, at three heights. The sector cuts were fixed angles; the four
  sizes (upper cut, lower cut, core radius, ring radius) were regressed per knee and started at
  the values the [attention analysis](2026-10-06-hard-cut-regions.md#the-shape-read-from-the-heads-attention-2026-10-06)
  placed them, in token pitches: +1.3, −0.9, 1.1 and 2.2 (about +18 mm, −12 mm, 15 mm and 30 mm in
  the spot-centred frame).
- **Region table**, from each label's map and its free report heads, made generous, medial and
  lateral mirrored:

  | Target | Sectors | Rings | Heights |
  |---|---|---|---|
  | Medial / Lateral Meniscus, Medial / Lateral OA | the three on its side | inner, outer, and its half of the core | tibia, joint |
  | PF OA, Effusion | the three front ones | inner, outer | joint, upper |
  | Synovitis | the three front ones | inner, outer, core | joint, upper |
  | ACL | all (core); anterior, posterior and the three lateral ones (inner ring) | core, inner | joint, upper |
  | MCL | medial, anteromedial | inner, outer | all |
  | Baker's | posterior, posteromedial, medial | inner, outer | all |
  | Contusion, Fracture | unrouted | | |

- **Report targets.** The tear places were single sectors in the joint band (anteromedial =
  anterior horn, medial = body, posteromedial = posterior horn, and the lateral mirror); edema by
  compartment and the facets as before.
- First step on four studies: token share per region 28% for a meniscus or OA side, 23% for the
  kneecap labels, 17% for ACL, 23% for MCL, 31% for Baker's; all four sizes received gradient. All
  nine guardrails passed.

### Run 24 result

Run `outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors` (run 24):

- ×4 macro **0.925067**, the highest of the line: 0.924845 for run 7 (+0.0002), 0.924251 for run
  23 (+0.0008), 0.924049 for the old recipe (+0.0010). ×1 0.917070 against 0.917361 for run 7.
  Within one sibling SD of run 7, so level with the best and not shown to be above it.
- Per label against run 7: Synovitis +0.0043, Baker's +0.0021, MCL +0.0018, Fracture +0.0016,
  Lateral Meniscus +0.0015, Lateral OA +0.0012, Medial OA −0.0002, Medial Meniscus −0.0007,
  Effusion −0.0011, PF OA −0.0016, ACL −0.0021, Contusion −0.0041.
- Frame at epoch 6: roll and turn 3.9° and 4.3° in absolute value; centre 4.8 mm down; nod +17.8°
  (it was about −15° in runs 21 to 23); upper cut +14.6 mm, lower cut +0.2 mm, core radius
  11.6 mm, ring radius 29.0 mm.
- Ten epochs took about 20 minutes with the diagnostics logged once per epoch (4.1 steps per
  second against 3.5 to 3.7).
- **S118** (run 24, ×4, τ 0.9): **0.953 public**, with S113 and S115; S112, S114, S116 and S117
  were 0.952 and the old recipe 0.950. The first notebook attempt errored on Kaggle after a minute
  with an empty log (the bundle ran locally), the second was refused for a notebook-name
  conflict, the third went through; one submission was used.
- Drawings of three knees
  (`outputs/analysis/s/side-routing-loop-20261006/side-routed-spot-frame-ring-sectors/regions-*.png`):
  side-to-side and height were right and the roll was 1° to 6°; the centre slid front to back
  (7 mm back in one knee, 10 mm forward in another, where the core ended up in front of the
  notch), the turn was 2° to 8°, and the nod was +10° to +14° where the three runs before had
  −15°. Two spots fix a line; the position along the menisci and the tilt around the line were
  not anchored.

### Attention temperature on the reference (run 24), ×4

Sweep at inference on fold 1's validation knees, with run 24's own code (the temperature also
divides the scores the frame's spots are read from, so it moves the frame slightly):

| τ | Macro | Largest movers against τ 1.0 |
|---:|---:|---|
| 0.70 | 0.92463 | MCL +0.0031, Fracture +0.0025; Synovitis −0.0057, Baker's −0.0034 |
| 0.80 | 0.92501 | MCL +0.0026, Fracture +0.0017; Synovitis −0.0034, Baker's −0.0022 |
| 0.90 | 0.92505 | MCL +0.0015, Fracture +0.0011; Synovitis −0.0017, Baker's −0.0010 |
| 1.00 | 0.92508 | |
| 1.10 | 0.92479 | MCL −0.0009, Fracture −0.0016 |
| 1.25 | 0.92442 | MCL −0.0029, Fracture −0.0033; Baker's +0.0012, Synovitis +0.0013 |

The macro is flat from 0.8 to 1.0 (0.00007 apart); the local ×4 score is the τ 1.0 one, and S118
was submitted at 0.9 like the earlier bundles. Sharper attention helps the small own-head labels
and softer helps the routed front and back-medial ones, and they cancel.

## Run 25: the frame on four anchors

Agreed with the user on 2026-10-06 after run 24 became the reference. On run 24's build, the
frame used two more spots, each the squared-attention centroid of a head, without gradient:

- **Notch:** where the ACL head looks over the whole knee. The centre's front-back position was
  the mean of the two meniscus spots and the notch along the level front axis; side-to-side and
  height still came from the midpoint of the meniscus spots.
- **Kneecap:** where the shared OA head looks in the front-upper part of the geometric frame. The
  front axis pointed from the centre to that spot, lowered by one learned angle shared by all
  knees (`kneecap_elevation`, starting at 26°, the elevation measured between the PF OA spot and
  the meniscus midpoint in run 22). This fixed the nod per knee; the per-knee nod regression and
  the three free landmarks were removed. The four sizes stayed regressed per knee.
- With untrained heads the kneecap spot is the centroid of the front-upper region, so the first
  frames were nodded by up to 26°.
- Logged: `kneecap_elevation_deg`. All nine guardrails passed.

Run `outputs/training/20261006-171738-side-routed-four-anchor-frame` (run 25):

- ×4 macro 0.924419 against 0.925067 for the reference (−0.0006); ×1 0.915438 against 0.917070
  (−0.0016). Not better; the worktree code went back to run 24's build (run 25's is in its source
  snapshot and `scratchpad/run25-build`).
- Per label against the reference: ACL +0.0028, Contusion +0.0027, PF OA +0.0009, Lateral
  Meniscus +0.0005, Effusion +0.0003, Medial OA 0.0000, Medial Meniscus −0.0001, Baker's −0.0009,
  Fracture −0.0022, MCL −0.0023, Lateral OA −0.0034, Synovitis −0.0059.
- Frame over the validation knees at epoch 7: nod −10.3° and stable from epoch 4 (the reference
  wandered between +8° and +18°); roll 3.8° and turn 6.0° in absolute value (4.5°, 5.6°);
  front-to-back shift 3.9 mm in absolute value, the same as the reference; the shared kneecap
  elevation went from 26° to 28.7°.
- So the kneecap anchor fixed the tilt, at −10°, and the notch did not tighten the centre. The
  reference's free tilt sat at the opposite sign (+10° to +18°), which is where the labels put
  it; the shared elevation is one slow number and could not follow in ten epochs.
- Drawing of study 45: the centre 6.8 mm forward (9.6 mm in the reference), roll −6°, and the
  lower cut risen above the centre (+6 mm) with the upper cut at +14 mm, leaving a joint band 8 mm
  thick. The two height cuts acted as the kneecap labels' floor and the menisci's ceiling, not as
  the joint line.

## The old recipe's attention against the reference's (2026-10-06)

`outputs/analysis/s/side-routing-old-vs-reference-20261006/` (`old.npz`, `reference.npz`,
`old-vs-reference.png`; scripts `extract_label_attention.py`, `compare_old_reference.py` in
`scripts/adhoc/side_routed_session/`). The old recipe (`slab-rule-and-positive-std-requeue`) and
run 24 on the 874 validation knees; both see identical images and tokens, so attention was
compared token by token. Positive knees, the ten most attended tokens.

| Label | Old model's tokens outside the reference's region | Same tokens in both top tens | Spots apart | ×1 AUC, reference − old |
|---|---:|---:|---:|---:|
| ACL | 35% | 40% | 10.0 mm | −0.0045 |
| Synovitis | 34% | 25% | 9.1 mm | 0.0000 |
| MCL | 29% | 43% | 9.5 mm | +0.0090 |
| Effusion | 24% | 47% | 7.6 mm | +0.0035 |
| PF OA | 13% | 45% | 6.0 mm | +0.0002 |
| Lateral OA | 13% | 35% | 3.9 mm | +0.0007 |
| Lateral Meniscus | 6% | 56% | 2.7 mm | +0.0013 |
| Baker's | 5% | 61% | 5.7 mm | −0.0010 |
| Medial OA | 4% | 32% | 9.7 mm | −0.0037 |
| Medial Meniscus | 2% | 61% | 1.3 mm | −0.0021 |
| Contusion, Fracture | 0% (unrouted) | 49% | 5.3, 4.2 mm | +0.0046, +0.0047 |

- **ACL.** The old model's spot was at −6.5, +2.5, −3.3 mm and the reference's at −6.6, −2.7,
  +5.3: the same place in the plane and 8.6 mm higher. ACL's region left out the tibia band, and
  the lower cut rose during training from 12 mm below the centre to the centre, so the lower part
  of the ligament and its tibial attachment were closed to it.
- **Medial Meniscus** read the same tokens as the old model (2% outside, 1.3 mm apart) and was
  0.0021 lower while Lateral Meniscus was 0.0013 higher: the effect of one head shared between
  339 and 135 positives, not of the region.
- MCL, Effusion and Synovitis read a narrower area than the old model and scored the same or
  better; the old recipe also restricted each meniscus to its side plus 10 mm across the midline.

## Run 26: ACL at all heights

The reference build (run 24) with one change: ACL's region (the core, and the inner ring of the
anterior, posterior and three lateral sectors) read at all three heights instead of leaving out
the tibia band. All nine guardrails passed.

Run `outputs/training/20261006-180530-side-routed-acl-all-heights` (run 26):

- ×4 macro 0.923636 against 0.925067 for the reference (−0.0014); ×1 0.915117 against 0.917070.
  The worktree code went back to the reference's build.
- Per label against the reference: Contusion +0.0021, Lateral Meniscus +0.0019, ACL +0.0010,
  Medial Meniscus +0.0010, PF OA +0.0009, Medial OA −0.0007, Effusion −0.0009, Baker's −0.0011,
  Lateral OA −0.0042, MCL −0.0050, Fracture −0.0050, Synovitis −0.0072.
- ACL got the tokens below the cut back and gained 0.0010, inside its noise (0.0014): the height
  of its region is not what held it below the old recipe.
- The lower cut, no longer held by ACL, ended at +9.6 mm with the upper cut at +15.3 mm
  (reference −1.8 and +13.7): a joint band 6 mm thick, above the joint line; it crossed the upper
  cut at epochs 3 and 4. Synovitis, which reads joint and upper, lost 0.0072 (three times its
  noise).

### ACL and Medial Meniscus were below the old recipe from run 1

ACL at ×4 across the line: old recipe 0.9458; run 1 (ACL on its own head, unrouted) 0.9415;
every later run between 0.9387 and 0.9446, run 14 (ACL unrouted on the region build) 0.9415.
Medial Meniscus: old recipe 0.9469, runs 0.9408 to 0.9466. So the gap dates from the first
build, not from any region. Outside the routing, the first build removed two parts of the old
recipe that had measured value when adopted:

- the half-knee loss (a shared tear detector, smooth maximum over the tokens of each half of the
  knee, trained on the meniscus labels at weight 0.075): +0.0031 at adoption, interval +0.0015 to
  +0.0048 ([knee-geometry campaign](../../campaigns-and-queues/2026-09-24-knee-geometry-campaign.md));
- the three-compartment OA detector (a linear detector pooled over the medial, lateral and front
  regions and added to the three OA logits through a learned scale): +0.0013, PF OA +0.0102
  ([near-misses campaign](../../campaigns-and-queues/2026-09-25-near-misses-campaign.md)).

Script: `label_history.py` in `scripts/adhoc/side_routed_session/`.

## Run 27, stopped

`outputs/training/20261006-183038-side-routed-routed-detectors` was stopped by the user in its
first minutes: the reference with the old recipe's tear detector and OA detector pooled over the
learned regions. Not pursued: both do what the shared heads already do (one reader shared across
the sides, on each side's region, with that side's label) and differ only in pooling by a soft
maximum instead of attention; their gains at adoption were measured against separate heads.
