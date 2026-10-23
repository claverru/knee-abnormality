# Side-routed heads: held cuts, runs 28 to 31, and the line's summary (2026-10-06)

**Run 24 stayed the reference: cuts placed between attention spots (run 28), only the three held
cuts with balanced pull (run 29, ×4 0.925098 but ×1 0.0016 lower), the searched 15-section
division (run 30) and the front half for PF OA (run 31) were all lower at ×1 or both multipliers.**
Over thirty runs the routing was worth +0.002 to +0.003 public (0.952 or 0.953 against 0.950) and
about +0.001 locally; nine labels moved no more than a rerun. The ResNet-50 on run 24's recipe
scored ×4 0.933209 and 0.957 public (S119).

## Runs at a glance

Fold 1 of `all_5fold_scanner_v2`, seed 1110484336, ResNet-18. Old recipe on this fold: ×4
0.924049, public 0.950 (S108, S111). Same-fold sibling runs differ by about 0.0004 of macro (SD).
Run 24 was the reference from 2026-10-06 19:20; the fixes-and-removals worktree was merged into
main by the user that evening, and every later run was compared with run 24. The 24 campaign runs
are in the [side-routing run notes](../../../../../autoresearch/campaigns/side-routing/run-notes.md);
the code of runs 10, 11 and 25 is in their source snapshots.

| # | Run (`outputs/training/…`) | Main change | ×4 macro | Public |
|---|---|---|---:|---:|
| 1 | `20261005-181012-side-routed-heads` | Shared heads routed by a learned boundary and side | 0.923102 | |
| 2 | `20261005-185535-side-routed-knee-frame` | Knee frame, learned offsets, position code | 0.923534 | S112 0.952 |
| 3 | `20261005-194836-side-routed-attention-prior` | Attention prior, extra auxiliary loss | 0.923085 | |
| 4 | `20261005-202001-side-routed-run4` | Prior kept, auxiliary loss removed | 0.923952 | S113 0.953 |
| 5 | `20261005-205251-side-routed-run5` | Octants from the fitted centre, learned rotation, no offsets | 0.924476 | S114 0.952 |
| – | campaign `separate-oa-frame-with-knee-level-term` | Second frame for OA, PF OA front-and-up, knee-level term | 0.924485 | S115 0.953 |
| 6 | `20261006-073158-side-routed-hard-octants` | One frame, hard cuts, front-and-up, knee-level term | 0.924432 | |
| 7 | `20261006-081303-side-routed-soft-gradient-own-routes` | Soft gradient through the cuts; MCL, Effusion, Baker's routed | 0.924845 | S116 0.952 |
| 8 | `20261006-084210-side-routed-study-frames` | Frame inferred per study (six regressed numbers) | 0.924320 | S117 0.952 |
| 9 | `20261006-092326-side-routed-fine-landmarks` | 14×14 tokens for the shared heads, landmark frame, slow frame | 0.922842 | |
| 10 | `20261006-095814-side-routed-fine-fused-routed-reports` | Deep features on the fine tokens, frame at head rate, routed v4 report targets | 0.924423 | |
| 11 | `20261006-103317-side-routed-study-frames-landmarks` | Run 8 plus the landmark frame, 7×7 tokens, v3.1 reports | 0.924119 | |
| 12 | `20261006-112944-side-routed-eighteen-regions` | Eighteen regions with learned sizes, overlap assignment, routed v4 reports, own position in the fixed frame | 0.923687 | |
| 13 | `20261006-115457-side-routed-regions-v31-reports` | Run 12 with the v3.1 reports on unrouted heads | 0.924122 | |
| 14 | `20261006-121814-side-routed-regions-acl-unrouted` | Run 13 with ACL unrouted | 0.923936 | |
| 15 | `20261006-124148-side-routed-regions-shared-frame` | Run 14 on one shared learned frame, no per-knee head | 0.924141 | |
| 16 | `20261006-130551-side-routed-regions-mirrored-position` | Run 14 with compartment-relative position for the shared heads | 0.924172 | |
| 17 | `20261006-132943-rn50-side-routed-regions` | ResNet-50 attempt, stopped by the user after three minutes | none | |
| 18 | `20261006-135237-side-routed-regions-soft-gradient-knee-level` | Run 12 with hard cuts and the soft gradient, knee-level term on every routed target | 0.923121 | |
| 19 | `20261006-141739-side-routed-regions-core-kept-by-sides` | Run 18 with the core no longer taken from the side labels | 0.923473 | |
| 20 | `20261006-144444-side-routed-regions-wider-table` | Run 19 with the wider region table, Synovitis unrouted | 0.923483 | |
| 21 | `20261006-151245-side-routed-regions-nod-per-knee` | Run 20 with per-knee centre, nod and sizes; roll and turn shared | 0.923947 | |
| 22 | `20261006-153716-side-routed-regions-kneecap-above-lower-cut` | Run 21 with PF OA and Effusion on the front above the lower cut | 0.924358 | |
| 23 | `20261006-160557-side-routed-meniscus-spot-frame-eight-sections` | Frame hung on the two meniscus spots; eight sections per level | 0.924251 | |
| 24 | `20261006-163650-side-routed-spot-frame-ring-sectors` | Spot frame with eight sectors centred on the axes, core and outer ring; region table from the attention | **0.925067** | S118 0.953 |
| 25 | `20261006-171738-side-routed-four-anchor-frame` | Run 24 with the notch and the kneecap as further anchors; no free tilt | 0.924419 | |
| 26 | `20261006-180530-side-routed-acl-all-heights` | Run 24 with ACL reading its region at all heights | 0.923636 | |
| 27 | `20261006-183038-side-routed-routed-detectors` | Run 24 with the old tear and OA detectors over the regions; stopped by the user | | |
| 28 | `20261006-184303-side-routed-midway-cuts` | Run 24 with the four cuts placed midway between attention spots, no gradient | 0.924384 | |
| 29 | `20261006-192417-side-routed-held-cuts-balanced` | Only the three cuts labels hold from both sides (midline, one height cut, core), equal pull per side | 0.925098 | |
| 30 | `20261006-200553-side-routed-mirrored-held-division` | Run 24 with the mirrored 15-section division found by the search (three heights, core 21 mm, four sectors) | 0.923564 | |
| 31 | `20261006-204225-side-routed-pf-oa-front-half` | Run 24 with PF OA and Effusion on the front half and the upper core, as in run 23 | 0.924552 | |
| – | `20261006-210555-rn50-side-routed-reference` | Refined ResNet-50 on run 24's build | 0.933209 | S119 0.957 |

Per-run details: [first to third builds](2026-10-05-first-to-third-builds.md),
[fourth and fifth](2026-10-05-fourth-and-fifth-builds.md), [runs 6 to 11](2026-10-06-octant-builds.md),
[runs 12 to 17](2026-10-06-eighteen-regions.md), [runs 18 to 23](2026-10-06-hard-cut-regions.md),
[runs 24 to 27](2026-10-06-spot-frame-reference.md).

## Run 28: cuts placed midway between the spots

A cut settles in an anatomical place only when labels pull on both sides of it (medial against
lateral, through the shared heads); a cut with a label on one side only drifts (the lower cut in
run 26), and labels with more cases pull harder. This build stopped moving the four region sizes
by gradient and placed them per knee from the heads' attention spots.

- Four spots per knee, squared-attention centroids without gradient: the two meniscus spots (as
  in the reference), the notch (ACL channel on its own tokens, whole knee) and the kneecap (OA
  channel over the fitted front-up region). Each had a spread, the weighted standard deviation of
  its tokens.
- A cut between two neighbours sat at `near + (far − near) · spread_near / (spread_near +
  spread_far)`, in the knee's frame: core radius between the notch's and the menisci's distance
  from the axis; ring radius between the menisci's and the kneecap's; upper cut between the
  menisci's and the kneecap's height; lower cut the mirror of the upper one about the menisci's
  height (nothing has a spot in the tibia).
- Removed: the shared `frame_size` and the per-knee size regression. The frame itself (centre on
  the meniscus spots, side axis, learned nod, shared shift and rotation) was the reference's, and
  the soft gradient still reached it through the token positions.
- With the reference's trained weights loaded into this build (146 validation knees,
  `midway_cut_values.py` in `scripts/adhoc/side_routed_session/`): upper cut +11.2 mm (SD 3.1),
  lower −12.3 (3.1), core radius 12.3 (2.0), ring radius 19.8 (3.0); joint band 23.5 mm, never
  empty. The reference had learned +13.7 and −1.8; the attention's best fit was +18, −12, core 15,
  ring 30.
- All nine guardrails passed.

Run `outputs/training/20261006-184303-side-routed-midway-cuts` (run 28):

- ×4 macro 0.924384 against 0.925067 for the reference (−0.0007, under two sibling SDs); ×1
  0.916204 against 0.917070 (−0.0009). Copy of the build in `scratchpad/run28-build`; the
  reference's in `scratchpad/run24-build`.
- Per label against the reference: Contusion +0.0020, Effusion +0.0012, MCL +0.0010, Synovitis
  +0.0008, PF OA +0.0001, Medial OA −0.0004, Medial Meniscus −0.0010, Baker's −0.0016, Lateral
  Meniscus −0.0017, ACL −0.0019, Lateral OA −0.0026, Fracture −0.0041. None beyond 1.5 of its own
  sibling SDs. The kneecap labels did not pay for their floor moving down 10 mm.
- The cuts held still over the epochs: lower −12.3 mm, upper +10.9, core radius 12.6, ring radius
  21.2 at the last epoch (reference −1.8, +13.7, 14.7, 26.5). The nod settled at +5° to +6°, where
  the reference swung between +8° and +18°.
- Drawing of study 45: the joint band lay on the joint line in both sagittal views and in the
  coronal ones, the first run where it did.
- ACL read the smallest region of the line here (core 12.6 mm, inner ring to 21.2 mm, against
  14.7 and 26.5): the ring cut was placed between the menisci (about 20 mm from the axis) and the
  kneecap spot (about 22 mm), two neighbours at nearly the same distance, so it fell on the
  meniscus spots themselves.

### Which targets hold each cut

Drawing: `outputs/analysis/s/side-routing-cut-forces-20261006/cut-forces.png`. A cut is held when
targets read only one side of it from each side. In the reference's division: medial against
lateral was held by labels on both sides; front against back by three labels in front and Baker's
alone behind; the core radius by ACL inside and MCL, Baker's, PF OA and Effusion outside. Not
held: the ring radius (ACL alone), the upper cut (menisci and OA from below, nothing above) and
the lower cut (kneecap labels and ACL from above, nothing below). Those three were the cuts that
drifted or landed badly in runs 24, 26 and 28. The model worked regardless because the cuts
started from the attention's fit, the regions were generous, and the attention inside each region
did the fine work.

## Run 29: only the held cuts, with equal pull from the two sides

Drawing per label: `outputs/analysis/s/side-routing-cut-forces-20261006/held-cuts-sectors-per-target-3d.png`.
The reference's construction (frame on the two meniscus spots, learned nod, sizes regressed per
knee, hard cuts with the soft gradient) with the division reduced to the three cuts that labels
hold from both sides:

- **Midline:** medial against lateral.
- **One height cut** (`cut_height`, from 0.45 pitch, about 6 mm above the centre, the middle of
  the reference's two learned cuts) in place of the two; the frame's nod tilts it.
- **Core radius** (from 1.1 pitch, as before). The ring radius was removed.
- Regions: Medial Meniscus, Medial OA, MCL and Baker's read the medial side below the cut and
  outside the core; Lateral Meniscus and Lateral OA the lateral side likewise; PF OA, Effusion and
  Synovitis everything above the cut outside the core; ACL the core at all heights, alone (the
  menisci and OA no longer read their half of it); Contusion and Fracture everything.
- Report targets: the routed questions followed their label's region; edema by compartment and
  the patellar facets used the three compartments; the tear places by horn kept their sectors
  below the cut, and sent no gradient to the three cuts.
- **Equal pull:** at each of the three cuts the soft gradient of the side that reads inside and of
  the side that reads outside were rescaled to their mean norm in every step (`BalancedSides`), so
  a side with more cases or more targets did not outweigh the other. Checked: sides at 323 and 3
  came out at 163 each; a silent side left the other unchanged.
- All nine guardrails passed.

Run `outputs/training/20261006-192417-side-routed-held-cuts-balanced` (run 29):

- ×4 macro 0.925098 against 0.925067 for the reference (+0.00003: equal); ×1 0.915496 against
  0.917070 (−0.0016). Five sections against 54. Copy of the build in `scratchpad/run29-build`.
- Per label against the reference: Effusion +0.0025, Medial OA +0.0024, Lateral Meniscus +0.0023,
  Contusion +0.0021, Fracture +0.0015, ACL +0.0013, Baker's +0.0001, PF OA −0.0002, Synovitis
  −0.0002, Medial Meniscus −0.0008, Lateral OA −0.0019, MCL −0.0085 (three of its sibling SDs; it
  lost its own corner and read the whole medial compartment).
- The frame nodded far more than in any earlier run: 62° at the last epoch, steady from epoch 8
  (reference 14°). With the cut 12 mm along the tilted axis, the single cut was closer to a
  front-against-back plane leaning backwards than to a height: the kneecap labels read the front
  and top, the menisci and OA sides the back and bottom.
- The core shrank from 14 mm to 9 mm, smaller than one token: with ACL alone inside, the outside
  wins. Roll 5.2° and turn 4.3° in absolute value, as in the reference.

### The finest division in which labels hold every cut (search, 2026-10-06)

Scripts `search_held_division.py` and `search_symmetric_division.py` in
`scripts/adhoc/side_routed_session/`; results and drawings in
`outputs/analysis/s/side-routing-cut-forces-20261006/` (`symmetric-division-<threshold>.json`,
`symmetric-division-per-target-3d.png`).

- Input: the old recipe's free attention (ten most attended tokens on positive knees) of the ten
  routed labels, centred per knee on the midpoint of its two meniscus spots. A label reads a part
  when it holds at least a threshold share of its attention; a cut is held when one label reads
  only the part before it and another only the part after. Heights, radii and sector angles were
  searched separately (3 mm and 15° steps, cuts at least 12 mm or 45° apart, every label keeping
  at least 80% of its attention), maximising the number of parts and then the weaker side's cases.
- Without mirroring the medial and lateral pairs: 18 sections at 10% and 15%, 15 at 20%. Its
  regions are not mirror images, which one shared head cannot read (the guardrail
  `shared_classifiers_blind_to_the_compartment` fails), so the search was repeated with sector
  cuts mirrored about the front-back axis and each pair reading the union of its two sides.
- Mirrored: 15 sections at 10% (three heights, five sectors, no core), 10 at 15% (two heights,
  four sectors, core), 15 at 20% (three heights at +9 and +27 mm, core 21 mm, four sectors cut at
  30°, 90°, 150° and 270° from medial through the front). The four sectors and the core are the
  same at 15% and 20%.
- Constant findings: three heights that are the labels' own levels (menisci, kneecap, pouch), not
  tibia, joint and femur; a core of 21 to 24 mm, not 14; four to six uneven sectors, one wide part
  at the back on each side.
- Weakest cuts at 20%: the front midline (Medial OA against Lateral OA, 95 cases), the core (ACL
  inside, 131), the upper cut (Effusion above, 244).

## Run 30: the mirrored 15-section division

The reference build with the division of the 20% search and no balancing of the gradient:

- Sizes per knee, learned as in the reference: upper cut from 2.1 pitch (27 mm), lower cut from
  0.7 pitch (9 mm), core radius from 1.65 pitch (21 mm). No ring radius.
- Sectors: front-medial 30° to 90°, front-lateral 90° to 150°, back-lateral 150° to 270°,
  back-medial 270° to 30°.
- Regions: each meniscus its half of the core and its back sector, bottom band; each OA side its
  half of the core and both sectors of its side, bottom and middle bands; PF OA the two front
  sectors, middle and top; Effusion the two front sectors, top (its own route `pouch`); Synovitis
  the core and the front sectors, middle and top; ACL the core, bottom and middle; MCL and Baker's
  the back-medial sector, bottom and middle.
- Report targets: tear place by horn on the front and back sector of its side in the bottom band,
  the body on both; the patellar facets on the front-medial and front-lateral sectors.
- All nine guardrails passed.

Run `outputs/training/20261006-200553-side-routed-mirrored-held-division` (run 30):

- ×4 macro 0.923564 against 0.925067 for the reference (−0.0015); ×1 0.914537 against 0.917070
  (−0.0025). Lower at both (copy of the build in `scratchpad/run30-build`).
- Per label against the reference: Contusion +0.0042, Lateral Meniscus +0.0009, Medial Meniscus
  +0.0007, Medial OA +0.0006, Baker's +0.0005, ACL −0.0002, Lateral OA −0.0011, PF OA −0.0022,
  Effusion −0.0031, Synovitis −0.0050, Fracture −0.0055, MCL −0.0078.
- The cuts stayed where the search placed them without any balancing: upper 27.0 mm, lower
  12.9 mm, core radius 20.2 mm at the last epoch (starts 27, 9, 21); nod 15°, as in the reference.
- The losses were on the labels whose regions were made tightest: Effusion (pouch only), PF OA and
  Synovitis (front sectors, middle and top), MCL (back-medial sector only). The menisci, read in
  the bottom band and their back sector with their half of a 20 mm core, were unchanged or
  slightly up. Held cuts hold; regions cut at 20% of the attention leave too much out.

## Run 31: PF OA and Effusion on the front half (launched by the user)

The reference build with one change: the route of PF OA and Effusion went from the three front
sectors outside the core above the lower cut (135° wide) to what it was in run 23: the front half
outside the core above the lower cut (180° wide) plus the core above the upper cut. Synovitis,
the patellar facets and everything else were the reference's.

Run `outputs/training/20261006-204225-side-routed-pf-oa-front-half` (run 31):

- ×4 macro 0.924552 against 0.925067 for the reference (−0.0005); ×1 0.916597 against 0.917070
  (−0.0005). Lower at both: not adopted (copy of the build in `scratchpad/run31-build`).
- Per label against the reference: Effusion +0.0026, ACL +0.0022, Lateral Meniscus +0.0014,
  Contusion +0.0011, PF OA +0.0004, Medial Meniscus −0.0005, Baker's −0.0006, Medial OA −0.0013,
  Synovitis −0.0016, Lateral OA −0.0023, MCL −0.0028, Fracture −0.0049.
- PF OA did not come back: 0.9037 against 0.9033 in the reference and 0.9067 in run 23. The width
  of its region is not what separates run 23 from run 24 for PF OA. Effusion, on the same route,
  gained 0.0026.

## ×1 beside ×4 (2026-10-06)

Over the line the gain from ×1 to ×4 was 0.0072 to 0.0090. Run 29 gained 0.0096, the largest: its
×4 score (0.925098) equalled the reference's while its ×1 (0.915496) was 0.0016 below. A run
counted as level with the reference only when both multipliers agreed, so run 29 was not, and the
five-section shape was not taken as a base. The seven public scores of the line (0.952 or 0.953)
are too coarse to say which multiplier predicts them. Script `x1_x4.py` in
`scripts/adhoc/side_routed_session/`.

## What the thirty runs show

- The routing is worth +0.002 to +0.003 on public (0.952 or 0.953 on all seven submissions against
  0.950) and about +0.001 locally, and that was there from the first builds. Local scores of the
  whole line lie between 0.9228 and 0.9251 at ×4 against a yardstick of 0.0004.
- Worked: shared heads on each side's region; hard cuts with a soft gradient; own routes for MCL,
  Effusion and Baker's (run 7); the frame on the meniscus spots with the division from the
  attention (run 24).
- Did not: finer tokens (9, 10); tight regions (18, 19, 30); more anatomical frames (21, 25, 28:
  stable frames, same or lower score); ACL through its region (14, 26); routed v4 reports against
  v3.1 (12, 13).
- Leaving evidence out costs and including too much is free; a frame that looks right does not
  score better; fold 1 cannot rank most of these variants.
- Unresolved when the line closed: ACL (−0.007) and Medial Meniscus (−0.003) below the old recipe
  since run 1; the first build's change that touches ACL is the token position (slice and
  in-image position replaced by the position code and the attention prior).

### The runs sorted, and which labels separate them

Script `rank_runs.py` in `scripts/adhoc/side_routed_session/`; ×1 and ×4 standardised and summed.

- Order: 24, 7, 10, 13, 8, 29, 11, 28, 23, 22, 6, 5, old recipe, 16, 25, 15, 12, 4, 14, 21, 20,
  26, 19, 30, 2, 18, 9, 3, 1. ×1 and ×4 correlate at 0.72 across the routed runs.
- Top six against bottom six, per label at ×4: PF OA +0.0040 (8 of its sibling SDs), Synovitis
  +0.0050 (2.3), Medial OA +0.0011 (1.4), Effusion +0.0015 (1.0), Contusion −0.0020 (−1.6), ACL
  and MCL 0.0000, the rest under one SD. PF OA and Synovitis carry about half of the macro's
  variance between runs (shares 0.20 and 0.28).
- Spread of each label across the 28 routed runs over its sibling SD: PF OA 5.8, Medial Meniscus
  1.8, Synovitis 1.6, the other nine 0.7 to 1.4. Thirty designs moved nine labels no more than a
  rerun does.
- PF OA: old recipe 0.8994; 0.9067 to 0.9076 in runs 4, 8, 10 and 23; 0.9033 in run 24 and 0.9031
  to 0.9042 in runs 25 to 29; 0.8924 with fine tokens (run 9). Synovitis: 0.878 to 0.880 in runs 1
  and 2, 0.8918 in run 24, 0.8926 in run 28. No run has both at their best.
- Against the old recipe the routed runs gain on PF OA, Effusion (0.9182 against 0.917 to 0.922)
  and Medial OA (0.9229 against 0.922 to 0.927) and lose on ACL (0.9458 against a mean of 0.9406
  with a spread equal to its noise) and Medial Meniscus (0.9469 against 0.941 to 0.947).

## The reference recipe on the ResNet-50 (launched by the user)

`outputs/training/20261006-210555-rn50-side-routed-reference`: `configs/refined-resnet50.yaml` (removed) on
the run 24 build, fold 1, seed 1110484336, ten epochs. ×4 0.933209, ×1 0.922934, against 0.925067
and 0.917070 for the ResNet-18 reference (+0.0081 and +0.0059). Submitted as **S119: 0.957
public** (×4, τ 0.9, batch 4, no reflection; bundle
`outputs/submission/exports/rn50-side-routed-ring-sectors-f1-x4-tau09`, record
`outputs/submission/records/rn50-side-routed-ring-sectors-f1-x4-tau09.json`). It equalled the best
public score at the time, S106 (five distilled ResNet-50 children, ×6, mirror TTA), with one model
at ×4. No ResNet-50 without routing exists on this fold and seed.
