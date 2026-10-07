# Prior work for the frame-and-targets campaign

The summary the researcher checks proposals against. Tables, drawings and every run's details
are in the [plan and runs](../../../docs/ongoing/side-routed-heads.md); the earlier campaigns on
the design are [side routing](../side-routing/INDEX.md) and
[routing and shifts](../routing-and-shifts/INDEX.md) (keeper's notes per run:
[run notes](../routing-and-shifts/run-notes.md)); learned view placement was studied in
[learnable views](../learnable-views/INDEX.md).

## The incumbent

Run 24 of the line, `outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors`: fold
1 of `all_5fold_scanner_v2`, seed 1110484336, ResNet-18, ten epochs. ×4 0.925067, ×1 0.917070,
0.953 public (S118). The ResNet-50 on the same recipe: ×4 0.933209, ×1 0.922934, 0.957 public
(S119, equal to the best submission). The old recipe without routing, same fold and seed: ×4
0.924049, ×1 0.916134, 0.950 public.

The routing-and-shifts campaign adopted one change on it, `height-cuts-from-the-spots-requeue`
(`outputs/autoresearch/routing-and-shifts/experiments/height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402`,
with `source/`, `changes.diff` and `training/`): the two height cuts placed per knee from the
meniscus and kneecap spots without gradient. ×4 0.925389, ×1 0.917411, 0.952 public (S120): level
with run 24 locally and one step lower on the public board. It is not part of this campaign's
base (user decision, 2026-10-07: the base is the model with a submission on both backbones); it
returns as the first seed. Most measurements of 2026-10-07 below were taken on its weights and
say so.

**How far near-identical runs spread.** Eight runs of the previous campaign that differed from
the height-cuts model only by a change that turned out inert scored 0.9244 to 0.9251 at ×4 and
0.9155 to 0.9165 at ×1. Run 24 (0.925067, 0.917070) is at the upper end of that range, so a run
level with it at both multipliers is a small gain over a typical rerun.

Its design: the frame of each knee is centred on the midpoint of two meniscus attention spots
(squared-attention centroids of the shared meniscus channel over the two fitted halves, no
gradient) plus one shared learned shift; its side axis is the line between them (roll and turn);
the forward tilt (nod) is a landmark tilt plus a per-knee regression; four sizes (upper cut,
lower cut, core radius, ring radius) are regressed per knee; all of these learn from the labels.
Eight 45° sectors centred on the axes, a core, an inner and an outer ring, three heights; hard
cuts with a soft gradient; routed v4.0 report targets; a knee-level term on every routed channel.
Geometry at its last epoch, validation means: upper cut +13.7 mm, lower cut −1.8 mm, nod 13.6°
(the height-cuts model: +10.8 mm, −12.9 mm, core 16.3 mm, ring 29.8 mm, nod 6.0°).

| Target | Region in the incumbent |
|---|---|
| Medial, Lateral Meniscus; Medial, Lateral OA | The three sectors of its side, both rings, its half of the core; tibia and joint bands |
| PF OA, Effusion | The three front sectors, both rings; joint and upper bands (route `kneecap`) |
| Synovitis | The same and the core (route `front_core`) |
| ACL | The core and the inner ring of the anterior, posterior and three lateral sectors; joint and upper |
| MCL | Medial and anteromedial sectors, both rings, all heights |
| Baker's | Posterior, posteromedial and medial sectors, both rings, all heights |
| Contusion, Fracture | Everything (not routed) |

## Yardstick

Standard deviation over same-fold sibling runs of one recipe: macro 0.0004. Per target: Fracture
0.0044, Lateral OA 0.0034, MCL 0.0029, Lateral Meniscus 0.0024, Synovitis 0.0022, Baker's 0.0015,
Effusion 0.0015, ACL 0.0014, Contusion 0.0013, Medial OA 0.0008, Medial Meniscus 0.0007, PF OA
0.0005. Going from ×1 to ×4 adds 0.0072 to 0.0090 over the line; ×1 and ×4 correlate at 0.72
across the routed runs.

## The runs of the line

One run each, fold 1, seed 1110484336. Sorted by ×1 and ×4 together.

| Run | What it was | ×1 | ×4 |
|---:|---|---:|---:|
| 24 | Spot frame, rings and sectors (incumbent) | 0.9171 | 0.9251 |
| 7 | Soft gradient, own routes for MCL, Effusion, Baker's | 0.9174 | 0.9248 |
| 10 | Fused fine tokens, routed v4 reports | 0.9167 | 0.9244 |
| 13 | Eighteen regions, v3.1 reports | 0.9169 | 0.9241 |
| 8 | Frame regressed per knee | 0.9166 | 0.9243 |
| 29 | Five sections (midline, one height cut, core), balanced pull | 0.9155 | 0.9251 |
| 31 | Run 24 with PF OA and Effusion on the front half | 0.9166 | 0.9246 |
| 11 | Run 8 plus landmarks | 0.9168 | 0.9241 |
| 28 | Cuts placed midway between attention spots | 0.9162 | 0.9244 |
| 23 | Spot frame, eight sections | 0.9162 | 0.9243 |
| 22 | Kneecap labels above the lower cut | 0.9161 | 0.9244 |
| 6 | Hard octants | 0.9159 | 0.9244 |
| 5 | Fifth early build | 0.9158 | 0.9245 |
| old | Old recipe, no routing | 0.9161 | 0.9240 |
| 16 | Mirrored position for shared heads | 0.9159 | 0.9242 |
| 25 | Four-anchor frame | 0.9154 | 0.9244 |
| 15 | Shared frame | 0.9156 | 0.9241 |
| 12 | Eighteen regions, overlap, v4 reports | 0.9159 | 0.9237 |
| 4 | Fourth early build | 0.9154 | 0.9240 |
| 14 | ACL unrouted | 0.9154 | 0.9239 |
| 21 | Roll and turn shared | 0.9152 | 0.9239 |
| 20 | Wider region table | 0.9155 | 0.9235 |
| 26 | ACL at all heights | 0.9151 | 0.9236 |
| 19 | Side labels keep their core half | 0.9152 | 0.9235 |
| 30 | Mirrored 15-section division, tight regions | 0.9145 | 0.9236 |
| 2 | Knee frame | 0.9145 | 0.9235 |
| 18 | Hard cuts, knee-level everywhere, tight table | 0.9151 | 0.9231 |
| 9 | Fine tokens, slow rate | 0.9151 | 0.9228 |
| 3 | Attention prior | 0.9146 | 0.9231 |
| 1 | First build | 0.9146 | 0.9231 |

Public scores of the line: 0.952 (S112, run 2), 0.953 (S113, run 4), 0.952 (S114, run 5), 0.953
(S115), 0.952 (S116, run 7), 0.952 (S117, run 8), 0.953 (S118, run 24).

## What separates the runs

- **Two targets decide the order.** Top six against bottom six at ×4: PF OA +0.0040 (8 sibling
  SDs), Synovitis +0.0050 (2.3); every other target under 1.5 SDs; ACL and MCL 0.0000; Contusion
  −0.0020.
- **Nine targets move no more than a rerun.** Spread across the 28 routed runs over the sibling
  SD: PF OA 5.8, Medial Meniscus 1.8, Synovitis 1.6, the other nine 0.7 to 1.4.
- **PF OA:** old recipe 0.8994; 0.9067 to 0.9076 in runs 4, 8, 10 and 23; 0.9033 in run 24; 0.9037
  in run 31, which gave it back run 23's region, so the width of its region is not the cause.
- **Synovitis:** 0.878 to 0.880 in runs 1 and 2; 0.8918 in run 24, 0.8926 in run 28.
- **Medial Meniscus:** 0.9457 to 0.9466 in runs 1 to 5 and 22, where it read its whole side;
  0.9433 to 0.9443 in runs 24 to 29; old recipe 0.9469.
- **ACL:** 0.9406 on average over the routed runs with a spread equal to its noise, against
  0.9458 for the old recipe; nothing varied so far touches it.
- **Against the old recipe** the routed runs gain on PF OA, Effusion and Medial OA and lose on
  ACL and Medial Meniscus.

## Measured, with drawings

- **Free attention of every target** (old recipe, ten most attended tokens on positive knees,
  874 validation knees): `outputs/analysis/side-routing-old-vs-reference-20261006/` (`old.npz`,
  `reference.npz`, `old-vs-reference.png`). Share of the old model's attention outside the
  incumbent's region: ACL 35% (95% of it below the lower cut), Synovitis 34%, MCL 29%, Effusion
  24%, PF OA 13%, Lateral OA 13%, the rest 6% or less.
- **Attention spots** in the fitted frame (medial, front, up; mm): Medial Meniscus (26, −7, −7),
  Lateral Meniscus (−29, −5, −7), ACL (−6, −2, −1), PF OA (−2, 32 to 36, 13), Effusion (−6, 26,
  32), Synovitis (−3, 13, 19); each a blob of 15 to 20 mm, the same with and without the finding.
- **Which targets hold each cut** (`outputs/analysis/side-routing-cut-forces-20261006/cut-forces.png`):
  the midline and the core radius have targets on both sides; the ring radius (ACL alone), the
  upper cut (nothing above) and the lower cut (nothing below) do not.
- **The finest division with every cut held, medial and lateral mirrored**
  (`symmetric-division-<threshold>.json` there): three heights at +9 and +27 mm above the
  meniscus spots (menisci, kneecap, pouch), a core of 21 mm, four sectors cut at 30°, 90°, 150°
  and 270° from medial through the front. Run 30 built it with regions cut at 20% of the
  attention: the cuts stayed where they started (27.0, 12.9, 20.2 mm) and the score fell on the
  targets whose regions were tightest.
- **Midway cuts** (run 28): the four sizes placed per knee between attention spots, no gradient.
  Joint band on the joint line, nod steady at 5° to 6° (incumbent 8° to 18°); Contusion +0.0020,
  Effusion +0.0012, MCL +0.0010, Synovitis +0.0008, ACL −0.0019 with the ring cut at 21 mm in
  place of 26.
- **Report answers usable as located targets** (`attributes_v5.md` in the worktree
  `knee-abnormality-jevk5-attributes-v4`): ACL tears 716 and PCL findings 259 in the core against
  MCL 874 and lateral ligaments 321 at the rim, all in the v4 answers; located edema targets
  explain 25% on unseen knees.

## Closed

| Idea | Result | Runs |
|---|---|---|
| Finer 14×14 tokens | −0.0022 and no gain when fused | 9, 10 |
| Tight region tables | −0.0019, −0.0016, −0.0015 | 18, 19, 30 |
| More anchors on the frame (notch, kneecap) | Stable nod, −0.0006 | 25 |
| ACL through the height of its region | ACL +0.0010 (inside noise); the lower cut then slid 10 mm up and Synovitis lost 0.0072 | 26 |
| The old recipe's slice-position and in-image codes on own-channel tokens | −0.0004 and −0.0009; routed targets −0.010 | side-routing campaign |
| A routed half-knee auxiliary loss | MCL −0.0128 | 3 |
| The old tear and OA detectors pooled over the regions | Same readers as the shared heads, differing only in pooling; not run | 27 |
| Routed v4 reports against v3.1 | No difference | 12, 13 |
| PF OA and Effusion on the front half | −0.0005 at both multipliers; PF OA +0.0004, Effusion +0.0026 | 31 |

## The routing-and-shifts campaign (2026-10-06 and 07, 21 decided runs)

One adoption, the height cuts from the spots (+0.0003 over run 24 at both multipliers). Deltas
against that model at ×4 and ×1; details per run in its [run notes](../routing-and-shifts/run-notes.md).

| Run | ×4 | ×1 | What it showed |
|---|---:|---:|---|
| Menisci read their whole side | −0.0010 | −0.0013 | |
| Held cuts with wider regions | −0.0010 | −0.0010 | |
| Shared ligament reader (ACL, PCL in the core; MCL, lateral ligaments at the rim) | −0.0007 | +0.0003 | ACL +0.0021 |
| Synovitis on Effusion's attention | −0.0012 | −0.0013 | |
| Contusion on the edema head | −0.0007 | −0.0011 | Its scale stayed at 0 |
| Shift per series type (light port, then unit parameter) | −0.0006, −0.0008 | −0.0010, −0.0012 | Shifts within 0.5 mm of zero |
| Shift per study, one point, label-trained | −0.0010, −0.0027 | −0.0015, −0.0046 | Drift; PF OA +0.0040 at 7.6 mm forward |
| Shift per study, per view, label-trained | −0.0008, −0.0021 | −0.0019, −0.0029 | |
| Shift per study trained on the meniscus spots | −0.0004 | −0.0022 | First shifts that differ per knee |
| The same with a spatial-map locator | +0.0001 | −0.0009 | Best ×4 of the campaign, 0.925530; MCL +0.0058 |
| Crops and frame on one learned centre | −0.0037 | −0.0062 | |
| Height cuts from the vertical spread of the spots | −0.0004 | −0.0009 | |
| ACL's ring cut removed | −0.0007 | −0.0009 | |
| Nod in the meniscus basis | −0.0007 | −0.0009 | |
| Compact-support cut gradients | −0.0003 | −0.0011 | |
| Report answers with their own attention | −0.0004 | −0.0006 | |

## Measured on 2026-10-07

All on the validation fold, written up with tables in
[the plan and runs](../../../docs/ongoing/side-routed-heads.md) (sections from "MCL in
spot-shift-locator-keeps-spatial-maps-k" onwards). Outputs under
`outputs/analysis/routing-and-shifts-frames-20261007/`.

- **A crop shift at prediction on shift-trained weights:** +0.0007 at ×4, +0.0001 at ×1; the MCL
  gain of the spatial-map run is 0.0040 in the trained weights and 0.0018 in the shift.
- **Random crop offsets in training** (3 mm disc, sunflower pattern at prediction;
  `outputs/training/20261007-081431-side-routed-crop-offsets-3mm`): ×4 −0.0004, ×1 −0.0026.
  Training with the offsets costs 0.0012 to 0.0020; closed.
- **An offset pattern at prediction on the height-cuts model's weights:** fourteen variants (by
  direction, radius 1.5 to 6 mm, constant shifts, two positions) average 0.0000 with a spread of
  0.00025; closed. What changes a score is the picture the backbone sees, not the frame or the
  token positions (frame and position swaps: +0.0038 for Lateral Meniscus with the anchor's
  frame, +0.0002 with only the pattern's frame). Sensitivity to a 3 mm move of the coronal crops,
  as a share of the gap between positives and negatives: 0.022 to 0.029 for the two menisci, the
  two OA sides and MCL, 0.007 to 0.014 for the rest; unrelated to the number of positives.
  Moving every coronal crop 3 mm medially costs Lateral OA 0.0043.
- **Crops cut by the knee's frame, the height-cuts model's weights, no training** (first look for the frame,
  then the 84 and 93 mm views re-cut):

  | Variant | ×4 | ×1 |
  |---|---:|---:|
  | Rotated to the frame | +0.0011 | −0.0002 |
  | Rotated the wrong way | −0.0007 | −0.0009 |
  | Rotated by half the angle | +0.0010 | +0.0007 |
  | Rotated, coronal views only | +0.0002 | +0.0002 |
  | Rotated, sagittal views only | +0.0010 | −0.0004 |
  | Centred on the frame's centre (deviation from the fold mean) | −0.0004 | |
  | Centred and rotated | −0.0007 | |

  Rotation applied: median 3.6° on coronal (90th percentile 8.7°), 5.9° on sagittal (11.5°).
  Centring moves a crop 5 to 7 mm at the median; it gains Medial Meniscus, ACL, Effusion and
  Contusion by about 0.002 each and costs Fracture 0.0088 and PF OA 0.0037. Half the angle on the
  two reference models: run 24 +0.0007, the ResNet-50 +0.0002.
- **The frame's angles across knees** (run 24's frames): nod median 14°, 1st to 99th percentile
  3° to 21°; roll −14° to 11°; turn −14° to 15°; single knees reach −25° of roll and −39° of
  turn. The height-cuts model's nod has a median of 6°. A 300 px canvas holds a 224 px crop turned by
  7.5° and moved by 9 mm, or turned by 15° and moved by 5 mm.
- **The spread of the frame's centre around the hand-set crop position:** 2.3 mm side to side,
  4.8 mm front to back, 7.4 mm up and down (standard deviations, spot noise of 4 to 6 mm
  included).
- **A femur segment with its own lean**
  (`outputs/training/20261007-101112-side-routed-femur-lean`, on the height-cuts model): the band above the upper cut laid
  out around an axis hinged at the cut, lean = a start from the kneecap spot plus a per-knee
  regression learned from the labels. ×4 −0.0002, ×1 −0.0009; Effusion +0.0021. The lean went to
  −25° for every knee within two epochs and the nod rose from 6° to 15°: the two angles traded
  against each other. On samples the lean does not follow the femur. The kneecap spot in the
  knee's frame: 21.8 mm in front of the axis (1.57 token pitches, SD 0.40) and 12.5 mm above the
  upper cut (0.90 pitches, SD 0.21), found in all 874 validation knees.
- **The bend between femur and tibia** could not be measured by rule (the limb's outline follows
  the calf, bright tissue follows the fat pads). By eye on eighteen sagittal slices the tibia
  stands close to vertical and the femur leans by 0° to 30°. The crops show about 3 cm of femur
  above the joint.
- **View sizes are at the acquired resolution** (84 mm T1/PD: 90% of series, 93 mm fluid: 90%;
  [review](../../../docs/data-preparation/plane-fluid-spacing-review.md)); sizes are fixed here.

## The v5 report answers

Extracted on 2026-10-07 in the worktree `knee-abnormality-jevk5-attributes-v4` (specification and
check results: its `attributes_v5.md`). The 34 kept questions are the v4 answers run 24 already
trains on; two questions are new. Their option probabilities, one column per option
named `attribute_<question>__<option>` beside `StudyInstanceUID`, are in the prepared file the
seeds name.

| Question | Options | On the 212 audited reports |
|---|---|---|
| `marrow_edema_bone` | `none`, `femur`, `tibia`, `femur_tibia`, `patella`, `patella_and_other`, `not_stated` | Exact option right in 176; as two yes/no targets the femur is right in 48 of 60 and the tibia in 58 of 72; femur and tibia swapped once in 37 single-bone reports; its typical error is `femur_tibia` where one bone or only the compartment is named |
| `extensor_tendon` | `none`, `quadriceps`, `patellar`, `both`, `not_stated` | 205 right; located precision 0.943 |

Counts in the 4,407 reports, estimated from the text: edema in the femur only 336, the tibia only
338, both 452, bone not named 769, no edema 2,512; with the compartment answer, medial femur 393,
medial tibia 349, lateral femur 354, lateral tibia 310. Tendon findings: quadriceps only 237,
patellar only 183, both 164. A located target needs about 200 studies in its smaller class (118
was not enough in v4), and a knee whose report says the finding is absent is a full-weight
negative.

The report heads live in the training module (`training/report_attributes.py`), not in the
model: their attention does not exist at prediction, so a frame cannot be read from it unless a
seed moves that head into the model.

## Reusable code

- Every campaign experiment has `source/` and `changes.diff` under
  `outputs/autoresearch/routing-and-shifts/experiments/<experiment>/`. The canvas loader (uint8
  canvases with a margin, normalised in the loader with the prepared images' bounds) and the in-model
  cutter are in `spot-shift-locator-keeps-spatial-maps-k-20261007T043843Z-632ceac9`.
- Manual runs keep their full source in `source_snapshot.zip`:
  `outputs/training/20261007-081431-side-routed-crop-offsets-3mm` (the canvas loader on the
  height-cuts model with a cutter that takes an offset per slice, no locator) and
  `outputs/training/20261007-101112-side-routed-femur-lean` (the upper band around its own axis
  in `frame()`, the kneecap spot's place returned by `study_frames`).
- `outputs/analysis/routing-and-shifts-frames-20261007/builds/rotation-on-run-24/`: the canvas
  loader and the crop rotation inside the model on run 24's code, this campaign's base (`modeling/model.py`: `ViewWindows.frame_angles`,
  the rotated cut in `ViewWindows.forward`, the first look in `KneeAbnormalityModel.forward`).
  `builds/scripts/frame_crops.py` has the centring as well and the controls.

## What is known about learned shifts (learnable-views campaign)

From the [learnable-views campaign](../learnable-views/INDEX.md) (47 runs on the old recipe,
fold 0; [build notes](../learnable-views/build-notes.md)):

- Views were cut in the model from a wider canvas, with the width, two offsets per view, a label
  bias and the slice budget learned by gradient. None was adopted; the hand 84 and 93 mm views
  with the label bias became the recipe.
- **Shifts per series type looked random:** means within ±2.5 mm of the start, 3 to 7 mm apart
  between runs.
- **Sizes pinned, shifts learned** (`hand-sizes-learned-shifts`, 8 mm shift jitter): ×4 0.932905,
  −0.0004 against its reference, 80% interval [−0.0014, +0.0005], ×1 0.9286.
- **Left open:** the affine augmentation moves every crop by up to 4% of its width per axis on
  70% of series, which trains the model to ignore shifts of the size the learned offsets move by.
- The loader anchors coronal and sagittal crops on a knee centre fitted across series; the
  side-routing measurements found the joint line varying by about ±5 mm between studies and 99%
  of same-plane series pairs moving under 2.7 mm.
- Nothing per study was tried there. In this line the frame per knee is read from the heads' own
  attention spots, which gives a per-study point without any marking.

Reusable code for topic 2 (older recipe; adapt, do not copy whole): the branch `view-optimisation`
(commit `b290da6`), whose `src/knee_abnormality/modeling/views.py` has the in-model cutter; and,
under `outputs/autoresearch/learnable-views/experiments/`, each with `source/` and `changes.diff`:
`hand-sizes-learned-shifts-20261003T183532Z-8d11d7a5`,
`knee-anchored-learned-offsets-20261002T194645Z-d4e331e1` and
`hand-views-routed-shift-learning-20261003T212020Z-0863da22` (built and reviewed, never run).
Runs of this line keep their full source in `source_snapshot.zip` in the run directory.
