# Prior work for the routing-and-shifts campaign

The summary the researcher checks proposals against. Tables, drawings and every run's details
are in the [plan and runs](../../../docs/ongoing/side-routed-heads.md); the first campaign on
the design is [side routing](../side-routing/INDEX.md), and learned view placement was studied
in [learnable views](../learnable-views/INDEX.md).

## The incumbent

Run 24, `outputs/training/20261006-163650-side-routed-spot-frame-ring-sectors`: fold 1 of
`all_5fold_scanner_v2`, seed 1110484336, ResNet-18, ten epochs. ×4 0.925067, ×1 0.917070, 0.953
public (S118). The old recipe without routing, same fold and seed
(`outputs/autoresearch/fixes-and-removals/experiments/slab-rule-and-positive-std-requeue-20261005T155303Z-59280bbc/training`):
×4 0.924049, ×1 0.916134, 0.950 public.

Its design: the frame of each knee is centred on the midpoint of two meniscus attention spots
(squared-attention centroids of the shared meniscus channel over the two fitted halves, no
gradient), its side axis is the line between them, the nod and four sizes (upper cut, lower cut,
core radius, ring radius) are regressed per knee; eight 45° sectors centred on the axes, a core,
an inner and an outer ring, three heights; hard cuts with a soft gradient; routed v4.0 report
targets; a knee-level term on every routed channel.

| Target | Region in the incumbent |
|---|---|
| Medial, Lateral Meniscus; Medial, Lateral OA | The three sectors of its side, both rings, its half of the core; tibia and joint bands |
| PF OA, Effusion | The three front sectors, both rings; joint and upper bands |
| Synovitis | The same and the core |
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
  874 validation knees): `outputs/analysis/s/side-routing-old-vs-reference-20261006/` (`old.npz`,
  `reference.npz`, `old-vs-reference.png`). Share of the old model's attention outside the
  incumbent's region: ACL 35% (95% of it below the lower cut), Synovitis 34%, MCL 29%, Effusion
  24%, PF OA 13%, Lateral OA 13%, the rest 6% or less.
- **Attention spots** in the fitted frame (medial, front, up; mm): Medial Meniscus (26, −7, −7),
  Lateral Meniscus (−29, −5, −7), ACL (−6, −2, −1), PF OA (−2, 32 to 36, 13), Effusion (−6, 26,
  32), Synovitis (−3, 13, 19); each a blob of 15 to 20 mm, the same with and without the finding.
- **Which targets hold each cut** (`outputs/analysis/s/side-routing-cut-forces-20261006/cut-forces.png`):
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

## Topic 2: what is known about learned shifts

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
