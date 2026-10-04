# 2026-10-03 learnable-views campaign

**Learning the views by gradient did not beat setting them by hand; what the learned recipe
contributed is the label bias, and that is on main.** The `learnable-views` autoresearch campaign
ran 47 scored runs on a RadImageNet ResNet-18 that learns each series type's view width and
shift, a per-view label bias and the slice budget. None met the adoption rule. On 2026-10-03 the
user made the hand 84/93 mm views, frozen, with the label bias the campaign's reference
(`hand-views-frozen-with-label-bias`: ×4 0.933307, ×1 0.9299, **0.950 public as S100**); the same
recipe built on main scores ×4 0.932477, ×1 0.9290 and **0.951 public (S101)** and is the recipe
in force ([adoption](../../modeling/current-model.md#adoption-history)). The campaign was stopped
by the user on 2026-10-03 23:40 CEST.

Fold 0 of `all_5fold_scanner_v2` (912 studies), `rebuild_state_v2` labels, seed 1533697213, ×4
predictions, paired report-group bootstrap with 2,000 draws. Adoption rule: the lower end of the
central 80% interval above zero
([protocol](../../../autoresearch/protocol.md#shared-evaluation-and-adoption)). Every result is
one run at one seed. Yardstick: the standard deviation over the learned runs is 0.0008 at ×4 and
0.0012 at ×1, which overstates run noise because the runs differ by mechanism; the reference
recipe trained twice (campaign build and main) differs by 0.0008 at ×4.

- **Campaign record** (contract, seeds, every proposal, decision summary):
  [learnable-views campaign](../../../autoresearch/campaigns/learnable-views/INDEX.md).
- **Build, checks, the runs before the campaign and two reviews of its first 33 runs:**
  [build notes](../../../autoresearch/campaigns/learnable-views/build-notes.md).
- **Code:** branch `view-optimisation`, commit `b290da6` (off `8ef12dc`); it is not on main.
- **Artifacts:** `outputs/autoresearch/learnable-views/` (state, and per run under
  `experiments/<run>/` the frozen source, `changes.diff`, config, logs, predictions and
  comparison), `outputs/analysis/learnable-views-review-20261003/` (review tables).

## Runs

"×1" is the selected checkpoint's validation macro AUC. Runs are in the order they were scored.

### Against the learned reference

Reference: `views-budget-shift-centre` (views, label bias and budget learned from the 120 mm
centre crop; ×4 0.932489, ×1 0.9273, 0.947 public as S94). It sits about 0.0006 above the average
learned run, so these deltas understate a mechanism by about that much.

| # | Run | What changes | ×4 macro | Δ | 80% interval | ×1 |
|---:|---|---|---:|---:|---|---:|
| 1 | `two-views-sagittal-fluid` | Two learned views on sagittal fluid (3 + 2 slices) | 0.932215 | −0.0003 | [−0.0018, +0.0012] | 0.9285 |
| 2 | `two-views-every-type` | Two learned views on every series type | 0.932373 | −0.0001 | [−0.0016, +0.0013] | 0.9253 |
| 3 | `knee-anchored-learned-offsets` | Every view anchored on the knee centre, offsets learned | 0.931991 | −0.0005 | [−0.0016, +0.0006] | 0.9260 |
| 4 | `knee-frame-3d-position` | 3D token position from a learned knee centre, in place of 1D and 2D | 0.932077 | −0.0004 | [−0.0019, +0.0010] | 0.9273 |
| 5 | `learned-centre-fit` | Knee-centre fit with a learned weight per series type | 0.933105 | +0.0006 | [−0.0005, +0.0017] | 0.9288 |
| 6 | `knee-frame-3d-position-with-1d-2d` | 3D position beside the 1D and 2D ones, learned centre | 0.932996 | +0.0005 | [−0.0007, +0.0017] | 0.9281 |
| 7 | `knee-frame-3d-position-ema` | 3D position with EMA weights (user exception) | 0.931784 | −0.0007 | [−0.0020, +0.0005] | 0.9272 |
| 8 | `computed-centre-3d-position` | 3D position from the computed centre, no learned centre | 0.932369 | −0.0001 | [−0.0015, +0.0013] | 0.9275 |
| 9 | `knee-frame-3d-position-centre-long-average` | 3D position, centre weights on a long gradient average | 0.930758 | −0.0017 | [−0.0032, −0.0004] | 0.9271 |
| 10 | `knee-frame-3d-position-centre-noise` | 3D position, knee centre displaced 4 mm at random in training | 0.931752 | −0.0007 | [−0.0023, +0.0007] | 0.9283 |
| 11 | `learned-slice-spread` | Spread of each view's slices along the stack learned | 0.930075 | −0.0024 | [−0.0043, −0.0005] | 0.9276 |
| 12 | `knee-frame-3d-position-first-order-oa-detector` | 3D position with the OA detector started first-order | 0.933443 | +0.0010 | [−0.0004, +0.0023] | 0.9272 |
| 13 | `views-learned-on-held-out-studies` | View values learned on a tenth of the studies the weights never see | 0.930089 | −0.0024 | [−0.0040, −0.0009] | 0.9258 |
| 14 | `view-optimizer-long-average` | View values on a gradient averaged over about 100 steps | 0.931994 | −0.0005 | [−0.0017, +0.0008] | 0.9277 |
| 15 | `views-start-late` | Views held for three epochs, then learned | 0.931843 | −0.0006 | [−0.0018, +0.0005] | 0.9257 |
| 16 | `label-bias-off` | Label bias removed | 0.930803 | −0.0017 | [−0.0030, −0.0004] | 0.9250 |
| 17 | `normal-jitter` | Jitter 4 mm and ±5% in place of 8 mm and ±10% | 0.931724 | −0.0008 | [−0.0022, +0.0006] | 0.9276 |
| 18 | `start-from-hand-recipe` | Learned views started from the hand 84/93 mm table | 0.931728 | −0.0008 | [−0.0022, +0.0006] | 0.9265 |
| 19 | `views-only-fixed-budget` | Slice budget fixed, views learned | 0.931941 | −0.0005 | [−0.0019, +0.0009] | 0.9273 |
| 20 | `views-frozen-late` | Views frozen after five epochs | 0.932306 | −0.0002 | [−0.0014, +0.0010] | 0.9270 |
| 21 | `learned-centre-soft-boundaries` | Learned centre with soft region cut-offs | 0.931920 | −0.0006 | [−0.0019, +0.0007] | 0.9282 |
| 22 | `sagittal-fluid-ordered-view-widths` | Two sagittal-fluid views kept ordered narrow and wide | 0.932908 | +0.0004 | [−0.0008, +0.0016] | 0.9291 |
| 23 | `first-order-oa-detector-image-views` | First-order OA detector on the reference's views | 0.931967 | −0.0005 | [−0.0017, +0.0006] | 0.9289 |
| 24 | `slice-budget-all-token-density` | Slice budget taught by every token's attention density | 0.931552 | −0.0009 | [−0.0023, +0.0004] | 0.9270 |
| 25 | `sagittal-fluid-evidence-conditioned-routing` | Ordered sagittal-fluid pair with routing conditioned on each view's evidence | 0.933143 | +0.0007 | [−0.0006, +0.0019] | 0.9301 |
| 26 | `learned-centre-with-slice-spread` | Learned centre with the learned slice spread | 0.931480 | −0.0010 | [−0.0024, +0.0003] | 0.9294 |
| 27 | `view-clipping-feasibility-gradient` | Gradient for views clipped at the image edge (feasibility) | 0.931634 | −0.0009 | [−0.0022, +0.0005] | 0.9275 |
| 28 | `ordered-sagittal-pair-with-learned-depth-spread` | Ordered sagittal-fluid pair with a learned depth spread | 0.932334 | −0.0002 | [−0.0014, +0.0011] | 0.9282 |
| 29 | `learned-centre-with-slice-spread-widen-only` | Learned centre, slice spread floored at its start | 0.932437 | −0.0001 | [−0.0013, +0.0011] | 0.9282 |
| 30 | `view-learning-through-photometric-jacobians` | View gradient carried through the photometric augmentations | 0.932897 | +0.0004 | [−0.0007, +0.0016] | 0.9277 |
| 31 | `learned-centre-spread-with-reference-mixture` | Learned spread mixed with the reference sampling law | 0.930405 | −0.0021 | [−0.0036, −0.0005] | 0.9266 |
| 32 | `view-clipping-inward-image-gradient` | Gradient for clipped views from the image inward | 0.931545 | −0.0009 | [−0.0022, +0.0004] | 0.9273 |
| 33 | `ordered-sagittal-pair-with-complementary-depth-routing` | Ordered sagittal-fluid pair, the two views routed to complementary depths | 0.931069 | −0.0014 | [−0.0028, −0.0002] | 0.9270 |
| 34 | `learned-per-axis-anchor-interpolation` | Anchor of each axis interpolated between image and knee, learned | 0.932210 | −0.0003 | [−0.0014, +0.0008] | 0.9284 |
| 35 | `oa-detector-3d-with-sagittal-evidence-routing` | First-order OA detector with 3D position, and the sagittal evidence routing | 0.932053 | −0.0004 | [−0.0019, +0.0010] | 0.9287 |
| 36 | `hand-views-frozen-with-label-bias` | **Hand 84/93 mm views frozen, label bias on (user seed)** | 0.933307 | +0.0008 | [−0.0007, +0.0023] | 0.9299 |

`views-only-fixed-budget` had an earlier attempt judged invalid; the row is its repair.

### Against the hand views with the label bias

Reference: run 36, ×4 0.933307, ×1 0.9299. It is a high draw of its recipe (0.932477 from main),
so these deltas overstate a loss by up to 0.0008.

| # | Run | What changes | ×4 macro | Δ | 80% interval | ×1 |
|---:|---|---|---:|---:|---|---:|
| 37 | `hand-views-sagittal-fluid-pair-learned-budget` | Second, wide sagittal-fluid view (3 + 2 slices), slice budget learned | 0.931352 | −0.0020 | [−0.0034, −0.0006] | 0.9271 |
| 38 | `hand-views-learned-budget` | Slice budget learned, nothing else | 0.931683 | −0.0016 | [−0.0030, −0.0004] | 0.9295 |
| 39 | `hand-views-label-bias-faster-rate` | Label bias at ten times the head rate | 0.933389 | +0.0001 | [−0.0004, +0.0006] | 0.9301 |
| 40 | `hand-sizes-learned-shifts` | View shifts learned with the sizes pinned, 8 mm shift jitter | 0.932905 | −0.0004 | [−0.0014, +0.0005] | 0.9286 |
| 41 | `hand-start-views-held-early` | Learned views from the hand table, held three epochs | 0.932965 | −0.0003 | [−0.0014, +0.0008] | 0.9282 |
| 42 | `view-size-menu-bias-guided` | Two slots per coronal and sagittal type drawing sizes from a menu, draws following the label bias | 0.933656 | +0.0003 | [−0.0010, +0.0017] | 0.9278 |
| 43 | `hand-views-training-randomness` | Jitter 8 mm and ±10% and random slice counts, nothing learned | 0.932427 | −0.0009 | [−0.0022, +0.0003] | 0.9282 |
| 44 | `hand-views-patellofemoral-axial-zoom` | Second axial-fluid view, 84 mm on the patellofemoral joint (3 + 3 slices) | 0.932017 | −0.0013 | [−0.0027, −0.0001] | 0.9262 |
| 45 | `hand-views-label-depth-bias` | Label bias per view, label and slice-depth bin | 0.933577 | +0.0003 | [−0.0001, +0.0006] | 0.9302 |
| 46 | `learned-centre-with-protected-strata-depth-transport` | Learned depth law of the slices with the outer strata fixed | 0.933354 | +0.0000 | [−0.0014, +0.0014] | 0.9276 |
| 47 | `ordered-sagittal-routing-with-shared-affine-exploration` | Ordered sagittal routing with shared affine exploration (scored, not decided: stopped) | 0.930760 | −0.0025 | [−0.0044, −0.0008] | 0.9273 |

Proposed and never trained: `learned-axis-specific-window-spacing` and
`photometric-views-with-quantile-aware-gradient` (blocked in preparation; the second cannot act
on frozen views), `view-size-menu-uniform` (retired by the user),
`hand-views-routed-shift-learning` (built and reviewed, not run),
`view-size-menu-with-deployment-replay` (queued).

## What the runs established

- **The hand view table is worth about 0.002 and gradient learning does not find it.** The hand
  views frozen against the learned final views frozen, same build: +0.0020 at ×4, +0.0028 at
  ×1. Learned widths drift to about 110 mm from any start (SD 8 to 15 mm between runs); from
  the hand table they widen again (run 18: coronal 84 to 105 and 93 to 111 mm). Holding them
  three epochs while the backbone adapts does not prevent it (run 41: 84 to 89–93 mm, 93 to
  102–107 mm, moved within two epochs of release), so the early drift is not an unadapted
  backbone. A zoomed view does not read better alone; the gain appears only in the pooled
  prediction, which a per-view gradient cannot see
  ([build notes](../../../autoresearch/campaigns/learnable-views/build-notes.md#what-separates-s93-from-the-learned-recipe-2026-10-03)).
- **A size menu prefers wider views too.** Run 42 draws each slot's size from 70, 84, 93, 105
  and 120 mm, the draws following the label bias: 70 mm ends at 0.07–0.11 in all eight slots,
  and the preferred sizes are 93–120 mm. It has the highest ×4 of the campaign (0.933656) with a
  low ×1 (0.9278): its gain from ×1 to ×4 is +0.0058, against +0.0034 for the reference.
  Validation uses one size per slot while training draws several, so its ×1 is not a clean
  comparison.
- **With the sizes pinned, placement learning is stable and worth nothing.** Run 40: shifts
  move up to 9 mm and settle after epoch 4; ×4 −0.0004, ×1 −0.0012.
- **The label bias is the learned recipe's own gain.** Removing it costs 0.0017 (run 16, interval
  below zero). It learns the same table in every run (59 of 72 entries keep their sign in 24 of
  27 runs). It is rate-limited but already at its useful size: at ten times the rate it grows
  seven-fold (mean 0.259 against 0.036, largest 0.91) with the same signs (correlation 0.98)
  and the same score (run 39: +0.0001, interval ±0.0005).
- **A label bias per slice depth is the nearest miss.** Run 45: +0.0003, 80% interval
  [−0.0001, +0.0006], ×1 0.9302 (the best of the campaign), ten of twelve labels up. The table
  is anatomical: for ACL on sagittal fluid the central bins rise (+0.11, +0.06) and the outer
  ones fall.
- **The slice budget learns a reproducible shift that is too small to matter**, a quarter of a
  slice in the learned recipe; on the hand views it moves at most 0.2 of a slice and the run
  loses 0.0016 (run 38, interval below zero), its gain from ×1 to ×4 falling to +0.0022.
- **A second view on a type costs the slices it takes.** A wide second sagittal-fluid view with
  the learned budget: −0.0020 (run 37). A zoomed patellofemoral axial-fluid view: −0.0013 (run
  44), with PF OA +0.0027 and the label bias routing PF OA to the zoomed view and Baker's cyst
  to the wide one as designed, but contusion −0.0051, MCL −0.0087 and fracture −0.0062 on the
  wide view's remaining three slices. In the learned recipe the ordered sagittal-fluid pairs
  gained the lateral meniscus (+0.004) and lost ACL (−0.002), and reached the hand recipe's ×1.
- **The learned recipe's training randomness does not transfer.** Run 43 (8 mm, ±10%, random
  slice counts, nothing learned): ×1 −0.0017, gain from ×1 to ×4 +0.0042, ×4 −0.0009.
- **Inside one run's noise:** knee anchors, the learned centre (it ends 0.2 to 1.0 mm from the
  computed one), the 3D position in place of or beside 1D and 2D, the first-order OA detector,
  EMA, view schedules and optimizers, clipping gradients, photometric Jacobians. **Negative:**
  views learned on held-out studies, the free slice spread (coronal spreads collapse to a
  quarter; MCL −0.015 to −0.019) and its mixtures, the centre on a long gradient average.
- **A lead, not a finding:** PF OA rose by +0.0023 to +0.0039 in six of the ten runs against
  the hand reference, each in a run that changes the views or the slices they get; none is separated
  from zero.
- **Early curves do not predict the result.** ×1 at epochs 0 to 5 has no correlation with the
  final ×4; runs with narrow views start 0.005 to 0.007 behind and finish ahead.

## Details of the runs against the hand reference

Per-label deltas are ×4 against run 36, unadjusted.

- **37, sagittal-fluid pair with learned budget.** Budget split between the two views stayed at
  3.0/2.0; other types moved at most 0.25 of a slice. Baker's cyst −0.0016 and PF OA +0.0001
  (the labels the wide view was for); lateral meniscus +0.0035; MCL −0.0103, fracture −0.0053.
  Bias entries on both views 0.005 to 0.03.
- **38, learned budget.** Coronal fluid 5.05 to 5.20, sagittal T1/PD 4.98 to 4.81. Fracture
  −0.0083, synovitis −0.0036, contusion −0.0034, PF OA +0.0027.
- **39, label bias at ten times the rate.** Eight labels up (MCL +0.0014, ACL +0.0009), Baker's
  cyst −0.0014, fracture −0.0013.
- **40, learned shifts.** End offsets (start in brackets): axial fluid horizontal 7.8 mm (0),
  axial T1/PD horizontal 5.8 (0), coronal fluid vertical 1.7 (10), coronal T1/PD vertical 3.7
  (10) and horizontal −3.7 (0), sagittal fluid horizontal 1.7 (7) and vertical 12.1 (10),
  sagittal T1/PD horizontal 11.4 (7) and vertical 6.4 (10). Both coronal views drop most of the
  10 mm vertical offset; the two sagittal views move in opposite directions horizontally.
  Lateral meniscus +0.0045, synovitis +0.0038, PF OA +0.0034, MCL −0.0041, contusion −0.0034.
  It trained with the 8 mm shift jitter, so its ×1 cost is not separated from the jitter's.
- **41, hand start held early.** Final widths: axial fluid 112 mm, axial T1/PD 128, coronal
  fluid 107, coronal T1/PD 93, sagittal fluid 102, sagittal T1/PD 89. Medial OA −0.0029, lateral
  meniscus +0.0023.
- **42, size menu.** Final draw probabilities (uniform is 0.20): coronal T1/PD slot 1 93 mm
  0.34 and 120 mm 0.30, slot 2 120 mm 0.43; coronal fluid slot 1 105 mm 0.60, slot 2 84 mm
  0.35 and 105 mm 0.29; sagittal T1/PD slot 1 93 mm 0.52, slot 2 105 mm 0.59; sagittal fluid
  slot 1 120 mm 0.54, slot 2 120 mm 0.36. Only coronal fluid splits into a wide and a narrow
  slot. Fracture +0.0054, PF OA +0.0026.
- **43, training randomness.** 56% of training studies had slice counts different from the
  fixed budget; half the windows were shifted beyond 4 mm. PF OA +0.0039 (95% [−0.0002,
  +0.0081]), synovitis −0.0039, MCL −0.0033, fracture −0.0032.
- **44, patellofemoral zoom.** The zoomed view sits 21 mm anterior of the knee centre, from
  typical anatomy; 0.2% of validation windows had to be moved back inside the image. Bias: PF
  OA +0.059 on the zoomed view and +0.015 on the wide one; Baker's cyst −0.102 and +0.112.
  ×1 0.9262.
- **45, label depth bias.** Mean absolute value 0.044, flat from epoch 7. ACL +0.0014,
  synovitis +0.0013, lateral OA +0.0010, fracture −0.0021. MCL on coronal T1/PD: one bin at
  −0.14.
- **46, protected-strata depth transport.** Synovitis +0.0051, lateral meniscus +0.0047,
  Baker's cyst +0.0034, fracture −0.0067, contusion −0.0057 (95% interval below zero).
- **47, shared affine exploration.** Scored (−0.0025, interval below zero); the campaign
  stopped before its decision was recorded.

## Submissions

| | Model | Fold 0 ×4 | Public |
|---|---|---:|---:|
| S94 | Learned reference | 0.932489 | 0.947 |
| S98 | Run 25, learned sagittal routing | 0.933143 | 0.947 |
| S99 | Runs 12, 25, 5, 6 and 22, equal weights | 0.935015 (blend) | 0.951 |
| S100 | Run 36, hand views with label bias | 0.933307 | 0.950 |
| S101 | The same recipe on main | 0.932477 | 0.951 |

Five learned-view ResNet-18 runs blended score what one hand-view run scores
([ledger](../../inference/submissions/ledger.md)).

## Retaking this line

- **Start from** the branch commit above for the learnable build, or from a run's frozen source
  for a single mechanism: `outputs/autoresearch/learnable-views/experiments/<run>/source`, with
  `changes.diff` against its parent. The run folders are named after the idea and its launch
  time. The disk cleanup does not delete them, and they are not in git.
- **Checkpoints kept** (selected checkpoint only): runs 36, 5, 6, 12, 22, 25, 39, 42, 45 and 46
  ([pruning record](../../architecture/configs-and-experiment-artifacts.md#checkpoint-pruning)).
- **Follow-ups worth a run** are in the backlog:
  [view table and label routing](../../backlog/big-bets.md#view-table-and-label-routing).
- **Resuming the campaign itself** needs the branch's worktree, where `configs/training.yaml`
  is the learnable recipe and the harness carries one change that main does not: the validation
  selection may differ between checkpoints when the learned loader state differs
  (`loader_state_sha256` in the triplet ledger). The exporter on the branch keeps
  `loader_state` in the checkpoint; a learned-view model cannot be predicted without it.
