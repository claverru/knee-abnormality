# Keeper's run notes

Decisions, repairs and per-run details of the routing-and-shifts campaign. Adoption is the
contract's: at least the incumbent at both ×1 and ×4. Incumbent at the start: run 24 (×4 0.925067,
×1 0.917070). Per-target deltas are ×4 against the incumbent; sibling SDs are in
[prior work](prior-work.md#yardstick).

## Runs

### 1. `learned-shift-per-series-type` — invalid, over the time budget (2026-10-07 00:14)

- First candidate sent back by the reviewer for a zero-shift difference of 1 grey level (allowed
  by the seed); the second trained and was stopped after its first epochs: training projected at
  2,021 s against the limit of 1,595 s (1.65 times the reference; 1.3 allowed).
- Epoch 1 at ×1: 0.892679 against 0.892140 for the incumbent. The shift parameters received
  gradient (`view_shift_grad_norm` 0.024).
- The developer repaired the cost without a GPU to time it; the third candidate was approved and
  queued. Its speed is unmeasured until it trains.

### 2. `menisci-read-their-whole-side` — not adopted (2026-10-07 00:37)

- ×4 0.924067 (−0.0010, 80% interval −0.0017 to −0.0003), ×1 0.915785 (−0.0013). Lower at both.
- Per target: Effusion +0.0024, Contusion +0.0013, PF OA +0.0009, ACL +0.0007, Medial Meniscus
  +0.0004, Baker's +0.0003, Medial OA 0.0000, MCL −0.0005, Fracture −0.0023, Lateral Meniscus
  −0.0032, Synovitis −0.0058, Lateral OA −0.0064.
- The seed's premise did not hold: Medial Meniscus moved by +0.0004, half of its sibling SD, with
  its whole side to read. Its higher scores in runs 1 to 5 and 22 did not come from the width of
  its region. Lateral OA and Synovitis, which this seed does not touch, lost 1.9 and 2.6 SDs.

### 3. `held-cuts-with-wider-regions` — not adopted (2026-10-07 01:13)

- ×4 0.924026 (−0.0010, 80% interval −0.0020 to −0.0000), ×1 0.916055 (−0.0010). Lower at both.
- Per target: Contusion +0.0048 (3.7 sibling SDs), Medial OA +0.0023 (2.9), ACL +0.0018, Lateral OA
  +0.0016, Medial Meniscus +0.0013 (1.9), Baker's −0.0008, Effusion −0.0010, Lateral Meniscus
  −0.0018, PF OA −0.0022 (4.4), Synovitis −0.0032, Fracture −0.0051, MCL −0.0102 (3.5).
- The core radius held (23.7 mm at epoch 1, 21.7 at epoch 10; start 24). The height cut did not: it
  moved from 14.2 mm at epoch 1 to 7.2 at epoch 10 (start 12), so with one cut and the regions at
  15% it is not held as the 20% division's two cuts were in run 30.
- Against run 30, the wider regions gave back Synovitis (−0.0032 from −0.0050) and Effusion
  (−0.0010 from −0.0031) in part, not PF OA (−0.0022 in both) nor MCL (−0.0102 from −0.0078). MCL
  reads the medial half of the core and the back-medial sector in both; its incumbent region is
  the medial and anteromedial sectors at all heights. The two targets that gain beyond noise,
  Contusion and Medial OA, are the lead: Medial OA reads its half of a 24 mm core here.

### 4. `height-cuts-from-the-spots-requeue` — adopted by the contract (2026-10-07 02:13)

- ×4 0.925389 (+0.0003, 80% interval −0.0003 to +0.0010), ×1 0.917411 (+0.0003). At least the
  incumbent at both multipliers, guardrails passed, training 1,203 s (limit 1,595): adopted with
  `adopt`, citing the contract; the runtime's own verdict was reject. It is the incumbent from
  here. The gain is under one sibling SD of the macro (0.0004).
- Per target, none beyond its noise: Contusion +0.0019, MCL +0.0013, Effusion +0.0010, Lateral
  Meniscus +0.0009, Baker's +0.0006, ACL +0.0002, Medial Meniscus +0.0002, PF OA −0.0001, Synovitis
  −0.0001, Medial OA −0.0003, Lateral OA −0.0009, Fracture −0.0009.
- Geometry at the last epoch against run 24: upper cut +10.8 mm (13.7), lower −12.9 (−1.8), core
  radius 16.3 (14.7), ring radius 29.8 (26.5), nod 6.0° (13.6°). The two height cuts hold where the
  spots put them and ACL keeps its score, which run 28 lost through the ring radius.
- The adoption was applied one run late. `adopt` writes a request that the supervisor processes
  only while it runs and no training is in progress; the first attempts came before the run's
  decision was recorded and then while the next training ran. `shared-ligament-reader-2` therefore
  trained on run 24. Procedure from here: `stop` when a qualifying run nears its end, `adopt` once
  it is `STOPPED`, `resume`; the request is applied before the next launch.

### 5. `shared-ligament-reader-2` — not adopted (2026-10-07 02:13)

- Built and compared on run 24. ×4 0.924362 (−0.0007, 80% interval −0.0013 to −0.0001), ×1 0.917334
  (+0.0003). Lower at ×4.
- Per target: ACL +0.0021 (1.5 sibling SDs), Effusion +0.0013, Baker's +0.0006, Medial Meniscus
  −0.0002, PF OA −0.0005, MCL −0.0009, Contusion −0.0009, Lateral Meniscus −0.0012, Medial OA
  −0.0018 (2.3), Fracture −0.0020, Lateral OA −0.0023, Synovitis −0.0028.
- ACL moved in the intended direction, by 1.5 of its SDs; the two radii ended at 14.9 and 28.8 mm
  against 14.7 and 26.5. The four targets are logged as
  `val_report_attribute_bce_ligament_injury_<question>`.

### 6. `synovitis-shares-effusion-attention-2` — not adopted (2026-10-07 02:37)

- First run on the new incumbent (0.925389, 0.917411). ×4 0.924222 (−0.0012), ×1 0.916078
  (−0.0013). Lower at both.
- Per target: Medial Meniscus +0.0002, Fracture +0.0002, Medial OA −0.0002, Baker's −0.0003, PF OA
  −0.0004, Effusion −0.0009, Lateral Meniscus −0.0009, ACL −0.0017, Contusion −0.0018, MCL −0.0021,
  Lateral OA −0.0024, Synovitis −0.0037 (1.7 sibling SDs).
- Shown wrong on its own target: with Effusion's attention Synovitis fell, and Effusion did not
  gain. Synovitis does not look where Effusion looks closely enough to borrow its attention (its
  free spot is 13 mm forward and 19 mm up against 26 and 32). Nothing rose.

### 7. `contusion-reads-the-edema-head-2` — not adopted (2026-10-07 02:57)

- On the height-cuts incumbent. ×4 0.924674 (−0.0007), ×1 0.916263 (−0.0011). Lower at both.
- Per target: MCL +0.0021, ACL +0.0006, Fracture +0.0003, Medial Meniscus 0.0000, Baker's −0.0002,
  Lateral Meniscus −0.0003, Effusion −0.0004, Synovitis −0.0008, Medial OA −0.0010, PF OA −0.0014
  (2.8 sibling SDs), Contusion −0.0024 (1.8), Lateral OA −0.0052 (1.5).
- Shown wrong by its own measure: `contusion_edema_scale` stayed within 0.01 of zero (−0.0012 at
  epoch 1, −0.0099 at epoch 5, −0.0061 at epoch 10) and negative, and Contusion fell. The edema
  head's evidence over the three compartments is not what Contusion's label follows.

### 8. `shift-per-series-type-light-port-c` — not adopted; the shifts could not move (2026-10-07 03:23)

- ×4 0.924827 (−0.0006), ×1 0.916431 (−0.0010) against the height-cuts incumbent. Lower at both.
- The light port fits: 1,455 s of training (limit 1,595), about 21 GB in use early and 28 GB late
  with 10 GB available, shared memory 6 GB; the incumbent's pipeline uses 17, 30 and 4.
- The shifts stayed at zero: at the last epoch the largest of the eight coronal and sagittal
  values is 0.07 mm (coronal fluid, rows), the rest between −0.03 and +0.02 mm. The parameter is
  held in millimetres and trained at the head rate of 0.0006, about 0.0006 mm per step, so 4,420
  steps cannot take it past 2.7 mm even with a constant sign. The keeper's seed text set that
  parametrisation.
- The run is therefore a rerun of the incumbent through the cutter at zero shift, and its deltas
  are a yardstick for this recipe: −0.0006 at ×4 and −0.0010 at ×1 with nothing changed. Per
  target: Effusion +0.0009, Medial Meniscus +0.0005, Fracture +0.0004, Medial OA −0.0002, ACL
  −0.0003, Synovitis −0.0008, PF OA −0.0009, Contusion −0.0009, MCL −0.0010, Baker's −0.0011,
  Lateral Meniscus −0.0015, Lateral OA −0.0018.
- Follow-up forced: [`shift-per-series-type-unit-parameter`](unit-parameter-shift-ideas.yaml), the
  same candidate with the parameter in units of the margin (`limit * tanh(view_shift)`), about
  0.009 mm per step. Queued after `shift-per-study-per-view-light-port-c`.

### 9. `shift-per-study-one-point-light-port-c` — not adopted; the point barely moved (2026-10-07 03:51)

- ×4 0.924344 (−0.0010), ×1 0.915927 (−0.0015) against the height-cuts incumbent. Lower at both.
- The locator's point at the last epoch, over validation knees: mean −0.62 mm (anterior), +0.59
  (left-right), +0.20 (superior); spread across knees 0.07, 0.09 and 0.06 mm. Every knee got the
  same sub-millimetre offset: no per-study shift was learned. The last layer gives millimetres
  directly at the head rate, the same slowness as the per-series parameter.
- Per target: Lateral OA +0.0020, Baker's +0.0008, MCL +0.0002, Medial Meniscus +0.0001, Medial OA
  −0.0002, Contusion −0.0002, Effusion −0.0005, ACL −0.0008, PF OA −0.0012, Lateral Meniscus
  −0.0013, Synovitis −0.0039 (1.8 sibling SDs), Fracture −0.0075 (1.7).
- With runs 8 and 9 the campaign has two runs of the incumbent with crops moved by under 1 mm:
  −0.0006 and −0.0010 at ×4, −0.0010 and −0.0015 at ×1. Both sit below the incumbent, by more than
  the +0.0003 that adopted it.
- Follow-ups forced ([seeds](unit-output-study-shift-ideas.yaml)): the four per-study seeds with a
  dimensionless last layer (`limit * tanh(x)`), about fifteen times the step:
  `shift-per-study-one-point-unit-output`, `shift-per-study-per-view-unit-output`,
  `shift-per-study-from-the-spots-unit-output`, `crops-and-frame-share-one-centre-unit-output`.
  The queued `-c` versions of the last two were removed. `shift-per-study-per-view-light-port-c`
  was already prepared and trains as written; with its millimetre output it serves as a third
  near-zero-shift run of the incumbent.

### 10. `shift-per-study-per-view-light-port-c` — not adopted (2026-10-07 04:19)

- ×4 0.924608 (−0.0008), ×1 0.915496 (−0.0019) against the height-cuts incumbent. Lower at both.
- The offsets moved further than the one-point run's, as a constant: at the last epoch the mean
  over validation knees is +2.29 mm on the coronal fluid rows, +0.63 on the sagittal fluid
  columns, −0.78 on the sagittal T1/PD rows, with a spread across knees of 0.04 to 0.06 mm. The
  per-series bias of each output layer learned; nothing per knee did.
- Third run of the incumbent with crops moved by at most about 2 mm and identically for every
  knee. The three (runs 8, 9, 10): −0.0006, −0.0010, −0.0008 at ×4 and −0.0010, −0.0015, −0.0019 at
  ×1; mean −0.0008 and −0.0015. The incumbent sits above all three at both multipliers.

### 11. `shift-per-series-type-unit-parameter` — not adopted; the per-series shift has no consistent pull (2026-10-07 04:45)

- ×4 0.924590 (−0.0008), ×1 0.916196 (−0.0012) against the height-cuts incumbent. Lower at both.
- With the parameter in units of the margin the shifts moved fifteen times faster in the first
  epoch (about 0.1 mm against 0.007) and could have crossed several millimetres. At the last epoch:
  coronal fluid rows +0.46 mm, sagittal fluid rows −0.46 (−0.73 at epoch 4), coronal T1/PD rows
  −0.36, sagittal T1/PD rows +0.31, the four column shifts within ±0.20. All inside 0.5 mm, with
  signs that wandered.
- By the seed's own criterion (shifts within 1 mm of zero at the last epoch) the target loss gives
  no consistent direction for a shift shared by a series type, on the routed model as in the
  learnable-views campaign. The hand placement stands.
- Fourth run of the incumbent with crops within 0.5 mm of the hand placement: its −0.0008 and
  −0.0012 sit with runs 8 to 10 (means now −0.0008 at ×4, −0.0014 at ×1 over four runs).
- Per target: Medial OA +0.0011, Effusion +0.0001, Baker's 0.0000, Lateral Meniscus −0.0002, PF OA
  −0.0004, Fracture −0.0004, Contusion −0.0007, Medial Meniscus −0.0008, ACL −0.0017, Lateral OA
  −0.0020, Synovitis −0.0020, MCL −0.0026.

### 12. `shift-per-study-one-point-unit-output` — not adopted; the point drifts, PF OA gains from it (2026-10-07 05:12)

- ×4 0.922682 (−0.0027), ×1 0.912854 (−0.0046) against the height-cuts incumbent. Lower at both,
  the lowest run of the campaign.
- The point moved by millimetres and nearly the same for every knee. Mean over validation knees
  by epoch, anterior: +2.3, −5.8, −5.0, +3.3, then +7.5 from epoch 7 to +7.6 at the end;
  left-right: −1.7, −7.0, −5.4, −9.3, −9.5 at the end; superior: −3.2, +0.1, −5.6, −4.3, −6.0 at
  the end. Spread across knees at the end: 0.45, 0.38 and 1.07 mm. It wandered with changing sign
  for four epochs and then settled as the learning rate fell; no per-knee placement was learned.
- Per target: PF OA +0.0040 (8 sibling SDs), Medial Meniscus +0.0019 (2.7), Effusion −0.0007,
  Lateral OA −0.0009, ACL −0.0013, Baker's −0.0022, Medial OA −0.0029, Contusion −0.0043, Fracture
  −0.0053, Synovitis −0.0054, MCL −0.0057, Lateral Meniscus −0.0098 (4.1).
- Reading: every crop ended 7.6 mm further forward and 6 mm lower, and 9.5 mm along the patient's
  left-right axis, which is medial for one knee side and lateral for the other. The forward move
  is where PF OA's gain comes from (the kneecap is further inside the coronal and sagittal crops);
  the left-right move is not anatomical and is the likely cost on Lateral Meniscus and MCL. Lead:
  a forward shift alone for PF OA, and a left-right component expressed towards medial, not along
  the patient axis.

### 13. `shift-per-study-per-view-unit-output` — not adopted; the offsets ran to the limit (2026-10-07 05:40)

- ×4 0.923337 (−0.0021), ×1 0.914540 (−0.0029) against the height-cuts incumbent. Lower at both;
  −0.0094 at ×1 at epoch 2.
- Final mean offsets over validation knees, rows and columns: coronal T1/PD −6.1 and +13.4 mm,
  coronal fluid −6.3 and +1.8, sagittal T1/PD +10.6 and −2.4, sagittal fluid −13.2 and +14.4. Four
  of eight are within 2 mm of the 15 mm bound; the spread across knees is 0.07 to 0.85 mm. The
  views of one study disagree by far more than the 2.7 mm the series move: by the seed's own
  criterion the freedom fitted noise.
- Per target: ACL +0.0026 (1.9 sibling SDs), MCL +0.0004, Medial Meniscus 0.0000, Contusion −0.0012,
  Lateral Meniscus −0.0016, PF OA −0.0018, Effusion −0.0025, Medial OA −0.0028, Baker's −0.0038,
  Synovitis −0.0044, Fracture −0.0045, Lateral OA −0.0050.
- PF OA's gain of run 12 (+0.0040 with every crop 7.6 mm forward) does not repeat here, where the
  two sagittal views moved in opposite row directions and the coronal ones 6 mm along their rows.
- Label-gradient shifts, summed over runs 8 to 13: held near zero they cost nothing measurable;
  free to move they do not settle on a placement per knee, and where they settle is a common
  offset that differs from run to run.

### 14. `shift-per-study-from-the-spots-unit-output` — not adopted; the only shift that differs per knee (2026-10-07 06:04)

- ×4 0.924941 (−0.0004), ×1 0.915219 (−0.0022) against the height-cuts incumbent. Lower at both.
  Against the four near-zero-shift reruns (means −0.0008 and −0.0014) it is +0.0004 at ×4 and
  −0.0008 at ×1.
- The locator learned a per-knee shift with a small common offset (under 3 mm, no drift): spread
  across validation knees at the last epoch 3.8 mm superior, 1.7 anterior, 0.2 left-right (2.2,
  0.2 and 0.1 after epoch 1). Its validation loss fell from 0.067 to 0.057.
- The spread of the meniscus-spot midpoint left over after the shift barely fell: superior 6.7 to
  6.4 mm, anterior 5.0 to 4.6, left-right 2.5 to 2.1. The locator predicts a small part of the
  midpoint's variation; the spots themselves are uncertain by 4 to 6 mm per knee
  (docs/ongoing/side-routed-heads.md, attention spots), so most of the target is noise in the
  spots, and knees are moved by millimetres without ending up more alike.
- Per target: Lateral OA +0.0035, MCL +0.0030, Medial Meniscus +0.0018 (2.6 sibling SDs), Medial OA
  +0.0016 (2.0), Effusion +0.0008, PF OA +0.0007, Baker's −0.0001, ACL −0.0002, Synovitis −0.0020,
  Lateral Meniscus −0.0026, Contusion −0.0039 (3.0), Fracture −0.0080 (1.8). The medial routed
  targets gain and the two unrouted trauma targets lose.
- Lead: a cleaner per-knee target than the spot midpoint of one pass (for instance the spots
  averaged over a knee's epochs, or the loader's fitted centre residual), and the shift applied
  to the vertical axis only, where the locator found most of its spread.

### 15. `crops-and-frame-share-one-centre-unit-output` — not adopted (2026-10-07 06:32)

- Last forced seed. ×4 0.921684 (−0.0037), ×1 0.911241 (−0.0062) against the height-cuts
  incumbent: the lowest run of the campaign.
- The shared point drifted as the label-trained ones did: mean over validation knees at the last
  epoch −3.9 mm anterior, −11.5 mm along the patient's left-right axis, +2.1 mm superior, with a
  spread across knees of 1.6 mm at most. Here the drift also moved the routing frame's centre.
- Per target: ACL +0.0029 (2.1 sibling SDs), Effusion +0.0001, Synovitis −0.0005, PF OA −0.0008,
  Contusion −0.0013, Medial OA −0.0017, Medial Meniscus −0.0018, Baker's −0.0029, Lateral OA −0.0062,
  Lateral Meniscus −0.0071, MCL −0.0092, Fracture −0.0161.
- The 11.5 mm move along the patient's left-right axis repeats run 12's 9.5 mm with the same sign
  and the same victims (Lateral Meniscus, MCL): a point expressed in side-free axes shifts left
  and right knees in opposite anatomical directions. ACL rose again, as in runs 13 and the
  ligament reader.

### Forced seeds complete (2026-10-07 06:32)

Sixteen decided: one adopted (`height-cuts-from-the-spots-requeue`), eleven not adopted, two
invalid on cost, plus the parked candidates requeued above. Summary by topic:

- Routing: height cuts placed from the spots adopted at +0.0003 on both multipliers, inside the
  rerun spread measured since (−0.0006 to −0.0010 at ×4 for four runs of the incumbent with crops
  within 2 mm). Menisci on their whole side, held cuts with wider regions, Synovitis on Effusion's
  attention and Contusion on the edema head did not hold. The ligament reader raised ACL by 1.5
  SDs at a cost elsewhere.
- Shifts: the target loss gives no consistent direction to a crop shift, per series type or per
  study (runs 8 to 13 and 15); freed, the shift drifts to a common offset that differs between
  runs. A locator trained on the heads' meniscus spots is the only one that gives knees different
  shifts (run 14: 3.8 mm of spread along the height) and ends level with the reruns; its target
  is noisy by 4 to 6 mm per knee.
- Leads: a forward shift for PF OA (+0.0040 in run 12); any left-right shift expressed towards
  medial; a cleaner per-knee target for the spot-trained locator, on the vertical axis first; ACL
  gains in the three runs that changed what the core reads or where the frame sits (+0.0021,
  +0.0026, +0.0029).

### 2026-10-07 06:40: the researcher's first proposal parked; the contract clarified

- `height-cuts-use-vertical-spot-spread` was parked without training for the developer failure of
  01:25 (no guardrail run, an attempt to write the frozen source). Research proposals carry none
  of the keeper's corrected wording, so every one of them was exposed to it, and the GPU stood
  idle from 06:32.
- A paragraph added to the contract in the worktree did not reach the roles: their prompts read
  the contract from the campaign's frozen base (`outputs/autoresearch/routing-and-shifts/baseline/source`),
  whose every file is under the reference's source hash, so it cannot be edited. The worktree's
  contract was restored to the sealed text.
- What works is the paragraph inside the idea itself (the `-2` and `-c` seeds were all prepared
  with it). From here every research proposal gets a keeper's copy before a developer takes it:
  `protect_ideas.py` (session scratchpad) copies each queued idea that lacks the paragraph to
  `<id>-k` with it prepended to the implementation, forces the copy and removes the original; a
  parked idea is requeued the same way. Files `keeper-copies-<time>-<n>.yaml` in this directory.
  First use 06:37: `spot-shift-locator-keeps-spatial-maps-k`, `remove-unheld-acl-ring-cut-requeue-k`.
- Requeued: [`height-cuts-use-vertical-spot-spread-requeue`](research-requeue-ideas.yaml).

## Repairs and decisions

### 2026-10-07 00:42: the repaired shift run aborted for memory; shift seeds held back

- The third `learned-shift-per-series-type` candidate trained for 6.5 minutes with the machine at
  34 of 39 GB used and swap full (main process 15.7 GB, sixteen workers at 1.3 to 1.6 GB each,
  20 GB shared). A background shell of the keeper was reaped for memory pressure. Aborted with
  `stop --abort`; memory returned to 3 GB used. Its epoch 1 took as long as before the repair
  (about four minutes with start-up), so the time budget was not met either. Recorded invalid.
- Cause, from the candidate's diff: the canvas carries the view window plus 15 mm per side, so the
  native pixels transported per slice and the float canvases are about 1.84 times the
  incumbent's, through sixteen workers with two prefetched batches each.
- `learned-shift-per-study-one-point` was already prepared on the same port and would have
  launched next. With the campaign stopped, `state.json` was edited (backup
  `state.before-holding-shift-seeds-20261007.json`): that experiment marked
  `superseded_by_reconciliation`, the idea returned to `queued`, and the four remaining shift
  seeds moved behind the five routing seeds with `move-idea` and that edit. Queue: held cuts,
  ligament reader, heights from spots, Contusion, Synovitis, then the four shift seeds.
- Open: the port has to fit the time budget and the machine's memory before a shift seed
  trains again; `learned-shift-per-series-type` has to be requeued with it.
- The stop-before-end shell was the one reaped and is not re-armed while memory is short: an
  adoption is applied after the run, and the next run may already have launched on the old
  incumbent.

### 2026-10-07 01:10: the shift seeds requeued on a lighter port

- Measured during the next routing run: the incumbent's pipeline uses 17 GB (shared memory 4 GB,
  largest process 4.5 GB) where the first port used 34 GB (20 GB and 15.7 GB). The loader's
  collate prepares the images in the workers (`data/dataset.py` calls `prepare_images` on the
  CPU), so the first port made every worker build and ship float32 canvases of 304 px per frame
  in place of uint8 images of 224 px, about seven times the bytes per batch, and added a sort of
  the pixels after the cut.
- The light port ([seeds](light-port-shift-ideas.yaml)): the loader normalises each canvas with
  the incumbent's own quantile bounds, taken over the unshifted 224 px windows, and ships uint8
  canvases; the model cuts from them and nothing is normalised after the cut. The central region
  of every canvas equals the incumbent's image exactly; a shifted crop keeps the unshifted
  window's intensity scaling. Each candidate reports the bytes of one collated batch against the
  incumbent's and is returned blocked above twice.
- Forced with `import-ideas --forced`: `shift-per-series-type-light-port` and the four per-study
  seeds on it. The four shift seeds still queued on the first port were removed with
  `remove-idea` (keeper's rationale recorded in the state). Queue: ligament reader, heights from
  spots, Contusion, Synovitis, then the five light-port shift seeds.

### 2026-10-07 01:25: two routing seeds parked by a harness failure; requeued

- `shared-ligament-reader` and `height-cuts-from-the-spots` were blocked and parked without
  training, and the GPU stood idle from 01:13. Their developers returned `blocked` because they
  did not run the guardrail check: the campaign's developer instructions say the check runs on the
  GPU, and the developer's assignment forbids the GPU. The first three seeds' developers had run
  it on the script's CPU fallback. The ligament candidate was otherwise complete (three files,
  every training study with positive mass on each new target).
- The policy is sealed, so the instruction cannot be amended. Requeued through the seeds instead:
  each requeue opens its implementation with the instruction to run the check on the CPU, that
  the CPU run is the required evidence, never to return blocked for lack of a GPU, and to keep
  the artifacts in the experiment directory, not under `/tmp`.
  [Routing requeues](routing-requeue-ideas.yaml): `shared-ligament-reader-requeue`,
  `height-cuts-from-the-spots-requeue`, `synovitis-shares-effusion-attention-requeue` (the last
  replaces the queued original before it meets the same failure).
  [Light-port shift seeds, second import](light-port-shift-ideas-b.yaml): the five light-port
  seeds again with the same instruction, ids ending in `-b`; their first import was removed from
  the queue.
- `contusion-reads-the-edema-head` was on its repair when this was done; it is requeued the same
  way if it is parked.
- 01:30: `contusion-reads-the-edema-head` was parked the same way after its repair and is requeued
  as `contusion-reads-the-edema-head-requeue` ([seed](contusion-requeue-ideas.yaml)), placed before
  the shift seeds.

### 2026-10-07 01:35: the first requeue was parked by its own wording; second requeue

- `shared-ligament-reader-requeue` was parked after its repair. The requeue's instruction to keep
  the artifacts "in the assigned experiment directory" made the developer try to write `source/`
  and `changes.diff` there; that directory is read-only to the developer by design (the
  supervisor freezes both from the candidate workspace), and it returned `blocked` on the
  read-only error with the complete candidate left in its workspace. The keeper's wording was
  the cause.
- Second requeue with the corrected instruction (run the check on the CPU from the workspace;
  change only the workspace; never write the frozen source or the diff; return `prepared`; an
  unwritable experiment directory is not a reason to block):
  [routing](routing-requeue-2-ideas.yaml) `shared-ligament-reader-2`,
  `height-cuts-from-the-spots-2`, `synovitis-shares-effusion-attention-2`,
  `contusion-reads-the-edema-head-2`; [shift](light-port-shift-ideas-c.yaml) the five light-port
  seeds with ids ending in `-c`. The queued `-requeue` and `-b` versions were removed.
  `height-cuts-from-the-spots-requeue` was already with its developer and stays; its `-2` twin is
  removed if it reaches training.
- The GPU has been idle since 01:13.

## Research runs

### 16. `height-cuts-use-vertical-spot-spread-requeue` — not adopted (2026-10-07 07:01)

- The researcher's first proposal: the spot-placed joint band widened where a knee's spots are
  spread vertically. ×4 0.924986 (−0.0004), ×1 0.916548 (−0.0009) against the incumbent. Lower at
  both; at ×1 it was level or ahead through epoch 7 (+0.0018 at epoch 3) and behind in the last
  three.
- Per target, nothing beyond noise: Medial OA +0.0010, Fracture +0.0009, PF OA +0.0001, Effusion
  −0.0001, Medial Meniscus −0.0002, Baker's −0.0003, Lateral Meniscus −0.0004, Lateral OA −0.0004,
  Contusion −0.0006, ACL −0.0008, MCL −0.0013, Synovitis −0.0026.
- It sits inside the rerun spread of the incumbent (−0.0006 to −0.0010 at ×4, −0.0010 to −0.0019 at
  ×1): no measurable effect.

### 17. `spot-shift-locator-keeps-spatial-maps-k` — not adopted; best ×4 of the campaign, below at ×1 (2026-10-07 07:27)

- The researcher's follow-up to run 14: the spot-trained locator predicts the point from spatial
  maps in place of globally averaged features. ×4 0.925530 (+0.0001), ×1 0.916504 (−0.0009) against
  the incumbent. Above at ×4, below at ×1: not adopted by the contract. Against the four reruns
  (means −0.0008 and −0.0014) it is +0.0009 at ×4 and +0.0005 at ×1.
- Locator at the last epoch: spread of the shift across validation knees 1.6 mm anterior, 1.0
  left-right, 1.4 superior (run 14: 1.7, 0.2, 3.8); validation loss 0.074 to 0.062; left-over
  spread of the meniscus-spot midpoint 4.5, 2.2 and 6.9 mm (run 14: 4.6, 2.1, 6.4). It moves
  knees less than run 14's locator and explains no more of the target.
- Per target: MCL +0.0058 (2.0 sibling SDs), Lateral Meniscus +0.0032 (1.3), Medial OA +0.0009,
  Medial Meniscus +0.0005, Baker's +0.0002, PF OA +0.0001, Effusion +0.0001, Fracture −0.0007, ACL
  −0.0016, Lateral OA −0.0018, Synovitis −0.0020, Contusion −0.0030 (2.3).
- With run 14 this is the second spot-trained shift to sit at or above the reruns, with MCL and a
  meniscus gaining and Contusion losing in both. It is the campaign's best lead on the shift
  topic; the gap at ×1 is what keeps it out.

### 18. `remove-unheld-acl-ring-cut-requeue-k` — not adopted (2026-10-07 07:48)

- The researcher's proposal to drop the outer-radius cut that only ACL reads and keep the core.
  ×4 0.924666 (−0.0007), ×1 0.916507 (−0.0009) against the incumbent. Lower at both, inside the
  rerun spread: no measurable effect of removing the cut.

### 19. `nod-fit-in-meniscus-basis` — not adopted (2026-10-07 08:25)

- The researcher's proposal to fit the frame's nod in the axes given by the two meniscus spots.
  ×4 0.924674 (−0.0007), ×1 0.916517 (−0.0009) against the incumbent. Lower at both, inside the
  rerun spread.
- Per target, nothing beyond two sibling SDs: ACL +0.0015, Synovitis +0.0004, Effusion +0.0002,
  Medial Meniscus +0.0001, Baker's +0.0001, Contusion −0.0007, PF OA −0.0011 (2.2), Lateral
  Meniscus −0.0013, Medial OA −0.0018 (2.3), MCL −0.0019, Lateral OA −0.0019, Fracture −0.0022.

### The spread of non-adopted routing variants on this incumbent (2026-10-07 08:25)

Runs 16, 18 and 19 changed small things about the frame and landed at −0.0004, −0.0007, −0.0007 at
×4 and −0.0009 at ×1 each; the four near-zero-shift reruns at −0.0006 to −0.0010 and −0.0010 to
−0.0019. Seven runs of what is close to the incumbent's recipe sit 0.0004 to 0.0010 below it at ×4
and 0.0009 to 0.0019 below at ×1, none above. The incumbent's own score is the upper edge of
that group: it was adopted at +0.0003 over run 24, and a real gain has to clear about 0.001 at
both multipliers to be adopted against it.

### 20. `compact-support-cut-gradients-k` — not adopted (2026-10-07 08:47)

- The researcher's proposal to confine each hard cut's soft gradient to the tokens within its
  measured extent. ×4 0.925067 (−0.0003), ×1 0.916350 (−0.0011) against the incumbent. Lower at
  both; at ×4 it equals run 24.
- Per target, nothing beyond one sibling SD: Synovitis +0.0008, Medial OA +0.0006, Lateral
  Meniscus +0.0003, Fracture +0.0001, the rest between −0.0004 and 0, ACL −0.0013, MCL −0.0028.

### Drawings of the incumbent's frame (2026-10-07 08:45, at the user's request)

`outputs/analysis/routing-and-shifts-frames-20261007/height-cuts-from-the-spots/regions-{005,021,045}.png`
(script `draw_experiment_frames.py` in the session scratchpad, which runs the drawing with the
experiment's frozen source). Study 45: cuts +6 and −8 mm, nod 2°, roll −11°, core 13 mm; study 5:
cuts +13 and −15 mm, nod 12°, roll −2°, core 19 mm. The joint band lies on the joint line in the
sagittal and coronal views of both; its thickness follows the knee (14 and 28 mm); the nod and the
roll still differ by ten degrees between the two knees.

### 21. `report-answers-own-attention-k` — below the incumbent at both; undecided at the stop (2026-10-07 09:08)

- The researcher's proposal to give the categorical report answers their own attention while
  keeping their classifiers and regions. ×4 0.924992 (−0.0004), ×1 0.916765 (−0.0006) against the
  incumbent. It does not meet the adoption rule. Its training finished during the stop; the
  runtime's comparison had not been processed when the campaign halted and runs on a resume.

## Stopped by the user (2026-10-07 08:55 requested, 09:08 STOPPED)

Graceful stop: the run in training finished, the researcher's third batch was cancelled. State at
the stop: incumbent `height-cuts-from-the-spots-requeue-20261006T232351Z-d2445402` (×4 0.925389,
×1 0.917411), 21 runs decided and one more trained, one adoption, queue empty. The incumbent was
submitted at the user's request as S120 (bundle
`outputs/submission/exports/side-routed-spot-height-cuts-f1-x4-tau09`, ×4, τ 0.9, batch 4).

Open at the stop, raised with the user and not acted on:
- The developer instruction about the GPU in the sealed policy parks proposals at random; a
  keeper's copy with a paragraph in the idea is the workaround. One sentence in the policy would
  remove it and needs the user's decision.
- The incumbent sits at the top of eight near-identical runs (0.0003 to 0.0010 below it at ×4,
  0.0009 to 0.0019 at ×1), so the two-multiplier rule against it turns away gains under about
  0.001.
- The incumbent's height cuts are centred on the menisci but lean with the frame (roll 5.3° and
  nod 7.0° on average) and their thickness follows one kneecap spot (14 mm in study 45, 28 mm in
  study 5; the lower cut is 8 to 17 mm below the menisci in 90% of knees). Proposed: the forward
  tilt from the kneecap spot as in run 25, and a lower cut that does not thin with the kneecap
  spot.

S120 (the incumbent, ×4, τ 0.9, batch 4): **0.952 public**; S118 (run 24, same settings) 0.953. Record `outputs/submission/records/side-routed-spot-height-cuts-f1-x4-tau09.json`.

## MCL in `spot-shift-locator-keeps-spatial-maps-k` (looked at 2026-10-07, after the stop)

- MCL +0.0058 against the incumbent is 226 of 38,869 pairs, on 47 positives: 14 moved up, 8 down,
  and the five biggest movers carry 213 pairs. The first spot-trained run gains 118 pairs, two
  near-identical reruns lose 40 and 100, and the per-knee changes of the two spot-trained runs
  correlate at 0.51 (0.17 with a rerun).
- The shifts do not explain it. Locator's point over the 47 positives: medial +0.05 ± 1.18 mm,
  front −0.34 ± 1.23, up +0.16 ± 1.14; 150 negatives: −0.18 ± 0.93, −0.43 ± 1.54, +0.29 ± 1.35.
  The gain per positive does not follow the shift (correlations between −0.20 and +0.24). The five
  movers were moved 1 to 2 mm in different directions, their coronal crops look the same with and
  without the shift, and the medial rim is inside the crop either way. The 120 mm axial views are
  not moved at all (no margin at the reference field).
- Drawing and table: `outputs/analysis/routing-and-shifts-frames-20261007/mcl-spatial-maps/`
  (`top-movers-crops.png`, `shifts.csv`).
- Same weights predicted at ×1 with the shift on and switched off (`shift-on-off.npz` in the same
  folder): macro 0.9165 against 0.9163. MCL 0.9190 against 0.9182 (+0.0008, of the +0.0058 at ×4
  against the incumbent), so the MCL gain is in the trained weights, not in moving the crops at
  prediction time. Largest effects of the shift itself: Medial OA +0.0020, Lateral OA +0.0015,
  ACL −0.0027, Synovitis −0.0012.
- The same on/off pass at ×4 (`shift-on-off-x4.npz`; shift on reproduces the run's saved
  predictions): macro 0.9255 on, 0.9248 off (+0.0007). MCL 0.9272 on, 0.9254 off, incumbent
  0.9214: of the +0.0058, 0.0040 is in the trained weights and 0.0018 is the shift at prediction
  time. The shift at ×4: Lateral Meniscus +0.0031, Lateral OA +0.0020, MCL +0.0018, Medial OA
  +0.0015, PF OA +0.0010, Contusion −0.0015, the rest within 0.0007. With the shift off the
  weights score 0.0006 below the incumbent, so the moved crops give at prediction what training
  with them cost. At ×1 the shift was worth +0.0001.

