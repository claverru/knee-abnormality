# Fixes-and-removals run notes

One entry per decided run: the result against the incumbent at its turn, the per-label deltas,
what the mechanism's own logs show, the adoption under the [contract](contract.md#adoption) and
the lesson. Keeper repairs are listed at the end. All scores are fold 1 of
`all_5fold_scanner_v2`, seed 1110484336, and are not comparable with fold-0 records.

## 1. `reflected-detail-view-positions` (adopted, forced)

- **Result:** ×4 0.923401 against 0.923102, **+0.000299** (80% [−0.000216, +0.000796], 95%
  [−0.000498, +0.001126]); ×1 0.914267 against 0.914860, −0.000593; 1,177 s. The shared rule
  rejects it; adopted by the contract's forced adoption and now the incumbent.
- **Curve at ×1:** within ±0.0005 of the reference at every epoch after the third (+0.0011 at
  epoch 3, −0.0005 at epochs 5 and 9).
- **Per label (×4):** MCL +0.0018, Fracture +0.0016, Baker's +0.0016, Lateral OA +0.0014; Lateral
  Meniscus −0.0021, Effusion −0.0007; the other six within ±0.0005. No label's 95% interval
  excludes zero.
- **Implementation:** as specified; the wide-frame x of mirrored non-sagittal detail tokens is
  reversed in `encode_batch`, and `reflect_batch` maps `view_to_wide` to
  `reversal @ V @ reversal`. The activation count of reflected detail tokens is positive.
- **Lesson:** correcting the positions of coronal detail tokens in half the training studies
  moves the macro by less than the interval's half-width, so the model draws little from the
  horizontal position of coronal detail tokens. The +0.0003 at ×4 and −0.0006 at ×1 bracket the
  size of a change that does nothing on this fold: read later deltas of that size as ties.

## 3. `centre-without-side` (adopted by the shared rule)

- **Result:** ×4 0.924815 against 0.923401, **+0.001414** (80% [+0.000269, +0.002629], 95%
  [−0.000400, +0.003238]); ×1 0.916467 against 0.914267, +0.002200. The 80% interval is above
  zero, so the runtime accepted it itself; it is the incumbent.
- **Curve at ×1:** ahead of the incumbent from epoch 6 on (+0.0036, +0.0021, +0.0020, +0.0018),
  mixed before (+0.0013, +0.0010, −0.0021, +0.0008, −0.0012).
- **Per label (×4):** Fracture +0.0073, MCL +0.0060, Contusion +0.0034, Synovitis +0.0025, ACL
  +0.0019, Medial Meniscus and Effusion +0.0013; Lateral OA −0.0051, Lateral Meniscus −0.0019;
  the rest within ±0.0003. No label's 95% interval excludes zero.
- **Implementation:** one line in `data/dataset.py`: validity no longer requires the side sign.
- **Lesson:** the gain is larger than the affected share (about 3.7% of studies) suggests for a
  crop change alone. Those studies also regain contrast mixing in training, and the change in
  which studies mix shifts the augmentation draws of every later batch, so part of the delta is
  a different random stream. The two labels that fall are the lateral ones, on which
  unknown-side studies still have no mask and no half-knee loss; worth reading again after the
  meniscus-mask removal.

## 4. `centre-outside-slab-rule` (not adopted)

- **Result:** ×4 0.924278 against 0.924815, **−0.000537** (80% [−0.001584, +0.000536], 95%
  [−0.002191, +0.001124]). Observed delta below zero: not adopted under the contract.
- **Curve at ×1:** within ±0.0011 of the incumbent at every epoch, ending +0.0001 at epoch 9.
- **Per label (×4):** Lateral Meniscus +0.0060, Lateral OA +0.0036, Synovitis +0.0036, ACL
  +0.0011; Fracture −0.0091 (95% [−0.0188, −0.0013]), MCL −0.0073, Contusion −0.0031.
- **Implementation:** the slab test applies for every fit method and the diagnostic is named
  `centre_outside_slab`; 20 series in 20 training studies change sampling.
- **Lesson:** a change that touches under 0.1% of series moved single labels by 0.006 to 0.009
  and the macro by 0.0005. That is the per-label noise of one run on this fold. The pattern is
  the mirror of seed 3's (Fracture and MCL down here, up there; the lateral labels up here,
  down there), so most of seed 3's per-label picture, and some part of its +0.0014, is the
  random stream and not the crop change. Read per-label deltas below about 0.01 as noise in
  this campaign. The fix is correct by reading and is listed for the user to take without a
  run.

## 2. `contrast-mixing-aligned-donor-requeue` (not adopted)

- **Result:** ×4 0.923971 against 0.924815, **−0.000844** (80% [−0.001465, −0.000209], 95%
  [−0.001813, +0.000143]). The 80% interval is below zero. The random draws are identical to
  the incumbent's (same mixing counts and lambdas per epoch), so the alignment is the only
  difference.
- **Curve at ×1:** ahead for three epochs (+0.0005, +0.0006, +0.0011), then behind from epoch 4
  (−0.0010, −0.0004, −0.0003, +0.0001, −0.0005, −0.0006).
- **Per label (×4):** ten of twelve down; Fracture −0.0028, MCL −0.0023, Synovitis −0.0016,
  Medial OA −0.0014; Lateral Meniscus +0.0015. All inside the per-label noise of this fold.
- **Mechanism logs:** mean in-plane shift between the two grids 3.9 mm; 88% of target pixels
  receive donor signal; mean pixel change 0.065 against 0.071 for the pixel-index blend.
- **Independent check of the alignment (2026-10-05, CPU, frozen source):** correct. Landmark
  error after alignment 0.01 px on synthetic grids of 84 and 93 mm with offsets and rotations
  (0.46 px with a 3° out-of-plane tilt), reflected cases identical. On 300 real target/donor
  pairs of 12 studies the aligned donor is closer to the target in every plane: gradient
  correlation 0.129 → 0.185 coronal 84←93, 0.121 → 0.186 coronal 93←84, 0.176 → 0.226 and 0.179
  → 0.226 sagittal, 0.212 → 0.247 axial; aligned wins in 62% to 80% of pairs. `patient_maps[n, 1]`
  maps the prepared 224 px crop to patient mm (`preparation.py:418-426`), the grid the images
  are on.
- **What the check adds:**
  - **The donor is usually another slice.** The nearest sampled slice of the other series is a
    median 4.0 mm away along the normal in coronal, 4.7 mm in sagittal and 6.1 mm in axial; half
    the pairs are more than 4 mm apart. Within 1.5 mm the alignment lifts the gradient
    correlation from 0.171 to 0.289 (coronal); beyond 8 mm it does nothing (0.087 → 0.083).
    Contrast mixing never blended the same anatomy in most pairs, before or after the fix.
  - **An uncovered frame.** With the 93 mm series as target, about 20% of the frame gets no
    donor and stays pure target, a rectangular contrast step the pixel-index blend did not
    have.
  - Two coronal series tilted 33–38° to each other in 2 of 12 studies share only a line; slice
    matching reads the patient axis at each crop's centre, which is up to 9.8 mm off the true
    perpendicular distance there.
- **Lesson:** the alignment is right and the score is lower, on one run. Two readings stay
  open: the coverage step hurts, or the mixing helps the ResNet-18 as a perturbation and a
  cleaner blend perturbs less. `contrast-mixing-aligned-feathered` separates them.

## 5. `feature-statistics-positive-std` (not adopted)

- **Result:** ×4 0.924641 against 0.924815, **−0.000174** (80% [−0.000623, +0.000251], 95%
  [−0.000889, +0.000474]). Observed delta below zero: not adopted.
- **Curve at ×1:** behind at eight of nine epochs by 0.0002 to 0.0015.
- **Per label (×4):** all within ±0.0012; Lateral OA +0.0012, PF OA +0.0008; Contusion and MCL
  −0.0008.
- **Mechanism log:** the clamp changed 0.41% of the (acquisition, channel) draws in the first
  epoch (5,245 draws over 9,335 augmented acquisitions), so the defect does fire: about one draw
  in 240 inverted a feature channel.
- **Lesson:** the smallest per-label spread of any run so far, as expected of a change to 0.4%
  of one augmentation's draws. Unlike the slab rule, which changed which slices are read, this
  one left the random stream almost intact, and the per-label deltas stayed under 0.0012 against
  0.006 to 0.009 there: the large per-label swings of runs 3 and 4 come from a changed data
  stream, not from the fix itself. Correct by reading; in `unadopted-fixes-together`.

## 6. `computed-centre-3d-position` (rejected, shared rule)

- **Result:** ×4 0.924037 against 0.924815, **−0.000778** (80% [−0.002063, +0.000489], 95%
  [−0.002752, +0.001254]).
- **Curve at ×1:** far ahead early (+0.0077, +0.0026, +0.0028 at epochs 1 to 3), level by epoch
  4, then behind by 0.0002 to 0.0016.
- **Per label (×4):** Lateral OA +0.0091 (95% [+0.0027, +0.0161]) and Lateral Meniscus +0.0060
  (95% [+0.0013, +0.0114]), the two lateral compartment labels and the only ones whose
  intervals exclude zero; MCL −0.0098, ACL −0.0044, Synovitis −0.0036, Baker's −0.0032. The soft
  macro rises (+0.0006) where the hard one falls.
- **Mechanism logs:** full position 96.3% of tokens, unknown side 3.7% (real y and z, learned
  x), unknown position 0%. OA detector mean absolute correction at the last epoch 0.031 medial,
  0.013 lateral and 0.014 patellofemoral against 0.277, 0.291 and 0.043 in the incumbent: the
  detector is nearly silent beside the 3D position.
- **Implementation:** as specified, ported from the 2026-10-02 run; `position_1d` and
  `position_2d` gone; the position follows the anatomy under reflection, so no mirroring step
  exists in training or TTA. The centre's y and z travel together under the `knee_center_y_mm`
  batch key, to be renamed if this encoding is ever merged.
- **Lesson (revised after the audit below):** a tie, not a loss. The first reading here, that
  the position costs the ligament labels and overfits, is withdrawn.

### Audit of every 3D run (2026-10-05, logs, predictions and weights of 11 runs)

- **The incumbent is a high draw.** The four runs built on it that change almost nothing
  (slab rule 0.924278, positive std 0.924641, aligned mixing 0.923971, the three together
  0.923997) average 0.92422; the incumbent is 0.924815, about 0.0006 above them. The 3D run's
  0.924037 sits among them. Every delta in this campaign since seed 3 is measured against that
  draw.
- **No overfitting.** The 3D run's validation soft BCE is lower (−0.0007 against +0.0002 ±
  0.0002 for the sibling runs) and its soft macro AUC higher (+0.0006; positive in 9 of 9 runs
  on the current recipe). Its predictions correlate with the incumbent's as a no-op sibling's
  do (logit correlation 0.9905 against 0.9904 and 0.9906).
- **The early lead is real and closes by epoch 3 in every run.** Epoch-0 ×1 delta today +0.0077
  (95% [+0.0044, +0.0108]; non-3D siblings +0.0006 ± 0.0007); ahead at epoch 0 in 7 of 9
  current-recipe runs. It goes with removing the 1D and 2D embeddings, not with adding the 3D
  one: seven of seven learnable-views replace runs start with a training loss 0.0056 to 0.0069
  below the reference, and the one run that kept 1D and 2D starts level (+0.0006). The fixed 1D
  sinusoid has norm 11.3 against 2.5 to 4.0 for the 3D term (logged in one run).
- **Position is nearly redundant with the image.** The position network barely trains (first
  layer +10% in norm, second +5%); the incumbent's 2D network stays at its initial norms; the
  compartment margin AUC is unchanged (meniscus 0.9666 to 0.9666, OA 0.9325 to 0.9339).
- **Per label, against the sibling runs:** Lateral OA +0.0073 at ×4 and +0.0103 at ×1 is the
  only robust gain; Synovitis −0.0039 repeats across campaigns (6 of 7 negative in
  learnable-views); MCL and ACL are not lower at ×1 (−0.0024 and −0.0005) and MCL is +0.0012 in
  learnable-views.
- **No subgroup carries the difference:** not the centre-fit RMS, the vendor, the side source or
  the number of series. Studies with an untrusted series are −0.0045 (95% [−0.0116, +0.0045],
  n 70).
- **The unknown-position vector never trains** (norm exactly 0.0; share 0.0 in all nine runs),
  so a test study without a centre would get an input the model never saw.
- **Across variants:** the seven replace runs average zero against their references; the two
  positives are the 3D position beside 1D and 2D (+0.0005) and with a first-order OA detector
  (+0.0010). Adding beats replacing, weakly.
- **Consequences:** `position-3d-beside-1d-2d` (the best-supported variant) and
  `position-embeddings-off` (no position at all, which would explain the early lead and could
  remove the two embeddings) are forced ahead of the three retakes. Of those, the audit finds
  the study dropout misdirected as a remedy (its real effect is training the unknown-position
  vector), the attention-only variant not indicated, and the coarse wavelengths a likely tie
  (the 8 and 16 mm features vary inside one 12 to 17 mm token cell and the network did not
  grow them).

## 7. `unadopted-fixes-together` (not adopted)

- **Result:** ×4 0.923997 against 0.924815, **−0.000817** (80% [−0.002048, +0.000437], 95%
  [−0.002670, +0.001046]). Observed delta below zero: not adopted. The three single deltas sum
  to −0.0016; the bundle lands on the aligned mixing's own −0.000844.
- **Curve at ×1:** within ±0.0012 of the incumbent, −0.0006 at epoch 9.
- **Per label (×4):** Fracture −0.0124 (95% [−0.0225, −0.0041]), MCL −0.0063, Contusion −0.0042;
  Lateral Meniscus +0.0076 (95% [+0.0004, +0.0159]), Synovitis +0.0032, Lateral OA +0.0022.
- **The per-label pattern is the slab-rule run's again** (Fracture −0.0091, MCL −0.0073,
  Contusion −0.0031, Lateral Meniscus +0.0060 there). Checked on the saved predictions whether
  the slab rule hurts the studies it touches: only 2 of the 874 validation studies have an
  affected series, and the largest prediction changes are in other studies. Mean absolute
  change of a study's Fracture probability against the incumbent: 0.016 in the slab-rule run and
  0.015 in this one, against 0.006 for the std clamp and 0.008 for the aligned mixing. So the
  slab rule changes the weights broadly, not a few validation studies: altering the sampling of
  20 training series shifts every later batch, and both runs share that one shifted stream.
  The repeated pattern is one noise draw seen twice, not two measurements.
- **Lesson:** on this fold a change of data stream alone moves a study's probability by about
  0.015 on average and a 60-positive label by 0.01. Two runs that share a stream change are not
  independent evidence. The slab rule and the clamp go forward as
  `slab-rule-and-positive-std`, adopted by user decision; the aligned mixing waits for the
  feathered run.

## 8. `contrast-mixing-aligned-feathered` (not adopted)

- **Result:** ×4 0.923754 against 0.924815, **−0.001061** (80% [−0.001702, −0.000415]); −0.0005
  against the 0.92422 mean of the do-nothing runs. ×1 0.915106 against 0.916467, −0.0014.
- **Curve at ×1:** ahead for three epochs (+0.0015, +0.0007, +0.0007), then behind, as the
  plain aligned run was.
- **Lesson:** fading the blend at the edge of the donor's coverage does not recover the aligned
  blend's deficit: aligned 0.923971, aligned and feathered 0.923754, both below the do-nothing
  mean. The coverage step is not the cause. What remains is that the pixel-index blend, which
  mixes anatomy a few millimetres off in plane and usually another slice along the normal,
  serves the ResNet-18 at least as well as a registered blend; the mixing acts as a
  perturbation, not as a contrast interpolation. Two runs, both small. The alignment stays out;
  the code is correct and kept in the frozen sources.

## 9. `oa-detector-off` (adopted by user decision, to revisit)

- **Result:** ×4 0.924347 against 0.924815, **−0.000468** (80% [−0.001017, +0.000110], 95%
  [−0.001321, +0.000412]); +0.0001 against the 0.92422 mean of the do-nothing runs. Observed
  delta against the incumbent below zero: not adopted under the contract.
- **Curve at ×1:** within ±0.0018 of the incumbent, −0.0007 at epoch 9.
- **Per label (×4):** the three OA labels, which the detector corrects, do not fall: Lateral OA
  +0.0029, PF OA +0.0004, Medial OA −0.0009. MCL −0.0048 (95% [−0.0087, −0.0014]), Fracture
  −0.0017, Contusion −0.0016; the rest within ±0.0008.
- **Implementation:** the correction is not added to the OA logits; the detector's parameters
  stay unused. The random stream is unchanged, so the per-label deltas are small, as for the
  std clamp.
- **Lesson:** the detector's own labels are level or up without it, so on this fold it is not
  earning its place; the macro deficit comes from labels it does not touch and equals the gap
  between the incumbent and its do-nothing siblings. The first removal the rule rejects where a
  comparison with the recipe's typical score would call it free. Worth one more run once the
  incumbent has changed (`slab-rule-and-positive-std`), if the user wants removals judged at
  that bar.

## 10. `position-3d-without-oa-detector` (rejected, shared rule)

- **Result:** ×4 0.924176 against 0.924347 (the incumbent without the detector), **−0.000171**
  (80% [−0.001328, +0.000927]); ×1 0.916171 against 0.915646, +0.000525. Since the detector's
  removal was adopted before this run, it is the plain 3D-replace variant on the new incumbent.
- **Curve at ×1:** +0.0076, +0.0020, +0.0019, +0.0017, −0.0004, −0.0005, +0.0007, +0.0011, +0.0007,
  +0.0005: the same opening as run 6, and ahead at the last four epochs, where run 6 was behind
  by 0.0002 to 0.0006.
- **Lesson:** a seventh tie at ×4 for the replace variant, at the recipe's typical score this
  time, with a small lead at ×1. Consistent with the audit: the encoding reaches the same model
  sooner. The two seeds that separate adding from removing come next
  (`position-3d-beside-1d-2d`, `position-embeddings-off`).

## 11. `slab-rule-and-positive-std-requeue` (adopted by user decision)

- **Result:** ×4 0.924049 against 0.924347, **−0.000297** (80% [−0.001535, +0.000897]); ×1
  0.916134 against 0.915646, +0.000488. Adopted whatever its delta by user decision; the
  incumbent since.
- **Lesson:** a tie, as expected of 20 series and 0.4% of one augmentation's draws. The two
  fixes are in the recipe; the aligned contrast mixing is not.
- **Order of events:** the adoption could only be requested once the run's decision was
  recorded, and the supervisor records a decision and launches the next training in one pass, so
  `position-3d-beside-1d-2d` trained on the incumbent without these two fixes. The adoption was
  applied when that run ended.

## 12. `position-3d-beside-1d-2d` (rejected, shared rule)

- **Result:** ×4 0.924598 against 0.924347 (the incumbent without the detector and without the
  two small fixes), **+0.000252** (80% [−0.000457, +0.000937]); the 80% interval reaches below
  zero, so the shared rule rejects it.
- **Curve at ×1:** +0.0010, −0.0002, +0.0005, +0.0021, +0.0012, +0.0006, +0.0005, +0.0007, +0.0007:
  no opening lead (the replace runs opened at +0.0077 and +0.0076), and ahead at every epoch
  from the third on.
- **Lesson:** with the 1D and 2D embeddings kept, the large early lead disappears, as in the
  one earlier run of this kind. The fast start of the replace runs goes with removing those
  embeddings, not with adding the 3D position. The 3D position beside them is the best of the
  three 3D variants run here (+0.0003 against −0.0008 and −0.0002) and the second positive run
  of this variant (+0.0005 in learnable-views), both inside one run's noise.

## 13. `position-embeddings-off` (not adopted)

- **Result:** ×4 0.923570 against 0.924049, **−0.000479** (80% [−0.001535, +0.000611], 95%
  [−0.002056, +0.001177]); soft macro level (+0.000003). Observed delta below zero: not adopted.
- **Curve at ×1:** +0.0047, +0.0027, +0.0013, +0.0014, −0.0004, +0.0002, −0.0005, −0.0016, −0.0006.
  Epoch-1 validation AUC 0.8880 against 0.8821 to 0.8832 for the standard recipe, 0.8831 for the
  3D position beside 1D and 2D, and 0.8898 and 0.8899 for the 3D position replacing them: the
  removal of the 1D and 2D embeddings alone gives about three quarters of the replace runs'
  opening lead, and the lead closes by epoch 5 here too.
- **Per label (×4):** the compartment labels that lost every source of side information do not
  fall: Medial OA +0.0001, Lateral OA +0.0017 (the OA detector is already off in this
  incumbent). Medial Meniscus −0.0011 and Lateral Meniscus −0.0023, which keep their mask;
  Synovitis −0.0035, Baker's −0.0028, ACL and MCL −0.0018; Contusion +0.0036, Effusion +0.0021.
  No 95% interval excludes zero.
- **Lesson:** the fast start of the 3D replace runs is mostly the removal of the 1D and 2D
  embeddings; the 3D position adds about 0.002 at epoch 1 and nothing measurable after. Without
  any position the recipe ends 0.0005 below the incumbent, inside one run's noise, so the
  embeddings are neither shown to matter nor shown to be removable. The expected cost on the OA
  compartment labels did not appear.
- **The campaign was stopped by the user after this run** (2026-10-05, about 19:25 CEST), with
  the seven removals and the three 3D retakes still queued, to plan the next steps.

## To revisit

- **The OA detector's removal (user, 2026-10-05).** `oa-detector-off` was adopted by user
  decision at −0.000468 against the incumbent (80% [−0.001017, +0.000110]) because it is level
  with the recipe's do-nothing runs (0.924347 against a mean of 0.92422) and its own three OA
  labels do not fall (Lateral OA +0.0029, PF OA +0.0004, Medial OA −0.0009). One run on fold 1.
  The detector was adopted on 2026-09-25 with PF OA +0.0102 and 0.944 against 0.940 public, so
  its removal should be checked again before it reaches a submission: a run with the detector
  restored on the campaign's final recipe, read on the three OA labels, and on the ResNet-50.
  **Public check (2026-10-05):** the pair scored the same, **0.950 with the detector (S108,
  `centre-without-side`) and 0.950 without it (S111, `oa-detector-off`)**, both fold 1 at ×4, τ
  0.9, no TTA. One rounding step is 0.001 and equivalent models have differed by about ±0.003
  on the public set, so this agrees with the local tie and does not prove the detector useless.
  S109 and its retry S110 (the same saved notebook of the detector-off bundle) ended in ERROR
  on Kaggle and were not counted against the daily limit; the same bundle uploaded again as a
  new dataset and notebook scored as S111, and a Kaggle run of it over all 4,407 training
  studies on two T4s in FP16 returned valid probabilities for every study. The failure was on
  Kaggle's side; its cause is unknown.
- **The adoption bar.** The incumbent of seeds 4 to 9 (`centre-without-side`, 0.924815) was
  about 0.0006 above runs that change almost nothing, so fixes and removals judged against it
  needed to beat a high draw. Decisions taken at that bar: the slab rule, the positive std, the
  aligned mixing and its feathered variant not adopted; the OA removal rejected by the rule and
  adopted by the user.

## Second chance and follow-ups (user and keeper seeds, 2026-10-05)

- **`position-3d-without-oa-detector`** (keeper): the missing cell of position encoding against
  detector, after the detector's corrections fell to a tenth beside the 3D position.

- **`unadopted-fixes-together`** (user): the slab rule, the aligned contrast mixing and the
  positive std in one run, after each missed alone by less than a no-op moves the macro on this
  fold. The slab rule also acts at inference, where the fold cannot judge its reach.
- **`contrast-mixing-aligned-feathered`** (keeper): the aligned blend with the weight faded out
  at the edge of the donor's coverage, to separate the coverage step from the alignment.

## Keeper repairs

- **2026-10-05, `slab-rule-and-positive-std` cancelled and requeued.** The user adopted
  `oa-detector-off` while that seed's first run, built on the previous incumbent, was three
  minutes into training. The campaign was stopped with `--abort`, the removal adopted (incumbent
  0.924347), and the seed forced again as `slab-rule-and-positive-std-requeue` on the new
  incumbent, still adopted by user decision whatever its delta.

- **2026-10-05, seed 2 cancelled and requeued.** `contrast-mixing-aligned-donor` had been
  prepared on the starting incumbent and began training the moment seed 1 finished, before
  seed 1's forced adoption could be applied (an adoption waits for a free GPU). Had it finished
  and been adopted, the incumbent would have lost seed 1's fix. The campaign was stopped with
  `--abort` 27 s into that training, seed 1 adopted, and the seed forced again as
  `contrast-mixing-aligned-donor-requeue` ([seed](contrast-mixing-requeue-ideas.yaml)) to be
  rebuilt on the new incumbent. The cancelled run is recorded as invalid.
