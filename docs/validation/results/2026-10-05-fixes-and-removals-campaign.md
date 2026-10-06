# 2026-10-05 fixes-and-removals campaign

**Four changes are on main: the detail-view position fix under reflection, the knee centre used
without a known side, the slab rule for every centre-fit method with a non-negative
feature-statistics std, and the removal of the OA detector.** The `fixes-and-removals`
autoresearch campaign repaired defects found by the
[code audits of 2026-10-05](2026-10-05-code-audit-measurements.md) and switched off mechanisms of
the ResNet-18 recipe one per run. It decided 13 runs (two more were cancelled and requeued) from
14:07 to about 19:25 CEST on 2026-10-05, when the user stopped it with seven removals and three
3D-position retakes still queued. One change met the shared rule (the centre without a side,
+0.0014); the other three were adopted by user decision at deltas between −0.0005 and +0.0003,
all inside one run's noise on this fold. The final incumbent scored ×4 0.924049 against 0.923102
for the reference, +0.0009.

Fold 1 of `all_5fold_scanner_v2` (874 studies), `rebuild_state_v2` labels, seed 1110484336, ×4
predictions, paired report-group bootstrap. Every result is one run at one seed, each against the
incumbent at its turn; none is comparable with a fold-0 record. Reference: `rn18-fold1`
(`outputs/training/20261005-113728-rn18-fold1`), the recipe in force trained from main: ×4
0.923102, ×1 0.914860, 1,179 s. Training-time range 0.5 to 1.2 of it.

- **Campaign record** (contract, prior work, seeds, decision summary):
  [fixes-and-removals campaign](../../../autoresearch/campaigns/fixes-and-removals/INDEX.md).
- **Per-run details** (per-label deltas, curves, mechanism logs, lessons, the audit of the 3D
  runs, the keeper's repairs):
  [run notes](../../../autoresearch/campaigns/fixes-and-removals/run-notes.md).
- **Code:** each run's source is under
  `outputs/autoresearch/fixes-and-removals/experiments/<run>/source/`. Main carries the
  incumbent's behaviour without the campaign's logging; see [adoption on main](#adoption-on-main).

## Runs

| # | Run | Change | ×4 | Against | Delta (80% interval) | Decision |
|---|---|---|---|---|---|---|
| 1 | `reflected-detail-view-positions` | Wide-frame x of mirrored detail tokens reversed in training; `view_to_wide` mirrored in reflection TTA | 0.923401 | 0.923102 | +0.000299 [−0.000216, +0.000796] | Adopted, forced (bug fix) |
| 3 | `centre-without-side` | Geometry validity no longer requires the side sign | 0.924815 | 0.923401 | +0.001414 [+0.000269, +0.002629] | Adopted by the rule |
| 4 | `centre-outside-slab-rule` | Slab test applied for every fit method | 0.924278 | 0.924815 | −0.000537 [−0.001584, +0.000536] | Not adopted alone |
| 2 | `contrast-mixing-aligned-donor-requeue` | Donor resampled onto the target grid | 0.923971 | 0.924815 | −0.000844 [−0.001465, −0.000209] | Not adopted |
| 5 | `feature-statistics-positive-std` | Resampled std clamped at zero | 0.924641 | 0.924815 | −0.000174 [−0.000623, +0.000251] | Not adopted alone |
| 6 | `computed-centre-3d-position` | 3D position in place of 1D and 2D | 0.924037 | 0.924815 | −0.000778 [−0.002063, +0.000489] | Rejected |
| 7 | `unadopted-fixes-together` | Slab rule, aligned mixing and clamp together | 0.923997 | 0.924815 | −0.000817 [−0.002048, +0.000437] | Not adopted |
| 8 | `contrast-mixing-aligned-feathered` | Aligned blend faded at the edge of the donor's coverage | 0.923754 | 0.924815 | −0.001061 [−0.001702, −0.000415] | Not adopted |
| 9 | `oa-detector-off` | OA detector's correction not added | 0.924347 | 0.924815 | −0.000468 [−0.001017, +0.000110] | Adopted by user decision |
| 10 | `position-3d-without-oa-detector` | 3D position in place of 1D and 2D, detector off | 0.924176 | 0.924347 | −0.000171 [−0.001328, +0.000927] | Rejected |
| 11 | `slab-rule-and-positive-std-requeue` | Slab rule and clamp, without the aligned mixing | 0.924049 | 0.924347 | −0.000297 [−0.001535, +0.000897] | Adopted by user decision |
| 12 | `position-3d-beside-1d-2d` | 3D position added beside 1D and 2D | 0.924598 | 0.924347 | +0.000252 [−0.000457, +0.000937] | Rejected |
| 13 | `position-embeddings-off` | No position embeddings | 0.923570 | 0.924049 | −0.000479 [−0.001535, +0.000611] | Not adopted |

`contrast-mixing-aligned-donor` and `slab-rule-and-positive-std` were cancelled minutes into
training because an adoption changed the incumbent under them, and requeued.

## What the runs established

- **The noise of one run on this fold.** Four runs built on `centre-without-side` that change
  almost nothing (runs 4, 2, 5 and 7) average 0.92422; that incumbent is 0.924815, about 0.0006
  above them. A change that alters which slices or studies are read shifts every later batch:
  the slab rule touches 20 training series and moved Fracture by −0.0091 and a study's Fracture
  probability by 0.015 on average, against 0.006 for the std clamp, which leaves the random
  stream intact. Read macro deltas under about 0.0006 and per-label deltas under about 0.01 as
  ties here; two runs that share a stream change are one draw, not two.
- **Centre without side (+0.0014, the one adoption by rule).** About 3.7% of studies (162 in
  training) have a fitted centre and no side; they now get knee-anchored crops and contrast
  mixing. Part of the delta is the changed random stream.
- **Reflection fix.** Correct by reading and by the
  [TTA measurement](2026-10-05-code-audit-measurements.md#f01-reflection-tta-detail-coordinates);
  the model draws little from the horizontal position of coronal detail tokens (+0.0003 at ×4,
  −0.0006 at ×1).
- **Slab rule and std clamp.** Both correct by reading and both ties: the slab rule changes
  sampling in 20 series of 20 training studies and 2 of 874 validation studies; the clamp
  changes 0.41% of the augmentation's draws (about one in 240 inverted a feature channel).
- **OA detector.** Without its correction the three OA labels do not fall (Lateral OA +0.0029,
  PF OA +0.0004, Medial OA −0.0009) and the macro is level with the do-nothing runs (0.924347
  against 0.92422). Public: **0.950 with the detector (S108, `centre-without-side`) and 0.950
  without it (S111, `oa-detector-off`)**, fold 1 at ×4, τ 0.9, no TTA. One fold-1 pair and one
  public pair; the detector was adopted on 2026-09-25 at PF OA +0.0102 and 0.944 against 0.940
  public, so the removal is to be [checked again](#open) before a submission relies on it.
- **Aligned contrast mixing loses twice** (−0.0008 with identical random draws, −0.0011
  feathered). The alignment is verified correct; the donor is a median 4.0 to 6.1 mm away along
  the normal, so mixing never blended the same anatomy and acts as a perturbation. The
  pixel-index blend stays.
- **3D position: three ties.** Replacing 1D and 2D scores −0.0008 and −0.0002, adding beside
  them +0.0003; the replace runs' lead in the first epoch (+0.0077) comes mostly from removing the
  1D and 2D embeddings (+0.0047 with no position at all) and closes by epoch 5. Lateral OA is
  the only label that gains robustly (+0.0073 against the sibling runs). With no position
  embeddings at all the recipe ends −0.0005, so they are neither shown to matter nor shown to
  be removable.

## Adoption on main

Main takes the behaviour of `slab-rule-and-positive-std-requeue` (user decision, 2026-10-06):

- **Reflection:** `encode_batch` reverses the wide-frame x of mirrored non-sagittal detail
  tokens; `reflect_batch` maps `view_to_wide` to `reversal @ V @ reversal`.
- **Centre without side:** `data/dataset.py` no longer requires the side sign for valid
  geometry.
- **Slab rule:** `build_pool` samples a series physically only when the centre projects inside
  its acquired range, for the `fit` method too; the diagnostic source is `centre_outside_slab`,
  and the sampling identity key is `reference_physical`.
- **Std clamp:** `feature_statistics_augment` clamps the resampled std at zero.
- **OA detector removed:** the detector, its scales, its regions, the anterior axis, the
  `trusted_geometry` tokens and their logging are gone from the model and the Lightning module,
  and `oa_shared_detector_regions` from the spatial-token identity. The campaign run only
  skipped the correction and kept the unused parameters, so main's model has fewer parameters
  than the incumbent's checkpoint. The detector was created under a forked random state, so its
  removal does not move the other initializations.
- **Not carried over:** the campaign's activation logging (sampling-source counts, clamp
  counts, reflected-token counts, centre-without-side counts).
- **Config:** `configs/training.yaml` and `configs/refined-resnet50.yaml` move to fold 1 with
  seed 819238850 (user decision, 2026-10-06).
- **Left in place:** the `knee_center_y_mm` batch key, which the model no longer reads.
  `scripts/adhoc/dump_label_attention.py`, `dump_input_saliency.py` and
  `score_tta_detail_coordinates.py` read the removed token fields and run only on sources from
  before this change.

Checkpoints trained before this change do not load on main. The control is the campaign
incumbent itself, without retraining (user decision, 2026-10-06); it exports from its archived
`source/` ([state](../../ongoing/state.md#current-control)). On 2026-10-06 the campaign's other
checkpoints were removed, except those of S108 and S111.

## Open

- **The OA detector's removal before a submission:** a run with the detector restored on the
  final recipe, read on the three OA labels, and on the ResNet-50
  ([to revisit](../../../autoresearch/campaigns/fixes-and-removals/run-notes.md#to-revisit)).
- **Seven removals never ran:** intensity augmentation, noise augmentation, the half-knee loss,
  the meniscus own-side mask, the plane-contrast readouts, the detail-view embedding and the
  report-attribute heads
  ([seeds 2](../../../autoresearch/campaigns/fixes-and-removals/seed-ideas-2.yaml),
  [seeds 3](../../../autoresearch/campaigns/fixes-and-removals/seed-ideas-3.yaml)); the two
  config-only ones are in the [overnight queue](../../backlog/overnight.md), the rest in
  [small changes](../../backlog/small-changes.md#removals-of-active-mechanisms).
- **Three 3D-position retakes never ran** (position dropped for 15% of studies, position in the
  attention path only, wavelengths from 32 mm); the audit expects little of them
  ([big bets](../../backlog/big-bets.md#position-and-knee-side)).
- **Two audit candidates never seeded:** runs broken by two instances per slice position, and a
  window fixed per series
  ([prior work](../../../autoresearch/campaigns/fixes-and-removals/prior-work.md#candidates-not-yet-seeded)).

## Submissions

| Submission | Model | Settings | Public |
|---|---|---|---|
| S108 | `centre-without-side` (fold 1, ×4 0.924815) | ×4, τ 0.9, no TTA | 0.950 |
| S109, S110 | `oa-detector-off`, one saved notebook and its retry | ×4, τ 0.9, no TTA | ERROR on Kaggle, not counted |
| S111 | `oa-detector-off` (fold 1, ×4 0.924347), same weights as a new dataset and notebook | ×4, τ 0.9, no TTA | 0.950 |
