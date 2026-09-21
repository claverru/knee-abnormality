# Shared-loss campaign

Current state, 2026-09-21: stopped on the user-adopted learned-strength incumbent
(seed 2263738815, fold-0 AUC 0.897443). The source/reference transition is recorded
in [the shared closure record](../../checkpoint-closure.md). Original reference,
budget and startup sections below describe the historical seal; they do not select
the current incumbent. Historical attempts retain their original references.

## Objective and scope

Improve mean per-label ROC-AUC by changing the training objective. Use **one loss,
or one combination of losses, shared across all twelve labels**. The user excluded
per-label loss selection on 2026-09-20 to avoid multiplying experiment variants.
Loss families, component coefficients, focusing/margin parameters, and schedules
must be shared. No label-specific switches, learned mixtures, coefficient vectors,
classifier banks, or per-label searches.

Within this boundary, explore pointwise and ranking objectives, shared hybrids,
loss weighting and normalization, training-only confidence treatment, and schedules
inside the original run. A common formula may use each label's training-fold class
counts, as the current class-balanced BCE already does; this is not permission to
choose different formulas or tune separate parameters for different labels.
Report every label as a diagnostic, without selecting a different objective for it.

Supporting loss computation and training-only state are allowed. Keep the encoder,
attention, classifiers, data preparation, triplet selection, augmentation, study
exposure, batch composition, optimizer, learning-rate schedule, and inference fixed.
No relabeling, new data, multi-model ensembles, additional training stages outside
the sealed run, or generic hyperparameter sweeps. A shared loss schedule must fit
inside the existing ten epochs, starting from the same pretrained initialization
policy; it is not an extra continuation from a fitted checkpoint.

**Only local training and local validation are authorized.** No submissions,
uploads, publication, or remote evaluation. Creating this campaign does not launch
it; the user starts the detached supervisor using [runtime operation](../../runtime.md).

## Fixed recipe and resource limits

- Use fold 0 of `all_5fold_v3`, the existing Steven v4 targets and observed masks,
  report groups, and checkpoint-selection/scoring rules. Preserve soft targets in
  pointwise terms. Define ranking pair eligibility and target semantics explicitly;
  hard positive/negative grouping for a pair term does not rewrite the source labels.
- Start from current class-balanced BCE and the control pinned in
  [campaign.yaml](campaign.yaml). Record the source snapshot, effective configuration,
  input identities, incumbent-matched training seed, and exact loss formula for every attempt.
- Keep the same training cohort and study visits, average realized valid triplets,
  ten epochs, and 4,380 optimizer steps. Keep stochastic training selection and
  deterministic unaugmented validation at image multiplier 1. Losses may not change
  preprocessing or validation selection identity. The existing triplet-accounting
  replay and gates remain mandatory even though this campaign changes losses.
- Seal the runtime ceiling at 115% of control B's recorded setup, training, and
  final prediction time, rounded up: **1,797 seconds** from the current artifacts.
  Configuration preparation, objective setup, training, final prediction, and CPU
  bootstrap comparison count toward it. The one-time baseline accounting replay,
  agent reasoning, queue waits, and source housekeeping follow the common protocol.
  An adopted candidate never raises this ceiling.
- All pair pools, component scales, and adaptive loss state use training data only.
  No fold-0 validation labels or gold labels enter objective fitting or weighting.
  Memory banks must record initialization, refresh, detachment, and stale-score
  handling; their construction runs inside measured training, with no external
  feature extraction or hidden precomputation. Extra model updates or data passes
  are outside the fixed schedule.
- Every new candidate inherits the incumbent's recorded seed; never retry seeds to recover a win.
  A genuine checkpoint resume retains its original seed. Bootstrap draws use the
  separate fixed resampling seed.

## Implementation and review

Candidate edits are limited to `configs/training.yaml`, loss configuration in
`src/knee_abnormality/config.py`, objective computation/state in
`src/knee_abnormality/training/lightning.py`, and helpers under
`src/knee_abnormality/training/losses/`. New settings belong under `trainer.loss`;
`trainer.class_balanced_bce` may change when the proposed objective replaces its
weighting. The numerical gate rejects other recipe changes. Neither scripts nor
the runtime, evaluator, accounting, or contract are candidate-editable.

The Lightning file also contains protected behavior. Review its diff to ensure
data loading, model construction, optimizer/scheduler, forward prediction,
validation, checkpoint selection, and accounting remain unchanged. Objective state
may be added without altering those semantics. The file allowlist and config gate
do not replace substantive source review.

Review missing-label masks, soft-target semantics, class weights, per-component
scales, and numerically stable float32 loss computation under mixed precision.
For ranking, exclude self-pairs, average usable pairs within each target, and
define the zero-pair case. Inspect pair coverage across all labels at microbatch
size eight; gradient accumulation does not create additional joint pairs. Log
pair coverage and component contributions when relevant. Review training-only
feedback and shared parameter shapes explicitly. No tests or GPU work by agents;
only the supervisor launches training after review and GPU checks.

## Candidate evaluation

Normally train one full-budget model per idea. Pin the current incumbent before
launch and compare saved predictions on the same studies using 2,000 paired
report-group bootstrap draws and the existing frozen invalid-resample rule.
Compute candidate-minus-reference macro AUC with a 95% interval.

| Result after contract checks | Decision |
|---|---|
| Valid macro AUC gain >= 0.001 and 80% interval lower bound > 0 | Adopt as the working incumbent (`provisional_accept`) |
| Valid result failing either adoption condition | Discard this implementation (`reject`); retain the incumbent |
| Missing evidence or contract breach | Invalid; no acceptance |

Retain the original control as evidence when the incumbent changes. Per-label
results explain tradeoffs but do not replace the fixed macro criterion or authorize
per-label objectives. Bootstrap uncertainty is conditional on the fitted models;
it excludes training-seed variation and repeated selection on fold 0. Acceptance
is provisional. Do not automatically repeat discarded runs or stop an otherwise
valid full-budget run for weak early AUC. Exceptional confirmations require a
recorded reason under the [protocol](../../protocol.md).
