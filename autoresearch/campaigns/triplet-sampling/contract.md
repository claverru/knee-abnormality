# Triplet-sampling campaign

Current state, 2026-09-21: stopped on the user-adopted learned-strength incumbent
(seed 2263738815, fold-0 AUC 0.897443). The source/reference transition is recorded
in [the shared closure record](../../checkpoint-closure.md). Original reference,
budget and startup sections below describe the historical seal; they do not select
the current incumbent. Historical attempts retain their original references.

## Objective and fixed constraints

Select more useful triplets for training while retaining both coverage and
exploitation. Explore selection mechanisms, distributions, redundancy, allocation,
and their evolution across training. Supporting code may change where necessary.
Backbones, losses, optimizers, image resolution, and generic hyperparameter sweeps
are outside this campaign. Keep the surrounding training recipe fixed.

**No submissions or publication. Only local training and local validation are
authorized.** Never submit to Kaggle, upload models or predictions, publish
datasets or notebooks, or launch remote evaluation jobs. Reading public research
and documentation is allowed.

- All training, including controls and confirmations, uses fold 0 of `all_5fold_v3`.
- Preserve the baseline's average realized valid triplets per study over the same
  training cohort and study visits. Redistributing counts between studies or types
  is allowed. Count real selections, excluding padding; record total selections and
  the denominator. Do not change cohort membership or study visit frequency to
  manipulate the average. `study_triplet_total: 26` is a floor, not this average.
- Training sampling is stochastic. Validation is deterministic, unaugmented, and
  uses image multiplier 1. Validation choices cannot depend on the training seed.
  If a candidate changes deterministic selection, record its identity explicitly.
- Every new candidate uses the current incumbent's recorded training seed. Do not
  optimize or retry seeds for a better score. A genuine resume retains its run's seed.
- Freeze the initial epochs, optimizer steps, and end-to-end resource ceiling for
  the campaign. Preparation and any auxiliary sampling computation count toward
  that ceiling. A winner cannot buy later improvements by increasing compute.
- Preserve labels, fold membership, grouping, scoring, and checkpoint-selection
  rules. Learned sampling feedback uses training data only.

## Candidate evaluation

Normally train one full-budget model per idea. Choose its reference incumbent
before launch. Use saved fold-0 predictions on the same studies for a paired
report-group bootstrap: resample whole groups, using identical draws for candidate
and reference, and compute candidate-minus-reference macro AUC. Start with the
existing 2,000-draw, 95% interval procedure; record its resampling seed separately
from the incumbent-matched training seed. Invalid resamples must follow one frozen rule.

The user replaced the comparison incumbent with control B on 2026-09-20:
`control-b-20260919T235434Z`, the `13-control-repeat` run under
`outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/`.
The runtime state pins this incumbent and its prediction hash. The next candidate
compares against B; a provisionally accepted candidate then becomes the incumbent
under the same rule below. Control A is supporting evidence. See
[reference evidence](prior-work.md#reference-evidence) for identities and provenance.
The configuration's `reference_run` remains the original sealing and budget
reference; `additional_evidence_runs` is an evidence inventory, not the runtime
incumbent selector. This replacement does not reseal or increase any limit.

| Result after contract checks | Decision |
|---|---|
| Valid macro AUC gain >= 0.001 and 80% interval lower bound > 0 | Adopt as the working incumbent (`provisional_accept`) |
| Valid result failing either adoption condition | Discard this implementation (`reject`); retain the incumbent |
| Incomplete evidence or contract breach | Invalid; no acceptance |

Per-label diagnostics inform interpretation but do not replace the fixed primary
metric. Report exact deltas; do not substitute the old generic 0.003 or 0.008
single-run thresholds for this campaign rule. The initial baseline remains a
reference even when the incumbent changes.

Bootstrap uncertainty is conditional on the fitted models. It excludes training
seed variation and does not correct repeated selection on fold 0. Acceptance is
provisional, not independent confirmation. Separate confirmation seeds require an explicit user-directed protocol; do not automatically rerun discarded ideas or retain pending branches.
Poor early-epoch scores alone do not justify terminating a model early.

## Initial sealing

The supervisor must capture and seal the source, resolved recipe, data identities,
reference predictions, measured baseline average triplets, counting tolerance,
resource ceiling with its measurement boundary, and evaluation settings. The count
tolerance is zero. The runtime ceiling is 115% of the original historical control's
recorded total, rounded up to 1,785 seconds. It covers configuration preparation,
training and its validation/prediction, and CPU comparison. Candidate agents cannot
change these limits or inherit a larger budget from the latest winner.

Verify the historical control against the sealed recipe and source. If it cannot
serve as a comparable reference, pause and resolve the control discrepancy before candidates.
The runner uses the current incumbent run's recorded seed for every new candidate.
The runtime obtains realized counts from a one-time full-schedule CPU replay; no
extra model is fitted to establish that count. See [runtime operation](../../runtime.md)
for the measurement boundary and [implementation status](../../status.md) for verification.
