# Ongoing work

A snapshot of where the project stands and what comes next, so the thread is not lost. Details
live in the linked documents; follow the [work-tracking rules](../AGENTS.md#work-tracking).

## State (2026-09-16)

- **Code:** the full-slice `00-baseline` recipe (fold 0 0.892763, public 0.934), restored on
  2026-09-16 ([current model](modeling/current-model.md)).
- **Best submission:** the reflection/sigma-40 champion, 0.936 public
  ([fast runtime](inference/submissions/2026-09-15-fast-champion-submission.md)).
- **Dropped:** learned crop selection ([record](history/crop-selection-investigation.md)).
- **Strategy:** iterate on fold 0; folds, ensembling, bigger images and TTA wait for the
  [final phase](backlog/endgame.md)
  ([validation strategy](validation/validation-strategy.md)).

## In progress

- Package restructure, `configs/training.yaml` rename, label manifests in `configs/labels/`,
  model-doc split and code-review refactors: done, awaiting commit.
- Docs [backlog](backlog/INDEX.md) replacing `future-work.md` and the root `overnight.md`: done,
  awaiting commit.

## Next

1. Next overnight queue: the user decides the
   [proposed list](backlog/overnight.md#next-queue-proposed-pending-the-users-decisions) (SWA tail
   learning rate, seed replicate); weight averaging is implemented first if the user asks
   ([small change](backlog/small-changes.md#weight-averaging)); then the launcher's run list is
   replaced and the user launches the queue. The saved-prediction comparison of `03-min-lr-0.1`
   with `00-baseline` is done: +0.00041, not established.
2. Data-loading speedups, shipped with the next retrain
   ([plan](inference/loading-speedup-plan.md)).
3. Submit the main-thread warm-up with the next submission; torch-free coordinator
   ([runtime changes](inference/loading-speedup-plan.md#runtime-changes-outside-the-preprocessing-identity)).
