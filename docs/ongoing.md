# Ongoing work

A snapshot of where the project stands and what comes next, so the thread is not lost. Details
live in the linked documents; follow the [work-tracking rules](../AGENTS.md#work-tracking).

## State (2026-09-17)

- **Code:** the full-slice recipe ([current model](modeling/current-model.md)) with series
  filtering and deterministic source ranking (preprocessing identity
  `ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`,
  [preparation](data-preparation/current-preparation.md#series-indexing-guards-and-exclusions)).
  Its seeds 42–44 scored 0.889573, 0.889650 and 0.892795 on fold 0 (mean 0.890673), a tie with
  the previous code's seed controls, and seed 42 scored 0.933 public
  ([result](validation/results/2026-09-17-ranked-filtered-result.md)).
- **Best submission:** the reflection/sigma-40 champion, 0.936 public
  ([fast runtime](inference/submissions/2026-09-15-fast-champion-submission.md)).
- **Noise:** run-to-run noise comes from training (standard deviation about 0.0015–0.0018); a
  single-run difference below about 0.003 carries no information, even at a matched seed.
- **Dropped:** learned crop selection ([record](history/crop-selection-investigation.md)).
- **Strategy:** iterate on fold 0; folds, ensembling, bigger images, TTA and weight averaging wait
  for the [final phase](backlog/endgame.md)
  ([validation strategy](validation/validation-strategy.md)). Test-time risks are listed in
  [risks](backlog/risks.md).

## Next

Experiments run one by one, each on top of the previous one's accepted state, compared over
seeds 42–44 against the ranked-filtered runs
([per-type budget evidence](validation/results/2026-09-17-plane-contrast-importance.md)):

1. **Missing-type fill:** when a plane–contrast type is absent, give its slots to the same plane's
   other contrast first, then coronal fluid, axial fluid and sagittal fluid
   ([entry](backlog/small-changes.md#per-type-budget-reallocation)).
2. **Per-type budget reallocation:** budgets A (9/18/15/21/15/18) and B (9/18/12/21/12/15), in
   the order axial T1/PD, axial fluid, coronal T1/PD, coronal fluid, sagittal T1/PD, sagittal
   fluid ([entry](backlog/small-changes.md#per-type-budget-reallocation)).

Also pending:

- **Robust study centre:** implemented behind `data.robust_center_outlier_mm`, and one seed-42
  run is queued ([overnight](backlog/overnight.md#next-queue)). The code changed the source
  digest, so the ranked-filtered checkpoints need their archived sources.
- Data-loading speedups, shipped with the next retrain
  ([plan](inference/loading-speedup-plan.md)).
- Submit the main-thread warm-up with the next submission; torch-free coordinator
  ([runtime changes](inference/loading-speedup-plan.md#runtime-changes-outside-the-preprocessing-identity)).
