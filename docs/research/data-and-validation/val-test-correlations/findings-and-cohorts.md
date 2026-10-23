# Validation–Kaggle correlations: findings and cohorts

Measured 2026-09-20 from saved artifacts, using CPU only. The investigation was read-only: no
training, image inference, checkpoint modification or submission was run. The sections are split
across [balanced-gain decomposition](balanced-gain-decomposition.md),
[label and study predictors](label-and-study-predictors.md) and [reproduction and limits](reproduction-and-limits.md).

## Findings and decisions

**The strongest new finding is that a large validation gain can concentrate on report-derived
negatives whose evidence is weak or unaddressed.** Ordinary ReXNet → balanced-BCE control-B gains
0.012911 validation macro AUC but moves public 0.936 → 0.933. Of that local gain, 92.8% involves
negative references between 0.1 and 0.5; 80.8% comes from addressed positives beating negatives
that Steven's original extractor marked unaddressed. These are two overlapping decompositions.

This is a plausible explanation for poor transfer, not proof of hidden-test errors or a causal
loss effect: seeds differ, validation uses fewer images than submission, and Kaggle exposes only
aggregate scores. We cannot identify which hidden-test label actually raised or lowered a score.
The older plateau also has a separate resolution problem: model differences are smaller than
measured run-to-run noise. Neither explanation alone covers every submission.

Recommendations made at the time, in priority order (not implemented by this analysis):

1. **Add a label-evidence audit alongside full-fold AUC.** Preserve original report coverage,
   label-source agreement and extreme-value diagnostics separately. Require a prospective gain
   to be inspected for dependence on unaddressed negatives. Keep full-fold AUC as the primary
   recorded metric; none of the alternative endpoints below is validated as a replacement.
2. **Resolve balanced BCE with matched comparisons.** Use ordinary/balanced pairs with the same
   source snapshot, seeds, exposure, optimizer budget and inference settings, replicated across
   seeds or folds. Existing repeated runs support endpoint sensitivity, but do not isolate cause.
3. **Validate finalists at submission settings.** Compare the same checkpoint at the intended
   image budget and attention temperature before attributing a public change to training.
   Existing token caches permit a useful CPU approximation for their three compatible models.
4. **Audit labels with the official image rubric.** Prioritize unaddressed negatives, contusion,
   MCL, Baker's cyst and fracture; distinguish report silence, explicit denial and severity.
   More independent image-adjudicated validation would be more informative than selecting a
   metric against this small submission history. Keep the current eleven gold cases diagnostic.
5. **Retain diversity and uncertainty in selection.** Inspect neighboring epochs and repeated
   runs; evaluate complementary predictions in ensembles. Do not select a label, subgroup or
   loss weight merely because its retrospective leaderboard correlation looks favorable.

Blanket removal of unaddressed training labels was already
[tried and rejected](../../../history/sharing/2026-09-13-label-masking-result.md), costing 0.011455 local
AUC. The finding motivated better diagnostics and controlled experiments, not that retry.

## Cohorts and checkpoint identity

All new calculations align predictions by `StudyInstanceUID` to fold 0 of `all_5fold_v3`:
**904 studies, 893 report-derived references and 11 official gold studies**, twelve targets.
Canonical AUC binarizes frozen targets at `q >= 0.5`. Checkpoints are selected by maximum saved
epoch macro AUC; alternative diagnostics do not reselect the checkpoint.

| Cohort | Models | Validation / submission | Macro AUC vs public, Pearson / Spearman |
|---|---:|---|---:|
| Historical comparable | 19 | ×2 / ×2, τ=1 | +0.907 / +0.620 |
| Historical plateau, submitted from Sep 13 | 14 of 19 | ×2 / ×2, τ=1 | +0.072 / +0.062 |
| Newer single-model set | 5 | ×1 / ×4, τ=0.75 | −0.672 / not used for selection |

The historical membership and exclusions are in the
[score manifest](../../../../scripts/adhoc/submission_public_scores.json) and
[earlier comparison](../../../validation/results/validation-and-ensembles/local-public-metric-comparison/2026-09-18-comparable-submissions.md).
Do not pool these inference regimes into one correlation. The newer five are a small descriptive
comparison, not a held-out statistical validation set.

| Newer model | Seed | Completed epoch | Saved validation AUC | Public | Submission record |
|---|---:|---:|---:|---:|---|
| Robust center 20, top-up 26 | 3167394480 | 10 | 0.888021 | 0.938 | [record](../../../../outputs/submission/records/knee-rc20-topup26-tau075-4x-20260918-v2.json) |
| EfficientNet-B0 fold 0 | 886913563 | 11 | 0.886354 | 0.934 | [record](../../../../outputs/submission/records/knee-b0-fold0-tau075-4x-20260919.json) |
| Ordinary ReXNet fold 0 | 471048644 | 10 | 0.883671 | 0.936 | [record](../../../../outputs/submission/records/knee-rexnet-fold0-tau075-4x-20260919.json) |
| Balanced control-B | 2263738815 | 9 | 0.896582 | 0.933 | [record](../../../../outputs/submission/records/control-b-20260920-x4-tau075.json) |
| Balanced, stratified sampling | 4061686476 | 10 | 0.897515 | 0.933 | [record](../../../../outputs/submission/records/stratified-sampling-20260920-x4-tau075.json) |

Records resolve to export manifests and exact source checkpoint paths. Completed epochs above
are one-based; prediction filenames/checkpoint epochs are zero-based. Control-B means
`overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat`, **with balancing enabled**.
The seed-42 `balanced-bce` run is a separate model; its failed submission attempts have no score.
