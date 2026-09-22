# Validation strategy

The default split is **`all_5fold_scanner_v1`**, protocol version 2, prepared on **2026-09-21**.
It has five stratified folds across **all 4,407 studies**, with roughly 20% held out for validation
in each fold. Report groups stay intact. Stratification balances the effective labels, gold
positives and negatives, `is_gold` and eight MRI scanner groups. Its evaluation targets are the
[combined labels](../public-labels/combined-labels.md), with gold labels where available. The
[protocol](project-validation-protocol.md#all_5fold_scanner_v1) gives the membership and scanner
counts. Only 209 of its 905 fold-0 validation studies were in `all_5fold_v3`'s fold 0, and four
evaluation targets changed. **No fold-0 score recorded before 2026-09-21 is comparable** with a run
on this split. `all_5fold_v3` and earlier split artifacts remain available for reproducing
historical runs.

Use full held-out-fold macro AUC as the development selection metric, with
report-derived diagnostics kept separate. This mixed-label
endpoint is not a public leaderboard score or an independently annotated
4,407-study reference endpoint. Repeated tuning remains development selection.
The eleven-study gold subset is recorded but decides nothing: across the comparable
submissions gold-only AUC and gold-only BCE carry no public-score signal, and gold AUC has three
times macro AUC's run-to-run noise
([comparison](results/2026-09-18-local-public-metric-comparison.md)).

The strategy selected on **2026-09-11** is to iterate on **fold 0** until near
the end of the competition. Use its complete 905-study validation predictions
for routine comparisons. Additional-fold training, full OOF scoring,
ensembling and, since 2026-09-16, bigger images are deferred to that
[later phase](../backlog/endgame.md); they are not
prerequisites for each experiment. Keep the revised five-fold membership fixed within comparisons;
compare against a recorded incumbent trained on this split; no automatic baseline
rerun is required for each experiment.

## Controlled comparisons

Change one factor against a matched control, and predefine the principal comparison and
checkpoint rule; never search seeds for an apparent improvement. **Candidates use the
current incumbent's recorded training seed.** Adopt a valid implementation only when
macro gain >= 0.001 and the central 80% paired-report-group bootstrap lower bound
is strictly positive; otherwise discard it. Keep the 95% interval as diagnostic
evidence. No automatic confirmation branches or control reruns follow a result.
This engineering policy does not establish cross-seed robustness; see the
[adoption protocol](../../autoresearch/protocol.md#adoption-policy). Preserve model
size, training budget, image dimensions, physical crop, slice budget and the configured
validation density (×1 since 2026-09-17) and prediction density (×4 since 2026-09-22) unless one of them is the factor. Training targets stay soft unless supervision
is the factor; validation memberships and
[frozen targets](project-validation-protocol.md#frozen-evaluation-targets-and-metrics)
never change. Record full-fold and per-target AUC, subgroup diagnostics, BCE, runtime, peak
VRAM and the distinct slices and TTA passes used, and keep all 904 validation predictions
with the run's [identities](../architecture/configs-and-experiment-artifacts.md). Export
parity is not required for each run or submission.

### What a comparison can resolve

- **Training runs** carry variability not captured by validation resampling. Historical
  seed controls and sigma sweeps showed spreads of several thousandths, under different
  recipes ([overnight results](results/2026-09-18-overnight-results.md),
  [sigma sweep](results/2026-09-19-fold-ensembles-and-sigma-sweep.md)). These observations
  motivate caution, not a universal 0.003 or 0.008 minimum meaningful gain. Matching
  seeds reduces one source of noise but does not make runs deterministic. The adopted
  rule above deliberately allows practical progression without claiming independently
  confirmed gains. Repeated selection against one control can still produce lucky winners.
- **Test-time variants of one frozen checkpoint** are far cheaper to resolve, because the paired
  bootstrap over studies removes training variability. Measured over 36 pairs, the standard error
  of Δ macro AUC scales with how much the variant moves the predictions: SE ≈ 0.067 × p^1.05,
  where p is the mean absolute probability change (R² 0.83). The 95% resolvable difference on full
  fold 0 is therefore about 0.0003 at p = 0.003, 0.001 at p = 0.01 and 0.002 at p = 0.02; on one
  half of the split about 1.46× larger, and 0.67–0.91× smaller when averaged over three
  checkpoints. Script and outputs:
  [snapshot](../history/series-and-sampling-sources/README.md#triplet-allocation),
  `outputs/analysis/detection-floor-20260917/`.
- **Validation multiplier:** scores at ×1 sit about 0.0035 below the same checkpoints at ×2, so
  runs are comparable only with a control at the same multiplier.
- **Prediction multiplier (since 2026-09-22):** each epoch is still validated at ×1 and the best
  one selected there, but the saved predictions that comparisons score come from that checkpoint
  at ×4 and attention temperature 1 (`validation.prediction_image_multiplier`). Models scale
  differently with more triplets: `oblique` was −0.0015 against `k_05` at ×1 and +0.0003 at ×4
  τ 0.75 ([result](results/2026-09-22-oblique-groups-result.md#temperature-and-count)). Compare
  only runs predicted at the same multiplier.
- **Against the leaderboard**, macro AUC is the best available local predictor (Pearson +0.907
  over nineteen comparable submissions) and BCE adds nothing on top of it, but neither resolves
  the submissions made since 2026-09-13, whose local spread is below one run's noise
  ([comparison](results/2026-09-18-local-public-metric-comparison.md)). Prefer a public result
  over a sub-0.005 fold-0 difference.

The user launches training and the
[encoding diagnostic](manual-encoding-usage-diagnosis.md) manually; agents schedule no
training, optimizer-step checks, automatic diagnostics, extra seeds or baseline reruns. Use
fresh artifact directories. Single-fold comparisons cannot resolve training-seed variability or
establish small repeatable gains.
