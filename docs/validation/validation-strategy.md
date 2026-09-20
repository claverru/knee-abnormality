# Validation strategy

The default split is **`all_5fold_v3`**, protocol version 2: five stratified folds
across **all 4,407 studies**, with roughly 20% held out for validation in each
fold. Prepared on **2026-09-12**, it stratifies gold positive/negative targets
alongside effective labels and `is_gold`, and requires both gold classes for
every target in every fold. No study is forced into every fold's validation set.
Earlier split artifacts remain available for reproducing historical runs.

Use full held-out-fold macro AUC as the development selection metric, with
report-derived diagnostics kept separate. This mixed-label
endpoint is not a public leaderboard score or an independently annotated
4,407-study reference endpoint. Repeated tuning remains development selection.
The eleven-study gold subset is recorded but decides nothing: across the comparable
submissions gold-only AUC and gold-only BCE carry no public-score signal, and gold AUC has three
times macro AUC's run-to-run noise
([comparison](results/2026-09-18-local-public-metric-comparison.md)).

The strategy selected on **2026-09-11** is to iterate on **fold 0** until near
the end of the competition. Use its complete 904-study validation predictions
for routine comparisons. Additional-fold training, full OOF scoring,
ensembling and, since 2026-09-16, bigger images are deferred to that
[later phase](../backlog/endgame.md); they are not
prerequisites for each experiment. Keep the revised five-fold membership fixed within comparisons;
rerun the baseline on this split before comparing new experiments.

## Controlled comparisons

Change one factor against a matched control, and predefine the principal comparison and
checkpoint rule; never search seeds for an apparent improvement. **Every run draws its own
recorded random seed, the control included**, so no comparison is made at a matched seed and a
single-run difference is read against the run-to-run spread below. Preserve model
size, training budget, image dimensions, physical crop, slice budget and the configured
validation density (×1 since 2026-09-17) unless one of them is the factor. Training targets stay soft unless supervision
is the factor; validation memberships and
[frozen targets](project-validation-protocol.md#frozen-evaluation-targets-and-metrics)
never change. Record full-fold and per-target AUC, subgroup diagnostics, BCE, runtime, peak
VRAM and the distinct slices and TTA passes used, and keep all 904 validation predictions
with the run's [identities](../architecture/configs-and-experiment-artifacts.md). Export
parity is not required for each run or submission.

### What a comparison can resolve

- **Training runs** carry run-to-run noise with a standard deviation of about **0.0029**, so a
  single-run difference below about **0.008** carries no information even at a matched seed. The
  earlier figure of 0.0015–0.0018 came from seed controls alone — the 2026-09-17 controls spanned
  0.0032 at ×2 and the 2026-09-18 pair 0.0026 at ×1
  ([overnight results](results/2026-09-18-overnight-results.md)) — and understated it. The
  2026-09-19 sigma sweep gives the better estimate: five fold-0 runs of a recipe whose only
  difference is a `sampling_sigma_fraction` between 0.60 and 1.10, which that sweep shows is inert,
  spanned 0.00674 with sd 0.00285
  ([fold ensembles and sigma sweep](results/2026-09-19-fold-ensembles-and-sigma-sweep.md)).
  Fold-0 ReXNet at the current recipe has now produced 0.88367–0.88848 across four seeds.
  **Most single-run deltas this project has recorded, in the 0.002–0.004 band, were never
  resolvable**; only the ensemble gain (+0.006) and class-balanced BCE (+0.0116) clear the bar. A
  queue of many runs against one control will produce an apparent winner by luck: `series-share`
  topped seventeen runs at +0.0044 over the control mean and then lost 0.004 publicly.
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
