# Validation strategy

The default split is **`all_5fold_v3`**, protocol version 2: five stratified folds
across **all 4,407 studies**, with roughly 20% held out for validation in each
fold. Prepared on **2026-09-12**, it stratifies gold positive/negative targets
alongside effective labels and `is_gold`, and requires both gold classes for
every target in every fold. No study is forced into every fold's validation set.
Earlier split artifacts remain available for reproducing historical runs.

Use full held-out-fold macro AUC as the development selection metric, with
gold-only and report-derived diagnostics kept separate. This mixed-label
endpoint is not a public leaderboard score or an independently annotated
4,407-study reference endpoint. Repeated tuning remains development selection.

The strategy selected on **2026-09-11** is to iterate on **fold 0** until near
the end of the competition. Use its complete 904-study validation predictions
for routine comparisons. Additional-fold training, full OOF scoring,
ensembling and, since 2026-09-16, bigger images are deferred to that
[later phase](../backlog/endgame.md); they are not
prerequisites for each experiment. Keep the revised five-fold membership fixed within comparisons;
rerun the baseline on this split before comparing new experiments.

## Controlled comparisons

Change one factor against a matched control, and predefine the principal comparison and
checkpoint rule; never search seeds for an apparent improvement. Preserve seed 42, model
size, training budget, image dimensions, physical crop, slice budget and 2× validation
density unless one of them is the factor. Training targets stay soft unless supervision
is the factor; validation memberships and
[frozen targets](project-validation-protocol.md#frozen-evaluation-targets-and-metrics)
never change. Record full-fold and per-target AUC, gold diagnostics, BCE, runtime, peak
VRAM and the distinct slices and TTA passes used, and keep all 904 validation predictions
with the run's [identities](../architecture/configs-and-experiment-artifacts.md). Export
parity is not required for each run or submission.

The user launches training and the
[encoding diagnostic](manual-encoding-usage-diagnosis.md) manually; agents schedule no
training, optimizer-step checks, automatic diagnostics, extra seeds or baseline reruns. Use
fresh artifact directories. Single-fold comparisons cannot resolve training-seed variability or
establish small repeatable gains.
