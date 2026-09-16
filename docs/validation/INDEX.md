# Validation and experiments

## Protocol and interpretation

| Read when | Document |
|---|---|
| Checking the default split, the development selection metric and the fold-0 iteration strategy | [Validation strategy](validation-strategy.md) |
| Using the `all_5fold_v3` split, membership and leakage controls, frozen targets, scoring or the previous split | [Project validation protocol](project-validation-protocol.md) |
| Checking the official macro ROC-AUC definition and public leaderboard evidence | [Official metric and public evidence](official-metric-and-public-evidence.md) |
| Interpreting how local validation scores related to public scores in earlier stages | [Local-to-public score relationship](local-to-public-score-relationship.md) |
| Choosing per-target diagnostics beyond macro AUC, or weighing how much one ranking mistake costs per target | [Target-level diagnostics](target-level-diagnostics.md) |
| Combining predictions with different scales or losses | [Ensemble interpretation](ensemble-interpretation.md) |
| Understanding baseline1's historical twelve-study gold validation | [Why baseline1 had twelve validation studies](why-baseline1-had-twelve-validation-studies.md) |
| Finding the 1A–1C sharing, supervision and hyperparameter evidence | [Sharing, supervision and hyperparameter evidence](sharing-supervision-hyperparameter-evidence.md) |

## Tools and procedures

| Read when | Document |
|---|---|
| Comparing a candidate fold run with a reference run: margins, cross-talk, vendor strata and bootstrap intervals | [Fold-run comparison](fold-run-comparison.md) |
| Measuring a checkpoint's reliance on embeddings and planes | [Manual encoding usage diagnosis](manual-encoding-usage-diagnosis.md) |
| Plotting epoch metrics from completed runs | [Plot validation metrics](plot-validation-metrics.md) |
| Launching the sequential overnight queue or reviewing the 2026-09-14 run list (next-queue candidates are in the [overnight queue](../backlog/overnight.md)) | [Overnight hyperparameter queue](overnight-hyperparameter-queue.md) |

## Results

| Read when | Document |
|---|---|
| Finding a dated experiment result or comparison, or checking how fold-0 AUC tracked the public leaderboard after the split change | [Experiment results](results/INDEX.md) |
| Checking the completed sagittal slice-allocation run | [Sagittal slice-allocation experiment](sagittal-slice-allocation-experiment.md) |
