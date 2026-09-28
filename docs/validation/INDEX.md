# Validation and experiments

## Protocol and interpretation

| Read when | Document |
|---|---|
| Choosing a split, screening fold and development metric, or checking the fold/seed rotation and controlled-comparison rules | [Validation strategy](validation-strategy.md) |
| Checking split membership (all_5fold_scanner_v2 and v1 scanner strata, all_5fold_v3), report leakage, frozen targets and scoring | [Project validation protocol](project-validation-protocol.md) |
| Checking the official metric and leaderboard evidence | [Official metric and public evidence](official-metric-and-public-evidence.md) |
| Interpreting local scores against public results | [Local-to-public score relationship](local-to-public-score-relationship.md) |
| Inspecting per-label AUC and the cost of individual ranking errors | [Target-level diagnostics](target-level-diagnostics.md) |
| Combining predictions with different scales or losses | [Ensemble interpretation](ensemble-interpretation.md) |
| Understanding baseline1's historical validation set | [Why baseline1 had twelve validation studies](why-baseline1-had-twelve-validation-studies.md) |
| Finding historical sharing and supervision evidence | [Sharing, supervision and hyperparameter evidence](sharing-supervision-hyperparameter-evidence.md) |

## Tools and procedures

| Read when | Document |
|---|---|
| Comparing paired fold runs with bootstrap intervals, vendor strata and cross-talk | [Fold-run comparison](fold-run-comparison.md) |
| Running post-training interventions on spatial embeddings and imaging planes | [Manual encoding usage diagnosis](manual-encoding-usage-diagnosis.md) |
| Plotting completed training runs | [Plot validation metrics](plot-validation-metrics.md) |
| Launching an overnight hyperparameter queue | [Overnight hyperparameter queue](overnight-hyperparameter-queue.md) |

## Results

| Read when | Document |
|---|---|
| Finding dated experiment results | [Experiment results](results/INDEX.md) |
| Reviewing sagittal slice-allocation experiments | [Sagittal slice-allocation experiment](sagittal-slice-allocation-experiment.md) |

## Common lookups

| Read when | Section |
|---|---|
| Checking fold membership and report-group leakage | [Membership and leakage controls](project-validation-protocol.md#membership-and-leakage-controls) |
| Reproducing fixed folds and scoring prediction CSVs | [Reproduce and score](project-validation-protocol.md#reproduce-and-score) |
