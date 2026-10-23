# Validation and experiments

| Read when | Document |
|---|---|
| Checking the final validation rules: the official metric, frozen targets and checkpoint selection, fold and seed rotation, controlled-comparison and adoption rules, what a comparison can resolve, membership and leakage controls, ensemble interpretation, and scoring prediction CSVs | [Validation protocol](validation-protocol.md) |
| Looking up a split by name: fold tables and stratification of `all_5fold_scanner_v2`, the v3 retarget, per-member rerolls, the gold-holdout and unseen-site splits, hashes, and the superseded splits with their dates | [Validation splits](validation-splits.md) |
| Interpreting local scores against public results: which metrics tracked the board, why gold-only diagnostics decide nothing, and the sharing-experiment generalization evidence | [Local-to-public score relationship](local-to-public-score-relationship.md) |
| Comparing paired fold runs with bootstrap intervals, re-predicting an older run, running the encoding-usage diagnostic, plotting runs, costing a reversed pair per label, or reading overnight-launcher outputs | [Run comparison and diagnostics](run-comparison-and-diagnostics.md) |
| Finding dated experiment results by topic | [Experiment results](results/INDEX.md) |

## Common lookups

| Read when | Section |
|---|---|
| Checking fold membership and report-group leakage | [Membership and leakage controls](validation-protocol.md#membership-and-leakage-controls) |
| Checking the fold and seed rotation rule | [Controlled comparisons](validation-protocol.md#controlled-comparisons) |
| Checking the rules for adopting a candidate | [Adoption rules](validation-protocol.md#adoption-rules) |
| Reproducing fixed folds and scoring prediction CSVs | [Reproduce and score](validation-protocol.md#reproduce-and-score) |
| Reading the fold table of the recipe's split | [all_5fold_scanner_v2](validation-splits.md#all_5fold_scanner_v2) |
