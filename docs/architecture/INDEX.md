# Architecture

| Read when | Document |
|---|---|
| Deciding where code belongs | [Competition repository architecture](code-organization.md) |
| Setting up Python and matching Kaggle dependencies | [Environment and dependency versions](environment.md) |
| Working with slice records and prepared-triplet geometry | [Shared geometry contract](shared-geometry-contract.md) |
| Tracing data through training and prediction | [Data flow and iteration speed](data-flow-and-iteration-speed.md) |
| Writing configs and identifying experiment artifacts | [Configs and experiment artifacts](configs-and-experiment-artifacts.md) |
| Finding script commands, catalog fields and maintenance procedures | [Script discovery and catalog maintenance](script-catalog.md) |
| Reviewing documentation retrieval measurements and their limitations | [Documentation navigation measurement](documentation-navigation-measurement.md) |

## Common lookups

| Read when | Section |
|---|---|
| Checking random seeds, explicit seeds and resume behavior | [Training seeds and fold selection](configs-and-experiment-artifacts.md#training-seeds-and-fold-selection) |
| Resolving config, CLI and label-manifest paths | [Path resolution](configs-and-experiment-artifacts.md#path-resolution) |
| Locating saved configs, checkpoints and run provenance | [Experiment artifacts](configs-and-experiment-artifacts.md#experiment-artifacts-and-reproducibility) |
| Checking which historical weights were removed and which remain usable | [Checkpoint pruning](configs-and-experiment-artifacts.md#checkpoint-pruning) |
