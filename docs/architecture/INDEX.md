# Architecture

| Read when | Document |
|---|---|
| Deciding where code belongs, or finding which module owns a responsibility, including why `modeling/model.py` stayed one module | [Competition repository architecture](code-organization.md) |
| Setting up Python and matching Kaggle dependencies | [Environment and dependency versions](environment.md) |
| Working with slice records and prepared-triplet geometry | [Shared geometry contract](shared-geometry-contract.md) |
| Tracing data through training and prediction | [Data flow and iteration speed](data-flow-and-iteration-speed.md) |
| Writing configs and identifying experiment artifacts, seeds, preprocessing identity and removed config options | [Configs and experiment artifacts](configs-and-experiment-artifacts.md) |
| Checking which checkpoints, bundle weights and caches each cleanup removed and kept | [Disk cleanup log](disk-cleanup-log.md) |
| Finding script commands and maintenance procedures | [Script index](../../scripts/INDEX.md) |

## Common lookups

| Read when | Section |
|---|---|
| Checking random seeds, explicit seeds and resume behavior | [Training seeds and fold selection](configs-and-experiment-artifacts.md#training-seeds-and-fold-selection) |
| Resolving config, CLI and label-manifest paths | [Path resolution](configs-and-experiment-artifacts.md#path-resolution) |
| Locating saved configs, checkpoints and run provenance | [Experiment artifacts](configs-and-experiment-artifacts.md#experiment-artifacts-and-reproducibility) |
| Checking which historical weights were removed and which remain usable | [Disk cleanup log](disk-cleanup-log.md) |
