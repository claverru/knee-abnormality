# Knee abnormality detection

Workspace for the [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
The baseline uses Steven Lee's v4 report-derived labels and five stratified
folds over all 4,407 studies, with about 20% validation per fold and separate
gold-label diagnostics.

## Development

From the repository root:

```bash
rtk proxy uv sync --locked
rtk proxy uv run --locked python -m knee_abnormality.config configs/resnet50.yaml
rtk proxy uv run --locked python -m scripts.tools.validation prepare
rtk proxy uv run --locked python -m scripts.pipeline.train configs/resnet50.yaml
```

Shared Python code lives in `src/knee_abnormality/` (`data/`, `modeling/`,
`training/`, `evaluation/` and `inference/` subpackages). `uv sync` installs it in
editable mode, so scripts and notebooks can import `knee_abnormality` while
source edits take effect immediately. Select `.venv/bin/python` in your editor
or notebook.
The config command prints the selected inputs and resolved paths. The default
config selects fold 0 for quick development; use `--validation.folds '[0,1,2,3,4]'` to
inspect or train a full comparison. Reload a checkpoint for offline test prediction with
`python -m scripts.pipeline.predict`; see the [pipeline index](scripts/pipeline/INDEX.md)
and each script's `--help` for complete commands.
Experiment YAML is validated with Pydantic; `pydantic-settings` parses explicit
CLI overrides. Environment variables and `.env` files do not override experiments.

The root `justfile` exposes the maintained four-command workflow. All artifact
and remote identities are explicit:

```bash
rtk proxy uv run --locked just train configs/resnet50.yaml --run-id RUN_ID
rtk proxy uv run --locked just export --checkpoint CHECKPOINT --output OUTPUT
rtk proxy uv run --locked just infer BUNDLE DATA_ROOT OUTPUT --batch-size 8 --num-workers 2
rtk proxy uv run --locked just submission-job start --bundle BUNDLE --dataset OWNER/DATASET --notebook OWNER/NOTEBOOK --record RECORD --label "Fold 0"
```

Each recipe passes its arguments to the module it names, so `just RECIPE --help`
documents it. `train` runs the YAML's folds (fold 2 in each model config) into a
fresh named run. `export` never selects or resumes a checkpoint implicitly. `infer`
imports prediction code from the bundle and accepts direct competition data mounts.
`submission-job start` versions a private Kaggle Dataset, runs the attached notebook
with Internet disabled, validates its output, and submits only that successful saved
version. A bare `just` lists the recipes without starting work.

- [Project architecture](docs/architecture/INDEX.md): where code belongs, component
  boundaries, configuration, reusable caches, and fast experiment iteration.
- [Documentation index](docs/INDEX.md): evidence, procedures, and decisions.
- [Script index](scripts/INDEX.md): find existing commands and place new scripts.
- [Ongoing work](docs/ongoing/INDEX.md): the next implementation steps.
- [Backlog](docs/backlog/INDEX.md): ideas not yet started, by experiment type.
- [Environment](docs/architecture/environment.md): dependency pins and setup details.

The maintained train/predict entry points live in `scripts/pipeline/`; label and
metadata preparation are shared setup code rather than a separate command.
Reusable acquisition and validation utilities live in `scripts/tools/`.
Metadata/label audits, environment probes, and one-off investigations belong
in `scripts/adhoc/`. See the
[script separation rules](docs/architecture/code-organization.md#maintained-workflow-versus-ad-hoc-work).
Data, credentials, weights, and generated run artifacts stay outside version
control.
