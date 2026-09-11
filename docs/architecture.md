# Competition repository architecture

Decision recorded on **2026-09-10**: optimize this repository for quick, traceable
competition experiments. Launch commands from the repository root. Separate
the maintained workflow in `scripts/pipeline/`, reusable utilities in
`scripts/tools/`, and one-off work in `scripts/adhoc/`.
Keep shared modules in `src/knee_abnormality/`, YAML configs validated with
Pydantic in `configs/`, and prepare image pixels on the fly for v1. `uv` manages
dependencies and installs the local package in editable mode through setuptools.

The baseline uses [Steven v4](public-label-audit.md#implication-for-the-first-baseline)
and the [all-study five-fold protocol](validation.md#project-validation-protocol).
Each fold holds out about 20% of all studies, without assigning folds by gold
status. The old reference-only split remains available for baseline1 reproduction.
The maintained baseline config uses a ReXNet-200 NAV ImageNet encoder with
three-channel adjacent slice triplets; the latest completed baseline3 artifact
uses ResNet50D RA2. Both pool usable acquisitions within each plane–contrast
slot, favor central triplets, and apply target-specific attention MIL across
all retained inputs. Validation doubles the configured per-series raw-slice
limits without changing training or test limits. The
[modeling notes](modeling.md#implemented-baseline) record the chosen
backbone, raw-slice limits, 224-pixel crops at 0.67 mm, clipping quantiles
`[0.005, 0.995]`, checkpoint-specific normalization and training defaults.

Prefer Lightning's built-in training, precision, optimization and callback
facilities wherever they fit this workflow. `LightningModule`,
`LightningDataModule`, `Trainer`, `ModelCheckpoint`, `EarlyStopping` and
`CSVLogger` manage the current training lifecycle. timm's public APIs select
the encoder and weights. The current backbone follows Lightning's training
transition and updates BatchNorm statistics; saved configs may instead select
evaluation-mode BatchNorm and timm activation checkpointing.

## Where code belongs

Start with the [script index](../scripts/INDEX.md) to find existing commands
before adding code. The repository's
[scripts-maintainer skill](../.agents/skills/scripts-maintainer/SKILL.md)
defines discovery and catalog maintenance. Each runnable script has one
catalog entry; detailed usage and findings stay in their topic docs.

Training and prediction entry points live in `scripts/pipeline/`.
Shared setup functions handle metadata/label preparation; a separate
`prepare.py` command is unnecessary. Reusable
support tools live in `scripts/tools/`; one-off investigations belong in
`scripts/adhoc/`. Shared modules belong in `src/knee_abnormality/` and use
imports such as `from knee_abnormality.config import load_config`. Add a
shared module when real code needs it, and split a growing file by
responsibility when that makes editing easier.

| Location | Responsibility | Status |
|---|---|---|
| `src/knee_abnormality/config.py` | Validate YAML with Pydantic, resolve paths, apply overrides and print effective settings | Implemented |
| `src/knee_abnormality/cli.py` | Define the shared explicit-only `pydantic-settings` policy for runnable CLIs | Implemented |
| `src/knee_abnormality/runtime.py` | Configure the CUDA allocator and move tensor batches between devices | Implemented |
| `src/knee_abnormality/provenance.py` | Stream and hash files used by audits, checkpoints, bundles and experiment records | Implemented |
| `src/knee_abnormality/metrics.py` | Compute per-target and macro AUC plus validation subgroup metrics | Implemented |
| `src/knee_abnormality/profiling.py` | Select bounded profile datasets and record reproducible profiling artifacts | Implemented |
| `configs/baseline.yaml` | Selected inputs, preprocessing, model, optimization, runtime, seed, paths, and development fold | Implemented |
| `scripts/INDEX.md`, `scripts/**/catalog.yaml` | Script discovery, usage metadata and index maintenance | Implemented; inventories live in the catalogs |
| `scripts/tools/` | Reusable acquisition, validation and repository utilities | Implemented; see the [catalog](../scripts/tools/catalog.yaml) |
| `scripts/adhoc/` | Metadata/label investigations, environment probes and future exploration | Implemented; see the [catalog](../scripts/adhoc/catalog.yaml) |
| `src/knee_abnormality/labels.py` | Polars joins, source fingerprints, gold/silver provenance and fold membership | Implemented |
| `src/knee_abnormality/checkpoints.py` | Validate saved input/model/target semantics and fingerprint actual training inputs for resumption | Implemented; legacy checkpoints support prediction with their saved config |
| `src/knee_abnormality/preprocessing.py` | DICOM decoding, physical ordering, spatial preparation and fixed intensity processing, with injectable decoding for profiling | Implemented |
| `src/knee_abnormality/augmentation.py` | Sample and apply shared affine, gamma/contrast and noise transforms for training and previews | Implemented |
| `src/knee_abnormality/dataset.py` | Set up the series index; select series and slices, prepare and collate inputs, and construct configured data loaders | Implemented |
| `src/knee_abnormality/model.py` | PyTorch encoder, guarded series dropout and study-level aggregation producing twelve logits | Implemented |
| `scripts/pipeline/train.py` | Lightning module/data loading, optimizer, checkpoints, validation outputs and training entry point | Implemented; baseline3 fold 0 completed, with earlier checkpoint corrections recorded in the [baseline review](baseline-review.md#implemented-corrections) |
| `scripts/pipeline/predict.py` | Load saved model/config, predict studies, write validation/submission tables | Implemented |
| `scripts/pipeline/export.py` | Build an atomic, hash-inventoried checkpoint/code/wheel bundle and generated inference notebook | Implemented; artifact checks remain checkpoint-specific |
| `scripts/pipeline/submit.py` | Version the private bundle, run the offline notebook, verify output and submit the exact successful version | Implemented; requires explicit remote IDs and a ready bundle |
| `justfile` | Expose the four maintained train, export, infer and submit commands without embedding Python orchestration | Implemented |
| `notebooks/` | Optional visual inspection and short investigations that call shared code | Create when needed; submission notebooks are generated inside bundles |
| `docs/` | Decisions, procedures, evidence, and work tracking | Implemented |
| `data/`, `outputs/` | Downloaded inputs, prepared metadata/labels and generated experiment artifacts | Local and Git-ignored |

This table defines responsibilities; planned files are not empty scaffolds.
Keep experiment logic explicit. A new backbone or pooling alternative can be
a small constructor/function in `src/knee_abnormality/model.py`; it does not
need a plugin registry or a parallel copy of the training loop. Put runnable
behavior behind a main guard so scripts can also expose useful functions.

### Maintained workflow versus ad-hoc work

`scripts/pipeline/` contains the commands used to reproduce a baseline and prepare a
submission. Improve these commands as the solution evolves; avoid accumulating
variants such as `train_v2.py` or `predict_final_fixed.py`. Select experiment
variants through configs and shared model code.

`scripts/tools/` contains reusable supporting commands: `download_data.py`
handles acquisition and resumption; `validation.py` creates fixed folds and
scores OOF predictions across experiments. Keep their arguments and outputs
stable as the pipeline develops. Extract shared functions into
`src/knee_abnormality/` when the training or preparation code needs them.

`scripts/adhoc/` contains focused investigations with descriptive names. The
existing metadata and public-label audits reproduce the evidence in the topic
docs; the environment probe records setup checks. These scripts
can call shared code, but the maintained workflow must not depend on them.
When an investigation becomes part of the solution, move its reusable behavior
into the relevant shared module or maintained entry point. Record useful
findings in the topic docs and keep generated artifacts under `data/` or
`outputs/`, not beside the script.

## Data flow and shared behavior

```mermaid
flowchart LR
    Raw[Raw MRI and series metadata] --> Dataset[Study dataset and series/slice selection]
    Dataset --> Prep[Shared on-the-fly pixel preparation]
    Gold[Official reference labels] --> Labels[Label and membership preparation]
    Silver[Steven v4 and provenance] --> Labels
    Folds[Saved folds] --> Labels
    Labels --> Train[Lightning training]
    Prep --> Train
    Train --> Weights[Checkpoint and resolved config]
    Weights --> Predict[Study prediction]
    Test[Unseen MRI] --> Dataset
    Prep --> Predict
    Predict --> Tables[OOF or submission table]
```

Label preparation and pixel preparation are independent. Label joins happen
once during setup, not inside `__getitem__`. V1 decodes and prepares pixels
while loading batches; preprocessing behavior follows the saved image recipe.

The dataset can retain multiple usable acquisitions per selected type and
supplies three-channel uint8 inputs containing consecutive non-overlapping
slices, triplet validity masks and the agreed metadata. Raw-slice limits are
divisible by three; positional encoding uses each triplet's middle slice. The
model applies acquisition-consistent augmentation and checkpoint-specific
normalization before the CNN; see [input semantics](modeling.md#three-channel-inputs).
Projected image features receive type and physical-position encodings before
target-specific attention produces twelve study logits. Supervision supplies
twelve targets and observation masks. Keep image padding and absent series
separate from missing target labels.

The model consumes the agreed series inputs and returns `[batch, 12]` logits.
Loss is applied after study aggregation. Raw reports, reference labels, file
paths and Polars operations do not belong in the model's forward pass.
Backbone-specific normalization must accompany the selected checkpoint.

Training and prediction share preprocessing, evaluation series/slice selection
and the PyTorch model. Training uses random slice sampling and
[acquisition-consistent augmentation](modeling.md#ten-epoch-schedule-and-augmentation);
evaluation uses a recorded deterministic policy with augmentation disabled.
The generated submission notebook invokes the packaged prediction command and
attaches the required code and weights instead of maintaining a second image
pipeline. It works offline with explicit input and output locations; follow
[submission requirements](inference-efficiency.md#verified-submission-requirements).

## Iteration speed

The pipeline uses on-the-fly pixel preparation without a persistent image cache. Measure
whether loading keeps up with the GPU before adding further infrastructure.

- **Index once.** Keep a study/series index with physical slice order, geometry,
  relative file locations and decode status. Reuse acquisition fingerprints
  and preparation manifests; do not hash every DICOM before each training run.
- **Select before decoding.** Skip absent types and unselected acquisitions.
  Fixed quantile clipping over each series' sampled slices allows decoding only
  those slices; see the
  [preparation notes](data-preparation.md#intensity-and-geometry) for the
  normalization tradeoff.
- **Prepare pixels in loader workers.** Use float32 for spatial/intensity
  arithmetic and fixed percentile clipping, then produce uint8 images.
  Prepare metadata and label arrays outside the batch loop. Tune worker count
  and prefetching from measured study throughput and memory use.
- **Profile before scaling.** Record data-loading time, studies/second, peak
  memory, GPU waiting and total runtime. Mixed precision and compact batches
  are first training candidates. Caching is a deferred option only if loading
  proves to be a bottleneck; compilation and additional models also need
  measured benefit.

Use progressively larger runs for different questions:

| Run size | Purpose | Interpretation |
|---|---|---|
| A few real studies/batches | Inspect decoded anatomy, batching, loss and runtime after a code change | A development execution; no competitive score claim |
| One fixed fold with a recorded budget | Iterate on throughput and screen an experiment affordably | Development evidence; repeated fold-0 tuning is not independent validation |
| All five fixed folds, complete study predictions | Compare promising changes with the same frozen evaluation labels, budget and seed | Pooled AUC on all 4,407 studies, plus separate gold/report-derived and per-target/fold results |

Partial validation batches cannot produce the complete held-out-fold endpoint.
Do not report a partial/single-class AUC as a full score. Use a fixed checkpoint
rule for comparisons. The existing `scripts/tools/validation.py` remains the owner
of final OOF scoring; reuse its metric logic rather than redefine it.

## Configs and development commands

Use one explicit YAML file per experiment. Copy the baseline for a new idea,
change the relevant settings, and retain the effective config with its outputs.
Keep numerical choices in configs, not scattered constants or notebook cells.
Introduce model, optimizer, sampling and runtime fields when those components
are implemented; there is no configuration inheritance framework.

`src/knee_abnormality/config.py` reads YAML with `yaml.safe_load` and validates
it against nested Pydantic models for experiment identity, paths, labels, and
validation settings.
Unknown or missing keys fail clearly; seeds must be integers in `[0, 2**32)`
and fold selections must be nonempty, distinct nonnegative integers. The models
are frozen after validation. Their nesting matches the YAML, for example
`config.labels.source` and `config.validation.folds`.

Use `pydantic-settings` for argument parsing and configuration in new CLIs,
following [project guidance](../AGENTS.md). The current inspection CLI accepts
only explicit arguments and disables environment, dotenv and secrets sources.
The YAML must be valid on its own; explicit overrides then take precedence and
are validated again. Keep this explicit source policy for experiment commands
so a shell environment cannot silently change an experiment.

Current commands, from the repository root:

```bash
rtk proxy uv run --locked python -m knee_abnormality.config configs/baseline.yaml
rtk proxy uv run --locked python -m knee_abnormality.config configs/baseline.yaml --folds 0,1,2,3,4
```

These print resolved settings as JSON, retaining the YAML's nested sections
and adding `config_file` and `split_dir`. They do not create run artifacts,
train or check dataset availability.
The current config selects `steven_v4`, split `all_5fold_v1`, seed 42 and
**fold 0 for quick development**. Selecting one fold does not change the
saved five-fold assignments. Preparation/training must check requested fold
IDs against `protocol.json` and verify the selected label-source fingerprint.

Configured paths are relative to the YAML file, so moving the checkout does
not require absolute paths in version control. `--data-root` and `--output-root`
override machine-specific storage locations; relative CLI overrides use the
current working directory. `label_manifest` points to the pinned source
manifest. Its `data/...` paths refer to the configured data root when consumed
by label preparation. The audit scripts resolve their
documented data and output arguments from the repository root.

`scripts/pipeline/train.py` and `scripts/pipeline/predict.py` use the same config
loader, with explicit CLI overrides for development limits, fold selection,
input/output locations and resume. Extend the schema alongside a new experimental option.

Launch the workflow from the repository root with
`rtk proxy uv run --locked python -m scripts.pipeline.train` or
`rtk proxy uv run --locked python -m scripts.pipeline.predict`, passing that
command's config/options. Run
`rtk proxy uv sync --locked` during setup to install `knee_abnormality` from
`src/` in editable mode. Entry points and notebooks import shared code from
that package. The `-m scripts...` commands locate entry points from the
repository root. Use the same form
for supporting tools and investigations, for example
`rtk proxy uv run --locked python -m scripts.tools.validation --help` and
`rtk proxy uv run --locked python -m scripts.adhoc.audit_metadata --help`.

## Experiment artifacts and reproducibility

Use one fresh directory per run under `outputs/<run_id>/`. Store the copied
input config, effective settings, source/split/preprocessing identifiers, code revision
and dirty-source snapshot, seed and dependency information. A name or Git SHA
alone does not identify an experiment made with uncommitted changes.

Keep per-fold checkpoints, logs and study predictions under `fold_<k>/`, with
`last.ckpt` for explicit resumption and the recorded selected checkpoint for
comparison. Put combined OOF predictions and metrics at the run root. Start
with local CSV/JSON logging; introduce a tracking service only when useful.
Resuming a run must check its config and input identities; a new experiment
gets a new directory instead of overwriting the previous result.

Best and latest checkpoints have separate callbacks; `last.ckpt` is refreshed
every completed epoch. Checkpoints embed the saved config, target order, fold
and input fingerprints, with the latter also written to `input_identity.json`.
Prediction rejects incompatible input/model settings or target order, and
validation checks the producing fold. Resume also checks numerical training
settings and the original input identities. Legacy checkpoints without those
identities support prediction using their saved config but cannot safely pass
the resume guard. See [implemented corrections](baseline-review.md#implemented-corrections)
for allowed runtime/storage changes and baseline1 compatibility.

Keep reusable metadata and label tables under `data/`, outside run directories.
Preserve original gold/silver
tables, use gold only for eligible training rows, and exclude held-out report
groups according to the saved manifest. The current split assigns every study
to exactly one validation fold; its complete OOF table has 4,407 rows. The
historical gold-only split's shared silver training pool cannot supply OOF
predictions. See [validation](validation.md) and
[the label audit](public-label-audit.md) for the full protocol.

## Current scope and next execution

The config, fold-aware labels, on-the-fly image preparation, study dataset,
model, training, prediction, export and remote-submission orchestration are
implemented. Existing frozen folds and source files are preserved. Baseline1
is the historical gold-only run; baseline3 is the current all-study fold-0 run.
Their results are recorded in [modeling](modeling.md#implemented-baseline)
and the [baseline1 review](baseline-review.md). Track the remote submission and
remaining folds in [ongoing work](ongoing.md). Use bounded real-data executions
and inspect their artifacts.
