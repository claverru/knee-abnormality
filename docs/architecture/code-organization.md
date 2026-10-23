# Competition repository architecture

Decision recorded on **2026-09-10**: optimize this repository for quick, traceable
competition experiments. Launch commands from the repository root. Separate
the maintained workflow in `scripts/pipeline/`, reusable utilities in
`scripts/tools/`, and one-off work in `scripts/adhoc/`.
Keep shared modules in `src/knee_abnormality/`, YAML configs validated with
Pydantic in `configs/`, and prepare image pixels on the fly for v1. `uv` manages
dependencies and installs the local package in editable mode through setuptools.

The maintained training config is `configs/resnet50.yaml`, validated against the Pydantic schema in
`config.py`; the per-family recipes of the final phase (`configs/<family>-*.yaml`: ResNet-50,
ConvNeXt, NextViT, FocalNet, EfficientViT and MambaVision) carry the same structure. The model
builds three-channel adjacent slice triplets from a timm encoder, pools usable acquisitions within
each plane–contrast slot and applies target-specific attention MIL across every visually valid
spatial token; the [current model](../modeling/current-model.md) records the choices and results,
the [model lineage](../history/model-lineage.md) the earlier baselines and the
[validation protocol](../validation/validation-protocol.md) the all-study five-fold split.

Training uses Lightning's built-in training, precision, optimization and callback facilities
(`LightningModule`, `LightningDataModule`, `Trainer`, `ModelCheckpoint`, `EarlyStopping`,
`CSVLogger`); timm's public APIs select the encoder and weights.

## Where code belongs

Open known script paths directly or start at the [script index](../../scripts/INDEX.md)
before adding code. The repository's
[scripts-maintainer skill](../../.agents/skills/scripts-maintainer/SKILL.md)
defines discovery, CLI conventions and index maintenance. A script's `--help` states its
purpose and options; detailed usage and findings stay in their topic docs.

Training and prediction entry points live in `scripts/pipeline/`.
Shared setup functions handle metadata/label preparation; a separate
`prepare.py` command is unnecessary. Reusable
support tools live in `scripts/tools/`; one-off investigations belong in
`scripts/adhoc/`. Shared modules belong in `src/knee_abnormality/` and use
imports such as `from knee_abnormality.config import load_config`. The
maintained pipeline never imports from `scripts/tools/` or `scripts/adhoc/`;
move behavior it shares with them into the package. Add a shared module when
real code needs it, and split a growing file by responsibility when that makes
editing easier.

### Package layout

`src/knee_abnormality/` has six subpackages (`data`, `modeling`, `training`, `evaluation`,
`inference`, `autoresearch`) with empty `__init__.py` files; callers import each owning module
directly. Top-level modules hold the cross-cutting pieces:

| Module | Responsibility |
|---|---|
| `config.py` | Validate YAML with Pydantic, resolve paths, merge dotted-flag overrides (`parse_experiment_arguments`) and print effective settings (`python -m knee_abnormality.config`); fixed recipe constants (`SLICES_PER_INPUT`), the inference defaults (2× test images, four studies per GPU, eight loader workers, eight I/O threads), the `Device` and `RuntimePrecision` choices and the `STUDY_HEADERS` table path |
| `cli.py` | The single `CLISettings` base class (explicit-only `pydantic-settings` parsing, boolean flags, usage and exit code 2 on argument errors) and the `describe`/`command` helpers for CLI and subcommand descriptions |
| `kaggle.py` | The competition name, the `KaggleId` type, the Kaggle Python and wheel pins for export, and the option models shared by Kaggle commands: `NotebookRun` (accelerator, time limit, polling) and `SubmissionRequest` (bundle, dataset, notebook, record, label, sequence) |
| `provenance.py` | File hashing, copying analysis sources to `source/<repository-relative path>`, and the Git revision and status recorded by training and export |
| `identity.py` | The preprocessing identity: fixed sampling, reflection, reliability and spatial-token semantics and a digest of the explicit `PREPROCESSING_SOURCES` list; required at prediction, validation, resume and export |
| `profiling.py` | Select bounded profile datasets and record reproducible profiling artifacts |

| Subpackage | Modules and responsibility |
|---|---|
| `data/` | `sources.py`: torch-free raw-directory discovery, series index and prediction study lists. `splits.py`: the study-ID column and target threshold, frozen split protocol checks and frozen evaluation targets. `labels.py`: pinned label-source loading, gold/silver training targets and the fold's training and validation frames. `dataset.py`: study/acquisition selection, preparation, collation and data loaders |
| `data/preprocessing/` | `dicom.py` parses headers and reads frames with the pydicom fallback; `series.py` indexes metadata-only slice records and centers physical positions; `geometry.py` validates series center lines and solves the plane-balanced study reference; `laterality.py` decides the imaged knee side; `filtering.py` applies the series exclusions; `sampling.py` indexes runs and selects triplets; `spatial.py` builds the cached resampling plans and the scipy resample; `intensity.py` the exact quantile ranks, bounds and normalization; `preparation.py` reads selected records into raw support rows and keeps decoder injection for profiling; `device.py` resamples, normalizes and gathers the uint8 inputs on CPU in loader workers or on the GPU after transfer |
| `modeling/` | `model.py`: timm backbone, study reflection and triplet augmentation, spatial tokens with additive embeddings and target-specific attention MIL. `spatial_tokens.py`: verified backbone anchors, visual validity, 2D coordinates and metadata embeddings. `augmentation.py`: reflection plus shared affine, gamma/contrast and noise transforms. `backbone_stages.py`: one stage adapter per backbone family (ResNet, ConvNeXt, FocalNet, EfficientViT, MambaVision and the others the model reads stage by stage). `staged_backbones.py`: the MambaVision-T wrapper. `families/nextvit.py`: the NextViT wrapper. `plane_readout.py`: the plane-contrast readout. `view_windows.py`: view windows and crop placement. `radimagenet.py`: RadImageNet weights. `anatomy_classes.py`: the bone and soft-tissue segmentation classes |
| `training/` | `lightning.py`: `KneeDataModule`, `KneeModule`, the optimizer groups and fixed BF16 precision. `checkpoints.py`: resume validation, checkpoint relocation and training input identities. `anatomy.py`: anatomy-target loss. `cuda.py`: CUDA allocator setup |
| `evaluation/` | `metrics.py`: per-target, macro, subgroup, bootstrap and compartment metrics. `scoring.py`: out-of-fold scoring against the frozen split. `predictions.py`: validated study-level prediction tables |
| `inference/` | `engine.py`: prediction datasets, device and precision policy, batch transfer, FP16 backbone chunks and the prediction loop. `checkpoint.py`: the single checkpoint reader, checkpoint validation and strict model construction. `processes.py`: serialized worker requests and one prediction process per visible GPU, without importing PyTorch. `bundle.py`: bundle schema, file inventory, verification and Kaggle data-root discovery. `reflection.py`: mirror TTA transform. `buffers.py`: pooled shared-memory pixel buffers. `compile_cache.py` and `startup_pages.py`: the shipped compile cache and the Kaggle startup page pre-read |
| `autoresearch/` | The autoresearch engine: `campaign/` (definitions, gates, rules, reports), `supervisor/` (scheduling, experiments, monitoring, prompts), `runtime/` (GPU, jobs, isolation, providers, remote machine, storage, usage), `scoring/`, `topics/` and `reporting/`, driven by `cli.py` and `watchdog.py` |

`KneeAbnormalityModel` in `modeling/model.py` stayed one `nn.Module` of about 2,000 lines whose methods share state through attributes; only the backbone stages (`backbone_stages.py`, `staged_backbones.py`, `families/`) and view windows (`view_windows.py`) were split out (2026-10-14). The rest falls into four groups (line counts of 2026-10-14): anatomical frame (`frame_rotation`, `rotation_angles`, `study_frames`, `anatomical_frames`, `frame`, `position_code`, ~270 lines), routing and regions (`routing_widths`, `position_priors`, `side_probability`, `_regions`, `route_sections`, routing weights, `_attention_pool`, ~390), soft-tissue routing (`soft_tissue_routing_weights`, `_soft_tissue_routing_weights`, `_soft_tissue_tokens`, `_soft_tissue_pool`, ~200) and image encoding (`_encode_images`, `_augment_images`, `_mix_contrast`, `_effective_presence`, `first_look_pose`, `_encode_crops`, ~600). Splitting them into submodules was proposed and not done: parameter names would gain a prefix (old checkpoints need a key map or their `source_snapshot.zip`), each submodule must be built where its parameters are made now to keep initial weights identical, and the hashed sources change the preprocessing identity.

Everything else in the repository:

| Location | Responsibility | Status |
|---|---|---|
| `configs/*.yaml` | `resnet50.yaml` is the baseline config; the other files are per-family and per-generation recipes (teachers, students, pretraining) | Implemented |
| `configs/labels/` | Pinned label-source manifests read by training and label tools | Implemented |
| `scripts/INDEX.md`, `scripts/*/INDEX.md` | Script discovery; the pipeline and tools indexes are generated by `scripts.tools.script_index` | Implemented |
| `scripts/tools/` | Reusable acquisition, validation, submission, profiling and repository utilities | Implemented; see the [tools index](../../scripts/tools/INDEX.md) |
| `scripts/adhoc/` | Frozen one-off investigations kept as the record behind their findings | Implemented; see the [ad-hoc index](../../scripts/adhoc/INDEX.md) |
| `scripts/pipeline/train.py` | Training entry point: fold setup, provenance and source snapshot, Lightning trainer and callbacks, selected-checkpoint validation predictions and OOF scoring | Implemented; earlier checkpoint corrections are recorded in the [baseline review](../history/baseline1-review.md#implemented-corrections) |
| `scripts/pipeline/predict.py` | Load saved model/config, partition studies across GPU worker processes, write validation/submission tables | Implemented |
| `scripts/pipeline/predict_worker.py` | Build the worker's data in a background thread while the main thread loads the model and runs the FP16 chunk warm-up, then predict the assigned studies | Implemented |
| `scripts/pipeline/export.py` | Build an atomic, hash-inventoried checkpoint/code/wheel bundle with the checked-in `scripts/pipeline/inference.ipynb` | Implemented |
| `scripts/pipeline/submit.py` | Version the private bundle, run the offline notebook, verify output and submit the exact successful version | Implemented; requires explicit remote IDs and a ready bundle |
| `justfile` | Expose the maintained setup, train, predict, export, infer, submit and autoresearch commands without embedding Python orchestration ([recipes](../inference/justfile-recipes.md)) | Implemented |
| `docs/` | Decisions, procedures, evidence, and work tracking | Implemented |
| `data/`, `outputs/` | Downloaded inputs, prepared metadata/labels and generated experiment artifacts | Local and Git-ignored |

The independent GPU coordinator and worker ([independent GPU processes](../inference/independent-gpu-inference-processes.md))
originated on branch `1C`; the 0.936 submission's [runtime provenance](../inference/submissions/early-submission-details.md)
preserves the reflection model and frozen notebook. Historical sharing model and audit sources are
[archived separately](../history/sources/INDEX.md).

`data/dataset.py` owns worker-local index lifetime, study/acquisition selection
and the study-reference cache. Slice records retain
native XYZ tuples and share repeated orientation/spacing/layout descriptors.
Native-pose validity, study-reference acceptance and sampling projection remain
separate decisions; the prepared series' geometry validity, now read only by previews, requires accepted series geometry and physical sampling, which needs an accepted study reference.
DICOM byte interpretation stays in `data/preprocessing/dicom.py`; spatial and intensity
modules perform no file I/O. Descriptors retain scalar metadata only, while
selected pixels are decoded afresh on every load. Reader dispatch is fixed,
with no configuration flag or persistent pixel/header cache. See the
[integration evidence](../data-preparation/production-dicom-integration.md) for
layout limits, equivalence checks, performance and per-worker memory bounds.

Keep experiment logic explicit. A new backbone family needs a stage adapter in
`src/knee_abnormality/modeling/backbone_stages.py`; a pooling alternative can be a
small function in `modeling/model.py`. Neither needs a parallel copy of the training loop. Put runnable
behavior behind a main guard so scripts can also expose useful functions.

### Maintained workflow versus ad-hoc work

`scripts/pipeline/` contains the commands used to reproduce a baseline and prepare a
submission. Improve these commands as the solution evolves; avoid accumulating
variants such as `train_v2.py` or `predict_final_fixed.py`. Select experiment
variants through configs and shared model code.

`scripts/tools/` contains reusable supporting commands: `download_data.py`
handles acquisition and resumption; `validation.py` creates fixed folds and
scores OOF predictions across experiments through the shared
`knee_abnormality.evaluation.scoring`. Keep their arguments and outputs
stable as the pipeline develops. Extract shared functions into
`src/knee_abnormality/` when the training or preparation code needs them.

Operational traps: launching a training with `KNEE_AUTORESEARCH_LIMITS` set makes it crash on `KeyError: 'input_slices'`; only `scripts/tools/preflight_training.py` may see that variable. `pgrep -f`/`pkill -f` patterns also match the ssh command line that runs them remotely.

`scripts/adhoc/` contains focused investigations with descriptive names. The
existing metadata and public-label audits reproduce the evidence in the topic
docs; the environment probe records setup checks. These scripts
can call shared code, but the maintained workflow must not depend on them.
When an investigation becomes part of the solution, move its reusable behavior
into the relevant shared module or maintained entry point. Record useful
findings in the topic docs and keep generated artifacts under `data/` or
`outputs/`, not beside the script.
