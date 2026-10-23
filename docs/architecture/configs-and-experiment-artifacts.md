# Configs and experiment artifacts

## Configs and development commands

Use one explicit YAML file per experiment. Copy `configs/resnet50.yaml` for a new idea,
change the relevant settings, and retain the effective config with its outputs.
Keep numerical choices that experiments vary in configs, not scattered constants or
notebook cells. Accepted fixed choices are code constants instead of single-value fields:
three slices per input (`SLICES_PER_INPUT`), the 256-wide projection and 128-wide attention
scorer (`modeling/model.py`), BF16 mixed precision on one automatically selected device
(`training/lightning.py`), physical millimeter positions centered on each acquisition, and
the inference defaults of 2× test images, four studies per GPU and eight loader workers
(`config.py`). Introduce a field only for a setting that is actually varied; there is no
configuration inheritance framework.

`src/knee_abnormality/config.py` reads YAML with `yaml.safe_load` and validates
it against nested Pydantic models for experiment identity, paths, labels, and
validation settings.
Unknown or missing keys fail clearly, so configs and checkpoints that still carry removed
fields must be used with their archived source; seeds must be integers in `[0, 2**32)`
and fold selections must be nonempty, distinct nonnegative integers.
`model.pretrained_weights: radimagenet` requires a backbone listed in `RADIMAGENET_WEIGHTS`
(`resnet50.a1_in1k`), and
`model.slice_channels: repeated_middle` feeds each triplet's middle slice on all three channels
([RadImageNet backbone](../modeling/backbones/resnet-radimagenet-backbones.md)). The models
are frozen after validation. Their nesting matches the YAML, for example
`config.labels.source` and `config.validation.folds`.

Use `pydantic-settings` for argument parsing and configuration in new CLIs,
following [project guidance](../../AGENTS.md). The shared `CLISettings` base in `cli.py` accepts
only explicit arguments and disables environment, dotenv and secrets sources, so a shell
environment cannot silently change an experiment. Argument errors print usage and exit
with code 2; any other failure raises with its traceback. Booleans are flags
(`--reflection-tta`, `--no-reflection-tta`). Each CLI states its purpose with
`model_config = describe(...)` and every field has a `description`, so `--help` documents
it; related actions are `CliSubCommand` subcommands run by `CliApp.run`.

`train.py` and the config inspector accept any experiment field as a dotted flag, such as
`--trainer.class-balance-exponent 0.5`, `--seed null`, `--no-model.pretrained` or
`--validation.folds '[0,1,2,3,4]'` (tuple and range fields take JSON). The overrides are
merged into the YAML before validation, so the YAML itself need not change for a one-change
run. Path overrides (`--paths.data-root`, `--paths.output-root`,
`--paths.label-manifest`) resolve against the current directory, while YAML paths resolve
against the YAML's directory. Training records the overrides in `provenance.json`;
`input_config.yaml` stays the unmodified YAML and `effective_config.json` the result.

Commands, from the repository root:

```bash
uv run --locked python -m knee_abnormality.config configs/resnet50.yaml
uv run --locked python -m knee_abnormality.config configs/resnet50.yaml --validation.folds '[0,1,2,3,4]'
```

These print resolved settings as JSON, retaining the YAML's nested sections
and adding `config_file`, `overrides` and `split_dir`. They do not create run artifacts,
train or check dataset availability.

### Training seeds and fold selection

`configs/resnet50.yaml`, the last baseline config, selects the `rebuild_state_v2` labels
([label rebuild](../public-labels/label-rebuild/INDEX.md)), split
[`all_5fold_scanner_v2`](../validation/validation-protocol.md), **fold 2** and seed **1110484336**.
Selecting one fold does not change the saved five-fold assignments. Autoresearch candidates
inherited their incumbent's saved training seed; preparation, review and launch could not select a
different seed. A null or missing seed in a manually supplied config draws a fresh 32-bit seed.
Effective configs and checkpoint hyperparameters retain the actual seed, and `--resume` adopts the
checkpoint's original seed. Matched seeds do not guarantee deterministic training.
Preparation/training verifies requested folds and the label-source fingerprint.

### Path resolution

Configured paths are relative to the YAML file, so moving the checkout does
not require absolute paths in version control. `--paths.data-root` and
`--paths.output-root` override machine-specific storage locations; relative CLI overrides
use the current working directory. `label_manifest` points to a pinned source
manifest in `configs/labels/` (`rebuild-state-v2-label-sources.json` in `configs/resnet50.yaml`;
the v3 tables have `rebuild-state-v3-label-sources.json`).

Paths recorded inside a label manifest are written relative to the project root and must
lie under `data/`, or be absolute. The label loader maps the leading `data/` to the
configured `paths.data_root` and rejects any other relative path, so the resolution does not
depend on where the manifest itself lives and follows a relocated or mounted data root.
Each table's SHA-256 is checked before use. The audit scripts resolve their
documented data and output arguments from the repository root.

### Training and prediction entry points

`scripts/pipeline/train.py` and `scripts/pipeline/predict.py` use the same config
loader, with explicit CLI overrides for development limits, fold selection,
input/output locations and resume. Extend the schema alongside a new experimental option.

Launch the workflow from the repository root with
`uv run --locked python -m scripts.pipeline.train` or
`uv run --locked python -m scripts.pipeline.predict`, passing that
command's config/options. Run
`uv sync --locked` during setup to install `knee_abnormality` from
`src/` in editable mode. Entry points and notebooks import shared code from
that package. The `-m scripts...` commands locate entry points from the
repository root. Use the same form
for supporting tools and investigations, for example
`uv run --locked python -m scripts.tools.validation --help` and
`uv run --locked python -m scripts.adhoc.audit_metadata --help`.

## Experiment artifacts and reproducibility

`outputs/` is Git-ignored and grouped by artifact kind:

| Directory | Contents |
|---|---|
| `training/` | Runs and overnight queues; `paths.output_root` in `configs/resnet50.yaml` points here |
| `analysis/`, `previews/`, `profiling/` | Investigation artifacts; the plotting tool's default PNG goes to `analysis/` |
| `submission/` | Inference bundles (`exports/`), their `runtimes/`, workflow `records/`, `verification/`, `prep/` scratch and offline codec wheels (`dependencies/`) |
| `research/` | External review material and scouting sweeps |
| `archive/` | Maintenance, planning and source snapshots |

Artifacts written before the 2026-09-16 reorganisation keep their original absolute paths
(`selected_checkpoint.json`, `effective_config.json`, bundle manifests, runtime provenance,
submission records, analysis comparisons); `outputs/archive/move-map.tsv` maps them to the
current locations. Only the `00-baseline`, `01-sigma30` and `03-min-lr-0.1` overnight runs and
`reflection-sigma40` kept a checkpoint, the best one; the other runs kept their records and
predictions, and stale bundles kept everything except `model.ckpt` and `wheels/`. A saved
configuration records an absolute `output_root`, so pass `--paths.output-root outputs/training` when
training from one saved before the reorganisation.

Use one fresh directory per run under `outputs/training/<run_id>/`. Store the copied
input config, effective settings, source/split/preprocessing identifiers, code revision
and dirty-source snapshot, seed and dependency information. A name or Git SHA
alone does not identify an experiment made with uncommitted changes.

Profiling and analysis snapshots preserve repository-relative paths under `source/`
(`provenance.copy_sources`).
External inputs, including configs, retain their resolved absolute path without
the filesystem root under `source/external/`; `/tmp/profile.yaml` becomes
`source/external/tmp/profile.yaml`. The manifest hashes these files too.

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
identities require their archived source/bundle, even for prediction with the
same config. Every new checkpoint and input identity records the preprocessing
algorithm (`ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`), the
reflection, sampling, series-filtering and spatial-token identities, fixed reliability values, the
selected backbone's spatial mapping, and a stable digest of the package-relative sources listed in
`identity.PREPROCESSING_SOURCES`: `data/dataset.py`, the eight `data/preprocessing/` modules
(including `filtering.py`), `modeling/augmentation.py`, `modeling/model.py` and
`modeling/spatial_tokens.py`. The `v10` label marks the 2026-09-17 series exclusions and fixed
source ranking; `v9` marked the 2026-09-16 package restructure. Checkpoints with any earlier
identity need their archived sources. The robust-centre code of 2026-09-17 changed the source
digest but left every other identity field unchanged. The three ranked-filtered checkpoints
(`ranked-filtered-seed42`, `-seed43`, `-seed44`) therefore no longer validate against the working
tree and need their archived sources (`source_snapshot.zip`). Their Kaggle bundles carry their
own code and are unaffected. Prediction does not compare the configured seed, because
evaluation sampling uses none; resume still requires the saved seed. Export validates the actual
copied sources; bundled prediction recomputes the identity without training data.
See [implemented corrections](../history/baseline1-review.md#implemented-corrections)
for allowed runtime/storage changes and baseline1 compatibility.

**Study slice total** (`data.study_slice_total`, required, **26 in `configs/resnet50.yaml`**, set on 2026-09-18) raises a study whose per-type budgets sum below that many training slices to
it, scaling the slots it has in proportion to their configured budgets, largest-remainder rounded
and capped by each slot's capacity; a study already at or above the floor is untouched. On fold 0
it lifts the ~29% of studies missing two types from about 21 triplets to 26 and leaves the rest
alone, so the mean rises from 25.4 to 26.7. It scales with the image multiplier, applies in
training, validation and inference, and changes the preprocessing identity, so it takes effect for
fresh runs only. The rejected **repeated-series share** (`data.repeated_series_share`) was removed
on 2026-09-25 ([overnight](../validation/results/campaigns-and-queues/2026-09-18-overnight-results.md)).

**Count-normalised attention** (`model.count_normalized_attention`) and the **robust study centre**
(`data.robust_center_outlier_mm`) were removed as config options (the latter on 2026-09-18, the
former in commit `665af0e`); configs and checkpoints that carry them fail validation and need their archived sources.

**Removed 2026-10-14** (no recipe used them; code at `7770ce9`): weight averaging
(`optimizer.weight_averaging`, its `live_state_dict`, `val_live_*` metrics and
`batch_norm_refresh_seconds`), the report-attribute heads (`trainer.report_attribute_loss_weight`,
`paths.report_attribute_manifest`), the scanner confusion head (`trainer.scanner_confusion_weight`),
`model.dropout`, `model.series_dropout`, `model.checkpoint_backbone`, `model.backbone_eval`,
`validation.unused_folds`, and the switches that every recipe had on (`model.plane_readout`,
`view_label_bias`, `anatomical_frame`, `meniscus_map`, `soft_tissue_routing`), whose behaviour is now
fixed. `paths.anatomy_targets`, `paths.meniscus_targets` and both loss weights are required. The
pre-clip total gradient norm is still logged as `grad_norm`.

Keep reusable metadata and label tables under `data/`, outside run directories.
Preserve original gold/silver
tables, use gold only for eligible training rows, and exclude held-out report
groups according to the saved manifest. The final split assigns every study
to exactly one validation fold; its complete OOF table has 4,407 rows. The
historical gold-only split's shared silver training pool cannot supply OOF
predictions. See [validation](../validation/INDEX.md) and
[the label audit](../public-labels/INDEX.md) for the full protocol.

Cleanups of checkpoints and caches are recorded in the [disk cleanup log](disk-cleanup-log.md).
