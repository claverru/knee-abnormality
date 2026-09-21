# Configs and experiment artifacts

## Configs and development commands

Use one explicit YAML file per experiment. Copy `configs/training.yaml` for a new idea,
change the relevant settings, and retain the effective config with its outputs.
Keep numerical choices that experiments vary in configs, not scattered constants or
notebook cells. Accepted fixed choices are code constants instead of single-value fields:
three slices per input (`SLICES_PER_INPUT`), the 256-wide projection and 128-wide attention
scorer (`modeling/model.py`), BF16 mixed precision on one automatically selected device
(`training/lightning.py`), physical millimeter positions centered on each acquisition, and
the inference defaults of 2× test images, four studies per GPU and sixteen loader workers
(`config.py`). Introduce a field only for a setting that is actually varied; there is no
configuration inheritance framework.

`src/knee_abnormality/config.py` reads YAML with `yaml.safe_load` and validates
it against nested Pydantic models for experiment identity, paths, labels, and
validation settings.
Unknown or missing keys fail clearly, so configs and checkpoints that still carry removed
fields must be used with their archived source; seeds must be integers in `[0, 2**32)`
and fold selections must be nonempty, distinct nonnegative integers.
`model.checkpoint_backbone` requires `model.backbone_eval: true`, because activation
checkpointing would otherwise rerun training-mode BatchNorm during recomputation. The models
are frozen after validation. Their nesting matches the YAML, for example
`config.labels.source` and `config.validation.folds`.

Use `pydantic-settings` for argument parsing and configuration in new CLIs,
following [project guidance](../../AGENTS.md). The shared `CLISettings` base in `cli.py` accepts
only explicit arguments, disables environment, dotenv and secrets sources, and raises
`SettingsError` so each script reports argument errors through its own handler.
The YAML must be valid on its own; explicit overrides then take precedence and
are validated again. Keep this explicit source policy for experiment commands
so a shell environment cannot silently change an experiment.

Current commands, from the repository root:

```bash
rtk proxy uv run --locked python -m knee_abnormality.config configs/training.yaml
rtk proxy uv run --locked python -m knee_abnormality.config configs/training.yaml --folds 0,1,2,3,4
```

These print resolved settings as JSON, retaining the YAML's nested sections
and adding `config_file` and `split_dir`. They do not create run artifacts,
train or check dataset availability.

### Training seeds and fold selection

The current config selects `steven_v4`, split `all_5fold_v3` and
**fold 0 for quick development**. Selecting one fold does not change the
saved five-fold assignments.

`configs/training.yaml` pins **2263738815**, matching the adopted learned-strength
reference. Autoresearch candidates inherit their current incumbent's saved training
seed; preparation, review and launch cannot select a different seed. A null or
missing seed in an explicitly supplied manual config still draws a fresh 32-bit
seed, but this is not the current research comparison policy. Effective configs
and checkpoint hyperparameters retain the actual seed. A `--resume` adopts its
checkpoint's original seed. Matched seeds do not guarantee deterministic training.
Preparation/training verifies requested folds and the label-source fingerprint.

### Path resolution

Configured paths are relative to the YAML file, so moving the checkout does
not require absolute paths in version control. `--data-root` and `--output-root`
override machine-specific storage locations; relative CLI overrides use the
current working directory. `label_manifest` points to a pinned source
manifest in `configs/labels/` (`public-label-sources.json` for training).

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

`outputs/` is Git-ignored and grouped by artifact kind:

| Directory | Contents |
|---|---|
| `training/` | Runs and overnight queues; `paths.output_root` in `configs/training.yaml` points here |
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
configuration records an absolute `output_root`, so pass `--output-root outputs/training` when
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

**Count-normalised attention** (`model.count_normalized_attention`, required, `false` in
`configs/training.yaml`) subtracts `log(that acquisition's valid tokens in the study)` from every
token's attention score before the masked softmax in `_attention_pool`, so a series' attention
mass no longer grows with the number of triplets it contributed. It applies in training,
validation and inference alike, and `false` is bit-identical to the earlier code. Because it lives
in `modeling/model.py`, which is in `identity.PREPROCESSING_SOURCES`, it changes the preprocessing
source digest and is a fresh-runs-only change; earlier checkpoints keep validating against their
archived sources.

**Study triplet total** (`data.study_triplet_total`, optional, **26 in `configs/training.yaml`
since 2026-09-18**) raises a study whose per-type budgets sum below that many training triplets to
it, scaling the slots it has in proportion to their configured budgets, largest-remainder rounded
and capped by each slot's capacity; a study already at or above the floor is untouched. On fold 0
it lifts the ~29% of studies missing two types from about 21 triplets to 26 and leaves the rest
alone, so the mean rises from 25.4 to 26.7. It scales with the image multiplier, applies in
training, validation and inference, and changes the preprocessing identity, so it takes effect for
fresh runs only. **Repeated-series share** (`data.repeated_series_share`, optional, unset) is the
rejected counterpart, kept only because its run exists
([overnight](../backlog/overnight.md#already-measured)).

**Robust study centre** was removed on 2026-09-18; `data.robust_center_outlier_mm` no longer
exists. Configs and checkpoints that carry it fail validation and need their archived sources, as
do the four runs that used it.

**Weight averaging** (`optimizer.weight_averaging`, required, `null` by default) is a
discriminated union: `null`, `{method: ema, decay}` or `{method: swa, epochs}`, one method per
run; a config validator rejects SWA `epochs` greater than `trainer.max_epochs` minus
`optimizer.warmup_epochs`. See the
[schedule doc](../modeling/schedule-and-augmentation.md#weight-averaging) for the averaging
semantics. When set, `KneeModule` keeps a deep-copied averaged model alongside the live one and
rewrites checkpoints on save: the saved `state_dict`'s `model.*` keys hold the averaged weights
(what selection, prediction, export and `compare_fold_runs` use, unchanged), and a new top-level
`live_state_dict` key holds the live, unaveraged weights, restored to `model.*` on resume and
dropped by export's checkpoint allowlist. Validation scores both models on the same batches every
epoch: `val_macro_auc`, `val_soft_target_bce` and `val_gold_macro_auc` always score the saved
(averaged) weights, while `val_live_macro_auc`, `val_live_soft_target_bce` and
`val_live_gold_macro_auc` score the live ones. The per-epoch files gain `epochs/live_metrics_epoch_*.json`
and `epochs/live_validation_epoch_*.csv` alongside the existing (unprefixed) ones. With
`weight_averaging: null`, none of this exists: no `live_state_dict`, no `val_live_*` metrics, no
`live_*` files. Independent of weight averaging, `KneeModule` always logs the pre-clip total
gradient norm as the `grad_norm` step metric, and `timing.json` always records
`batch_norm_refresh_seconds` (0 unless a SWA run actually refreshed BatchNorm statistics that
epoch).

Keep reusable metadata and label tables under `data/`, outside run directories.
Preserve original gold/silver
tables, use gold only for eligible training rows, and exclude held-out report
groups according to the saved manifest. The current split assigns every study
to exactly one validation fold; its complete OOF table has 4,407 rows. The
historical gold-only split's shared silver training pool cannot supply OOF
predictions. See [validation](../validation/INDEX.md) and
[the label audit](../public-labels/INDEX.md) for the full protocol.

### Checkpoint pruning

On 2026-09-20, user-authorized cleanup removed **249 obsolete `.ckpt` files (33.412 GiB)**
from completed superseded/rejected experiments, unused resume checkpoints and stale submission
exports. The [cleanup manifest](../../outputs/archive/maintenance/checkpoint-pruning-20260920T191234Z/manifest.json)
lists every removed and retained path, reason, size and file identity. It records 48 retained
checkpoints (4.381 GiB); all other artifacts, including predictions, metrics, configs, source
snapshots and submission manifests, were preserved.

Retained weights cover the current control B, supporting control A and sealed historical baseline;
the best score/efficiency bundles and their selected source models; pending submissions/loading
benchmarks; attention-temperature cache parents; and the held unevaluated plane-coupling candidate
with its parent. Retaining that parent's weights does not restore the withdrawn stratified-sampler
acceptance. Both campaign states were stopped and unchanged by cleanup.

Historical `selected_checkpoint.json` and bundle manifests retain their original provenance paths;
they do not establish that pruned weights are still available. Consult the cleanup manifest before
attempting to resume, re-export or rerun inference on a historical experiment. Its saved predictions
and metrics remain available for comparisons. Linux free space increased by 33.412 GiB; Windows C:
free space remained about 52 GiB because the WSL virtual disk was not compacted.

## Current scope and next execution

The config, fold-aware labels, on-the-fly image preparation, study dataset,
model, training, prediction, export and remote-submission orchestration are
implemented. Existing frozen folds and source files are preserved. The
[current model](../modeling/current-model.md) records the `00-baseline` fold-0 reference;
earlier baselines, including the historical gold-only baseline1, are in the
[model lineage](../history/model-lineage.md) and the
[baseline1 review](../history/baseline1-review.md). Track the next runs in
[ongoing work](../ongoing/INDEX.md); remaining folds are deferred to the
[final phase](../backlog/endgame.md#ensembling). Use bounded real-data
executions and inspect their artifacts.
