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
checkpointing would otherwise rerun training-mode BatchNorm during recomputation.
`model.pretrained_weights: radimagenet` requires a backbone listed in `RADIMAGENET_WEIGHTS`
(`resnet50.tv_in1k`, `resnet18.a1_in1k`, `resnet10t.c3_in1k`), and
`model.slice_channels: repeated_middle` feeds each triplet's middle slice on all three channels
([RadImageNet backbone](../ongoing/radimagenet-backbone.md)). The models
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

Current commands, from the repository root:

```bash
rtk proxy uv run --locked python -m knee_abnormality.config configs/training.yaml
rtk proxy uv run --locked python -m knee_abnormality.config configs/training.yaml --validation.folds '[0,1,2,3,4]'
```

These print resolved settings as JSON, retaining the YAML's nested sections
and adding `config_file`, `overrides` and `split_dir`. They do not create run artifacts,
train or check dataset availability.

### Training seeds and fold selection

The current config selects [`combined_v1`](../public-labels/combined-labels.md), split
[`all_5fold_scanner_v1`](../validation/validation-strategy.md) and
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
not require absolute paths in version control. `--paths.data-root` and
`--paths.output-root` override machine-specific storage locations; relative CLI overrides
use the current working directory. `label_manifest` points to a pinned source
manifest in `configs/labels/` (`combined-label-sources.json` for training; it lists both
`combined_v1` and `steven_v4`).

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

**Count-normalised attention** (`model.count_normalized_attention`, required, `false` in
`configs/training.yaml`) subtracts `log(that acquisition's valid tokens in the study)` from every
token's attention score before the masked softmax in `_attention_pool`, so a series' attention
mass no longer grows with the number of triplets it contributed. It applies in training,
validation and inference alike, and `false` is bit-identical to the earlier code. Because it lives
in `modeling/model.py`, which is in `identity.PREPROCESSING_SOURCES`, it changes the preprocessing
source digest and is a fresh-runs-only change; earlier checkpoints keep validating against their
archived sources.

**Study slice total** (`data.study_slice_total`, required, **26 in `configs/training.yaml`
since 2026-09-18**) raises a study whose per-type budgets sum below that many training slices to
it, scaling the slots it has in proportion to their configured budgets, largest-remainder rounded
and capped by each slot's capacity; a study already at or above the floor is untouched. On fold 0
it lifts the ~29% of studies missing two types from about 21 triplets to 26 and leaves the rest
alone, so the mean rises from 25.4 to 26.7. It scales with the image multiplier, applies in
training, validation and inference, and changes the preprocessing identity, so it takes effect for
fresh runs only. The rejected **repeated-series share** (`data.repeated_series_share`) was removed
on 2026-09-25 ([overnight](../validation/results/2026-09-18-overnight-results.md)).

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
epoch: `val_macro_auc`, `val_soft_macro_auc`, `val_soft_target_bce` and `val_gold_macro_auc`
always score the saved (averaged) weights, while `val_live_macro_auc`, `val_live_soft_macro_auc`,
`val_live_soft_target_bce` and `val_live_gold_macro_auc` score the live ones. The per-epoch files gain `epochs/live_metrics_epoch_*.json`
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

Cleanups run through the [disk-cleanup skill](../../.agents/skills/disk-cleanup/SKILL.md), which
holds the deletion rules, and
[`free_disk`](../../scripts/tools/free_disk.py), which applies them and writes the manifest under
`outputs/archive/maintenance/disk-cleanup-<UTC>/`. Record each run below.

**2026-10-02:** the first run of the skill removed **180 entries (25.789 GiB)**
([manifest](../../outputs/archive/maintenance/disk-cleanup-20261002T171047Z/manifest.json)); free
space rose by 24.74 GiB while a training run was writing, and a separate `uv cache prune` removed
1.4 GiB more. 81 `.ckpt` files (9.262 GiB) remain under `outputs/`.

- **Checkpoints (16.606 GiB):** 50 `last.ckpt` of finished runs (8.889 GiB) and 42 selected
  checkpoints of recorded runs named neither in `docs/ongoing/` nor by a submitted bundle
  (7.717 GiB): the 28 remaining `smart-crops` and 11 `mixed-field` experiments,
  `plane-contrast-readouts`, `refined-rn50-scaled-backbone-lr` and `rn18-stochastic-pooling-g05`.
  The first two were retained on 2026-10-01 and are gone now.
- **Other:** 29 analysis arrays (5.454 GiB: the `token-cache-20260917`,
  `token-cache-allocation-20260917` and `site-probes-20260930` caches and two smaller files),
  ten cached Hugging Face backbones (1.396 GiB), 23 scratchpads of finished sessions (1.021 GiB),
  Codex build folders (0.866 GiB), the download chunk cache (0.375 GiB) and the wheels of 14
  unsubmitted bundles (0.071 GiB).
- **Retained:** the 25 source checkpoints of submitted bundles, the eight selected checkpoints
  named in `docs/ongoing/` (`no-anatomical-bias`, `control-check`, `budget-learnable-recipe` and
  the five view runs), and every submitted bundle's weights.
- **Kept for a decision:** two crashed `two-stream-paired` attempts and
  `views-budget-jitter-centre`, which have no selection record; the four
  `rn18-l2sp-no-confusion-folds-1-4` teachers, `attributes-v4-w005-candidate` and
  `attributes-v4-w02-candidate`, whose results are recorded nowhere in the docs; and
  `outputs/analysis/slice-budget-20261002/scratch/bundle-copy/model.ckpt`, outside any run folder.

The two cleanups below predate the skill and were user-authorized, hand-reviewed removals of
obsolete `.ckpt` files. Each manifest lists every removed and
retained path with its reason, size and file identity; all other artifacts, including predictions,
metrics, configs, source snapshots and submission manifests, were preserved, and the stopped
campaign states were unchanged.

Historical `selected_checkpoint.json` and bundle manifests retain their original provenance paths;
they do not establish that pruned weights are still available. Consult the latest manifest before
attempting to resume, re-export or rerun inference on a historical experiment. Its saved predictions
and metrics remain available for comparisons.

**2026-10-01:** removed **435 files (63.336 GiB)** and retained 84 (10.567 GiB)
([manifest](../../outputs/archive/maintenance/checkpoint-pruning-20261001T164846Z/manifest.json)).

- **Removed:** every `last.ckpt` of a completed run, retained runs included (170 files, 27.8 GiB);
  rejected, invalid and superseded experiments of the stopped campaigns (103, 17.6 GiB) and of
  `outputs/training` and `outputs/kaggle-training` (85, 13.3 GiB); and superseded submission
  exports with their two standalone sources (77, 4.6 GiB): timing, loading-arm and τ-sweep
  bundles and candidates that lost publicly.
- **Retained, selected checkpoint only:** the runs named in the
  [project state](../ongoing/state.md#current-control) (`no-anatomical-bias`, both Refined
  ResNet-50 runs, `plane-contrast-readouts`, `rn18-l2sp-no-confusion` with its fold 1–4 teachers,
  `rn18-oof-blend`, `rn18-no-series-dropout`, `rn18-more-triplets`, `rebuild-state-v2-control`,
  `scanner-confusion-control`); the v1-split models at 0.947–0.948 public
  (`scanner-confusion-detached-head`, `half-stack-control`, `rebuild-state-candidate`); the ten
  2026-09-19 ensemble members; `control-check`; the three `attributes-v4` candidates; and the 31
  export checkpoints of those models, including `aux-site-bias-f3`, whose source was already gone.
- **Not reviewed:** `outputs/autoresearch/mixed-field` (22 files, 4.2 GiB), an active campaign.
- **Superseded retentions:** the held plane-coupling candidate with its parent, the
  attention-temperature cache parents, control A, control B and the reflection reference kept on
  2026-09-20 are gone; their campaigns cannot resume on the current split.

**2026-09-20:** removed **249 files (33.412 GiB)** from completed superseded or rejected
experiments, unused resume checkpoints and stale submission exports, retaining 48 (4.381 GiB)
([manifest](../../outputs/archive/maintenance/checkpoint-pruning-20260920T191234Z/manifest.json)).
Windows C: free space did not grow after either cleanup unless the WSL virtual disk is compacted.

**2026-10-04:** by hand at the user's request, from the stopped `learnable-views` campaign only:
removed **86 checkpoints (14.72 GiB)**, every `last.ckpt` and the selected checkpoint of every
run that is not needed, retaining 10 selected checkpoints (1.70 GiB)
([manifest](../../outputs/archive/maintenance/learnable-views-checkpoints-20261003T220314Z/manifest.json)).
Kept: the campaign reference `hand-views-frozen-with-label-bias` (S100), the five members of S99
(one of them S98), and the four runs above the reference at ×4 (size menu, label depth bias,
label bias at ten times the rate, protected-strata depth transport). Predictions, metrics,
configs, source snapshots and the campaign state are untouched.

**2026-10-04, 22:14 CEST:** `scripts.tools.free_disk`, run after the `refined-radimagenet`
campaign stopped: **19.22 GiB freed**
([manifest](../../outputs/archive/maintenance/disk-cleanup-20261004T201421Z/manifest.json)). The
user asked for the campaign's checkpoints; the script applies its rules to every worktree, so it
also removed what the rules covered elsewhere.

- *From the campaign (6.43 GiB):* 19 `last.ckpt` (3.51 GiB) and 16 selected checkpoints of
  recorded runs (2.92 GiB). Kept: the incumbent `radimagenet-input-statistics` (named in
  `docs/ongoing/`), the two manual slice-teacher runs (results not in the campaign state) and the
  two cancelled runs without a selection record (`crossd-3d-stem`, `acs-gated-stem-block1`).
- *Outside the campaign, by rule (12.79 GiB):* 29 `last.ckpt` of finished runs in
  `outputs/training` (7.39 GiB); 21 selected checkpoints of recorded runs no longer named in any
  `docs/ongoing/` (3.46 GiB): **the ten `learnable-views` checkpoints retained by hand earlier the
  same day** (reference S100, the S99 members, four runs above the reference), the ReXNet-200
  folds 1–4 and `rn18-l2sp-no-confusion` folds 1–4 teachers, `attributes-v4-w005` and `-w02`
  candidates and `budget-learnable-recipe`; 41 export folders of unsubmitted models (1.82 GiB);
  three unused Hugging Face models (0.11 GiB); ten finished-session scratchpads (0.05 GiB).
  Submitted bundles in the ledger and their source checkpoints are untouched.
- *Failed:* one `.ruff_cache` inside a Codex debate (permission denied, 4 KB).
- *Kept for a decision:* six `model.ckpt` outside a fold folder (slice-budget scratch copy and
  the `ens5-learnable-views-f0` dataset), four `mixed-field` two-stream checkpoints and the two
  stopped ResNet-50 heated runs (no selection record), the ten `oof-blend-v2` children, the
  ResNet-10t reference, five `learnable-views`-era runs in `outputs/training` and the
  `oof-blend-v2` export members (results not recorded in `docs/`).
- *uv cache:* `uv cache prune` was skipped: another uv process held the cache lock.

**2026-10-05, about 12:10 CEST:** removed by hand on the user's request, outside the script:
the 72 checkpoint files of the `pretraining-input` campaign (36 runs, 5.7 GiB) and six of the
seven left in `refined-radimagenet` (the two manual slice-teacher runs and the two cancelled
runs). Kept: the `radimagenet-input-statistics` incumbent, the reference of the ResNet-18
comparisons.

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
