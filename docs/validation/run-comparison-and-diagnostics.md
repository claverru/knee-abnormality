# Run comparison and diagnostics

Tools for comparing completed runs and inspecting a trained model, and the arithmetic for reading
per-label AUC. Rules for what a comparison can decide are in the
[validation protocol](validation-protocol.md).

## Fold-run comparison

Compare a candidate run with a reference run on CPU, from saved predictions and checkpoint tensors
read raw, without building a model, so runs from retired model code still compare:

```bash
env CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m scripts.tools.compare_fold_runs --reference outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/00-baseline --candidate outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/03-min-lr-0.1 --output-dir outputs/analysis/m/min-lr-0.1-vs-00-baseline --allowed-config-differences name,optimizer.minimum_lr_ratio
```

The example compares the `03-min-lr-0.1` candidate with its `00-baseline` control. Add
`--allowed-input-identity-differences preprocessing_identity` when the runs come from different
sources. Allowed configuration differences are dotted keys or prefixes (default `name`); allowed
input-identity differences are top-level keys. `--reference-identity-exceptions` removes dotted
input-identity fields from both runs before the check; autoresearch uses it for a campaign's
user-authorized reference exceptions. Any other difference fails, as do selected predictions that do
not reproduce their recorded macro AUC: the `prediction_macro_auc` in `selected_checkpoint.json` for
runs that predict at another multiplier than their epoch validation (×4 since 2026-09-22), otherwise
the selected epoch's. Runs predicted at different multipliers differ in
`validation.prediction_image_multiplier` and are rejected unless that key is allowed; re-predict the
older run with [`repredict_run`](../../scripts/tools/repredict_run.py) instead. The output directory
must be fresh; besides `comparison.json` it holds `source/` copies of the analysis code.
`comparison.json` records:
- selected epochs, full-fold and per-target AUC, soft-target BCE, and gold and report-derived
  metrics;
- family means for all targets, the meniscus and OA pairs and the other eight targets;
- exclusive-positive margin AUCs and lateral cross-talk for both runs, with the shared definitions of
  the [encoding diagnostic](#manual-encoding-usage-diagnosis);
- vendor strata with positive counts, prediction changes, epoch curves, timing and peak memory,
  parameters present in only one run and changed `src/` and `configs/` members.

Paired report-group bootstrap draws (2,000, seed 20260914) give candidate-minus-reference intervals
for each family and both margins. They exclude training-seed and checkpoint-selection variability.
Vendor comes from the saved study-header audit and is an acquisition proxy. The tool generalizes the
historical knee-side analysis, which stays unchanged.

### Re-predicting an older run

`repredict_run` derives a new run directory whose selected checkpoint (hard-linked) is re-predicted
on the validation fold at another multiplier, attention temperature 1. It uses the GPU for one
validation pass (about 70 s at ×4 on fold 0), so check GPU usage first:

```bash
uv run --locked python -m scripts.tools.repredict_run --run outputs/training/k_05 --output outputs/training/k_05-x4 --image-multiplier 4
```

It runs with the repository's sources, so the run's preprocessing identity must match them; older
runs need their archived sources. It refuses an existing output and never modifies the source
run. Only `validation_predictions.csv`, `selected_checkpoint.json` (with
`prediction_image_multiplier` and `prediction_macro_auc`), the prediction time in `timing.json` and
`validation.prediction_image_multiplier` in `effective_config.json` change; epoch metrics, logs and
the ×1 epoch score stay as trained, and `repredicted.json` records the source. Predictions match what
`scripts.pipeline.train` writes at the same multiplier.

## Manual encoding usage diagnosis

After training a run, the separate command was run manually against the selected checkpoint:

```bash
uv run --locked python -m scripts.adhoc.diagnose_encoding_usage outputs/training/run --output-dir outputs/analysis/encoding-usage-run --workers 8
```

`--checkpoint` selects an explicit checkpoint; otherwise the run's fold selection is used. It runs
inference on the whole held-out fold, on the GPU by default, so check GPU usage first. The output
directory must be fresh; the package and the script are copied to `source/` under their
repository-relative paths. Configuration, source/checkpoint and input identities are checked and
recorded. No training weights are changed and no callback or training recipe launches the command.
It was written for the spatial-token model ([model](../modeling/current-model.md)) and preprocessing
identity `v9`; later model changes were not rechecked against it (unverified). Earlier checkpoints need their archived sources, including every checkpoint trained
before the 2026-09-16 package restructure, the retired token graph's and those of the 2026-09-14
overnight queue, `00-baseline` among them.

Every variant reuses one batch's projected backbone features and deterministic held-out images, then
recomputes the embedding sum, LayerNorm and MIL with the same visual token membership. The graph
variants and graph statistics of the `00-baseline` version were removed with the graph. The seven
full-fold variants are:
- intact;
- without the 1D, 2D, plane or contrast embedding;
- without the anatomical bias, which zeroes the medial axis while keeping the independent heads and
  learned family strengths; and
- without all position information (1D and 2D embeddings and the anatomical bias).

Embedding removals affect both attention and the position-rich pooled values. The attention-prior
variants were removed with the [prior](../history/model-lineage.md#retired-attention-prior) on
2026-09-16; the [attention-prior result](results/pooling-and-attention/2026-09-15-attention-prior-result.md)
keeps their measurements. No production removal flags were added. The complete held-out fold supplies
per-study predictions, per-target and macro AUC, soft-target BCE and absolute probability changes;
undefined target/subset AUCs are omitted as null values.

Axial-, coronal- and sagittal-only variants pool the intact tokens of one plane by excluding other
planes' tokens from attention. They cover only studies with visually valid tokens of that plane. Their
prediction files, metrics, per-target AUCs and bootstrap intervals are paired with intact on the same
studies, and study counts are recorded. Every variant also reports compartment metrics from frozen
targets thresholded at 0.5, for the medial/lateral meniscus and OA pairs:
- **Margin AUC:** the AUC of logit(medial) - logit(lateral) for medial-only against lateral-only
  exclusive-positive studies, with a paired bootstrap interval.
- **Lateral cross-talk:** flag the top-k lateral predictions, where k is the lateral positive count; it
  is the share of flagged lateral negatives that are medial-positive, with counts and the
  medial-positive base rate among lateral negatives.

**The champion plane-only baseline was skipped** by user decision on 2026-09-14; the pre-graph
diagnostic was never run on `reflection-sigma40`. Plane-only margins exist only for later checkpoints,
so a sagittal-only gain cannot be compared with the champion. The champion's full-fold references come
from its saved predictions through the fold-run comparison: macro AUC 0.892360, margins 0.9690
(meniscus) and 0.8791 (OA), and lateral cross-talk 68.9% (31/45) and 63.9% (53/83) against
medial-positive base rates of 36.7% and 21.6%.

Paired study-level bootstrap resampling retains all twelve targets together; the default is 2,000
replicates with seed 20260914 and percentile 95% intervals on variant-minus-intact differences, with
valid replicate counts recorded. The intervals describe this checkpoint and held-out fold, not
training-seed variability. The intact output is compared with the saved validation probabilities;
investigate any disagreement before interpreting small intervention effects.

The pre-graph version replaced 3D with `unknown_3d` and measured the
[knee-side checkpoint](results/pooling-and-attention/2026-09-14-knee-side-embedding-result.md#side-interventions-the-checkpoint-ignores-knee-side);
its source is snapshotted under `outputs/analysis/e/encoding-usage-knee-side-20260914/source/`, and its
side interventions (suppression, swapping and seeded corruption) are preserved by the
[archived patch](../history/sources/knee-side/README.md). The token-graph version added graph and
edge-type removals and per-plane edge statistics; it is preserved with the
[historical token-graph sources](../history/sources/token-graph/README.md), and its artifacts are under
`outputs/analysis/e/encoding-usage-token-graph/` and `outputs/analysis/e/encoding-usage-graph-forced/`.

These interventions measure reliance. Embeddings and planes overlap and interact, effects need not
add, and they cannot be assigned unique contribution percentages. Removals never made in training can
create unfamiliar inputs. Attention weights or embedding magnitudes alone are not contribution scores;
separately trained models would be needed to establish learning benefits. The original spatial-token
decisions and method references are in the [completed plan snapshot](../history/spatial-token-plan.md).

## Plot validation metrics

Plot epoch metrics from completed run folders with:

```bash
uv run --locked python -m scripts.tools.plot_validation_metrics baseline0 baseline1 baseline4
```

The command writes `outputs/analysis/v/validation_metrics.png` by default (overwritten on every call;
`--output` selects another PNG) with six panels in two rows. The top row holds the ranking metrics,
where higher is better: hard macro AUC, [soft macro AUC](results/labels-and-losses/2026-09-23-validation-auc-variants.md)
and gold macro AUC. The bottom row holds training BCE and validation soft-target BCE side by side for
reading overfitting, then the per-target AUC at the best epoch. Runs with several folds draw one line
per fold.

Autoresearch experiments keep their run in a `training/` folder; the legend names them by their
experiment folder without its timestamp and hash. Runs with no epoch metrics yet are skipped with a
note, so a campaign's control and all its experiments plot with one glob:

```bash
uv run --locked python -m scripts.tools.plot_validation_metrics k05-clean \
  outputs/autoresearch/noisy-labels/experiments/*/training --output outputs/analysis/n/noisy-labels.png
```

Soft macro AUC is computed from each epoch's saved `validation_epoch_*.csv` against the run's frozen
split targets, so runs trained before it was logged are plotted too; the split named in
`effective_config.json` must load. Training BCE is the epoch-mean `train_loss_epoch` from
`logs/version_*/metrics.csv`; a fold without it fails. It is the loss as optimized, including
class-balance weights, so compare it only across runs with the same class-balance setting.

The per-target panel draws the 12 target AUCs from each series' best macro-AUC epoch as grouped bars,
with a dashed horizontal line at that epoch's macro AUC so targets above and below the run average are
visible at a glance. Bars start at the 0.05 step below the lowest plotted AUC and the axis label states
that floor, which keeps 0.001 differences readable; one or two series also get per-target value
labels. The run legend in the first panel names the best epoch each bar group comes from. Undefined
gold macro AUCs (`null` in epoch JSON) appear as gaps; full-validation AUC, BCE and per-target AUCs
must remain numeric.

## Cost of a reversed pair

A target's ROC-AUC is the fraction of its (positive, negative) study pairs ranked correctly, so with
`P` positives and `N` negatives, reversing one pair costs `1/(P x N)` of that target's AUC and
**`1/(12 x P x N)` of the macro AUC**. Targets with few positives carry the most expensive mistakes,
while the target with the largest total deficit (1 - AUC) pulls the average down most; the two
rankings need not agree.

Fold 0 of `all_5fold_v3` (904 studies, positives are targets >= 0.5, as in `validation_metrics`),
computed on 2026-09-16:

| Target | Positives | Negatives | Pairs | Macro AUC per reversed pair |
|---|---:|---:|---:|---:|
| Fracture | 63 | 841 | 52,983 | 1.573 x 10^-6 |
| Synovitis | 115 | 789 | 90,735 | 9.18 x 10^-7 |
| MCL | 134 | 770 | 103,180 | 8.08 x 10^-7 |
| Lateral Meniscus | 138 | 766 | 105,708 | 7.88 x 10^-7 |
| Contusion | 153 | 751 | 114,903 | 7.25 x 10^-7 |
| ACL | 184 | 720 | 132,480 | 6.29 x 10^-7 |
| Baker's | 216 | 688 | 148,608 | 5.61 x 10^-7 |
| Lateral OA | 241 | 663 | 159,783 | 5.22 x 10^-7 |
| Medial OA | 329 | 575 | 189,175 | 4.41 x 10^-7 |
| Medial Meniscus | 356 | 548 | 195,088 | 4.27 x 10^-7 |
| Effusion | 525 | 379 | 198,975 | 4.19 x 10^-7 |
| PF OA | 406 | 498 | 202,188 | 4.12 x 10^-7 |

- A fracture pair costs 3.8 times a patellofemoral OA pair.
- **Example:** the [crop224 run](results/crops-and-views/2026-09-16-crop224-result.md) lost 0.014250
  fracture AUC against `00-baseline`, about 755 reversed pairs, which alone costs 0.001187 macro AUC,
  more than that run's whole -0.001139 gap.
- A table shared on 2026-09-16 from an unrecorded source (881 studies per target, 43 MCL and 47
  fracture positives, possibly another split or the sparser official labels) made the same point: one
  MCL pair cost 5.23 times a medial-meniscus pair there, while synovitis had the largest total
  deficit. Its counts do not match this split.

Secondhand competition notes, not verified here: one unnamed target had F1 around 0.2 at threshold 0.5
despite high aggregate AUC and about 0.7 under focal loss (a target-level F1 change, not an AUC
improvement; other thresholds were suggested); errors were said to concentrate in four targets, and a
review of the worst 40 cases suspected that about half were label mismatches, by model-assisted
inspection and not independent adjudication.

## Overnight launcher

`scripts/adhoc/train_overnight.py` ran a queue of one-change experiments sequentially against an
already-trained control run (`CONTROL_RUN`). The runs it executed and their results are in
[campaign and queue results](results/campaigns-and-queues/INDEX.md); this section describes the
script, kept in `scripts/adhoc/` as the record of those runs.

```bash
nohup uv run --locked python -m scripts.adhoc.train_overnight > outputs/training/overnight.out 2>&1 &
```

`--prepare-only` writes the configurations without training; `--config PATH` selects another training
YAML (default `configs/resnet50.yaml`). Every invocation creates a new queue and never resumes an
earlier one. The script trains and plots without invoking Codex or submitting to Kaggle. Before
training it:

- refuses to start if any GPU compute process is running (`nvidia-smi --query-compute-apps=pid
  --format=csv,noheader` lists a PID); skipped only when `nvidia-smi` is missing;
- raises the soft open-file limit to the inherited hard limit and records before/after limits and the
  starting source fingerprint in `runtime.json` (`--prepare-only` skips both). The first background
  attempt on 2026-09-12 stopped at batch 44 of epoch 0 in all seven runs with file-descriptor
  exhaustion: the service had a soft limit of 1,024 against 1,048,576 in the shell;
- fingerprints every tracked and untracked-but-not-ignored file under `src/`, `scripts/` and
  `configs/` (content hashes via `git ls-files --cached --others --exclude-standard`) and recomputes
  it before every run, stopping the queue if it changed, because each run imports the working tree
  fresh and a mid-queue edit would silently change later runs.

The launcher snapshots the training config into validated YAML files under
`outputs/training/overnight-<UTC timestamp>-<suffix>/configs/`. Split, fold and seed come from the
config, so every run shares the control's. The same seed and training sources reproduce bit-identically,
so the control is a trained run and not retrained: the launcher refuses a queue whose base config
differs from the control's saved `effective_config.json` in anything but `name` and
`paths.output_root` (the two manifest paths are compared from their `configs/` directory on, so a
control trained from a worktree still matches). Each candidate differs from it only by its listed
override.

Each queue has `runs/<name>/` for normal training artifacts, `logs/<name>.log`, `status.tsv`
(`experiment`, `started_at`, `finished_at`, `wall_seconds`, `exit_code`, `run_dir`, `log`) and
`summary.tsv`, rewritten after every run so a crashed night still has it. `summary.tsv` has one row per
finished run, read from `fold_<k>/selected_checkpoint.json`, `fold_<k>/epochs/*metrics_epoch_*.json`
and `fold_<k>/timing.json`:

| Column | Source |
|---|---|
| `selected_epoch` | Epoch number (1-indexed) parsed from `selected_checkpoint.json`'s checkpoint filename |
| `best_val_macro_auc` | `selected_checkpoint.json`'s `score` |
| `last_val_macro_auc` | `macro_auc` of the highest-numbered `epochs/metrics_epoch_*.json` |
| `delta_best_vs_control`, `delta_last_vs_control` | Difference against the control run's best/last |
| `prediction_macro_auc`, `delta_prediction_vs_control` | `selected_checkpoint.json`'s ×4 `prediction_macro_auc`, the adoption metric, and its difference against the control run |
| `soft_macro_auc_at_selected` | `soft_macro_auc` from the selected epoch's metrics file |
| `soft_target_bce_at_selected`, `gold_macro_auc_at_selected` | `soft_target_bce` and `gold.macro_auc` from the selected epoch's metrics file |
| `training_seconds`, `optimizer_steps` | `timing.json` |
| `peak_cuda_memory_gib` | `timing.json`'s `peak_cuda_allocated_bytes` / 2^30 |

A failed experiment is recorded and the queue continues; the final exit code is nonzero if any
training or plotting command failed or the queue stopped early on a source-fingerprint change. SIGINT
or SIGTERM stops the queue and terminates its active subprocess group. Successful runs are plotted
together in `validation_metrics.png`. At the end the launcher prints a `compare_fold_runs` command
for every completed candidate against `CONTROL_RUN`, allowing `name`, `paths.output_root` and the
candidate's own override; the control is also the first row of `summary.tsv` and of the plot.
Bootstrap comparisons are manual CPU follow-ups.

**Budget:** a fixed five-hour elapsed-time deadline covers preparation, training and plotting; remaining
runs stay unstarted when less than 35 minutes remain (`RUN_ALLOWANCE_SECONDS`). An active subprocess
is terminated at the deadline with fifteen seconds reserved for graceful shutdown. Timed-out runs
record exit code 124; a queue stopped for budget or changed source returns 3. Finished artifacts
remain available. As a size reference, a ten-epoch EfficientNet-B3 run took about 33 minutes (1,937 s of
training plus ×4 prediction) and a twenty-epoch run about 65. Use `--prepare-only` to validate
configs; repository policy prohibits writing or running tests.

**Already measured.** User decisions of 2026-09-16 on the ordinary-BCE recipe: `min_lr 0.1` dropped
after three unestablished measurements, series dropout 0.1 dropped (0.2 and 0.3 flat), reflection
0.25, intensity-off and batch size 4 dropped. No delta recorded before 2026-09-19 is a prior for later
recipes: those runs used ordinary BCE, and class-balanced BCE alone moves fold 0 by about 0.008. The
[revisits campaign](results/campaigns-and-queues/2026-10-01-revisits-campaign.md) ran 18
hyperparameter values on ResNet-18 without a gain. Keep `validation_image_multiplier: 1` in
comparisons.
