# Overnight hyperparameter queue

The candidates for the next queue are tracked in the [overnight queue](../backlog/overnight.md).

## Launcher

Run the fourteen prepared experiments sequentially with:

```bash
nohup rtk proxy uv run --locked python -m scripts.tools.train_overnight > outputs/training/overnight.out 2>&1 &
```

Use `--prepare-only` to write the fourteen configurations without starting training, or
`--config PATH` to select another training YAML (default `configs/training.yaml`). Every
invocation creates a new queue; preparation does not reserve a queue for a later command.

The script trains and plots results without invoking Codex or submitting to Kaggle. Before
training, it:

- Refuses to start if any GPU compute process is already running
  (`nvidia-smi --query-compute-apps=pid --format=csv,noheader` lists a PID); the check is skipped
  only when `nvidia-smi` itself is missing.
- Raises the process soft open-file limit to the inherited hard limit and saves the before/after
  limits, plus the starting source fingerprint, in `runtime.json`. `--prepare-only` leaves process
  limits and the fingerprint check out entirely.
- Fingerprints every tracked and untracked-but-not-ignored file under `src/`, `scripts/` and
  `configs/` (content hashes via `git ls-files --cached --others --exclude-standard`, so
  `__pycache__` and other gitignored caches are excluded). Before every run, it recomputes the
  fingerprint and stops the queue with a clear message if it changed, because every run imports
  the working tree fresh and an edit mid-queue would silently change later runs.

The first background attempt on 2026-09-12 stopped at batch 44 of epoch 0 in all seven runs and
produced no validation results or checkpoints. Three logs reported file-descriptor exhaustion; the
service had a soft limit of 1,024, compared with 1,048,576 in the shell. This is the likely shared
cause. The corrected queue below completed all seven runs with the raised limit. The subsequent
Codex resume was interrupted. Failed outputs were deleted at the user's request, and the resume
hook was removed in favor of manual launching. The 2026-09-15 queue completed seven of sixteen
runs before the machine or WSL stopped; the run list below replaces it.

The launcher snapshots the current training config into fourteen validated YAML files under a
unique `outputs/training/overnight-<UTC timestamp>-<suffix>/configs/` directory. It fixes
`all_5fold_v3`, fold 0 and validation image multiplier 1; every run carries its own recorded random seed. All other settings come
from the training config at launch, which since 2026-09-19 means
`trainer.class_balanced_bce: true`, `data.sampling_sigma_fraction: 0.867` and
`data.study_triplet_total: 26`. Each run starts fresh with exactly the listed overrides from
`00-control`, its comparison control.

The [completed 2026-09-20 queue](results/2026-09-20-overnight-hyperparameter-results.md)
has two controls and twelve independent candidates. All fourteen finished in 6 h 8 m
with no adoption. The script retains this last run list until a new queue is selected.
Each invocation draws distinct 32-bit seeds from operating-system randomness for
all fourteen runs, including both controls, prints them and stores them in each
YAML. `--prepare-only` creates a separate preview queue; a later launch generates
new seeds. Ten epochs and patience ten are fixed, with unchanged triplet budgets.

| Run | Override from control |
|---|---|
| `00-control` | None |
| `01-head-lr-0.0012` | `optimizer.head_lr: 0.0012` |
| `02-intensity-p-0.75` | `data.intensity_augmentation.probability: 0.75` |
| `03-backbone-lr-0.0003` | `optimizer.backbone_lr: 0.0003` |
| `04-dropout-0.1` | `model.dropout: 0.1` |
| `05-gamma-0.70-1.30` | `data.intensity_augmentation.gamma_range: [0.70, 1.30]` |
| `06-contrast-0.80-1.20` | `data.intensity_augmentation.contrast_range: [0.80, 1.20]` |
| `07-affine-p-0.5` | `data.affine_augmentation.probability: 0.5` |
| `08-head-lr-0.0024` | `optimizer.head_lr: 0.0024` |
| `09-rotation-7` | `data.affine_augmentation.rotation_degrees: 7` |
| `10-scale-0.97-1.03` | `data.affine_augmentation.scale_range: [0.97, 1.03]` |
| `11-weight-decay-0.03` | `optimizer.weight_decay: 0.03` |
| `12-warmup-2` | `optimizer.warmup_epochs: 2` |
| `13-control-repeat` | None; independent seed |

All candidates start from the same current checked-in recipe, not from earlier
queue winners. The completed 2026-09-20 queue used the original sampler. The
[subsequent ten-epoch sampler promotion](results/2026-09-20-stratified-sampling-promotion.md)
was reverted after inconclusive confirmation, restoring the original sampler for
future training. Saved runs retain their own source snapshots.
The previous five-run queue and its seeds remain in the
[2026-09-19 results](results/2026-09-19-untried-hyperparameter-queue.md).

Each queue has `runs/<name>/` for normal training artifacts, `logs/<name>.log` for training
output, `status.tsv` for per-run timing and exit codes, and `summary.tsv` for per-run metrics,
rewritten after every run so a crashed night still has it. The launcher prints each experiment's
start, log path and completion status.

`status.tsv` columns: `experiment`, `started_at`, `finished_at`, `wall_seconds`, `exit_code`,
`run_dir`, `log`.

`summary.tsv` columns (one row per finished run, read from the run's `fold_0/selected_checkpoint.json`,
`fold_0/epochs/*metrics_epoch_*.json` and `fold_0/timing.json`):

| Column | Source |
|---|---|
| `selected_epoch` | Epoch number (1-indexed) parsed from `selected_checkpoint.json`'s checkpoint filename |
| `best_val_macro_auc` | `selected_checkpoint.json`'s `score` |
| `last_val_macro_auc` | `macro_auc` of the highest-numbered `epochs/metrics_epoch_*.json` |
| `delta_best_vs_control`, `delta_last_vs_control` | Difference against `00-control`'s best/last, blank until `00-control` has finished |
| `live_best_val_macro_auc`, `live_last_val_macro_auc` | Best/last `macro_auc` from `epochs/live_metrics_epoch_*.json`; blank unless `optimizer.weight_averaging` is set |
| `soft_target_bce_at_selected`, `gold_macro_auc_at_selected` | `soft_target_bce` and `gold.macro_auc` from the selected epoch's metrics file |
| `training_seconds`, `batch_norm_refresh_seconds`, `optimizer_steps` | `timing.json` |
| `peak_cuda_memory_gib` | `timing.json`'s `peak_cuda_allocated_bytes` / 2^30 |

A failed experiment is recorded and the queue continues; its final exit code is nonzero if any
training or plotting command failed, or if the queue stopped early on a source-fingerprint change.
SIGINT or SIGTERM stops the queue and terminates its active subprocess group. Successful runs are
plotted together in `validation_metrics.png` at the end. Every invocation creates fresh
directories and never resumes an earlier queue.

At the end, the launcher prints `compare_fold_runs` commands for every completed
candidate against each completed control, plus control B against control A.
Allowed config differences include the fresh seed and the candidate's own override.
The summary deltas remain relative to `00-control`. Bootstrap comparisons are
manual CPU follow-up commands; see the [comparison procedure](fold-run-comparison.md).

**Budget:** recent runs took about 26 minutes each, so fourteen should take roughly
6–7 hours. The queue has a fixed eight-hour elapsed-time deadline covering preparation,
training and plotting. It leaves remaining runs unstarted when less than thirty
minutes remain. An active subprocess is terminated at the deadline, with fifteen
seconds reserved for graceful shutdown before forced termination. Timed-out runs
record exit code 124; a queue stopped for budget or changed source returns 3.
Finished artifacts remain available. Bootstrap commands printed afterward are not
executed inside that budget. Nothing submits or publishes results.

**Preparation verification:** use `--prepare-only` to validate all fourteen configs
without launching training. Repository policy prohibits writing or running tests.
