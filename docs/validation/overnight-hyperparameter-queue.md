# Overnight hyperparameter queue

The candidates for the next queue are tracked in the [overnight queue](../backlog/overnight.md).

## Launcher

Run the fifteen prepared experiments sequentially with:

```bash
nohup rtk proxy uv run --locked python -m scripts.adhoc.train_overnight > outputs/training/overnight.out 2>&1 &
```

Use `--prepare-only` to write the fifteen configurations without starting training, or
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

The launcher snapshots the current training config into fifteen validated YAML files under a
unique `outputs/training/overnight-<UTC timestamp>-<suffix>/configs/` directory. It fixes
`all_5fold_v3` and fold 0; seed 42 unless a run overrides it. All other settings come from the
training config at launch. Each run starts fresh with exactly the listed overrides from
`00-control`, its comparison control.

| Run | Overrides |
|---|---|
| `00-control` | none |
| `01-control-seed-43` | `seed: 43` |
| `02-control-seed-44` | `seed: 44` |
| `03-ema-0.999` | `optimizer.weight_averaging: {method: ema, decay: 0.999}` |
| `04-swa-3` | `optimizer.weight_averaging: {method: swa, epochs: 3}` |
| `05-epochs-20` | `trainer.max_epochs: 20`, `trainer.patience: 20` |
| `06-beta2-0.99` | `optimizer.beta2: 0.99` |
| `07-affine-off` | `data.affine_augmentation.probability: 0` |
| `08-slower-head` | `optimizer.head_lr: 0.0003` |
| `09-warmup-0` | `optimizer.warmup_epochs: 0` |
| `10-clip-0.001` | `data.lower_quantile: 0.001`, `data.upper_quantile: 0.999` |
| `11-clip-0.01` | `data.lower_quantile: 0.01`, `data.upper_quantile: 0.99` |
| `12-grad-clip-0.5` | `trainer.gradient_clip_val: 0.5` |
| `13-grad-clip-5` | `trainer.gradient_clip_val: 5.0` |
| `14-noise-0.3` | `data.noise_augmentation.probability: 0.3` |

See the [results](results/2026-09-17-overnight-results.md) of this list, the [backlog](../backlog/overnight.md#already-measured) for why
each run was chosen and the [user's decisions](../backlog/overnight.md#user-decisions-2026-09-16)
about what was dropped from this list.

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

At the end of the queue (and again if you re-read `status.tsv`/`summary.tsv` later), the launcher
prints one `compare_fold_runs` command per completed non-control run against `00-control`, each
with `--allowed-config-differences name,<changed keys>` set from that run's own overrides — for
example `04-swa-3` prints `--allowed-config-differences name,optimizer.weight_averaging`. Run
these manually once you want the paired saved-prediction bootstrap comparison; see the
[fold-run comparison doc](fold-run-comparison.md).

**Time estimate:** about 7.7 hours. Ten-epoch runs are about 28 minutes each (the 2026-09-15
queue averaged 29.9 min); the EMA run (`03-ema-0.999`) is about 32 minutes for the extra
validation forward; the SWA run (`04-swa-3`) is about 36 minutes for the extra forward plus three
BatchNorm refreshes; the 20-epoch run (`05-epochs-20`) is about 55 minutes. This is an estimate,
not a measurement.

**Verification:** `--prepare-only` was run once to confirm all fifteen configs load through
`ExperimentConfig` and differ from `configs/training.yaml` (seed 42, `all_5fold_v3`, fold 0)
exactly by the overrides in the table above; the verification queue directory was deleted
afterward.
