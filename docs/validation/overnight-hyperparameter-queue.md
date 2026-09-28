# Overnight hyperparameter queue

The candidates for the next queue are tracked in the [overnight queue](../backlog/overnight.md).

## Launcher

Run the prepared experiments sequentially with:

```bash
nohup rtk proxy uv run --locked python -m scripts.tools.train_overnight > outputs/training/overnight.out 2>&1 &
```

Use `--prepare-only` to write the configurations without starting training, or
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

The launcher snapshots the training config into validated YAML files under a unique
`outputs/training/overnight-<UTC timestamp>-<suffix>/configs/` directory. Split, fold and seed come
from the config, so every run shares the control config's split, fold and seed. Same seed and same training sources reproduce bit-identically,
so the control is an already-trained run (`CONTROL_RUN`), not retrained: the launcher refuses to
prepare a queue whose base config differs from the control's saved `input_config.yaml` in anything
but `name` and `paths.output_root`. Each candidate differs from it only by its listed override.

### EfficientNet-B3, second round (ran 2026-09-27)

Ran 2026-09-27 as queue `outputs/training/overnight-20260927T101441Z-_1ey2phz`; results in the
[B3 results page](results/2026-09-27-b3-overnight-queue.md#second-round-learning-rates-and-schedule).

Base and control: `02-backbone-lr-0.00045` of the
[first B3 queue](results/2026-09-27-b3-overnight-queue.md) (×4 0.904429), that is the recipe with
`model.backbone: efficientnet_b3.ra2_in1k`, DSU off (ReXNet-only) and `optimizer.backbone_lr:
0.00045`. The arms vary learning rates and the schedule (linear warmup from
`warmup_start_ratio`, then cosine decay to `minimum_lr_ratio`); augmentation strength is left out
(user decision). Every signal in the first queue was one run below 0.002, so each basis is a
hypothesis.

| Run | Override from control | Basis |
|---|---|---|
| `01-head-lr-0.0009` | `optimizer.head_lr: 0.0009` | Both LRs ×2 was the best ×1 run (+0.0019); this scales the head with the backbone by 1.5 |
| `02-head-lr-0.0004` | `optimizer.head_lr: 0.0004` | Lower side of the head rate, untested on B3 |
| `03-warmup-2` | `optimizer.warmup_epochs: 2` | The higher backbone rate won; a longer warmup may steady it (B0 +0.0007) |
| `04-warmup-0` | `optimizer.warmup_epochs: 0` | Other side: full rate from the first step |
| `05-minimum-lr-ratio-0.1` | `optimizer.minimum_lr_ratio: 0.1` | Higher cosine floor; the control selected epoch 8 of 10 |
| `06-minimum-lr-ratio-0.001` | `optimizer.minimum_lr_ratio: 0.001` | Other side: anneal nearly to zero |
| `07-epochs-20` | `trainer.max_epochs: 20` | +0.0011 ×1 at 3e-4 (×4 −0.0002); last, so the deadline cannot cut a shorter run |

Not queued: further backbone-rate values (3e-4 to 6e-4 lie within 0.0006), 15 epochs (−0.0012),
and augmentation strengths. The ReXNet `trainer.max_epochs: 12` run in
the [overnight queue](../backlog/overnight.md) needs its own queue afterwards.

Each queue has `runs/<name>/` for normal training artifacts, `logs/<name>.log` for training
output, `status.tsv` for per-run timing and exit codes, and `summary.tsv` for per-run metrics,
rewritten after every run so a crashed night still has it. The launcher prints each experiment's
start, log path and completion status.

`status.tsv` columns: `experiment`, `started_at`, `finished_at`, `wall_seconds`, `exit_code`,
`run_dir`, `log`.

`summary.tsv` columns (one row per finished run, read from the run's `fold_<k>/selected_checkpoint.json`,
`fold_<k>/epochs/*metrics_epoch_*.json` and `fold_<k>/timing.json`):

| Column | Source |
|---|---|
| `selected_epoch` | Epoch number (1-indexed) parsed from `selected_checkpoint.json`'s checkpoint filename |
| `best_val_macro_auc` | `selected_checkpoint.json`'s `score` |
| `last_val_macro_auc` | `macro_auc` of the highest-numbered `epochs/metrics_epoch_*.json` |
| `delta_best_vs_control`, `delta_last_vs_control` | Difference against the control run's best/last |
| `prediction_macro_auc`, `delta_prediction_vs_control` | `selected_checkpoint.json`'s ×4 `prediction_macro_auc`, the adoption metric, and its difference against the control run |
| `soft_macro_auc_at_selected` | `soft_macro_auc` from the selected epoch's metrics file |
| `live_best_val_macro_auc`, `live_last_val_macro_auc` | Best/last `macro_auc` from `epochs/live_metrics_epoch_*.json`; blank unless `optimizer.weight_averaging` is set |
| `soft_target_bce_at_selected`, `gold_macro_auc_at_selected` | `soft_target_bce` and `gold.macro_auc` from the selected epoch's metrics file |
| `training_seconds`, `batch_norm_refresh_seconds`, `optimizer_steps` | `timing.json` |
| `peak_cuda_memory_gib` | `timing.json`'s `peak_cuda_allocated_bytes` / 2^30 |

A failed experiment is recorded and the queue continues; its final exit code is nonzero if any
training or plotting command failed, or if the queue stopped early on a source-fingerprint change.
SIGINT or SIGTERM stops the queue and terminates its active subprocess group. Successful runs are
plotted together in `validation_metrics.png` at the end. Every invocation creates fresh
directories and never resumes an earlier queue.

At the end, the launcher prints a `compare_fold_runs` command for every completed candidate against
`CONTROL_RUN`, allowing `name`, `paths.output_root` and the candidate's own override; the control is
also the first row of `summary.tsv` and of the plot. Bootstrap comparisons are manual CPU
follow-up commands; see the [comparison procedure](fold-run-comparison.md).

**Budget:** a ten-epoch B3 run takes about 33 minutes of wall time (1,937 s of training plus ×4
prediction) and a twenty-epoch run about 65, so the seven runs need about 4.4 hours; the
twenty-epoch run starts at about 3.3 hours. The queue has
a fixed five-hour elapsed-time deadline covering preparation, training and plotting, and leaves
remaining runs unstarted when less than 35 minutes remain. An active subprocess is terminated at
the deadline, with fifteen seconds reserved for graceful shutdown before forced termination.
Timed-out runs record exit code 124; a queue stopped for budget or changed source returns 3.
Finished artifacts remain available. Nothing submits or publishes results.

**Preparation verification:** use `--prepare-only` to validate the configs without launching
training. Repository policy prohibits writing or running tests.
