# Overnight hyperparameter queue

The candidates for the next queue are tracked in the [overnight queue](../backlog/overnight.md).

## Launcher

Run the sixteen planned experiments sequentially with:

```bash
rtk proxy uv run --locked python -m scripts.adhoc.train_overnight
```

Use `--prepare-only` to write the sixteen configurations without starting
training, or `--config PATH` to select another training YAML (default `configs/training.yaml`). Every invocation
creates a new queue; preparation does not reserve a queue for a later command.

The script trains and plots results without invoking Codex or submitting to
Kaggle. Before training, it raises the process soft open-file limit to the
inherited hard limit and saves the before/after limits in `runtime.json`.
`--prepare-only` leaves process limits unchanged.

The first background attempt on 2026-09-12 stopped at batch 44 of epoch 0 in
all seven runs and produced no validation results or checkpoints. Three logs
reported file-descriptor exhaustion; the service had a soft limit of 1,024,
compared with 1,048,576 in the shell. This is the likely shared cause. The
corrected queue below completed all seven runs with the raised limit. The
subsequent Codex resume was interrupted. Failed outputs were deleted at the
user's request, and the resume hook was removed in favor of manual launching.

The launcher snapshots the current training config into sixteen validated YAML files
under a unique `outputs/training/overnight-<UTC timestamp>-<suffix>/configs/` directory.
It fixes `all_5fold_v3`, fold 0 and seed 42. The 2026-09-14 queue also fixed
`model.graph_layers: 0`, disabling the since
[retired token graph](../history/model-lineage.md#retired-token-graph). The
launcher no longer sets it, and the run list below is still the 2026-09-14 queue.
All other settings come from the training config at launch. Each run starts fresh with at most one change from
`00-baseline`, its comparison control. None changes pixels, image count or model size.
Settled earlier searches are not repeated: lower LR, faster head, weight decay
0.003/0.03, 20 epochs, two-epoch warmup, effective batch 16, sigma 60/80/120,
slice reallocation, no noise, stronger intensity, shared normalization and
attention dropout.

| Run | Change from `00-baseline` |
|---|---|
| `00-baseline` | None |
| `01-sigma30` | `data.sampling_sigma_mm: 30` |
| `02-higher-lr` | `optimizer.backbone_lr: 3e-4`, `optimizer.head_lr: 9e-4` |
| `03-min-lr-0.1` | `optimizer.minimum_lr_ratio: 0.1` |
| `04-series-dropout-0.3` | `model.series_dropout: 0.3` |
| `05-wd-0.05` | `optimizer.weight_decay: 0.05` |
| `06-affine-strong` | Affine rotation 15°, translation 0.08, scale `[0.9, 1.1]` |
| `07-crop130` | `data.spacing_mm: 130 / 224` |
| `08-clip-0.01` | Quantiles 0.01 / 0.99 |
| `09-epochs12` | `trainer.max_epochs: 12`, `trainer.patience: 12` |
| `10-clip-0.001` | Quantiles 0.001 / 0.999 |
| `11-warmup-0` | `optimizer.warmup_epochs: 0` |
| `12-slower-head` | `optimizer.head_lr: 3e-4` |
| `13-affine-off` | Affine probability 0 |
| `14-intensity-off` | Intensity probability 0 |
| `15-backbone-eval` | `model.backbone_eval: true` |

Each queue has `runs/<name>/` for normal training artifacts, `logs/<name>.log`
for training output and `status.tsv` for completed run exit codes. The launcher
prints each experiment's start, log path and completion status. A failed
experiment is recorded and the queue continues; its final exit code is nonzero
if any training or plotting command failed. SIGINT or SIGTERM
stops the queue and terminates its active subprocess group. Successful runs
are plotted together in `validation_metrics.png` at the
end. Every invocation creates fresh directories and never resumes an earlier
queue. At about 29 minutes per ten-epoch run, the sixteen-run queue should take
about eight hours; this is an estimate, not a measurement.
