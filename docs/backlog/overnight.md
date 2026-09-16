# Overnight queue

Hyperparameter-only runs: values of existing training options on the
[current recipe](../modeling/current-model.md), one change per run against a fresh control (seed 42,
`all_5fold_v3`, fold 0). A [small change](small-changes.md) enters this queue once it is
implemented; other ideas belong to the other [backlog](INDEX.md) documents. The queue runs
unattended and the user launches it, detached from agent sessions; the
[queue launcher](../validation/overnight-hyperparameter-queue.md) owns the procedure. Record
results in [validation results](../validation/results/INDEX.md), then remove the finished entries
from this file.

## Next queue (proposed, pending the user's decisions)

Base: `configs/training.yaml` at launch. The order is the priority order, so stopping early loses
only the least informative runs. The launcher's run list still describes the 2026-09-14 queue and
must be replaced first.

| Run | Change | Why | Est. min |
|---|---|---|---:|
| `00-control` | None | Required: the restored code's training random stream and source identity differ from the original `00-baseline`, whose checkpoints load only through their archived sources | 30 |
| `01-min-lr-0.1` | `optimizer.minimum_lr_ratio: 0.1` | Replicates the only candidate: +0.00041 now, +0.0018 on 1C ([below](#candidate-notes)) | 30 |
| `02-ema-0.999` | EMA, decay 0.999 | Untested; averaged versus live weights paired within the run. Needs the [weight-averaging small change](small-changes.md#weight-averaging) | 34 |
| `03-swa` | SWA over the last 3 epochs | Untested; paired within the run. Needs the same small change; pairing with `minimum_lr_ratio: 0.1` is undecided | 38 |
| `04-epochs-12` | `trainer.max_epochs: 12`, `trainer.patience: 12` | `00-baseline` peaked at its last epoch | 36 |
| `05-epochs-15` | `trainer.max_epochs: 15`, `trainer.patience: 15` | Same; sets the budget for later runs | 45 |
| `06-backbone-eval` | `model.backbone_eval: true` | Never ran; it gates `model.checkpoint_backbone` for [bigger images](endgame.md#bigger-images) | 30 |
| `07-beta2-0.99` | `optimizer.beta2: 0.99` | A new axis ([optimizer research](../research/optimizer-research.md#32-second-experiment-beta2099) A2) | 30 |
| `08-reflection-0.25` | `data.reflection_augmentation.probability: 0.25` | 0.5 was never tuned | 30 |
| `09-affine-off` | `data.affine_augmentation.probability: 0` | Strong affine was the largest effect (−0.0024) | 30 |
| `10-slower-head` | `optimizer.head_lr: 3e-4` | Never ran; the faster head lost on 2026-09-12 | 30 |
| `11-clip-0.001` | `data.lower_quantile: 0.001`, `data.upper_quantile: 0.999` | Never ran; keeps bright fluid signal | 30 |
| `12-clip-0.01` | `data.lower_quantile: 0.01`, `data.upper_quantile: 0.99` | Never ran; the other side | 30 |
| `13-warmup-0` | `optimizer.warmup_epochs: 0` | Never ran; the two-epoch warm-up lost −0.0010 on 1C | 30 |
| `14-series-dropout-0.1` | `model.series_dropout: 0.1` | 0.3 lost −0.0005; the other side | 30 |
| `15-intensity-off` | `data.intensity_augmentation.probability: 0` | Never ran; stronger intensity tied on 1C | 30 |

- **Time:** about 8.5 hours (512 ± 25 min). Ten-epoch runs took 29.9 min on average on
  2026-09-15; the other rows are estimated at 175 s per epoch plus 45 s of overhead, and the
  averaging runs add a second validation forward and, for SWA, BatchNorm refreshes. If the queue
  runs long, drop from the bottom: `15`, `14`, `13`, then `12`.
- **Optional seed replicate** (the user's call): a `seed: 43` control, inserted as `01-seed-43` so
  the noise estimate survives an early stop, with `15-intensity-off` moved to a later queue.
- **Undecided SWA tail learning rate:** SWA averages iterates at a non-vanishing learning rate.
  With `minimum_lr_ratio: 0.1` the ratio falls from 0.33 to 0.10 over epochs 8–10; at the current
  0.01 it falls from 0.26 to 0.01, almost converged and close to what EMA 0.999 already averages. Pairing makes the comparison
  with `00-control` two-factor, while the run's live curve replicates `01-min-lr-0.1`; the
  alternative is SWA on the base schedule.
- **Free noise measurement:** the EMA run's live curve repeats `00-control`, and a paired SWA
  run's repeats `01-min-lr-0.1`. With `trainer.deterministic: false` their differences measure GPU
  nondeterminism, a lower bound on seed noise; identical curves show that averaging does not touch
  training.
- **Launcher changes proposed with the new list:** accept `optimizer.weight_averaging: null` when
  merging overrides, stop if `src/`, `scripts/` or `configs/` change mid-queue, refuse to start
  while another GPU process runs, record start, finish and wall time per run, and write a summary
  table with deltas against `00-control` after every run.

## Candidate notes

- **Longer schedule, 12–15 epochs** (untested). `00-baseline` peaked at its last epoch, as did
  `02-higher-lr`, `04-series-dropout-0.3` and `05-wd-0.05`; `09-epochs12` never ran
  ([overnight results](../validation/results/2026-09-15-overnight-results.md)). 20 epochs was
  settled on an earlier recipe. The cosine decay stretches with `trainer.max_epochs`, and
  `trainer.patience` must be at least the epoch count. Cost: 1.2–1.5× the 1,670 s training time.
- **`min_lr 0.1`** (measured twice, not established). `03-min-lr-0.1`
  (`optimizer.minimum_lr_ratio: 0.1`) selected epoch 8 at 0.893171, +0.000408 over
  `00-baseline` (0.892763, epoch 10), and held 0.892674 at epoch 10; single seed. The
  saved-prediction [fold-run comparison](../validation/fold-run-comparison.md)
  (`outputs/analysis/min-lr-0.1-vs-00-baseline/comparison.json`, 2,000 paired report-group
  bootstrap draws) gives macro AUC +0.00041 with a 95% interval of [−0.0018, +0.0027]. Family
  means move by −0.0017 (meniscus), −0.0024 (OA) and +0.0016 (other eight targets); the largest
  per-target changes are contusion +0.011, MCL +0.009, Baker's cyst −0.009 and lateral meniscus
  −0.005. The same change gained +0.001787 on the older 1C recipe
  ([2026-09-14 results](../history/sharing/2026-09-14-overnight-hyperparameter-results.md)).

## Candidates for later queues

- **`optimizer.backbone_lr: 1e-4`, head `6e-4`** (research A1; weak). Backbone 1e-4 was part of
  both losing 2026-09-12 configurations, the control (1e-4/3e-4) and the faster head (1e-4/1e-3,
  −0.0012), while doubling both rates to 2e-4/6e-4 gained +0.0045
  ([2026-09-12 results](../validation/results/2026-09-12-overnight-results.md)), and 3e-4/9e-4 now
  ties. `10-slower-head` covers the ratio from the other side.
- **`trainer.gradient_clip_val` 0.5 and 5.0**, after the
  [gradient-norm logging](small-changes.md#gradient-norm-logging) shows whether 1.0 ever binds; a
  run at 5.0 may be a no-op.
- **Reflection probability 0.75**, if `08-reflection-0.25` moves the score. Per study, 0.25 and
  0.75 give the same augmentation diversity; 0.25 stays closer to the unreflected validation and
  test distribution.
- **Weight-averaging follow-ups:** EMA decays 0.998 and 0.9995 if 0.999 wins, EMA with
  `minimum_lr_ratio: 0.1`, and SWA on the base schedule.
- **20 epochs**, if `05-epochs-15` beats `04-epochs-12`.
- **Other values of the existing augmentation families**, after `09-affine-off` and
  `15-intensity-off` (hypothesis): probabilities and strengths of the intensity, affine and noise
  families, one per run, such as the milder side of the strong affine that lost −0.0024. Ordering,
  interactions and new families are [small changes](small-changes.md#augmentation-families).
- **Low value:** `data.sampling_sigma_mm: 50`, since 30 tied with 40 (+0.000039) and 60, 80 and 120
  were settled worse; `optimizer.weight_decay: 0.001`, since 0.003 and 0.03 are settled, 0.05 lost
  −0.0008 and 0.003 gained only +0.0002 on the 2026-09-12 recipe.
- **Out of scope:** `model.channels_last` is a speed question for a bounded, user-launched
  `profile_training.py` measurement, not an overnight slot. NHWC kernels change bf16 outputs only
  in low-order bits. A 2026-09-11 channels-last ReXNet-100 profile
  (`outputs/profiling/rexnet100-diagnosis-repeat-20260911`) ran at 0.54 studies/s, against 35.43
  in the later [production-path profile](../modeling/training-runtime-and-memory.md); the design
  pass attributes this to about 10 s of cuDNN setup for every new flattened image count, which
  MIL batches change every step.

## Already measured (2026-09-15, on `00-baseline`)

`01-sigma30` +0.000039, `02-higher-lr` −0.000222, `03-min-lr-0.1` +0.000408,
`04-series-dropout-0.3` −0.000492, `05-wd-0.05` −0.000837, `06-affine-strong` −0.002389, and
the `crop130` rerun −0.000263
([overnight results](../validation/results/2026-09-15-overnight-results.md)). That queue completed
seven of sixteen runs: `07-crop130` stopped during epoch 8 when the machine or WSL stopped, and
runs `08-clip-0.01` through `15-backbone-eval` never started. Earlier settled searches are listed
in the [launcher doc](../validation/overnight-hyperparameter-queue.md).
