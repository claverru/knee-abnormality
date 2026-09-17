# Overnight queue

Hyperparameter-only runs: values of existing training options on the
[current recipe](../modeling/current-model.md), one change per run against a fresh control (seed 42,
`all_5fold_v3`, fold 0). A [small change](small-changes.md) enters this queue once it is
implemented; other ideas belong to the other [backlog](INDEX.md) documents. The queue runs
unattended and the user launches it, detached from agent sessions; the
[queue launcher](../validation/overnight-hyperparameter-queue.md) owns the procedure. Record
results in [validation results](../validation/results/INDEX.md), then remove the finished entries
from this file.

## Next queue

One run, launched on its own rather than by the queue launcher. The 2026-09-17 queue finished
([results](../validation/results/2026-09-17-overnight-results.md)); build the following queue from
"Candidates for later queues" below.

- **`robust-center-seed42`** (`data.robust_center_outlier_mm: 20.0`, the
  [robust study centre](big-bets.md#implementation-2026-09-17); otherwise
  `configs/training.yaml`, seed 42, fold 0). Config:
  `outputs/training/configs/robust-center-seed42.yaml`. It is paired with
  `outputs/training/ranked-filtered-seed42` (0.889573).

  ```bash
  rtk proxy uv run --locked just train robust-center-seed42 outputs/training/configs/robust-center-seed42.yaml 0
  ```

  Expectation: a tie. The inference-only check measured −0.0001 on fold 0 (mean of three
  control checkpoints); on the `ranked-filtered-seed42` checkpoint it measured −0.00035. Seed
  noise is 0.0015–0.0018, so one run cannot separate the two.

## User decisions (2026-09-16)

- **Dropped `min_lr 0.1`** (measured three times, not established). The 2026-09-16 replicate on
  the restored code scored 0.892868 (+0.0001 against `00-baseline`,
  [result](../validation/results/2026-09-16-restored-code-hyperparameter-runs.md)). The original
  `03-min-lr-0.1` (`optimizer.minimum_lr_ratio: 0.1`) selected epoch 8 at 0.893171, +0.000408 over
  `00-baseline` (0.892763, epoch 10), and held 0.892674 at epoch 10; single seed. The
  saved-prediction [fold-run comparison](../validation/fold-run-comparison.md)
  (`outputs/analysis/min-lr-0.1-vs-00-baseline/comparison.json`, 2,000 paired report-group
  bootstrap draws) gives macro AUC +0.00041 with a 95% interval of [−0.0018, +0.0027]. Family
  means move by −0.0017 (meniscus), −0.0024 (OA) and +0.0016 (other eight targets); the largest
  per-target changes are contusion +0.011, MCL +0.009, Baker's cyst −0.009 and lateral meniscus
  −0.005. The same change gained +0.001787 on the older 1C recipe
  ([2026-09-14 results](../history/sharing/2026-09-14-overnight-hyperparameter-results.md)).
- **Dropped series dropout 0.1.** 0.2 (current) and 0.3 (−0.0005) are flat within noise, so the
  other side is low value.
- **Replaced epochs 12 and 15 with epochs 20** (`05-epochs-20`), tested directly against
  `00-control` instead of as two intermediate budgets.
- **Dropped reflection 0.25, intensity-off and batch size 4** for this queue.
- **Moved `model.backbone_eval` to [endgame.md#bigger-images](endgame.md#bigger-images)**: it
  only matters as the gate for `model.checkpoint_backbone`, needed for the bigger-images work, not
  as an isolated setting on the current recipe. `baseline0`/`baseline1` (ResNet-50D) trained with
  it on without a recorded comparison against it off.

## Candidates for later queues

- **`optimizer.backbone_lr: 1e-4`, head `6e-4`** (research A1; weak). Backbone 1e-4 was part of
  both losing 2026-09-12 configurations, the control (1e-4/3e-4) and the faster head (1e-4/1e-3,
  −0.0012), while doubling both rates to 2e-4/6e-4 gained +0.0045
  ([2026-09-12 results](../validation/results/2026-09-12-overnight-results.md)), and 3e-4/9e-4 now
  ties. `08-slower-head` covers the ratio from the other side.
- **Other values of the existing augmentation families**, after `07-affine-off` (hypothesis):
  probabilities and strengths of the intensity, affine and noise families, one per run, such as
  the milder side of the strong affine that lost −0.0024, and reflection probabilities other than
  0.5. Ordering, interactions and new families are
  [small changes](small-changes.md#augmentation-families).
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

## Already measured

On the restored code, 2026-09-17 queue
([result](../validation/results/2026-09-17-overnight-results.md)), paired with its seed-42
control (0.892995): three control seeds span 0.0032 (0.889837–0.892995, standard deviation
0.0016), so a bootstrap interval that clears zero can still be seed luck. EMA 0.999 +0.0002 and
SWA over 3 epochs +0.0001 at the best epoch (+0.0005 at the last); `clip 0.001` −0.0003,
`gradient_clip_val 5` −0.0003, warm-up 0 −0.0015, `gradient_clip_val 0.5` −0.0016, head LR 3e-4
−0.0017, noise 0.3 −0.0019, `clip 0.01` −0.0021, `beta2 0.99` −0.0025, 20 epochs −0.0027 (peaks at
epoch 12, then overfits), affine off −0.0041 (interval excludes zero).

On the restored code (2026-09-16, user-launched, no fresh control yet;
[result](../validation/results/2026-09-16-restored-code-hyperparameter-runs.md)): `min_lr 0.1`
+0.0001, series dropout 0.3 +0.0001 and 0.25 −0.0025, 20% fewer slices −0.0041 (20% faster
training) and 35% fewer slices −0.0051 (37% faster). Replicates of the same setting differed by
0.0003–0.0007, but the 2026-09-17 seed controls spread by 0.0032, so single-run deltas of this
size are noise.

On `00-baseline` (2026-09-15):

`01-sigma30` +0.000039, `02-higher-lr` −0.000222, `03-min-lr-0.1` +0.000408,
`04-series-dropout-0.3` −0.000492, `05-wd-0.05` −0.000837, `06-affine-strong` −0.002389, and
the `crop130` rerun −0.000263
([overnight results](../validation/results/2026-09-15-overnight-results.md)). That queue completed
seven of sixteen runs: `07-crop130` stopped during epoch 8 when the machine or WSL stopped, and
runs `08-clip-0.01` through `15-backbone-eval` never started. Earlier settled searches are listed
in the [launcher doc](../validation/overnight-hyperparameter-queue.md).
