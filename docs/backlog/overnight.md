# Overnight queue

Hyperparameter-only runs: values of existing training options on the
[current recipe](../modeling/current-model.md), one change per run against a fresh control
(`all_5fold_v3`, fold 0, one recorded random seed per run). A [small change](small-changes.md)
enters this queue once it is implemented; other ideas belong to the other [backlog](INDEX.md) documents. The queue runs
unattended and the user launches it, detached from agent sessions; the
[queue launcher](../validation/overnight-hyperparameter-queue.md) owns the procedure. Record
results in [validation results](../validation/results/INDEX.md), then remove the finished entries
from this file.

## Next queue

The class-balance exponent sweep is complete, run by hand on 2026-09-21–22 and judged on the
public leaderboard: unweighted 0.943, k=0.25 0.944, k=0.5 0.945, k=0.75 0.942, k=1 0.940
([result](../validation/results/2026-09-21-class-balance-exponent.md)). No run is queued.

The [2026-09-20 fourteen-run queue](../validation/results/2026-09-20-overnight-hyperparameter-results.md)
completed in 6 h 8 m. No change is adopted: milder scaling was the best candidate
but inconclusive against both controls; control B beat every candidate. The head-LR,
intensity, affine, dropout, decay and warm-up arms are recorded under already measured.
The earlier [five-run queue](../validation/results/2026-09-19-untried-hyperparameter-queue.md)
also yielded no adoption.

**No delta recorded before 2026-09-19 is a prior for a new queue.** Those historical runs used
ordinary BCE; the recipe in force uses class-balanced BCE, which by itself moves fold 0 by about
0.008. The [result links](#already-measured) also include the newer balanced-BCE comparisons.
Keep `validation_image_multiplier: 1` in comparisons.

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

- **The epoch budget is reopened under this loss.** Three of the four arms on 2026-09-19 selected
  epoch 10, the last one, and so did both positives; the control peaked at 9. The measurements
  that closed the axis were all made under ordinary BCE, where the peak came earlier.
- **Denser per-type budgets** (hypothesis; the only untested side of a measured axis). Raise
  `data.series_types[*].max_images` above today's 18/18/15/15/15/15 at a fixed sampler width and
  crop. Only the cheaper side has been measured and it loses monotonically: 15/15/12/12/12/12
  −0.0041 and 12/12/9/9/9/9 −0.0051, 20% and 37% faster
  ([runs](../validation/results/2026-09-16-restored-code-hyperparameter-runs.md)). Public evidence
  points the same way, with a CoAtNet pipeline gaining 0.917 → 0.926 → 0.928 → 0.932 as slice
  density rose at a fixed reach
  ([public research](../research/public-research.md#slice-density-and-the-board-2026-09-18-re-sweep)),
  but a second team whose pipeline pools per-series slots through masked attention — our structure,
  not the window-scoring one — measured **−0.0045** going denser on the same axis
  ([counter-report](../research/public-research.md#the-2026-09-20-re-sweep)), so the public evidence
  no longer points one way and our own cheaper-side losses carry this candidate on their own.
  **Memory rules it out as a one-change run:** `sigma-fraction-0867` already peaks at 20.8 GiB
  with `study_triplet_total: 26`, well above the 18.9 GiB of the era that produced the 23.5 GiB
  estimate, so 21/21/18/18/18/18 exceeds the card. It needs a second change (accumulation, or
  `model.checkpoint_backbone`) to fit, and stops being a one-change run.
  `data.study_triplet_total` is the
  other knob on the same axis and 32 is already measured (+0.0036, inside the control spread).
- **Other values of the existing augmentation families**, after `07-affine-off` (hypothesis):
  probabilities and strengths of the intensity, affine and noise families, one per run, such as
  the milder side of the strong affine that lost −0.0024, and reflection probabilities other than
  0.5. Ordering, interactions and new families are
  [small changes](small-changes.md#augmentation-families).
- **`trainer.accumulate_grad_batches: 2`** (untried; two-sided). It buys an effective batch of 16
  at unchanged memory but halves optimizer steps to 2,190, the axis the b0 queue says helps.
- **Low value:** `data.sampling_sigma_mm: 50` is retired with the fixed-width sampler;
  `optimizer.weight_decay: 0.001`, since 0.003 and 0.03 are settled, 0.05 lost
  −0.0008 and 0.003 gained only +0.0002 on the 2026-09-12 recipe.
- **Out of scope:** `model.channels_last` is a speed question for a bounded, user-launched
  `profile_training.py` measurement, not an overnight slot. NHWC kernels change bf16 outputs only
  in low-order bits. A 2026-09-11 channels-last ReXNet-100 profile
  (`outputs/profiling/rexnet100-diagnosis-repeat-20260911`) ran at 0.54 studies/s, against 35.43
  in the later [production-path profile](../modeling/training-runtime-and-memory.md); the design
  pass attributes this to about 10 s of cuDNN setup for every new flattened image count, which
  MIL batches change every step.

## Already measured

Completed runs belong to their result records. These exclusions prevent them being queued again
as untried settings:

- The September 20 queue's twelve candidate arms completed with no adoption, including
  balanced-BCE head LR 0.0012/0.0024 and stronger intensity probability/gamma/contrast.
- Shared sampler scale 0.60–1.10 and the two sampler-shape proposals have been measured; no
  fraction or shape repeat is queued here.
- More epochs under ordinary BCE, fixed-width sigma 50, and rejected attention/readout variants
  are not fresh queue candidates. The epoch question under balanced BCE remains open above.
- Explicit user decisions above remain in force; the result history does not reopen them.

| Recipe | Completed comparisons |
|---|---|
| Balanced BCE | [September 20 fourteen-run queue](../validation/results/2026-09-20-overnight-hyperparameter-results.md) |
| Balanced BCE | [September 19 five-run queue](../validation/results/2026-09-19-untried-hyperparameter-queue.md) |
| Ordinary BCE | [September 19 folds, epoch arms and sigma sweep](../validation/results/2026-09-19-fold-ensembles-and-sigma-sweep.md) |
| Ordinary BCE | [September 18 queue](../validation/results/2026-09-18-overnight-results.md) and [B0 queues](../validation/results/2026-09-18-b0-hyperparameter-queues.md) |
| Ordinary BCE | [September 17 queue and EMA/SWA](../validation/results/2026-09-17-overnight-results.md) |
| Ordinary BCE | [September 16 restored-code runs](../validation/results/2026-09-16-restored-code-hyperparameter-runs.md) |
| Ordinary BCE | [September 15 queue](../validation/results/2026-09-15-overnight-results.md) |
| Sampling campaign | [Sampler-shape outcomes](../data-preparation/peripheral-coverage.md#what-is-settled-and-what-is-not) |

Earlier settled searches are linked from the [launcher guide](../validation/overnight-hyperparameter-queue.md).
