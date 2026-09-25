# Overnight queue

Hyperparameter-only runs: values of existing training options on the
[current recipe](../modeling/current-model.md), one change per run against the incumbent (fold 0
of `all_5fold_scanner_v1`, the incumbent's recorded training seed). The
[queue launcher](../../scripts/tools/train_overnight.py) still pins `all_5fold_v3` and must move to
the current split before the next queue ([next](../ongoing/next.md)). A [small change](small-changes.md)
enters this queue once it is implemented; other ideas belong to the other [backlog](INDEX.md) documents. The queue runs
unattended and the user launches it, detached from agent sessions; the
[queue launcher guide](../validation/overnight-hyperparameter-queue.md) owns the procedure. Record
results in [validation results](../validation/results/INDEX.md), then remove the finished entries
from this file.

## Next queue

- **`trainer.max_epochs: 12`** (queued 2026-09-24, user decision) against the current control
  `knee-geometry-merged-2` (seed 638120755, ×4 0.900955). Under ordinary BCE, 15 epochs gained
  +0.0023 and +0.0025 on ReXNet and +0.0029 on B0, all inside seed noise
  ([overnight results](../validation/results/2026-09-18-overnight-results.md),
  [epoch arms](../validation/results/2026-09-19-fold-ensembles-and-sigma-sweep.md)). Under
  class-balanced BCE several runs selected their last or second-to-last epoch, and the
  [noisy-labels campaign](../validation/results/2026-09-23-noisy-labels-campaign.md) found the
  model does not memorize its labels within ten epochs.

**No delta recorded before 2026-09-19 is a prior for a new queue.** Those historical runs used
ordinary BCE; the recipe in force uses class-balanced BCE, which by itself moves fold 0 by about
0.008. The [result links](#already-measured) also include the newer balanced-BCE comparisons.
Keep `validation_image_multiplier: 1` in comparisons.

## Candidates for later queues

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
- **Low value:** `optimizer.weight_decay: 0.001`, since 0.003 and 0.03 are settled, 0.05 lost
  −0.0008 and 0.003 gained only +0.0002 on the 2026-09-12 recipe.

## Already measured

Completed runs belong to their result records. These exclusions prevent them being queued again
as untried settings:

- The September 20 queue's twelve candidate arms completed with no adoption, including
  balanced-BCE head LR 0.0012/0.0024 and stronger intensity probability/gamma/contrast.
- Shared sampler scale 0.60–1.10 and the two sampler-shape proposals have been measured; no
  fraction or shape repeat is queued here.
- More epochs under ordinary BCE, fixed-width sigma 50, and rejected attention/readout variants
  are not fresh queue candidates. The epoch question under balanced BCE remains open above.
- User decisions of 2026-09-16, on the ordinary-BCE recipe: `min_lr 0.1` dropped after three
  unestablished measurements ([result](../validation/results/2026-09-16-restored-code-hyperparameter-runs.md));
  series dropout 0.1 dropped, since 0.2 and 0.3 are flat; reflection 0.25, intensity-off and batch
  size 4 dropped; `model.backbone_eval` moved to [bigger images](endgame.md#bigger-images). The
  result history does not reopen them.

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
