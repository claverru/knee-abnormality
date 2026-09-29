# Overnight queue

Hyperparameter-only runs: values of existing training options on the
[current recipe](../modeling/current-model.md), one change per run against the incumbent, at its
fold and recorded training seed. The
[queue launcher](../../scripts/tools/train_overnight.py) takes split, fold and seed from the
control config. A [small change](small-changes.md)
enters this queue once it is implemented; other ideas belong to the other [backlog](INDEX.md) documents. The queue runs
unattended and the user launches it, detached from agent sessions; the
[queue launcher guide](../validation/overnight-hyperparameter-queue.md) owns the procedure. Record
results in [validation results](../validation/results/INDEX.md), then remove the finished entries
from this file.

## Next queue

Empty. The ReXNet-era entries were cleared on 2026-09-29, when the champion became the RadImageNet
ResNet-18 ([state](../ongoing/state.md#recipe-in-force)). More epochs gained about +0.001 on
ResNet-18 at 14 epochs for 40% more time. More input per study, meaning denser per-type budgets
or more triplets, is [deferred to the endgame](endgame.md#test-time-augmentation) for iteration
speed. The augmentation families were searched by the
[augmentation campaign](../validation/results/2026-09-26-augmentation-campaign.md), and the batch
and weight-decay candidates rested on B0 and 2026-09-12 evidence.

**No delta recorded before 2026-09-19 is a prior for a new queue.** Those historical runs used
ordinary BCE; the recipe in force uses class-balanced BCE, which by itself moves fold 0 by about
0.008. ReXNet-200 and earlier backbones measured everything below; ResNet-18 measurements are in
the [RadImageNet campaigns](../../autoresearch/campaigns/radimagenet-rn18/findings.md).
Keep `validation_image_multiplier: 1` in comparisons.

## Already measured

Completed runs belong to their result records. These exclusions prevent them being queued again
as untried settings:

- The September 20 queue's twelve candidate arms completed with no adoption, including
  balanced-BCE head LR 0.0012/0.0024 and stronger intensity probability/gamma/contrast.
- Shared sampler scale 0.60–1.10 and the two sampler-shape proposals have been measured; no
  fraction or shape repeat is queued here.
- More epochs under ordinary BCE, fixed-width sigma 50, and rejected attention/readout variants
  are not fresh queue candidates.
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
