# 2026-09-19 untried-hyperparameter queue

**No arm cleared the run-to-run floor, and the informative result is the control.** Five runs on
2026-09-19, fold 0 of `all_5fold_v3`, validation ×1, each one unseeded run with its seed in its
config. Every arm was a value no run had ever used: a scan of the 143 saved
`effective_config.json` files put `optimizer.epsilon` at 1e-8 in 143 of 143, `optimizer.beta1` at
0.9 in 143 of 143, `data.intensity_augmentation.probability` at 0.5 in 143 of 143 and
`data.noise_augmentation.std_range` at `[0.005, 0.02]` in 142 of 142.

Queue: `outputs/training/overnight-20260919T120130Z-i2oqlnls/` (`status.tsv`, `summary.tsv`,
`configs/`, `runs/`, `validation_metrics.png`), launched with `scripts/adhoc/train_overnight.py`.
Total 2 h 11 m, 1,532–1,557 s of training per run, 19.0–19.6 GiB peak.

This is the first queue on the [class-balanced BCE](../../modeling/supervision.md#fixed-class-balanced-bce)
recipe, so no earlier delta is a prior for it: only `balanced-bce` carried that loss before tonight.

## Runs

| Run | Change | Seed | Epoch | Macro AUC | vs control | Gold | Soft BCE |
|---|---|---:|---:|---:|---:|---:|---:|
| `04-noise-std-0.01-0.04` | `data.noise_augmentation.std_range: [0.01, 0.04]` | 10764577 | 10 | 0.895091 | +0.00105 | 0.9403 | 0.4814 |
| `03-beta1-0.95` | `optimizer.beta1: 0.95` | 663550682 | 10 | 0.894715 | +0.00068 | 0.9392 | 0.4792 |
| `00-control` | none | 927831896 | 9 | 0.894037 | — | 0.9503 | 0.4834 |
| `01-eps-1e-6` | `optimizer.epsilon: 1e-6` | 1491446251 | 10 | 0.893985 | −0.00005 | 0.9337 | 0.4837 |
| `02-intensity-p-0.25` | `data.intensity_augmentation.probability: 0.25` | 788474613 | 9 | 0.893221 | −0.00082 | 0.9396 | 0.4810 |

Reference on the same recipe, seed 42, from an earlier source snapshot:
[`balanced-bce`](2026-09-19-balanced-bce-result.md) 0.895282 at epoch 8.

## Paired bootstrap

2,000 paired report-group draws over 852 groups, saved-prediction comparisons against `00-control`
([procedure](../fold-run-comparison.md)). All three intervals contain zero.

| Candidate | Macro AUC delta | 95% interval | Meniscus | OA | Other eight | Artifact |
|---|---:|---|---:|---:|---:|---|
| `04-noise-std-0.01-0.04` | +0.00111 | [−0.00260, +0.00474] | +0.0074 | −0.0018 | +0.0002 | `outputs/analysis/04-noise-std-0.01-0.04-vs-00-control/` |
| `balanced-bce` | +0.00125 | [−0.00275, +0.00535] | +0.0072 | −0.0016 | +0.0005 | `outputs/analysis/balanced-bce-vs-00-control/` |
| `03-beta1-0.95` | +0.00071 | [−0.00281, +0.00428] | +0.0043 | −0.0056 | +0.0013 | `outputs/analysis/03-beta1-0.95-vs-00-control/` |

Largest per-target moves: `04` Baker's +0.0098, contusion −0.0096, medial meniscus +0.0085;
`03` MCL −0.0211, fracture +0.0178, Baker's +0.0108; `balanced-bce` MCL −0.0172, lateral meniscus
+0.0112, contusion +0.0076.

Comparing `balanced-bce` needed three allowances, all recorded in its `comparison.json`. Its
`validation_predictions.csv` does not reproduce its recorded AUC to the tool's 1e-12 tolerance —
it is a re-prediction of the selected checkpoint under the older source snapshot and matches
`validation_epoch_007.csv` to a mean absolute difference of 1e-5 — so the comparison ran against a
staged copy carrying the epoch-7 predictions. That run never wrote `fold_0/timing.json`, so the
record holds an explicit marker instead of runtime figures, and `--allowed-input-identity-differences
preprocessing_identity` was needed for its differing `sources_sha256`. Seed and source change
together there; their contributions are not separable.

## Findings

- **Class-balanced BCE replicates.** `00-control` at 0.894037 sits 0.00125 from `balanced-bce`
  at a different seed and a different source snapshot, and both stand about 0.008 above every
  ordinary-BCE run on the same sampler (`rexnet-fold0` 0.883671, `sigma-fraction-0867` 0.886486).
  The 2026-09-19 adoption rested on one run against historical references; it now has two runs.
- **Nothing to adopt from the arms.** The two positives, `04` at +0.00105 and `03` at +0.00068,
  are below the 0.00175 pooled replicate standard deviation and their intervals span zero.
- **Weaker intensity augmentation loses**, the only negative worth noting at −0.00082. The family
  is load-bearing, so the untested direction is the stronger side, not the weaker one.
- **The budget binds.** Three of the four arms and both positives selected epoch 10, the last one;
  the control peaked at 9 and `balanced-bce` at 8. Under this loss the curves are still rising at
  the end of the budget, which the [closed epoch axis](../../backlog/overnight.md#candidates-for-later-queues)
  measured only under ordinary BCE.

## Run-to-run spread

Comparing the six balanced-BCE fold-0 runs with the six ordinary-BCE fold-0 ten-epoch runs
(`rexnet-fold0`, the four sigma-sweep arms, `sigma-fraction-0867`):

| Measure | Balanced BCE | Ordinary BCE |
|---|---:|---:|
| Best macro AUC, standard deviation across runs | 0.00078 | 0.00258 |
| Best macro AUC, range across runs | 0.00206 | 0.00674 |
| Macro AUC standard deviation over each run's last four epochs, averaged | 0.00067 | 0.00079 |
| Per-target AUC standard deviation across runs, averaged over the twelve | 0.00447 | 0.00476 |

Individual label AUCs vary as much as they ever did; what tightened is their average. If the
twelve labels moved independently, both groups would show a macro standard deviation near 0.0014;
the ordinary group sits at 1.74× that and the balanced group at 0.55×, so label deviations drift
together under ordinary BCE and partly cancel under balanced BCE.

**This is suggestive, not established.** The ordinary group contains the sigma sweep, 0.60 to
1.10, a far larger perturbation than this queue's epsilon, beta1 and augmentation nudges, and one
that plausibly moves every label the same way, so part of the three-fold gap belongs to the arms
rather than the loss. The within-run and per-target rows are the parts that survive that
objection, and the within-run gap is small. The working rule in
[project state](../../ongoing/state.md) — pooled standard deviation 0.00175, single-run
differences below about 0.003 uninformative — stands until more controls land on this recipe.
Every future queue adds one for free.
