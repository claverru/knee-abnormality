# 2026-09-20 overnight hyperparameter results

**All fourteen runs completed in 6 h 8 m; no hyperparameter change is adopted.**
The two controls differ by 0.003360 AUC, and control B beats every candidate.
The strongest candidate, milder scale augmentation, is inconclusive against both controls.

## Recipe and execution

Queue: `outputs/training/overnight-20260919T235434Z-8ol_1c30/`.
All training exit codes are zero; the final `validation_metrics.png` exists.
Training ran from 2026-09-19 23:54:34 UTC to 2026-09-20 06:02:12 UTC,
about 1,571–1,580 seconds per run. Each run completed 4,380 optimizer steps,
with 19.0–20.0 GiB peak allocated GPU memory.

Fixed recipe: ReXNet-200, class-balanced BCE, Steven v4, `all_5fold_v3` fold 0,
validation x1, ten epochs, current main-tree Gaussian sampler at fraction 0.867,
study-total floor 26, unchanged per-type budgets. The autoresearch stratified sampler
was not used. Each run received a distinct random seed saved in its effective config.
Candidates differ from the control only by the named override and run/seed fields.

The queue retains configurations, source snapshots, input identities, selected
checkpoints, epoch metrics, predictions, `runtime.json`, `status.tsv`, and `summary.tsv`.

## Results

The control mean is the mean of their individual AUCs, **0.894902**, not an ensemble.
The table uses saved selected-checkpoint scores; recomputed prediction AUCs below differ
only at float32 rounding precision. Epochs count from one.

| Run | Seed | Epoch | Macro AUC | Delta vs control mean |
|---|---:|---:|---:|---:|
| `00-control` | 1133376617 | 7 | 0.893222 | -0.001680 |
| `01-head-lr-0.0012` | 3116037075 | 8 | 0.892640 | -0.002261 |
| `02-intensity-p-0.75` | 701916628 | 9 | 0.894401 | -0.000500 |
| `03-backbone-lr-0.0003` | 2088294760 | 10 | 0.893492 | -0.001410 |
| `04-dropout-0.1` | 911887175 | 9 | 0.894426 | -0.000476 |
| `05-gamma-0.70-1.30` | 3148850708 | 9 | 0.894224 | -0.000678 |
| `06-contrast-0.80-1.20` | 1154472597 | 10 | 0.889906 | -0.004996 |
| `07-affine-p-0.5` | 121408477 | 10 | 0.892986 | -0.001916 |
| `08-head-lr-0.0024` | 506593792 | 8 | 0.891483 | -0.003419 |
| `09-rotation-7` | 1810929613 | 9 | 0.894372 | -0.000529 |
| `10-scale-0.97-1.03` | 1664739758 | 10 | 0.895868 | +0.000966 |
| `11-weight-decay-0.03` | 831833751 | 9 | 0.893593 | -0.001309 |
| `12-warmup-2` | 639911864 | 7 | 0.893261 | -0.001641 |
| `13-control-repeat` | 2263738815 | 9 | 0.896582 | +0.001680 |

## Paired bootstrap

Five comparisons were run on saved predictions, using 2,000 paired resamples of
852 report groups and seed 20260914. All 2,000 draws were valid in each comparison.
This resampling seed is separate from training seeds. Differences are candidate
minus reference, and intervals are the bootstrap 2.5th and 97.5th percentiles.
Only the best candidate, strongest apparent regression and the control pair were
bootstrapped; no claim is made that every candidate received a bootstrap comparison.

| Candidate | Reference | AUC delta | 95% interval | Evidence |
|---|---|---:|---|---|
| Milder scale | Control A | +0.002646 | [-0.000955, +0.006478] | Artifact (`outputs/analysis/o/overnight-20260920-scale-vs-00-control/comparison.json`, deleted) |
| Milder scale | Control B | -0.000714 | [-0.004289, +0.002950] | Artifact (`outputs/analysis/o/overnight-20260920-scale-vs-13-control-repeat/comparison.json`, deleted) |
| Stronger contrast | Control A | -0.003316 | [-0.007352, +0.000775] | Artifact (`outputs/analysis/o/overnight-20260920-contrast-vs-00-control/comparison.json`, deleted) |
| Stronger contrast | Control B | -0.006675 | [-0.010681, -0.002738] | Artifact (`outputs/analysis/o/overnight-20260920-contrast-vs-13-control-repeat/comparison.json`, deleted) |
| Control B | Control A | +0.003360 | [-0.000219, +0.007073] | Artifact (`outputs/analysis/o/overnight-20260920-control-repeat/comparison.json`, deleted) |

Both scale-versus-control comparisons and the control-to-control comparison are
inconclusive at 95%: each macro-AUC interval includes zero. The scale comparisons
have matching input identities and differ in configuration only by scale range,
training seed and run name; the controls differ only by seed and name. Retain
`data.affine_augmentation.scale_range: [0.95, 1.05]` in the current recipe.

## Findings

- **Keep the recipe unchanged.** Milder scaling `[0.97, 1.03]` is the best candidate
  at 0.895868, but only +0.000966 above the control mean and below control B. Both
  macro-AUC bootstrap intervals include zero. It is a lead, not an established gain.
- **Wider contrast `[0.80, 1.20]` is not worth adopting.** It scores 0.889906,
  approximately -0.004996 against the control mean. Its macro interval is negative
  against control B but includes zero against A. This supports avoiding the setting,
  without claiming a seed-independent causal regression.
- **Higher head LR did not improve these runs.** The 0.0012 and 0.0024 arms score
  0.892640 and 0.891483. Both are below both controls; keep 0.0006. These are two
  different seeds, not a cleanly established monotonic learning-rate response.
- **Other apparent positives depend on choosing the weaker control.** Dropout 0.1,
  intensity probability 0.75, gamma `[0.70, 1.30]` and rotation 7 degrees all
  fall below the two-control mean. Do not stack them into a new recipe.
- **Two controls matter.** Their 0.003360 spread exceeds the largest candidate gain
  against control A. This is observed seed variation, not a precisely estimated
  noise distribution or a universal rejection threshold. The bootstrap conditions
  on fitted models and does not capture seed variability or correct selection
  among twelve candidates.

## Follow-up and tooling

No further training or submission was launched, and the autoresearch campaign stays stopped.
The ordinary queue is complete; these arms left the actionable queue and stay
recorded here to avoid accidental repetition. No incumbent or default hyperparameter changed.

The printed post-run comparison commands used positional run paths although the
comparator requires `--reference` and `--candidate`. The launcher was changed to print the
correct named arguments. This affected only the printed follow-up commands, not
training, scoring or the comparisons recorded above.
