# Plot validation metrics

Plot epoch metrics from completed run folders with:

```bash
rtk proxy uv run --locked python -m scripts.tools.plot_validation_metrics baseline0 baseline1 baseline4
```

The command writes `outputs/analysis/validation_metrics.png` by default with six
panels in two rows. The top row holds the ranking metrics, where higher is better:
hard macro AUC, [soft macro AUC](results/2026-09-23-validation-auc-variants.md)
and gold macro AUC. The bottom row holds training BCE and validation soft-target BCE
side by side for reading overfitting, then the per-target AUC at the best epoch. Use
`--output` to select another PNG path; the default is overwritten on every call. Runs with
several folds draw one line per fold.

Autoresearch experiments keep their run in a `training/` folder; the legend names them by
their experiment folder without its timestamp and hash. Runs with no epoch metrics yet are
skipped with a note, so a campaign's control and all its experiments plot with one glob:

```bash
rtk proxy uv run --locked python -m scripts.tools.plot_validation_metrics k05-clean \
  outputs/autoresearch/noisy-labels/experiments/*/training --output outputs/analysis/noisy-labels.png
```

Soft macro AUC is computed from each epoch's saved `validation_epoch_*.csv` against
the run's frozen split targets, so runs trained before it was logged are plotted
too. The split named in `effective_config.json` must load. Training BCE is the
epoch-mean `train_loss_epoch` from `logs/version_*/metrics.csv`; a fold without it fails. It
is the loss as
optimized, including class-balance weights, so only compare it across runs that
use the same class-balance setting.

The per-target panel draws the 12 target AUCs from each series best macro-AUC
epoch as grouped bars, with a dashed horizontal line at that epoch macro AUC so
targets above and below the run average are visible at a glance. Bars start at
the 0.05 step below the lowest plotted AUC and the axis label states that floor,
which keeps 0.001 differences readable; one or two series also get per-target
value labels. The run legend in the first panel names the best epoch each bar
group comes from.

Undefined gold macro AUCs (`null` in epoch JSON) appear as gaps; full-validation
AUC, BCE and per-target AUCs must remain numeric.

A dashboard workflow linked ideas, experiments, local metrics, and Kaggle
submissions. Models described as `tabfpn3` and simpler
alternatives were fit to local/public results to predict promising combinations.
No training data, model definition, or predictive-accuracy result for this
meta-modeling workflow was recorded.

Interactive notebooks were useful for inspecting image/label pairs and
preprocessing. A reported failure mode of extended automated experimentation
was producing more benchmark runs without explaining why models with similar
metrics behaved differently. The actionable lesson is to compare their
predictions and failure cases, not only collect aggregate scores.
