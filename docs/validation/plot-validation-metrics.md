# Plot validation metrics

Plot epoch metrics from completed run folders with:

```bash
rtk proxy uv run --locked python -m scripts.tools.plot_validation_metrics baseline0 baseline1 baseline4
```

The command writes `outputs/analysis/validation_metrics.png` by default with four
panels: full-validation macro AUC, gold macro AUC, per-target AUC at the best
epoch, and soft-target BCE. Use `--output` to select another PNG path.

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
