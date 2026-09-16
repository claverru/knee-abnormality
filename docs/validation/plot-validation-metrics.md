# Plot validation metrics

Plot epoch metrics from completed run folders with:

```bash
rtk proxy uv run --locked python -m scripts.tools.plot_validation_metrics baseline0 baseline1 baseline4
```

The command writes `outputs/analysis/validation_metrics.png` by default and shows the
full-validation, gold and report-derived macro AUCs plus soft-target BCE. Use
`--output` to select another PNG path. Undefined gold or report-derived macro
AUCs (`null` in epoch JSON) appear as gaps; a subgroup with no defined epochs
has no visible curve. Full-validation AUC and BCE must remain numeric.

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
