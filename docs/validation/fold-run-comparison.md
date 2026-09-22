# Fold-run comparison

Compare a candidate run with a reference run on CPU, from saved predictions and
checkpoint tensors, without building a model:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m scripts.tools.compare_fold_runs --reference outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/00-baseline --candidate outputs/training/overnight-20260914T220805Z-ozsz4qbu/runs/03-min-lr-0.1 --output-dir outputs/analysis/min-lr-0.1-vs-00-baseline --allowed-config-differences name,optimizer.minimum_lr_ratio
```

The example compares the `03-min-lr-0.1` candidate with its `00-baseline` control. Add
`--allowed-input-identity-differences preprocessing_identity` when the runs come from different
sources. Allowed configuration differences are dotted keys or prefixes (default `name`);
allowed input-identity differences are top-level keys. Any other difference fails,
as do selected predictions that do not reproduce their recorded macro AUC: the
`prediction_macro_auc` in `selected_checkpoint.json` for runs that predict at another
multiplier than their epoch validation (×4 since 2026-09-22), otherwise the selected epoch's.
Runs predicted at different multipliers differ in `validation.prediction_image_multiplier` and
are rejected unless that key is allowed; re-predict the older run with
[`repredict_run`](../../scripts/tools/repredict_run.py) instead.
The output directory must be fresh. `comparison.json` records:
- selected epochs, full-fold and per-target AUC, soft-target BCE, and gold and
  report-derived metrics;
- family means for all targets, the meniscus and OA pairs and the other eight targets;
- exclusive-positive margin AUCs and lateral cross-talk for both runs, using the
  [encoding diagnostic](manual-encoding-usage-diagnosis.md)'s shared definitions;
- vendor strata with positive counts, prediction changes, epoch curves, timing and peak
  memory, parameters present in only one run and changed `src/` and `configs/` members.

Paired report-group bootstrap draws (2,000, seed 20260914) give candidate-minus-reference
intervals for each family and both margins. They exclude training-seed and
checkpoint-selection variability. Vendor comes from the saved study-header audit and is
an acquisition proxy. The tool generalizes the historical knee-side analysis, which
stays unchanged.
