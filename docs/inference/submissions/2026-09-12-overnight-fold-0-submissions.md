# Overnight fold-0 submissions

All three submissions from `outputs/training/overnight-20260911T234416Z-8du9a4nh`
completed successfully, confirmed through the Kaggle API on 2026-09-12.
They use notebook version 1, 2× images and batch size two. Epochs below are
numbered from one. Each record pins the uploaded bundle manifest and dataset
version, notebook version, submission ID, API score and workflow history.

| Run | Selected epoch | Submission ID | Workflow record |
|---|---:|---:|---|
| `02-longer` | 12 | 56181804 | [Record](../../../outputs/submission/records/knee-overnight-02-longer-20260912.json) |
| `04-higher-lr` | 10 | 56181821 | [Record](../../../outputs/submission/records/knee-overnight-04-higher-lr-20260912.json) |
| `05-faster-head` | 10 | 56181919 | [Record](../../../outputs/submission/records/knee-overnight-05-faster-head-20260912-v2.json) |

The selected checkpoint for the overnight winner is
`outputs/training/overnight-20260911T234416Z-8du9a4nh/runs/04-higher-lr/fold_0/best-epoch=09-val_macro_auc=0.88294.ckpt` (deleted on 2026-09-16),
SHA-256 `900add74a5473bf6c220b6287f90c96e46881a37776abbc53ec1d7d20a24e670`.
Its immutable bundle is
[`outputs/submission/exports/knee-overnight-04-higher-lr-20260912`](../../../outputs/submission/exports/knee-overnight-04-higher-lr-20260912).
The other bundles use the corresponding run names under `outputs/submission/exports/`.
All exports passed metadata, strict model-loading and bundle integrity checks;
optional local inference was not requested. Successful scored submissions
establish hidden-data execution for these artifacts, but the API does not
provide official hidden-run runtime and no efficiency ranking is inferred.
