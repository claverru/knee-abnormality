# 140-mm crop submission

`crop140mm` epoch 8 (checkpoint index 7) completed scoring at **0.925** in
submission `56182948`, notebook version 1. The
[workflow record](../../../outputs/submission/records/knee-crop140mm-20260912.json)
pins the dataset version, notebook version, submission response and API score.
The [immutable bundle](../../../outputs/submission/exports/knee-crop140mm-20260912) uses
224-pixel inputs at 0.625 mm spacing, 2× images and batch size two.

Checkpoint:
`outputs/training/crop140mm/fold_0/best-epoch=07-val_macro_auc=0.88612.ckpt` (deleted on 2026-09-16),
SHA-256 `f3fd7b72bed1315297d54ee92742965ebb9c892ae0683c717bf20ad0947d3a92`.
Bundled-source SHA-256:
`babcebfccc5561c7f813acd8c1f68effaf8652e30d3b01428b8f5e2c9b6195ff`.
The manifest records that bundled source differs from the training snapshot;
export occurred during preprocessing package changes. Metadata, strict loading
and bundle integrity checks passed; optional local inference was not requested.
The completed score confirms this artifact's hidden-data execution. Workflow
elapsed times include scheduling and polling, and are not official inference
runtime. This remains the frozen 140-mm control; the overnight higher-LR
artifact remains the 150.08-mm comparison reference.
