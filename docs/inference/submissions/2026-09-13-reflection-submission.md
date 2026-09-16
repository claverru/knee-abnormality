# Reflection submission

The selected `spatial-tokens-reflection` fold-0 checkpoint (epoch 9, index 8)
was exported on 2026-09-13 to `outputs/submission/exports/knee-reflection-20260913` at
2× images and batch 2. Checkpoint SHA-256 is
`bc55feaa24cd6e6880bbb4ce0f0f452a9ea3302b5894928b117d508852bb928b`.
Metadata, strict CPU model loading and bundle integrity checks passed; bundled
source matches the training snapshot. Export took 0.70 seconds, with optional
local inference not requested.

The detached submission service launched at 17:24:48 UTC using private dataset
`claverru/knee-reflection-20260913`, version 1, and private offline notebook
`claverru/knee-reflection-20260913-inference`, version 1, Kaggle script version
`349584477`. Its output passed verification and submission **56211972** completed
with **0.931 public AUC**, confirmed through the Kaggle API on 2026-09-13.
The runner recorded completion at 17:41:06 UTC in the
[workflow record](../../../outputs/submission/records/knee-reflection-20260913.json).
This was **+0.001** over 3dsampling and the best public result at that time.
The later [sigma-40 submission](2026-09-14-reflection-sigma-40-submission.md) reached 0.936.
The [saved validation comparison](../../validation/results/2026-09-13-reflection-validation-comparison.md)
shows +0.001012 macro AUC versus that previous champion, with lower BCE and lower
eleven-study gold AUC. The API provides no official hidden-data inference runtime.
