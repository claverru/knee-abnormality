# 3dsampling submission

The selected `3dsampling` fold-0 checkpoint (epoch 9, index 8) was exported on
2026-09-13 to `outputs/submission/exports/knee-3dsampling-20260913` at 2× images and batch 2.
Checkpoint SHA-256 is
`3a63b49d59d33be6b3f254639572aa3c34606a397107e5bcaa9ac1cdb42d6819`.
Export readiness and bundle integrity checks passed, with preparation taking
0.55 seconds and `local_inference: not_requested`.

The detached automatic submission job uses private dataset
`claverru/knee-3dsampling-20260913` and notebook
`claverru/knee-3dsampling-20260913-inference`. Dataset version 1 supplied
notebook version 1, Kaggle script version `349556156`. Its output was verified
and submission **56210103** completed with **0.930 public AUC**, confirmed
through the Kaggle API at 15:46:48 UTC on 2026-09-13. This is **+0.001** over
the previous spatial-token champion. Exact identities and transitions are in
the [workflow record](../../../outputs/submission/records/knee-3dsampling-20260913.json).
The [saved validation comparison](../../validation/results/2026-09-13-3dsampling-validation-comparison.md)
shows +0.000458 macro AUC versus the previous champion, with slightly worse BCE.
