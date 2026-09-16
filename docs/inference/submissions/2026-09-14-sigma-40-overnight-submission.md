# Sigma-40 overnight submission

The selected `02-central-sampling` checkpoint from
`outputs/training/overnight-20260913T214822Z-74udtekr` is epoch 10 (index 9), with
0.892538 full-fold AUC, the 1C architecture, reflection probability 0.5 and
sampling sigma 40 mm. On 2026-09-14 it was exported from its archived training
source to `outputs/submission/exports/knee-sigma40-20260914`, at 2× images and batch 2.
Checkpoint SHA-256 is
`4035f331070361a81c59f753e0210f0c75d5554755c4a60d78d643acb7e4fa7e`.
Strict loading and bundle checks passed; bundled source matches the training
snapshot. Export took 0.69 seconds without optional local inference.

The user-authorized automatic submission service launched on 2026-09-14 with
private dataset `claverru/knee-sigma40-20260914` and notebook
`claverru/knee-sigma40-20260914-inference`. The
[workflow record](../../../outputs/submission/records/knee-sigma40-20260914.json) tracks
notebook verification, competition submission and scoring. Notebook version 1
completed and its three-row example output passed verification. Submission
**56224822** was accepted at 07:15:03 UTC and completed with **0.931 public AUC**,
tying reflection at the reported precision. The saved workflow record contains
the completed API response and Kaggle script version `349713159`.
