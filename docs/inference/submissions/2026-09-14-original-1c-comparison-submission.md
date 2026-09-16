# Original 1C comparison submission

The original `disease-compartment-attention` selected checkpoint (epoch 10,
index 9; 0.889430 local macro AUC) was exported on 2026-09-14 from its archived
training runtime to `outputs/submission/exports/knee-1c-20260914`. Checkpoint SHA-256 is
`fc34d31860aee0958a8a676be104e0e6f0c6ebd6a2528a2583dbf42ee80506e4`.
This is the original 1C experiment with sigma 80 mm. The separate sigma-40
submission below uses an overnight hyperparameter candidate.

The user requested **2× images, batch size 8 and four workers** for this
comparison. The bundle manifest records those runtime settings, and the notebook
reads its worker default from that manifest. The runtime adjustment and refreshed
notebook hash are recorded before upload; model and prediction source still match
the training snapshot. Metadata, strict CPU model loading and bundle integrity
checks passed. No local inference or GPU work was performed.

The automatic submission service launched with private dataset
`claverru/knee-1c-20260914` and private notebook
`claverru/knee-1c-20260914-inference`. Its
[workflow record](../../../outputs/submission/records/knee-1c-20260914.json) preserves submission
**56224993**, script version `349714936`: its hidden rerun failed with an
unhandled-error message and no score. The visible notebook processed only three
studies, so it never exercised a full batch of eight. GPU memory pressure is a
possible cause; Kaggle's returned error did not provide a traceback to confirm it.
The response reports `status: COMPLETE` alongside `errorDescription`; the runner used on branch `1C`
treats that combination as failure, and saved-status inspection correctly
reports failure for the original record without rewriting its history.

The user-requested batch-4 retry retained the checkpoint, frozen source, 2×
images and four workers in `outputs/submission/exports/knee-1c-batch4-20260914`. Submission
**56225269**, script version `349717410`, completed with **0.928 public AUC**;
the [retry record](../../../outputs/submission/records/knee-1c-batch4-20260914.json) recorded
completion at 07:54:50 UTC. It used the previous single-GPU inference path.
Original 1C improves on 1B's 0.925 by 0.003 and trails reflection and sigma 40
at 0.931 by 0.003. These are separate trained models; the result does not isolate
the causal effect of the compartment embedding.
