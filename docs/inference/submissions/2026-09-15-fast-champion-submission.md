# Fast champion submission

The unchanged reflection/sigma-40 champion checkpoint (SHA-256
`9526b28955ff67e0401aa3038ad659fd73aa797475f2c50d7ea79ce9ed2b0c31`, 2× images) was resubmitted on
2026-09-15 with the runtime changes selected by the
[Kaggle inference profiling](../2026-09-15-kaggle-inference-profiling.md): 16 total loader workers
and `auto` precision resolving to FP16 with 64-image channels-last backbone chunks on T4. The
runtime `outputs/submission/runtimes/knee-reflection-sigma40-fast-20260915` records the overlaid
`inference.py`, `predict.py` and `export.py`; bundle `outputs/submission/exports/knee-reflection-sigma40-fast-20260915`
keeps the champion checkpoint and preprocessing identity.

Competition submission **56262847 completed with public AUC 0.936**, equal to the FP32 champion.
The dataset is `claverru/knee-sigma40-fast-20260915` and the notebook
`claverru/knee-sigma40-fast-20260915-inference`; the
[workflow record](../../../outputs/submission/records/knee-sigma40-fast-20260915.json) holds the transitions.
The visible notebook ran for 2m 32s; `submission.runtime.json` confirms two Tesla T4 processes in
FP16 with 64-image chunks, batch 4 and eight loader workers each. Submission to scored completion
took **10m 05s**, against 18m 08s for the original champion and 16m 07s and 16m 37s for the two
FP32 four-worker submissions earlier on 2026-09-15. Those waits include queueing and scoring, so
they only indicate the direction of the hidden-test runtime change.
