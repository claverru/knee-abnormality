# Champion without 3D at inference

This submission tests whether the champion relies on its 3D embedding on the hidden
test data. It uses the unchanged `reflection-sigma40` checkpoint (SHA-256
`9526b28955ff67e0401aa3038ad659fd73aa797475f2c50d7ea79ce9ed2b0c31`), with every
token's 3D embedding replaced by the learned `unknown_3d` vector at inference.

The isolated runtime `outputs/submission/runtimes/knee-reflection-sigma40-unknown3d-20260915`
copies the champion runtime. It adds one forward hook in `inference.py`, outside the
preprocessing identity hash, so checkpoint and preprocessing identities are unchanged;
its `runtime-provenance.json` records the change. The bundle
`outputs/submission/exports/knee-reflection-sigma40-unknown3d-20260915` uses 2× images, four
studies per GPU and four total workers. Its `model.ckpt` is byte-identical to the
champion bundle's, and `inference.py` is its only code difference. On the three local
example studies (one RTX 3090, BF16) predictions differed from the champion bundle by a
mean of 0.008 and at most 0.024.

Competition submission **56249900 completed with public AUC 0.936**, equal to the
champion at the three decimals Kaggle reports. The notebook ran for 2m 01s on two Tesla
T4s in FP32 with batch 4, and scoring took 16m 37s. The first launch failed when
Kaggle's `SaveKernel` returned HTTP 400 for a 52-character notebook title, before any
notebook run or competition submission. The retry used notebook
`claverru/knee-sigma40-unknown3d-20260915-inference` with the same dataset
`claverru/knee-reflection-sigma40-unknown3d-20260915`, and its
[workflow record](../../../outputs/submission/records/knee-reflection-sigma40-unknown3d-20260915-retry.json)
holds the transitions. `submit.py` now rejects notebook titles longer than 50 characters
before any remote call.

The champion therefore does not measurably rely on 3D at inference on the public test
data, consistent with the fold-0 removal diagnostics. This does not show whether 3D
helped during training. The [00-baseline](2026-09-15-overnight-00-baseline-submission.md) public
score of 0.934, trained without 3D, is more likely explained by 3D's training-time
effect or run-to-run training variability than by inference-time use. Differences
below the public score's rounding cannot be resolved.
