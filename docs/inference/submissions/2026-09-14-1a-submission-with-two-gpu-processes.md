# 1A submission with two GPU processes

The selected `meniscus-classifier` checkpoint (epoch 9, index 8; 0.891192
local macro AUC) was exported on 2026-09-14 to
`outputs/submission/exports/knee-1a-dual-gpu-20260914`. Checkpoint SHA-256 is
`d124774eecc83cff0d54bd3f42c59e5adb4ab4a53da82eb5df5121888aa5323a`.
Its model identity is `spatial_tokens_shared_meniscus_classifier_v1`: only
the meniscus classifier weights are shared; target attention and biases remain
independent, and OA classifiers remain independent.

The isolated runtime in `outputs/submission/runtimes/knee-1a-dual-gpu-20260914` starts
from 1A's training archive and adds the branch-1C prediction coordinator, worker,
process module, inference metadata partitioning and exporter. All other package
Python files match the archive exactly. `runtime-provenance.json` records the
archive hash and each overlaid file hash. The bundle reports changed inference
source separately from its unchanged model/preprocessing identity.

Requested settings are **2× images, four studies per GPU batch and four loader
workers total**, using [independent GPU processes](../independent-gpu-inference-processes.md).
Strict CPU model loading, metadata and bundle integrity checks passed; export
preparation took 0.55 seconds. No local inference, GPU work or tests ran.

The authorized submission service uses private dataset
`claverru/knee-1a-dual-gpu-20260914` and private notebook
`claverru/knee-1a-dual-gpu-20260914-inference`. Its
[initial workflow record](../../../outputs/submission/records/knee-1a-dual-gpu-20260914.json)
and saved notebook log show two visible GPUs, followed by the request-loading
failure described above. The notebook failed before inference and before
competition submission.

The corrected retry uses runtime and bundle paths ending in `-v2`, preserving
the first attempt. Only `scripts/pipeline/predict_worker.py` changed in the
bundled Python source; the checkpoint and inference settings remain the same.
The same private dataset and notebook receive new versions. The
[retry workflow record](../../../outputs/submission/records/knee-1a-dual-gpu-20260914-v2.json)
tracks notebook verification, automatic competition submission and scoring.
Notebook version 2 completed successfully and validated its three example rows.
Both **Tesla T4s** were used, with FP32 inference, batch limit 4 and two loader
workers each. GPU 0 predicted two studies in 8.81 seconds; GPU 1 predicted one
in 4.89 seconds. Worker wall time, including startup and model loading, was
24.64 seconds; the complete prediction subprocess took 83.65 seconds and the
notebook's measured setup/inference/validation total was 90.62 seconds.
These example timings include startup overhead and do not establish hidden-data
throughput or a speedup over single-GPU inference. The downloaded
`submission.runtime.json` records both worker identities, counts and timings.
Competition submission **56226273** was accepted at 08:25:06 UTC on
2026-09-14 and completed with **0.930 public AUC**, 0.001 below reflection,
0.005 above 1B and 0.002 above original 1C. The saved workflow observed completion
967.03 seconds after submission, about **16m07s**. This includes queuing, scoring
and polling; it is not an isolated inference measurement. See the
[cross-experiment investigation](../../history/sharing/generalization-investigation.md) for the local/public
comparison and runtime limits.
