# Reflection sigma-40 submission

The selected `reflection-sigma40` checkpoint (epoch 7, index 6;
**0.892360168 validation AUC**) was exported on 2026-09-14 to
`outputs/submission/exports/knee-reflection-sigma40-20260914`. Its checkpoint SHA-256 is
`9526b28955ff67e0401aa3038ad659fd73aa797475f2c50d7ea79ce9ed2b0c31`.
The [validation comparison](../../validation/results/2026-09-14-reflection-sigma-40-result.md)
records the architecture-by-sigma results.

This submission retains the accepted **2× images, four studies per GPU batch
and four total loader workers**, with one process per visible GPU on Kaggle's
two-T4 allocation. Its isolated runtime is
`outputs/submission/runtimes/knee-reflection-sigma40-20260914`: it starts from this run's
training archive and uses the existing prediction coordinator, corrected
worker, process module and exporter from branch `1C` at `3b8c2cc`. The
reflection model constructor is preserved in `inference.py`; all other archived
package Python files match exactly. No source in the active checkout or
existing notebook was changed. The runtime's `runtime-provenance.json` pins
the archive and overlaid files. The coordinator, worker and process module
match the successful 1A dual-GPU v2 bundle.

Strict CPU checkpoint loading, metadata and bundle-integrity checks passed;
the bundle is ready with no unmet prerequisites. Export preparation took
0.723 seconds excluding interpreter startup. No local prediction, GPU work
or tests were run.

The user authorized submission, and the detached workflow launched on
2026-09-14 at **12:14:15 UTC**. Its private dataset is
`claverru/knee-reflection-sigma40-20260914`; its notebook is
`claverru/knee-reflection-sigma40-20260914-inference`. The
[workflow record](../../../outputs/submission/records/knee-reflection-sigma40-20260914.json)
and adjacent `-runner.log` record the complete transitions and scored result.
Notebook version 1 started at **12:15:00 UTC**. Competition submission
**56230573 completed at 12:34:41 UTC with public AUC 0.936**, a **+0.005**
gain over reflection sigma 80 and 1C sigma 40, both 0.931.

The downloaded `submission.runtime.json` confirms **two Tesla T4 GPUs in
FP32**, each configured for batch 4 and two loader workers. The visible
notebook contains three studies, partitioned 2+1; it does not measure sustained
four-study batches or hidden-test throughput. The visible notebook wait was
**1m 31s**, submission-to-observed-completion was **18m 08s**, and the full
workflow took **20m 26s**. Those waits include queue, setup, scoring and polling
delays; GPU profiling remains deferred.
