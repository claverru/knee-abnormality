# Independent GPU inference processes

This runtime was implemented on branch `1C` at `3b8c2cc`, overlaid onto the frozen
[reflection/sigma-40 submission](submissions/2026-09-14-reflection-sigma-40-submission.md), and ported to
`meniscus-soft-coupling` on 2026-09-15, byte-identical to that submission's runtime
(`outputs/submission/runtimes/knee-reflection-sigma40-20260914`) at the time. The same day the
[Kaggle profiling](2026-09-15-kaggle-inference-profiling.md) added per-batch timing, raised the
default to **16 total loader workers**, and made `auto` precision resolve to FP16 on CUDA devices
without native BF16 (T4), where workers run the backbone in zero-padded 64-image channels-last
chunks (`use_fixed_backbone_chunks`). The `export` and `infer` recipes default to four studies per
GPU and 16 total workers. The `main` branch and 1C's submission error handling in `submit.py` were
not changed.

Start order and GPU-side work since 2026-09-22 on branch `inference-opt`
([measurement](2026-09-22-kaggle-x4-startup-and-gpu-preparation.md)):

- **Coordinator:** `predict.py` does not import torch. It counts GPUs through the CUDA driver
  (`cuda_device_count` in `inference/processes.py`) and spawns the workers about three seconds
  after launch.
- **Worker start:** `predict_worker.py` builds the series index in a background thread while the
  main thread imports torch and the dataset code, then forks the loader workers at niceness 10
  (`INFERENCE_LOADER_NICENESS`), then imports the model stack and loads the model while they
  prepare the first batches. The optional `wandb` import that `timm` attempts is skipped.
- **CUDA:** each worker sets blocking synchronization on the primary context before creating it,
  so the main thread sleeps rather than spinning a core while the GPU works.
- **Backbone:** under FP16, `use_fixed_backbone_chunks(model, image_size)` warms the 64-image
  channels-last chunk on a side stream and captures it as a CUDA graph; each chunk copies into the
  graph's fixed input, replays and copies out.
- **Images:** loader batches carry the shared buffers of raw 16-bit rows, and `predict_loader`
  page-locks them in place (`cudaHostRegister`), copies them asynchronously, builds the uint8 images
  on the GPU with `prepare_images` and releases the lock after the batch. Inference loaders do not
  pin memory; pinning every prefetched raw batch exhausted host memory.
- **Precision:** `auto` resolves to native BF16 where supported, FP16 on other CUDA devices and
  FP32 on CPU.

The prediction coordinator partitions study IDs across all visible CUDA devices
before dataset collation. Each child process sees one GPU through
`CUDA_VISIBLE_DEVICES`, loads its own complete model — every member, for an
ensemble bundle — and predicts its assigned studies. Ensemble members run
forward on the same decoded, device-resident batch and their per-label sigmoid
probabilities are averaged before the batch is recorded; see
[model bundle and offline environment](model-bundle-and-offline-environment.md#multi-checkpoint-ensemble-bundles)
for the bundle-side contract. With two T4s, the default is **four studies per GPU batch**, with eight
DataLoader workers per process: eight studies can be in flight across the GPUs,
with 16 loader workers total. `--batch-size` controls each device's batch;
`--num-workers` is divided across processes. A single visible GPU receives the
whole worker budget; CPU inference uses one process with CUDA hidden. Small
datasets use at most one process per study, avoiding empty partitions.

Study assignment is deterministic and disjoint. The coordinator checks exact
study coverage, target order, finite probabilities in [0, 1] and validation
folds, then restores input study order before publishing the merged CSV.
Worker failures abort publication and trigger process cleanup; the parent
includes the failed worker's log tail in its error. The adjacent
`submission.runtime.json` records active devices, assigned GPU IDs, GPU names,
study counts, precision, backbone chunk size, batch size, loader workers, startup and
setup timings, and per-batch loader wait, transfer and forward times per process.

`src/knee_abnormality/inference/processes.py`
owns serialized requests and subprocess lifecycle without importing PyTorch.
`scripts/pipeline/predict_worker.py` owns execution
through the shared PyTorch inference engine (`inference/engine.py`). This boundary lets future
ONNX or TensorRT work replace worker model execution while retaining study
assignment and output merging. Training continues to use Lightning.

Export bundles include both entry points and the shared process module. Their
manifest records `batch_size_scope: per_device`, `num_workers_scope: total`
and `parallelism: one_process_per_visible_gpu`; the bundled notebook reads the
recorded batch and worker defaults. The prepared original-1C bundle is
`outputs/submission/exports/knee-1c-dual-gpu-20260914`, using the same checkpoint hash
recorded below. Strict CPU model loading and bundle integrity checks passed;
local inference was not requested. Updated inference source differs from its
training snapshot, which the manifest records explicitly; model and
preprocessing identity checks still pass.

The first 1A Kaggle execution subsequently exposed a worker request-loading
error: JSON-mode validation rejected resolved `Path` objects returned by the
experiment path validators. The corrected archived worker decodes JSON and applies Python-mode
validation, matching the configuration loader. Export fresh bundles to include
this fix; the initial prepared `knee-1c-dual-gpu-20260914` bundle predates it.

The local GPU was occupied, so implementation used no local inference, training
or tests. The subsequent [1A notebook](submissions/2026-09-14-1a-submission-with-two-gpu-processes.md)
confirmed successful prediction on two T4s. Its three-study example does not
measure sustained throughput or a full four-study batch. Each model and
four-study batch must fit independently on its GPU; GPU memory is not pooled.
