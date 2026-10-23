# Independent GPU inference processes

This runtime was implemented on branch `1C` at `3b8c2cc`, first used in the
[reflection/sigma-40 submission](submissions/early-submission-details.md)
and ported to the then-current model branch on 2026-09-15. The same day the
[Kaggle profiling](profiling/2026-09-15-kaggle-inference-profiling.md) added per-batch timing, raised the
default to 16 total loader workers (`INFERENCE_NUM_WORKERS` was 8 at the end) and made `auto` precision resolve to FP16 on CUDA devices
without native BF16 (T4), where workers run the backbone in zero-padded 64-image channels-last
chunks (`use_fixed_backbone_chunks`). The `export` and `infer` recipes default to four studies per
GPU.

Start order and GPU-side work added from 2026-09-22 and merged into `main`
([measurement](profiling/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md)):

- **Coordinator:** `predict.py` does not import torch. It counts GPUs through the CUDA driver
  (`cuda_device_count` in `inference/processes.py`) and spawns the workers about three seconds
  after launch.
- **Worker start:** `predict_worker.py` builds the series index in a background thread while the
  main thread imports torch and the dataset code, then forks the loader workers at niceness 10
  (`INFERENCE_LOADER_NICENESS`), then imports the model stack and loads the model while they
  prepare the first batches. The optional `wandb` import that `timm` attempts is skipped.
- **CUDA:** each worker sets blocking synchronization on the primary context before creating it,
  so the main thread sleeps rather than spinning a core while the GPU works.
- **Backbone:** under FP16 each member's backbone is held in FP16 (`model.backbone.half()`; the
  rest of the model, whose side head takes FP32 tokens, stays FP32). A bundle exported with
  `--backbone eager` stops there. With the default `graph`, `use_fixed_backbone_chunks(model,
  image_size)` compiles the backbone
  with `torch.compile` (default mode, static shapes), warms the 64-image channels-last chunk on a
  side stream, which runs the compile, and captures it as a CUDA graph; each chunk copies into the
  graph's fixed input, replays and copies out. A cold compile costs about 14–30 s for a family's
  first member at model load and about 0.05 s for each later member of the same family; a
  `compile-cache.bin` shipped in the bundle (`inference.compile_cache`) removes most of it
  ([compile](profiling/2026-10-19-torch-compile-backbone.md)).
- **Images:** the prediction worker's loader skips loader-side preparation
  (`prepare_on_loader=False`), so its batches carry the shared buffers of raw 16-bit rows.
  `predict_loader` page-locks them in place (`cudaHostRegister`), copies them asynchronously,
  builds the uint8 images on the GPU with `prepare_images` and releases the lock after the batch.
  Inference loaders do not pin memory; pinning every prefetched raw batch exhausted host memory.
- **Precision:** `auto` resolves to native BF16 where supported, FP16 on other CUDA devices and
  FP32 on CPU.

The prediction coordinator partitions study IDs across all visible CUDA devices
before dataset collation. Each child process sees one GPU through
`CUDA_VISIBLE_DEVICES`, loads its own complete model — every member, for an
ensemble bundle — and predicts its assigned studies. Ensemble members run
forward on the same decoded, device-resident batch and their per-label sigmoid
probabilities are averaged before the batch is recorded; see
[model bundle and offline environment](model-bundle-and-offline-environment.md#multi-checkpoint-ensemble-bundles)
for the bundle-side contract. With two T4s, the default is **four studies per GPU batch**; the 2026-09-15 to
2026-09-17 submissions used eight DataLoader workers per process (16 loader workers
total, eight studies in flight across the GPUs), and the `INFERENCE_NUM_WORKERS`
default at the end was 8 in total. `--batch-size` controls each device's batch;
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
setup timings, and per-batch loader wait, transfer and forward times per process. Forward time
covers GPU image preparation and every member's pass, both passes under reflection TTA; each
worker also records its `member_count` and `peak_allocated_gib`.

`src/knee_abnormality/inference/processes.py`
owns serialized requests and subprocess lifecycle without importing PyTorch.
`scripts/pipeline/predict_worker.py` owns execution
through the shared PyTorch inference engine (`inference/engine.py`). This boundary lets future
ONNX or TensorRT work replace worker model execution while retaining study
assignment and output merging. Training continues to use Lightning.

Export bundles include both entry points and the shared process module. Their
manifest records `batch_size_scope: per_device`, `num_workers_scope: total`
and `parallelism: one_process_per_visible_gpu`; the bundled notebook reads the
recorded batch and worker defaults.

The first 1A Kaggle execution exposed a worker request-loading error: JSON-mode validation
rejected resolved `Path` objects returned by the experiment path validators. The corrected worker
decodes JSON and applies Python-mode validation, matching the configuration loader; bundles exported
afterwards carry the fix. The 1A notebook confirmed successful prediction on two T4s
([details](submissions/early-submission-details.md)); its three-study example did not measure
sustained throughput or a full four-study batch. Each model and four-study batch must fit
independently on its GPU, because GPU memory is not pooled.
