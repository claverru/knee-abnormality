# 2026-10-13 TensorRT backbone and two-model timing

What a second ensemble member costs at inference, where the GPU loop spends its time, and what a
TensorRT engine for the ResNet-50 backbone changes. **A TensorRT FP16 engine runs the ResNet-50
backbone 2.1× faster with near-equal predictions; one model then waits on the loaders, and two
TensorRT ResNet-50s take 343 s against 303 s for one ResNet-50 today.**

All runs: `profile_kaggle_inference`, two T4s, four vCPUs, 1,300 studies of
`all_5fold_scanner_v2`, ×4, FP16, four studies per GPU, 8 loader workers. One run per row; hosts
differ by about ±15 s on identical code. Bundles: S148 (`rn50-joint-fastmri-f3-x4-tau09`) and S152
(`convnext-rn50-joint-f3-x4-tau09-b`). Outputs under `outputs/profiling/kaggle-*-20261013`.

## Whole runs

| Run | Command, s | Forward per GPU, s | Loader wait per GPU, s | Model load, s |
|---|---:|---:|---:|---:|
| ResNet-50 (S148) | 302.5 | 245.9, 231.7 | 15.5, 20.4 | 2.3 |
| ConvNeXt-Tiny + ResNet-50 (S152) | 510.7 | 433.7, 431.6 | 14.2, 14.1 | 10.6 |
| ResNet-50 on TensorRT | 318.3 | 175.8, 179.5 | 91.1, 85.8 | 1.5 |
| ResNet-50 on TensorRT, 12 workers | 301.2 | 195.0, 189.2 | 37.1, 38.6 | 13.3 |
| Two ResNet-50s on TensorRT | 343.4 | 264.1, 260.0 | 17.5, 17.8 | 10.5 |
| S152 with its ResNet-50 on TensorRT | 433.7 | 361.2, 346.3 | 12.0, 12.1 | 16.7 |
| Two ResNet-50s on TensorRT, shared-memory files and warmed libraries | 330.3 | 263.8, 263.3 | 15.0, 14.4 | 2.2 |
| The same, second host | 323.2 | 266.5, 262.9 | 4.0, 2.9 | 2.3 |

- **A second member is not free.** ConvNeXt-Tiny adds 208 s to the ResNet-50, about 0.003 of AUC
  at the [exchange rate](efficiency-leaderboard.md#the-exchange-rate), for the 0.001 S152 gained.
- **One model on TensorRT is loader-bound.** The forward pass falls by about 60 s per GPU and the
  loader wait rises by about 70 s: the loaders deliver about 4.9 studies per second with 8 workers
  and the GPU would take more. The single-model TensorRT row also hashed the 750 MB runtime at
  start (verification 9.5 s against 3.7 s); the runtime is excluded from the bundle hashes since.
- **Two TensorRT ResNet-50s cost 41 s over one ResNet-50 today:** about 23 s of forward time per
  GPU, 8 s of model loading and the rest in setup. The run loaded the same weights twice, so it
  measures time only.
- **Loader workers stay at 8.** Twelve recovered about 30 s of prediction time and 16 exhausted the
  14 GB of shared memory (`unable to allocate shared memory`, studies lose series and then fail);
  the user ruled more workers out for memory.
- **Loaders at idle scheduling priority lose:** forward 231, 227 s but loader wait 39, 39 s and
  315.3 s for the command, against 302.5 s.

## Batch hand-over and library warm-up

Two changes outside the hashed sources, both with identical predictions, measured on the
two-ResNet-50 TensorRT bundle.

- **Batches handed over as shared-memory files.** `predict_worker` sets the `file_system` sharing
  strategy. Receiving a batch cost the main thread a median 102–107 ms with file descriptors and
  15–17 ms with files, with the GPU idle meanwhile. In a rotated comparison (three notebooks, each
  variant once per host, study third and position; six worker runs per variant) the loader wait
  per GPU was 1.1 s against 5.7 s on 433 studies, and it fell in every worker run (0.9–2.3 s
  against 4.9–7.3 s); forward time did not move (88.3 against 87.8 s).
- **TensorRT libraries read ahead.** The first load of the runtime from the dataset took 8.5–18.7 s
  in the workers and under 1.1 s once cached. The coordinator now reads the libraries through on a
  background thread from its first line; model load for two members fell from 10.5 s to 2.2–2.3 s.
- **With both, two TensorRT ResNet-50s take 323 and 330 s** against 343.4 s before and 302.5 s for
  one ResNet-50 today. The first host stalled on storage 15 times for 23 s in total; the second
  did not.
- **Loaders at `SCHED_IDLE` with two members:** forward 85.1 against 87.8 s and wait 6.2 against
  5.7 s per GPU in the same rotated comparison, inside the spread between workers. Not adopted.
- **Building a member costs 1.2–2.5 s** (model construction), reading its checkpoint 0.2–0.3 s.
- **Preparing the next batch while the GPU works is not possible outside the hashed sources:**
  preparation and both model calls each end in a forced synchronization, so the main thread has no
  free window, and the feeder-thread form is [closed](../backlog/loading-speedup/closed.md#closed-2026-09-22).

## The GPU loop

`mainloop` condition, 16 batches of four studies, synchronized, one T4, no loaders running.

| Per batch, ms | ResNet-50 | ConvNeXt-Tiny |
|---|---:|---:|
| Backbone, first look and read | 761 | 1,042 |
| Rest of the first look and the read | 145 | 131 |
| `prepare_images` | 189 | shared |
| Transfer, page-lock and release | 86 | shared |

- **The backbone is 85–89% of a member.** A batch passes 624 images through each backbone in
  10.75 chunks of 64, 133 of them in the first look: 70.5 ms per chunk for the ResNet-50 and 97 ms
  for ConvNeXt-Tiny. Padding the first look and the read separately wastes about one chunk in
  eleven.
- **Production runs the one-model loop at 1.47 s per batch against 1.18 s here.** The main thread
  uses 153 ms of CPU per batch (page-lock and release 43, read 42, first look 30, preparation 24)
  and meets about 72 synchronization points, on two physical cores shared with the loaders.
- **Inside the rest:** `_encode_crops` outside the backbone about 82 ms for its two calls,
  `view_windows` 35 ms, pooling 30 ms. Inside `prepare_images`: `_resample` 117 ms over 25 groups,
  `_native_blocks` 25 ms, `_region_values` 22 ms.

## Candidates measured

| Change | Result | Predictions | Ships |
|---|---|---|---|
| TensorRT FP16 engine, ResNet-50 | 68.7 → 32.4 ms per chunk; one model 1.110 → 0.770 s per batch | max 0.013, mean 0.0004 | engine and worker only |
| Dynamic-batch engine (no padding) | 33.6 ms per chunk, 9.3 ms for 16 images; 0.745 s per batch | max 0.011, mean 0.0004 | with the engine |
| TensorRT FP16 engine, ConvNeXt-Tiny | 90.6 → 88.3 ms per chunk | — | superseded: channels-first at level 5 pays ([families](2026-10-18-tensorrt-backbone-families.md)) |
| Batch norm folded into the convolutions | backbone 761 → 725 ms per batch | max 0.010, mean 0.0002 | included in the engine |
| Dense-matrix `_resample` | `prepare_images` 188 → 135 ms per batch; 3,178 of 1.58 billion pixels move by one level | max 0.0005 | hashed source: next retrain |
| Loaders at `SCHED_IDLE` | slower | identical | dropped |

Two runs of identical code differ by up to 0.0015 in a prediction (mean 0.0001), which is the
yardstick for the prediction columns. None of the near-equal changes has been scored on a
validation fold.

## The TensorRT path

Removed on 2026-10-19 ([why](2026-10-19-bundle-size-and-session-startup.md)); kept as a record.

- **Engine.** `scripts.tools.build_tensorrt_engines` uploads the bundle, builds one dynamic-batch
  FP16 engine per member of an adopted family ([families](2026-10-18-tensorrt-backbone-families.md))
  in a T4 notebook with internet (TensorRT 11.4.0.106, strongly typed
  half-precision network, batch 1 to 64; about 37 s per engine, 47 MB), and installs the engines
  beside the checkpoints (`model.engine`, `models/member-NNN.engine`) with the runtime under
  `tensorrt/`. Engines are specific to the GPU model and the TensorRT version. Its first run, on a
  copy of the S148 bundle, built the engine in 37.9 s; on random inputs its features differ from
  the PyTorch backbone by 0.0008 on average against a mean magnitude of 0.21.
- **Runtime.** Only `libnvinfer.so.11` (670 MB), the plugin and parser libraries and the Python
  bindings are needed to run an engine; the full install is 6.7 GB, nearly all of it builder
  resources. The runtime imports from the bundle with internet off, in 0.1 s plus 0.1 s per engine
  when warm. It is left out of the bundle hashes.
- **Worker.** `predict_worker` uses `use_tensorrt_backbone` for a member whose engine file exists
  and the CUDA graph otherwise. The lean runtime does not load these engines, and a
  version-compatible build failed.

## Not done

- A validation-fold score of the TensorRT path against the CUDA graph.
- The dense `_resample` in `device.py` and its parity audit.
- The startup imports: 60% of the model-stack import is `torch._dynamo` and `sympy` through
  torchvision, which timm requires; kornia is about 14% and sits in a hashed source.
- The shared-memory-file hand-over measured on a single model.
