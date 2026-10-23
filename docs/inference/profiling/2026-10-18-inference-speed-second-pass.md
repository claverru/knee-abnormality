# 2026-10-18 Inference speed, second pass

Where S156 TensorRT ×4 spends its 269 s after the [buffer pool](2026-10-17-loader-buffer-pool.md),
and what was tried against each part. **Kept: the worker exits as soon as it has reported (teardown
9.4 → 0.7 s) and a startup page pre-read (worker start 8.4 → 3.3 s, model ready about 6 s
earlier); a script kernel instead of the notebook is about 40 s slower per session. Without
TensorRT (removed 2026-10-19) S156 ×4 runs 318.9–321.0 s against 332.9 s. The GPU loop itself is
the floor: 0.9 s of GPU work per batch.**

All runs: Kaggle 2×T4, S156 TensorRT ×4, the 1,300 studies of `all_5fold_scanner_v2`,
`fp16-b4-w8`, one run each; hosts differ by about ±15 s on identical code, so timestamps of the
stages are compared rather than whole commands. Outputs under `outputs/profiling/kaggle-s156-*`.

## Where the time goes

| Part | Seconds | Source |
|---|---:|---|
| Notebook wrapper before and after the command | about 23 | S165's submission log: kernel start 10–13 s, setup cell 3 s, output check and HTML render 7 s |
| Coordinator imports and bundle verification | 6 | `predictions.runtime.json` |
| Worker start (interpreter, `import torch`) | 8.4 | |
| Index, loader fork, model-stack import, model load | 26 | |
| Prediction per GPU | 218 | of it forward 190, loader wait 22–28 |
| Teardown after the last batch | 9.4 | |

- **Alone on a GPU the loop is GPU-bound:** with prefetched batches and no loaders it runs 0.91 s
  per batch and the GPU is busy 92% of the time (`torch.profiler`, 12 batches,
  `kaggle-s156-loop-lab-1-20261017`). Synchronised stages per batch: backbone 0.57 s, image
  preparation 0.19 s, the rest of the model 0.15 s, the copy 0.05 s. A batch makes 74 synchronising
  calls and about 4,250 kernel launches.
- **In production the GPU is busy the same 0.9 s per batch** (67% of 218 s over 163 batches) but a
  batch takes 1.17 s. An instrumented run (`kaggle-s156-trt5-x4-instr-20261017`) put the rest on
  page-locking the buffers the pool does not cover: 50–85 ms per batch for 2.6–4.9 buffers of
  about 32 MB, 8–14 s per GPU.

## Tried

| Change | Result | Kept |
|---|---|---|
| Worker calls `os._exit(0)` after printing its result | the last worker's teardown 9.4 → 0.7 s | yes |
| Startup page pre-read: the coordinator reads the library pages the imports touch (0.82 GiB of system files in `inference/kaggle_startup_pages.json`, recorded with the `page-lab` condition on Kaggle's v171 image), 16 threads, from its first line (`inference.startup_pages`) | worker start 8.4 → 3.3 s, model ready at 34.0 s against 37.9–42.5 s on the TensorRT bundle; without TensorRT 9.8 → 3.3 s and 46.7 → 39.1 s (`kaggle-s156-branch-pages-20261019-1`) | yes |
| Model-stack import before forking the loaders, during indexing | model ready 34.0 → 28.2 s, command 261.2 against 261.6 s | unproven |
| Script kernel instead of the notebook (3 visible test studies, the same bundle, alternated) | session 102 and 91 s against 61 and 51 s: the log is about 8 s shorter but Kaggle spends 53–62 s around a script session against 4–13 s around a notebook (`kaggle-kernel-type-20261019`); a first comparison of log spans alone had suggested 14 s saved | no |
| Batch 8 per GPU, prefetch 1 | 293.3 s; forward unchanged, loader wait 44–53 s; predictions move up to 0.028 | no |
| Two or three processes per GPU | 11.5 against 10.9 s per 48 studies alone; 13.2 against 12.0 s under four CPU burners | no |
| Copying unpooled buffers into a pinned staging area instead of page-locking them | 106 / 74 ms per batch against 85 / 50 ms | no |
| Dedicated CUDA stream (TensorRT warns about the default stream) | 5.31 against 5.23 s alone, 5.87 against 5.79 s with loaders | no |
| Loaders at nice 19, CUDA spin or yield synchronisation | within the ±8% between repeated arms | no |
| Main thread at nice −10 or `SCHED_FIFO` | not permitted in Kaggle's container | — |
| Pre-reading every file of the imported packages (5.9 GiB) | 22 s of reading for a 22 s cold import | no |
| `sys.modules["pandas"] = None` (sklearn imports it optionally) | import self time 18.3 against 17.9 s | no |
| Moving `fold_prediction_macro_auc` so the worker skips sklearn | `model.py` imports `evaluation.metrics` itself | no |

- **Real loaders cost the loop about 11%** in a lab with loader processes running beside it
  (5.79 against 5.23 s, `kaggle-s156-contention-lab-1-20261017`); no scheduling change recovered
  it.
- **The model-stack import stays near 15 s** even with warm pages and no loaders. `py-spy` shows
  the worker's main thread competing with its own indexing threads for the GIL; the imports
  themselves are about 6 s of CPU.
- **Cold imports are I/O-bound:** 31–59 s cold against 6.4 s warm in a GPU notebook, with 0.7 GB
  read; the page list, recorded with `mincore` after a warm import (`page-lab` condition), reads
  in 6.5 s on 16 threads, and two concurrent cold imports took 10.4 s with it against 66.8 s
  without.

## Without TensorRT

The branch's kept changes on S156 ×4 without an engine (buffer pool, early exit, slimmed
checkpoint, page pre-read), one run each:

| Run | Command | Model ready | Forward per GPU | Loader CPU |
|---|---:|---:|---:|---:|
| S156, 2026-10-16 | 332.9 s | 44.5 s | 282, 274 s | 522 CPU-s |
| Branch without the page pre-read | 321.0 s | 46.7 s | 266, 269 s | 478 CPU-s |
| Branch | 318.9 s | 39.1 s | 253, 277 s | — |

The pool gains little here: without the engine S156 waits on the GPU (loader wait 2.5–3 s per GPU),
so the loader savings only relieve CPU contention. Predictions match S156's (max \|Δ\| 0.004).

## Open

- Re-record `kaggle_startup_pages.json` with the `page-lab` condition when Kaggle's image changes.
- The 74 synchronising calls per batch sit in `model.py` (hashed): removable only with a retrain.
