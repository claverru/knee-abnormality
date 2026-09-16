# Kaggle inference profiling

Measured on 2026-09-15 on Kaggle, with the reflection/sigma-40 champion checkpoint
(SHA-256 `9526b28955ff67e0401aa3038ad659fd73aa797475f2c50d7ea79ce9ed2b0c31`) and 2× images.
Bundle `outputs/submission/exports/knee-reflection-sigma40-profiling-20260915` (dataset
`claverru/knee-sigma40-profiling-20260915`) adds only timing instrumentation to the
champion runtime; its checkpoint and preprocessing identity are unchanged. Artifacts are in
`outputs/profiling/kaggle-inference-20260915` and `outputs/profiling/kaggle-loading-20260915`,
produced by [`profile_kaggle_inference`](../../scripts/adhoc/catalog.yaml).

## Setup

Each prediction condition ran the bundle's unchanged `predict` entry point in its own fresh
notebook, so every run read DICOMs cold, like the hidden test. All runs used the same
904 fold-0 studies, presented through a symlinked test layout so image counts are enumerated
from disk as on Kaggle. The machine had two Tesla T4s, four vCPUs (cgroup quota 4, Xeon
2.00GHz) and 32 GB RAM. Competition data is on an **NFS mount** (`rsize` 512 KiB).
Each condition is a single run; machine-to-machine NFS variance is visible in indexing times
(24–64s for the same work).

## Prediction conditions

Two GPUs, batch 4 per GPU. Loader wait, forward and indexing are per GPU process (mean of two).

| Precision | Loader workers (total) | Command | Worker wall | Studies/s | Indexing | Loader wait | Forward | s per 1000 images | CPU cores | iowait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 4 | 599s | 561s | 1.61 | 24s | 442s | 74s | 3.2 | 1.38 | 28% |
| FP16 | 4 | 672s | 635s | 1.42 | 45s | 385s | 188s | 7.8 | 1.67 | 22% |
| FP16 | 8 | 501s | 463s | 1.95 | 40s | 193s | 210s | 8.5 | 2.54 | 15% |
| FP16 | 16 | 467s | 428s | 2.11 | 42s | 114s | 249s | 9.9 | 2.85 | 9% |
| **FP32** | **16** | **380s** | **342s** | **2.64** | 28s | 227s | 69s | 2.9 | 2.97 | 6% |
| FP32 | 24 | 423s | 383s | 2.36 | 64s | 234s | 64s | 2.6 | 2.73 | 6% |

- **Loader workers:** 16 total workers (8 per GPU) cut FP32 worker wall by 39%, with predictions
  identical to 4 workers. At 16 the CPU is the limit (about 3 of 4 cores, 6% iowait); 24 gives
  no further loader gain.
- **FP16 is slower on T4 with variable batch shapes.** Autocast raises forward time from 3.2 to 7.8 s
  per 1000 images, and it worsens as loader workers compete for CPU, so FP16 forward is CPU-bound
  ([cause](#why-fp16-is-slower)). FP32 forward is
  GPU-bound (GPU utilization matches the forward share of time). FP16 accuracy matched:
  macro AUC 0.892379 versus FP32 0.892344, max prediction difference 0.0025.
- **Parity:** T4 FP32 macro AUC 0.892344 versus 0.892360 for the local BF16 validation predictions
  (max difference 0.021, mean 0.002).
- **Fixed costs per run:** coordinator imports take 34–36s before `main` (0.6 cores, cold package
  reads); each worker then imports for about 12s; each worker indexes the *whole* study list
  (24–64s for 904 studies, mostly NFS directory listing). Peak GPU allocation at batch 4 is
  5.7 GiB in FP32.

## Loading stages

A CPU notebook (four vCPUs, Xeon 2.20GHz, a different NFS server) ran the bundled dataset
sequentially in one process on 100 seeded fold-0 studies, timing each stage by wrapping the
bundle's functions. A second pass over the same studies with fresh dataset objects had a warm
OS page cache; the difference isolates NFS latency.

| Stage | Cold | Warm | Cold − warm |
|---|---:|---:|---:|
| Slice headers (19,191 files, every slice of every candidate series) | 146.1s (72%) | 35.0s | 111.1s |
| Frame reads (11,930 selected slices, all uncompressed) | 25.5s | 12.5s | 13.0s |
| Resample and crop | 8.2s | 7.6s | 0.6s |
| Geometry and triplet sampling | 7.0s | 7.3s | −0.3s |
| Quantiles and normalization | 7.6s | 7.6s | 0 |
| Series listing index | 3.1s | 0.6s | 2.5s |
| Other, including collation | 5.6s | 4.8s | 0.7s |
| **Per study** | **2.03s** | **0.76s** | |

Headers cost 7.6 ms each cold and 1.8 ms warm, so pydicom parsing alone is 46% of warm CPU.
About 192 headers are read per study, but only 119 frames are decoded.
Thread scaling on separate cold studies:

| Operation | 1 thread | 4 threads | 16 threads | 64 threads |
|---|---:|---:|---:|---:|
| Header read and parse, files/s | 114 | 347 | 336 | 321 |
| Full file read, files/s | 96 | — | 872 | — |

Raw reads scale with threads (NFS latency), but header parsing plateaus at about 340 files/s from
four threads because it holds the GIL.

## Notebook-level experiments

These change only the profiling notebook's cells around the unchanged `predict` command, at FP32
and batch 4. Totals are from launch to exit; package pre-read time is added before launch.

| Run | Workers | Coordinator import | Indexing per worker | Loader wait per process | Transfer per process | Worker wall |
|---|---:|---:|---:|---:|---:|---:|
| Plain | 16 | 35.6s | 28.4s | 227s | 0.5s | 342s |
| Plain, `NvidiaTeslaT4Highmem` | 16 | 49.4s | 27.0s | 213s | 0.5s | 329s |
| Parallel series listing only | 16 | 43.7s | 9.0s | 255s | 0.5s | 355s |
| Pre-read packages and DICOMs, local bytecode cache | 8 | 48.7s | 5.8s | 325s | 0.5s | 418s |
| Pre-read packages and DICOMs, local bytecode cache | 16 | 55.0s | 8.0s | 113s / 373s | 248.6s / 0.6s | 473s |

- **High-memory T4 is the same machine** (four vCPUs, 32 GB), so that run replicates plain 16 workers:
  command 381s versus 380s. Coordinator import varies from 35 to 49s between machines.
- **Package pre-read is rejected.** 32 threads read 7.1 GiB in 31s, but coordinator import still
  took 49–55s.
- **DICOM pre-read is rejected.** A notebook thread pool listed every series directory, then read
  whole files at most 150 studies ahead of the loaders' open files. Studies average 124 MB, so
  the pre-read pushed 112 GiB through 32 GB of RAM. At 16 workers one GPU process's
  host-to-device transfers stalled for 249s in total (up to 5s per batch), consistent with memory
  reclaim delaying pinned-memory transfers; at 8 workers loader wait stayed high with 20% iowait.
  With 16 workers the CPU, not I/O, is the limit, so warm files cannot help much.
- **A local bytecode cache (`PYTHONPYCACHEPREFIX`) is rejected:** worker startup stayed at 12–14s.
- **Listing series directories in parallel shortens indexing but not the run.** The listing took
  about 7s and cut per-worker indexing from about 27s to 6–9s, yet the listing-only run's loader
  wait was higher (255s versus 213–227s) and its worker wall was 355s, within or above the plain
  runs' 329–342s. Per-worker indexing in the pipeline remains the fix for indexing time.
- **TPU notebooks are excluded:** only CPU and GPU notebooks can submit, and Kaggle's TPU image lacks
  several bundle dependencies ([requirements](submission-requirements-and-output-contract.md#verified-submission-requirements)).

## Why FP16 is slower

A single-T4 notebook loaded the champion once and timed full-model forward passes on eight real
batches (32 fold-0 studies, 1,596 images) already on the GPU, with no loader workers. "New shapes"
is the first pass for a precision, when every backbone batch size is unseen; "repeated" reruns the
same batches.

| Seconds per 1000 images | New shapes | Repeated shapes | Repeated, 3 CPU-burning processes |
|---|---:|---:|---:|
| FP32 | 3.24 | 2.48 | 2.52 |
| Autocast FP16, context per batch | 18.53 | 1.37 | 1.40 |
| Autocast FP16, one context for all batches | — | 1.38 | — |

- Every unseen backbone input shape costs FP16 about 3.4s of host-side cuDNN setup (the main thread
  ran at one full CPU core), versus about 0.15s for FP32. In production the backbone batch size, the
  images in each four-study batch, changes almost every batch, so FP16 kept paying it; that also
  explains why it slowed as loader workers took CPU.
- With repeated shapes FP16 is 1.8× faster than FP32, and CPU contention does not slow either.
  Re-casting weights per autocast context costs nothing measurable.
- The profiler agreed: over one repeated pass FP16 used 2.29s of self CPU and 4.60s of device time,
  versus 4.07s and 8.16s for FP32, despite 19,392 versus 15,320 aten calls.

### Fixed-size backbone chunks

A second single-T4 notebook wrapped `backbone.forward_features` to run fixed-size image chunks,
zero-padding the last, on the same kind of eight real batches (180–210 images each). Chunked
runs came first, so their first pass includes the one-time setup for the chunk shape.

| Seconds per 1000 images | First pass | Repeated | Max difference from FP32 |
|---|---:|---:|---:|
| FP32, unchunked | 2.60 | 2.64 | — |
| FP32, chunks of 128, channels-first / channels-last | 3.20 / 6.07 | 3.22 / 3.16 | 0 |
| FP16 autocast, unchunked | 20.29 | 1.45 | 0.0027 |
| FP16, chunks of 128, channels-first | 6.55 | 1.74 | 0.0027 |
| FP16, chunks of 128, channels-last | 7.66 | 1.60 | 0.0032 |
| FP16, chunks of 64, channels-last | 7.67 | 1.59 | 0.0029 |

- One chunk shape turns FP16's per-shape setup into a one-time cost of about 8s per process;
  afterwards FP16 with 64-image chunks and channels-last needs 40% less forward time than FP32
  (1.59 versus 2.64), and 37% less with three CPU-burning processes (1.67 versus 2.66, chunks of 128).
- Padding each batch to a multiple of the chunk size wastes about 16–28% of backbone work, which is
  why chunked FP16 (1.59–1.74) trails unchunked FP16 with repeated shapes (1.45).
- Chunking leaves FP32 predictions unchanged and makes FP32 slower (padding without a setup saving);
  channels-last helps FP16 only. FP16 prediction differences match the full fold-0 runs, whose
  macro AUC moved by +0.00004.
- The forward pass is about a quarter of prediction time at 16 loader workers, so the end-to-end
  gain needs a full pipeline run; the GPU process holds a CPU core while launching kernels, so a
  shorter forward also returns CPU to CPU-bound loaders.

### FP16 with chunks in the full pipeline

Bundle `outputs/submission/exports/knee-reflection-sigma40-fp16chunks-20260915` (dataset
`claverru/knee-sigma40-fp16chunks-20260915`) adds `use_fixed_backbone_chunks` to the runtime: FP16
workers run 64-image channels-last backbone chunks, while FP32 is unchanged. Both runs used 16
loader workers and started together on separate machines.

| Run | Command | Worker wall | Studies/s | First-batch forward | Steady forward, s per 1000 images | Loader wait per process | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 control | 379s | 340s | 2.66 | 1.8–3.8s | 2.64–2.68 | 229s | 0.892344 |
| FP16 with chunks | 358s | 326s | 2.77 | 26–34s | 1.67–1.84 | 210s | 0.892380 |

- Steady forward time falls by a third, but cuDNN setup for the chunk shape takes 26–34s on the
  first batch while 16 loader workers compete for CPU (8s on an idle machine), consuming most of
  the forward saving over 904 studies.
- Loader wait falls because GPU processes spend less CPU launching kernels; worker wall improves by
  14s (4%), below the three plain FP32 16-worker runs (329–342s).
- FP16 macro AUC is 0.000036 above FP32 with a 0.0025 maximum prediction difference.

### Warming up the chunk shape

Two runtimes load the model and run one zero 64-image FP16 chunk while the worker indexes.
Warming up in a background thread (`knee-reflection-sigma40-warmup-20260915`) did not help: the
first real batch still took 29–35s, so the cuDNN setup did not carry over to the inference thread.
Indexing in the background thread and warming up on the main thread
(`knee-reflection-sigma40-warmup-main-20260915`) did.

| FP16 chunks, 16 workers | Command | Worker wall | First-batch forward | Total forward per process | Loader wait per process |
|---|---:|---:|---:|---:|---:|
| No warm-up | 358s | 326s | 26–34s | 68–72s | 210s |
| Warm-up in background thread | 384s | 344s | 29–35s | 66–72s | 223s |
| Warm-up on main thread | 359s | 318s | 1.1–1.8s | 40–41s | 237s |

Model loading and warm-up took 13.5s alongside 23s of indexing. Total forward time fell by about
40%, but loader wait rose by a similar amount: with CPU-bound loaders, faster forward passes mostly
turn into waiting, so worker wall improved by 8s over chunks without warm-up and 22s over the
simultaneous FP32 control. Macro AUC stayed 0.892380.

## Implications

- Use 16 total loader workers on Kaggle's two-T4 machine: no retrain, identical predictions.
- CPU is then the limit. The largest remaining costs are header parsing (CPU) and header I/O
  latency. Reading raw bytes with threads and parsing only the needed tags would attack both;
  indexing only each worker's studies and skipping directory counts removes most indexing time.
  These edit preprocessing-identity files.
- On T4, FP16 pays off only with fixed backbone input shapes; test FP16 with 64-image chunks and
  channels-last in the full pipeline at 16 workers before switching.
