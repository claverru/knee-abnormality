# Kaggle x4 profile: startup, CUDA graph and GPU preparation

Kaggle A/B runs of the full inference command on 2026-09-22, taking **429.5 s to about 270 s
(−37%)** with prepared images bit-identical to production. Supersedes the
[post-CPU-cluster profile](2026-09-18-kaggle-x4-profile-after-cpu-cluster.md) as the runtime
picture of 2026-09-22. The code was developed on branch `inference-opt` (based on `ab49830`) and
reached `main` as commit `4d23276`.

Every run: two T4s, four vCPUs (two physical cores), 1,300 studies (fold 0 of
`all_5fold_scanner_v1` padded from other folds), ×4, τ 0.75, batch 4 per GPU, 16 loader workers,
8 io threads, FP16. Retrain-free arms use the `k_05` checkpoint; arms that change hashed
preprocessing sources use a seeded random-weights timing checkpoint built from it
(`make_timing_checkpoint --seed 20260922`), so their predictions are not scored. Artifacts are
`outputs/profiling/kaggle-*-20260922/`; each arm's bundle is in `outputs/submission/exports/`.

## Results

| Arm | Change | Command | Loader CPU | GPU-process CPU |
|---|---|---:|---:|---:|
| Baseline | production code (`k05` bundle) | 429.5 s | 1,096 | 196 |
| B | torch-free coordinator, index built during imports, loader workers forked before model load, blocking CUDA sync, optional `wandb` import skipped | 358.0 s | 906 | 188 |
| C | B + CUDA graph for the 64-image backbone chunk | 332.4 s | 912 | 109 |
| C − malloc | C without the 2026-09-19 glibc tunables | 316.2 s | 893 | 104 |
| E | C − malloc + niced loaders, fork before the timm/sklearn import | 314.2 s | 896 | 125 |
| GPU prep | E + [device image preparation](#gpu-preparation), no pinned memory | 331.8 s | 601 | 176 |
| Trim | GPU prep + [row-trimmed payload reads](#row-trimmed-reads) | 283.5 s | 592 | 168 |
| **Direct** | trim + [reads into shared memory and a page-locked copy](#direct-reads-and-page-locked-copies) | **273.3 s** | **538** | 196 |

CPU columns are CPU-seconds from launch to the last batch. Predictions of B, C, C − malloc and E
are bit-identical to the baseline's. The direct arm was the best, and its code is what the
branch holds; two runs of the same reads with a feeder thread gave 278.7 and 264.3 s, so its
state was about 270 s. Loader wait is then ~27 s per GPU and the GPU process's own loop is
the critical path.

Rejected on the same setup, all single runs:

| Arm | Command | Why it is rejected |
|---|---:|---|
| D: nice + early fork, with malloc tunables | 435.8 s | a slow-NFS host (63% busy, 36% iowait); uninformative, repeated as E |
| F: E + `prefetch_factor` 4 | 330.3 s | no gain over E; more shared memory |
| Trim + 16 io threads | 308.7 s | no gain over 8 |
| G: import the whole model stack before the fork | 303.9 s | model load stretched to 34 s against niced loaders |
| G with 12 loader workers | 297.8 s | no gain over 16 |
| Trim + page-locked copy on a feeder thread | 286.2, 269.4 s | copy 252 → 33 ms per batch, but the thread's GIL use stretched the main loop by ~25 s per GPU; the inline copy is kept |
| Direct reads with that feeder thread | 278.7, 264.3 s | no better than the inline copy at 273.3 s |
| Legacy cuDNN algorithm selection (`TORCH_CUDNN_V8_API_DISABLED=1`) | probe | warm-up 10.9 → 0.8 s, but each 64-image chunk 96 → 197 ms |
| `torch.backends.cudnn.benchmark` | probe | warm-up 19 s at the same chunk speed |
| GPU prep with pinned batches | killed | pinning every prefetched raw batch took each GPU process to 8.8 GB RSS and the host OOM-killed a worker |

**Host variance is the limit on these comparisons.** Identical loader code measured 592–696
loader CPU-seconds across hosts, and NFS speed alone swung D by ~120 s. A single-run difference
under ~15 s is not a finding; compare CPU-seconds and setup timestamps as well as wall time.

## Startup

The baseline spent **61 s** before its first forward pass: 32.6 s of coordinator import, 8.6 s of
worker import, 17.6 s of model load and a 14 s first-batch wait.

- **The coordinator imported torch only to count GPUs.** It now counts them through the CUDA
  driver (`inference/processes.py`), imports in 0.2 s, and spawns workers ~3 s after launch.
- **Model load was mostly a CPU-bound first FP16 chunk.** A standalone probe put the warm-up at
  9.4–10.3 s of CPU per process (~0.9 s of it disk), read and construction under 1.3 s, CUDA init
  0.5 s. Cold imports cost 21.5 s against 7.4 s warm; `timm` alone is 3.3 s warm, of which
  `wandb` (imported optionally by `timm.utils.summary`) is 1.6 s.
- **Loader workers now fork before the model loads**, so the first batch is waiting when the
  model is ready. After the fork the model-stack import and load take ~41 s, because four vCPUs
  are two hyperthreaded cores and niceness cannot stop a loader sharing the main thread's core.
  Setup is ~57 s of the trim arm's 283.5 s; see [remaining leads](#remaining-leads).

## GPU-process CPU

- **Default CUDA scheduling spins.** With one context on four logical CPUs the driver spins a
  full core while the GPU works: CPU/wall 1.00 against 0.56 with blocking sync, 0.70 against 0.25
  under four CPU burners, GPU time +1%. Workers set `CU_CTX_SCHED_BLOCKING_SYNC` on the primary
  context before creating it.
- **Eager launches cost ~20–40 ms of CPU per 64-image chunk.** The backbone chunk is now a
  captured CUDA graph (`use_fixed_backbone_chunks`), which cut GPU-process CPU 188 → 109 CPU-s and
  forward wall ~142 → ~105 s per GPU.

## Loader CPU

A py-spy profile of all 16 loader workers in arm E attributed the loader main threads as follows;
the 2026-09-15 loading cell overstated preprocessing because it runs without the production
environment.

| Share of loader main-thread samples | % |
|---|---:|
| `resample_crop` (sparse matmul 18%, CSR column slicing 8.6%, casts and allocations) | 39.1 |
| header parse (`parse_prefix`) | 12.8 |
| quantiles | 9.5 |
| collate (4.1 of it the `share_memory_()` copy) | 7.5 |
| `select_triplets` | 6.6 |
| normalize | 4.4 |
| `summarize_series` | 4.1 |
| frame decode | 3.3 |

- **The 2026-09-19 support-limited conversion bought nothing on Kaggle**: frame reads plus resample
  stayed at 23.6 against 23.4 CPU-seconds per 100 studies. Frame-read CPU is mostly the kernel's
  NFS read cost.
- **The glibc tunables cost memory for no gain in production**: 893 against 912 loader CPU-seconds
  without them, loader RSS 1,220 against 1,437 MiB, so they are removed. Locally they make
  `resample_crop` 2.5× faster in a tight loop, which production does not reproduce.
- **`share_memory_()` after `torch.zeros` still copies**: Torch promotes a private tensor by
  allocating shared storage and copying it. Decoding now reads into shared storage and collation
  passes it through ([direct reads](#direct-reads-and-page-locked-copies)).

### GPU preparation

Resampling, quantiles and normalization move from the loader workers to the start of the batch on
the GPU (`data/preprocessing/device.py`, called by `predict_loader` and Lightning's
`on_after_batch_transfer`). Loader workers ship each retained frame's raw 16-bit rows of the
filter support and its plan key; frames that are not 8/16-bit or whose rescale exceeds 1e30 are resampled on CPU and
shipped as float32 regions.

It is **bit-exact by construction**: the scipy CSR product accumulates taps in index order with
separate float32 multiply and add, which the device code repeats tap by tap; quantiles are exact
order statistics with the same float32 interpolation; clip, shift, scale and round-half-even are
elementwise. Evidence:

- synthetic: 0 of 250,880 pixels differ for the tap loop, and end-to-end collate + device
  preparation matches the old `_linear_quantiles`/`_normalize_pixels` path on uint16, int16 with
  rescale, uint8 and float32-fallback series;
- **Kaggle parity: 300 studies, 1,656 series, 0 mismatches** in prepared images, rectangles,
  positions, patient maps and geometry validity, against the old pipeline with Kaggle's own scipy;
- census of all 4,407 train studies at ×4: 24,365 series, **0 preparation failures**, so the one
  behavioral difference — a constant series is now dropped on the device instead of re-sampled —
  never triggers on known data.

Loader CPU fell 896 → 601 CPU-s; the run then became NFS-bound (72% busy, 27% iowait). Raw blocks
are ~54 MB per study against ~7.5 MB of uint8 images, so inference no longer pins batches, and the
pageable copy costs ~43 s of GPU-process wall per GPU.

### Row-trimmed reads

The resampling plan depends only on header fields, so the payload read now fetches only the rows
the vertical filter touches. Bit-exact (same Kaggle parity: 0 mismatches), 6–19% fewer payload
bytes depending on geometry, and it took the command from 331.8 s to 283.5 s.

Single-process cold pass over 100 studies, CPU-seconds, old path against trim:

| Stage | Before | Trim |
|---|---:|---:|
| resample, quantiles, normalize | 19.0 | 0 |
| frame reads | 11.7 | 14.2 |
| headers and series indexing | 12.9 | 11.1 |
| sampling | 2.4 | 2.7 |
| **pass** | **51.6** | **37.2** |
| collate (wall) | 2.8 | 8.2 |

### Direct reads and page-locked copies

Each decode call now reads its frames' payload rows with `os.preadv` straight into one shared int16
buffer and masks the stored bits in place, so frames reference that buffer and collation passes the
buffers through without copying. It removes the per-frame `bytes` object, the masked copy and the
collate copy. Loader CPU fell ~13% (592 → 494–545 CPU-seconds over three runs) and loader RSS
~35% (2.0–2.3 → 1.4 GB); Kaggle parity again found 0 mismatches over 300 studies and 1,656
series. Frames keep their full stored rows, so a batch carries ~439 MB.

The GPU process page-locks those shared buffers in place (`cudaHostRegister`) before an
asynchronous copy and releases them after the batch, which took the copy from ~252 ms to ~33 ms
per batch without pinned copies.

The GPU loop per batch, synchronized and uncontended (`mainloop` probe, 16 batches):

| Step | ms |
|---|---:|
| page-lock | 26 |
| transfer | 36 |
| `prepare_images` (of which the quantile sort 34) | 142 |
| model forward | 589 |
| release | 10 |

Under production contention the same loop takes ~1.12 s per batch.

## Remaining leads

- **The model forward is ~75% of the GPU loop.** ReXNet-200 runs 96 ms per 64-image chunk in
  FP16; the backbone alone is ~96 s per GPU at ×4.
- **`prepare_images`, ~142 ms per batch.** Order statistics by `topk` on the few ranks instead of
  a full sort, and fewer small launches, are exact candidates worth ~50–80 ms.
- **Overlap the copy without a thread**: fetch and copy batch *i+1* on a side stream before
  launching batch *i*'s forward, worth up to ~36 ms per batch.
- **Setup, ~50–57 s.** The model-stack import after the fork (~26 s cold) and the 9 s CPU-bound
  cuDNN warm-up run on a core shared with loaders; the warm-up cannot be cut without slowing the
  forward.
- **Headers are ~30% of loader CPU**, spread over many small functions.
- **Training memory is untested.** Training still pins memory, and its batches now carry raw
  16-bit rows, several times the old image bytes.
- **Tooling:** py-spy `--subprocesses` loses the loader workers once they fork early, falling
  300 s behind with 4k samples; the loading cell's stage table remains the loader attribution.
