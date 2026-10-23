# 2026-10-17 loader buffer pool

Where loader CPU goes on Kaggle and what reusing pixel buffers changes. **Loader workers reuse
their shared-memory pixel buffers from a pool capped at 768 MiB per worker: S156 TensorRT ×4 takes
268.9 s against 297.6 s (−28.7 s, about 0.0004 AUC at the
[exchange rate](2026-10-16-efficiency-exchange-rate.md)). Direct (`O_DIRECT`) reads cut CPU
further but are slower in wall time and were dropped.**

## Kaggle environment

- `/kaggle/input` is NFS v3 with 512 KB reads; every open costs a `GETATTR` and readahead pulls
  whole files, about 107 MB per study for what we use.
- CPU notebook: 4 vCPU Xeon 2.2 GHz, 32 GB, kernel 6.18, transparent huge pages `madvise`, shared
  memory huge pages `never`. `/dev/shm` holds 15 GB.
- The S156 TensorRT ×4 run (2×T4, 8 loader workers) is CPU-saturated: idle 4%, system 42%, nice
  30%; each GPU process uses 126 CPU-s and the loaders about 0.43 CPU-s per study.

## Loader lab

`profile_kaggle_inference` condition `loader-lab` (CPU notebook) runs loader variants in
subprocesses (`LOADER_LAB_PASS`) on 100 studies, three rounds in rotating order, cold cache through
`POSIX_FADV_DONTNEED`, 600 s per pass, then a fingerprint pass (SHA-256 of the prepared batch
tensors) per variant; `analyze` writes `lab_summary`. Outputs:
`outputs/profiling/kaggle-loader-lab-{1,2,3b,4,5,6,7}-20261017`, base profile
`kaggle-s156-loading-20261017` (100 cold studies, 62.5 s wall, 42 CPU-s; reads dominate, a DICOM
header parses in about 0.17 ms).

| Variant, 100 studies | Loader CPU | Fingerprints |
|---|---|---|
| Read each file once, whole-file direct read, random order | no gain | identical |
| `MAP_POPULATE` | −5% once, not reproduced | identical |
| Pre-population on a thread | none | identical |
| **Reused pixel buffers** | **40.9 → 30.7 CPU-s (−25%)**, wall 56.9 → 50.9 s | identical |
| `O_DIRECT` reads (prefix and payload) | −11 to −12% | identical |
| Reused buffers and `O_DIRECT` | 43.4 → 28.8 CPU-s (−34%), system 24.2 → 8.9 | identical |

Page faults on fresh shared-memory buffers, about 9 GB per 100 studies, were 20–25% of loader CPU.

## The pool

`knee_abnormality.inference.buffers`, installed in each loader worker by `predict_worker`
(non-hashed module; it replaces `preparation.shared_empty`).

- Each worker holds 64 slots of shared storage, rounded up to 4 MB, best fit, for `torch.int16`
  buffers only (CPU-prepared `uint8` images keep unwritten padding and broke parity).
- A shared `uint8` flag array (workers × 64) releases slots explicitly: the worker sets a flag when
  a batch takes a slot, the GPU process clears it after the batch is consumed. Releasing through
  weak references corrupted 8 of 12 studies locally.
- The GPU process page-locks each pooled storage once (`PinnedBuffers`) and unregisters it when the
  slot's storage is replaced.
- **Cap, `POOL_BYTES` = 768 MiB per worker.** Beyond it a request is served unpooled.

## The shared-memory bug

The first pool kept each slot's largest storage and never shrank. With 8 workers `/dev/shm` peaked
at 14.0 GiB and `_new_shared` raised `unable to allocate shared memory` 7 times in one run. The
replacement deleted the slot's key before allocating, so a failed allocation, caught as a
preparation error, left a slot without a key: its buffers reported slot −1, the pending slot was
freed while still in flight and was overwritten. GPU-side checksums caught 3 corrupted buffers in
one run, and the next replacement of such a slot raised `KeyError` and crashed the run. This
explains the earlier diverging predictions (study …496260885546, Baker's cyst 0.0334 against
0.1757).

Fix: allocate first, then drop the old key, and cap the pool. An instrumented full run with a
1 GiB cap had no failed allocation, no stale slot and no checksum mismatch; `/dev/shm` peaked at
12.1 GiB.

## Timing

Kaggle 2×T4, the same 1,300 studies of `all_5fold_scanner_v2`, `fp16-b4-w8`, one run each. Hosts
differ by about ±15 s on identical code.

| Variant | Output | Command | Loader CPU | Loader wait per GPU |
|---|---|---:|---:|---:|
| S156 TensorRT ×4 | `kaggle-s156-trt5-x4-20261016` | 297.6 s | 537 CPU-s | 27, 14 s |
| **Fixed pool** | `kaggle-s156-trt5-x4-pfnd-r1-20261017` | **268.9 s** | 447 CPU-s | 28, 22 s |
| Fixed pool and `O_DIRECT` | `kaggle-s156-trt5-x4-poolfix-r2-20261017` | 323.0 s | 413 CPU-s | 100, 96 s |
| Old pool and `O_DIRECT`, 16 I/O threads | `kaggle-s156-trt5-x4-pooldirect-io16-20261017` | 319.5 s | 363 CPU-s | 102, 104 s |

- **Parity of the pool against the base:** max |Δ| 0.0067 on a synovitis study (…624156291171)
  that also moves between two base-loader runs; 4 studies above 0.0015. The two studies that
  diverged before the fix match exactly.
- **`O_DIRECT` loses on wall time.** Its reads are byte-identical (6,295 files, no fallback), but
  synchronous direct reads lose readahead and the loaders wait on NFS; 16 I/O threads do not help.
  Removed.

## Not done

- A validation-fold score with the pool.
- The same reuse in the training loaders (local disk, so the gain will differ).
