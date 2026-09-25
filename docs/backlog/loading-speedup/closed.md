# Closed and rejected loading changes

Formulations that were built, measured or ruled out. **Do not retry these as specified.** If a new
idea resembles one of them, say exactly how it differs and why the objection recorded here does not
apply. Open work is in [candidates](candidates.md).

## Closed as measured regressions

All four were built, audited for correctness and measured slower (2026-09-17/18 probes). The
pure-CPU [`resample.py`](../../../scripts/adhoc/kaggle_loading_probes/resample.py) and
[`quantiles.py`](../../../scripts/adhoc/kaggle_loading_probes/quantiles.py) probes measured them
locally ([how to run](measuring.md#probe-on-kaggle-one-kernel-per-change)); the whole-series
resample groups slices by shape, spacing, slope and intercept so each group shares one pair of
cached axis filters.

| Candidate | Result |
|---|---|
| Stacked-sparse resample, one pass per series | bit-exact, **4.8x slower** |
| Batched-dense resample | **2.6x slower**, and not bit-exact |
| Coarse-histogram quantiles | exact bounds, **1.67x slower** on real pixels |
| Fused clip/shift/scale/round normalize | 3.679 against 3.658 ms per acquisition, not bit-exact |

**The lesson that outlives the changes:** the same histogram selection measured **1.87x faster on
synthetic gamma noise**. Real MRI slices carry a large near-zero background spike, so the bin holding
the 0.005 rank contains a large share of the values and the in-bin partition costs more than one
partition over everything. Exact bounds hid the regression entirely. See
[how to measure](measuring.md#measure-intensity-work-on-real-pixels).

**Correction, 2026-09-19 — the second reason once given here was wrong.** This paragraph used to
argue that the baseline is already near memory bandwidth, on the grounds that `_linear_quantiles`
makes one in-place multi-kth `partition` call whose introselect reuses pivots so the floor/ceil
neighbours are nearly free, and that NumPy 2.0 routes `partition` through Intel x86-simd-sort.
Replacing that one call with a partition at the high rank, a partition of its prefix at the low rank
and two tail reductions measured **1.44–1.63x faster with bit-identical bounds on 24 real
acquisitions** ([evidence](../../history/loading-speedup-ranked-by-gain.md#measurement-behind-rank-5)), so the multi-kth
call was not the floor that argument assumed. What still closes, and is unaffected: coarse-histogram
selection is 1.67x slower on real pixels, and `np.quantile` does strictly more work than either
formulation (a NaN scan and a float64 lerp), so neither of those can win. The exact single-rank
reformulation shipped on 2026-09-19, and on 2026-09-22 quantiles moved to the GPU with the rest of
the [image preparation](../../inference/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md#gpu-preparation).

## Threading loses when there is no latency to hide

Threads pay only for I/O latency they can hide, which is why they win on Kaggle's NFS and lose on
local NVMe. Both directions are measured.

Local warm files, per-file throughput: pydicom 3,036 → 325 files/s at four threads; the raw parser
13,776 → 2,461, with CPU per file rising 13x and 7.6x.

Local dataloader, 256 studies, batch 8, 16 workers, page cache dropped before the cold arm
(`outputs/profiling/frames-cold-serial-20260918/`, `frames-cold-threads4-20260918/`):

| Arm | Pass 0, cold | Pass 1, warm |
|---|---:|---:|
| serial | 46.2 studies/s, 5.54 s | 113.1 studies/s, 2.26 s |
| 4 read threads, decode serial (what ships) | 45.6 studies/s, 5.61 s | 82.3 studies/s, 3.11 s |
| 4 decode threads (whole `decode_slice`) | 39.3 studies/s, 6.51 s | 65.4 studies/s, 3.91 s |

**Threading only the payload reads is flat locally; threading whole `decode_slice` calls costs 15%
and is not what ships.** The difference isolates the cause: the earlier regression came from the CPU
half competing for cores, not from the parallel reads. The warm-arm losses were also the leading
explanation for `prepare_series` running
[slower warm than cold](attribution.md#prepare_series-slower-warm-than-cold-machine-dependent-not-code),
which two Kaggle runs showed and a third did not reproduce.

## Ruled out by arithmetic or by the environment

- **`dicomsdl`.** 0.097–0.110 ms per open-plus-27-fields locally, against 0.0259 ms for the raw
  parser with the [struct scan](candidates.md#bit-exact-batch-into-the-retrain) — 3.7–4.2x slower,
  before the offline wheel and the field-for-field equality audit it would still need. It does
  expose the pixel-data offset (`getDataElement("PixelData").offset()`), so the option was real;
  it is speed that closes it.
- **Sidecar, parquet or DB header cache.** Headers are parsed exactly once per file per run
  (`_slice_index` plus one study per worker), so a path-keyed cache has no within-run reuse. The
  image-count parquet is absent on Kaggle precisely because the test studies are unseen before
  submission.
- **Local SSD staging of test DICOMs.** 1,300 studies at ~124 MB is about 161 GB, against roughly
  20 GB in `/kaggle/working` and 60 GiB in `/kaggle/tmp`; it does not fit either. Each byte is read
  once, so staging is an NFS read plus a local write plus a local read. This is the already-rejected
  notebook pre-read (112 GiB through 32 GB of RAM, 249 s of stalled transfers) with a different
  destination, which does not change the arithmetic.
- **One-time conversion to PNG, npy, zarr, hdf5 or webdataset.** Zero for Kaggle inference: the
  hidden test DICOMs do not exist before submission. A cache of *prepared* images is additionally
  ruled out because crop and quantiles depend on the per-item selection. A cache of *raw slice
  pixels plus parsed geometry* would be bit-exact — the sampler interpolates in-plane only, not
  through-plane — but it can only serve training, where the whole benefit is epoch zero (4.25 s →
  3.24 s of a 6.83 s cold pass) and steady-state batches are pure queue drain. Worth roughly 1–4 s
  per training run and nothing at inference.
- **NFS client tuning** — `nocto`, `nconnect`, `actimeo`, `lookupcache`, `rdirplus`, rsize,
  readahead, RPC slot tables. All are mount-time and root-only, and `/kaggle/input` is a
  platform-managed read-only mount inside an unprivileged container. `readdirplus` would not help an
  `open()` under close-to-open semantics anyway, and NFSv4.2 already negotiates rsize/wsize to
  1 MiB, so the only documented lever left is client concurrency, which the thread pool already
  exploits.
- **Removing a per-file `stat`.** There is none to remove: `list_series` reads only `entry.name`.
  The per-file floor is `open()`'s close-to-open `GETATTR`, removable only by `nocto`, which is
  unreachable.
- **`list_series` sorting names before building `Path` objects** (4.51 → 0.67 µs/file, 0.3 s
  cold-100). Correct and measured, but it lands inside `build_series_index`, which since the
  2026-09-18 index changes is 7.1 s per worker hidden behind 15.5 s of model load. It buys nothing
  at run scale, and the cold-100 probe is not what the
  [efficiency prize](../../inference/efficiency-leaderboard.md#the-exchange-rate) measures.
- **RAM or page-cache warm-up, and notebook pre-listing.** Measured: indexing fell from 27 s to
  6–9 s but worker wall rose to 355 s against 329–342 s. Setup holds no one-off mount penalty
  either — cold minus warm is per-directory dentry misses, so a warm-up thread would just do the
  same `scandir`s. Threading the listing *inside* `build_series_index` is a different formulation,
  and it shipped 2026-09-18.
- **A coordinator that builds the index once and passes it down**, by shm or a counts parquet. The
  serial build would move onto the critical path; today it hides behind model load. The worker-filter
  plus threading gets the same result without IPC.
- **Deferring the listing into the loader workers**, with `StudySeries.slots` as a cached property
  that lists and sorts on first touch. It was worth ~41 s/worker only while it subsumed the worker
  filter and the threaded listing, and both
  [shipped 2026-09-18](attribution.md#what-the-shipped-changes-bought). What is left of indexing is
  5–8 s hidden behind 12.9–13.4 s of model load, so the wall saving is zero, against a real cost: a
  missing directory would fail inside a loader worker rather than at setup. The copy-on-write
  argument survives at half size — each worker now holds only its own studies — and does not carry
  the change on its own.
- **Removing the unused `patient_maps` from the model hot path.** `prepare_series` builds them and
  `collate_studies` drops them, so they are genuinely dead weight for the model — but measured at
  0.01% of an item's CPU (68 tensors, 33,480 elements over 12 studies). Only
  `scripts/adhoc/inspect_physical_preparation.py` (removed; last in commit `df4b6cd`) read them, so removing the field costs a script
  port and an identity bump for nothing.
- **Trimming the diagnostics and sampling metadata carried on every batch.** Measured at 0.14 ms and
  0.22 MiB per batch of four studies, against 82.8 MiB of tensors in the same batch — 0.2% of the
  payload either way, and the tensors move through shared memory rather than pickle.
- **Packed collation to remove padding.** Padding amplification is **1.445** (194,778 padded triplet
  slots against 134,750 real ones, identical across all three x4 runs), so 31% of the image tensor
  is zeros — about 8 GB per run. It costs the zero-fill and copy inside `collate_studies`, which the
  cold probe puts at 0.95 s per 100 studies in total, so the recoverable part is on the order of a
  second of wall. Ordering each worker's studies by triplet count captures most of it without
  touching the model contract.

- **Prefetching the next study's bytes while the GPU works.** Already present via
  `prefetch_factor`, and with 231–238 s of loader wait there is no spare loader capacity to prefetch
  with.
- **Swapping the CPU resample backend.** At the production shape the current code is 0.308 ms; PIL
  mode-F BILINEAR with `box` is 0.412 ms and bit-identical, torch CPU `interpolate(antialias=True)`
  0.323 ms and bit-identical, `scipy.ndimage.zoom(order=1)` 0.978 ms and it aliases, torch without
  antialias 0.086 ms and it aliases. The two bit-exact options are *slower*; the fast one is wrong.
  Our hand-written filter is the standard PIL triangle filter, which is why all three agree
  bit-for-bit. cv2, SimpleITK and MONAI would each add an offline wheel and import time.
- **`readinto` into a reused buffer, mmap, and contiguous-volume assembly.** `readinto` measured
  199–226 µs/frame against 154–189 for plain serial reads; mmap 203 µs/frame with no win, and over
  NFS it faults page-by-page instead of issuing one large read, with close-to-open coherence caveats
  and `SIGBUS` if the file changes. Contiguous assembly measured no better, and the allocator
  env vars that were meant to capture the same benefit measured no production gain
  ([closed 2026-09-22](#closed-2026-09-22)).
- **Tiled or cache-blocked normalize, a manual tiled transpose, and a transpose-free tap loop.**
  Blocking one or four slices through all five ufuncs is no faster at N = 16/26/40/64; a manual
  tiled transpose does not beat `np.ascontiguousarray`; the tap-loop pass 2 is bit-exact but 1.85x
  slower. `np.empty` instead of `np.zeros` saves 0.8 µs per slice, negligible.
- **Inline logsumexp in the sampler.** A known speedup in general, but SciPy's implementation is not
  the textbook max-trick: an inline version shifted targets by 1e-16 to 8e-14 and **flipped
  selections in 1,067 of 3,200 harness cases**. Rejected on correctness.
- **Replacing `linear_sum_assignment` with a greedy or small-N exact method.** It costs 5 µs on the
  real (24,55) matrix. Pointless, and it would put the tie-breaks at risk.
- **numba or torch for the sampler.** Compilation is per process and `cache=True` only shortens it,
  so all 16 loader workers pay warm-up, plus a new offline wheel, for the ≤1.6 s that would remain.
- **Early exit from the header scan.** The last wanted tag is element 62 of 63 before PixelData,
  because DICOM orders elements by increasing tag so group 0x0028 necessarily abuts 0x7FE0. A
  stop-after-0x0028 variant measured 0.0327 against 0.0324 ms — nothing.
- **Dropping header tags.** None is droppable: all 29 appear in at least 191 of 1,200 files and each
  feeds the pixel descriptor, the pose or the
  [series filtering rules](../../data-preparation/current-preparation.md#series-indexing-guards-and-exclusions).
- **GPU frame decoding** (DALI, nvJPEG2000). Our frames are uncompressed Explicit VR Little Endian,
  single-frame, MONOCHROME2 — 1,419 of 1,419 sampled series — so there is nothing to offload.
- **Also closed earlier:** autocast FP16 with variable shapes, more than 16 loader workers,
  notebook pre-reading of DICOMs or packages, a local bytecode cache, and TPU notebooks (not allowed
  for submission).

## Closed 2026-09-22

Measured on Kaggle at ×4 over 1,300 studies, one run each, on hosts that vary by about ±15 s
([profile](../../inference/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md)).

| Candidate | Result |
|---|---|
| glibc tunables (`MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_` at 256 MiB, `MALLOC_ARENA_MAX=2`), shipped 2026-09-19 unmeasured | 912 against 893 loader CPU-seconds with and without them, loader RSS 1,437 against 1,220 MiB; removed. They make `resample_crop` 2.5× faster in a local tight loop, which production does not reproduce |
| Support-limited float32 conversion, shipped 2026-09-19 | no Kaggle gain: frame reads plus resample 23.6 against 23.4 CPU-seconds per 100 studies; superseded by GPU preparation |
| `share_memory_()` on the zeroed collate tensor, shipped 2026-09-19 | Torch promotes a private tensor by allocating shared storage and copying, so it only moved the copy; collation now allocates shared storage directly |
| Pinning raw batches in the DataLoader after GPU preparation | the pin thread keeps a pinned copy of every prefetched ~240 MB batch; each GPU process reached 8.8 GB RSS and the host killed a worker |
| 16 io threads once the run is NFS-bound | 308.7 s against 283.5 s at 8 |
| `prefetch_factor` 4 at inference | 330.3 s against 314.2 s at 2 |
| Importing the whole model stack before forking the loaders, with niced loaders | 303.9 s against 283.5 s; model load stretched to 34 s |
| 12 loader workers | 297.8 s against 303.9 s at 16 in the same arm, inside host variance |
| Page-locked copy on a feeder thread | the copy fell 252 → 33 ms per batch, but the thread's GIL use stretched the GPU loop ~25 s per GPU; 278.7 and 264.3 s against 273.3 s inline |
| `TORCH_CUDNN_V8_API_DISABLED=1` to skip the 9 s warm-up | warm-up 10.9 → 0.8 s, but each 64-image FP16 chunk 96 → 197 ms |
| `torch.backends.cudnn.benchmark` | warm-up 19 s at the same chunk speed |
