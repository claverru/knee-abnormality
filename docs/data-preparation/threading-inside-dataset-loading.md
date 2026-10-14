# Threading inside Dataset loading

Production loading retains sequential preparation inside eight persistent
DataLoader worker processes. The isolated
[threading profiler](../../scripts/adhoc/profile_threaded_loading.py) tested two to
four slice-decoding threads inside each worker while preserving physical order,
sampling RNG order, clipping and production error handling. The prototype is
technically viable, but the measured workload did not show a reliable speedup.
Keep it outside production unless a new storage or codec profile demonstrates a
loader bottleneck.

## Standalone loading measurement

Use standalone DataLoader throughput to compare loading implementations:
consume each CPU batch immediately and divide completed studies and selected
slices by elapsed wall time. This includes DICOM indexing, decoding, resampling,
cropping, joint clipping, collation and worker-to-parent transfer. It excludes
memory pinning, host-to-device transfer, GPU augmentation/normalization and
model execution. Training-step throughput answers the separate question of
whether the loader holds up the GPU; prefetching can hide loading differences.

The 2026-09-10 comparison used the previous single-slice RGB recipe and 512
fold-0 training studies from
`all_5fold_v1`, Steven v4 labels and seed 42: four maximum-load studies plus
508 sampled studies, then a fixed seeded permutation. It retains batch size
eight, 224-pixel crops, 0.8 mm spacing, `[0.005, 0.995]` clipping, physical
positions and slice limits `[17, 16, 15, 15, 15, 15]`. Each configuration runs
in a separate process, with persistent workers and prefetch factor two.
Two passes repeat epoch-zero random slice choices; the second reuses each
worker's header index. OS page cache is uncontrolled, so the first pass must
not be interpreted as a cold-disk measurement.

There are 64 batches per pass. The first 16 are excluded from the primary
throughput metric, leaving 384 studies and 28,351 slices per measured pass.
Full-pass wall time, including iterator startup/reset, is recorded separately.
The hardware is a Ryzen 9 9950X3D with 32 logical CPUs under Linux/WSL.
DataLoader workers and each explicit task thread use one PyTorch compute
thread; `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are
also set to one. The calling process uses 16 PyTorch threads.

The profiler is historical since the 2026-09-22 row-trimmed reads: its threaded arm patches
`read_frame`, which the native decode path no longer calls, so run it against the source of
`ab49830` or earlier. It subclasses the dataset at `_decode_series` and leaves production loading
unchanged. `--decode-threads` threads whole `decode_slice` calls, which is more than the shipped
change; `--read-threads` is the shipped form, prefetching payloads on a pool with `read_frame`'s
validation and decoding serially (the 2026-09-18 cold comparison is in
[closed loading changes](../backlog/loading-speedup/closed.md#threading-loses-when-there-is-no-latency-to-hide)).
Use a fresh output directory:

```bash
rtk proxy env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.profile_threaded_loading configs/resnet50.yaml --mode loader --workers 8 --decode-threads 1 --samples 512 --warmup-batches 16 --output-dir outputs/profiling/threading-loader-new
rtk proxy env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.profile_threaded_loading configs/resnet50.yaml --mode loader --workers 16 --decode-threads 1 --read-threads 4 --samples 256 --warmup-batches 1 --passes 2 --output-dir outputs/profiling/frames-cold-readthreads-new
rtk proxy uv run --locked python -m scripts.adhoc.profile_threaded_loading configs/resnet50.yaml --mode audit --samples 8 --output-dir outputs/profiling/threading-audit-new
rtk proxy uv run --locked python -m scripts.adhoc.profile_threaded_loading configs/resnet50.yaml --mode gpu --workers 8 --decode-threads 1 --samples 256 --output-dir outputs/profiling/threading-gpu-new
```

Warmup batches are excluded from `studies_per_second`. For a cold run, drop the page cache first,
pass `--warmup-batches 1` and read `all_studies_per_second` and `pass_wall_seconds`, which cover the
whole pass; the manifest records `Cached` from `/proc/meminfo` so a warm run is recognisable. Audit
mode compares every collated field with production without CUDA. GPU mode runs bounded optimizer
steps with seeded random weights and saves no checkpoint or validation metric.

Artifacts are under `outputs/profiling/threading_20260910/`. Each run retains
configuration, study order, split/label hashes, dependency versions, source
snapshots, per-batch counts and timings. The eight-study real-input audit in
`audit/` found identical hashes for all collated tensor fields with one, two
and four threads; study IDs and failure records matched too. Those studies
had no preparation failures, so this does not establish runtime coverage of
corrupt-file fallback handling.

## Results and decision

Retain production's eight process workers with sequential slice preparation.
This slice-thread prototype does not demonstrate a reliable loading speedup
on the measured machine. Four-worker threading reduced measured throughput;
the apparent initial eight-worker benefit became small after repeating the
comparison in reverse order.

| Worker processes | Slice threads per worker | First pass, studies/s | Second pass, studies/s | Second pass, slices/s |
|---:|---:|---:|---:|---:|
| 8 | 1, production preparation | 27.63 | 29.17 | 2,154 |
| 4 | 1, production preparation | 19.75 | 27.83 | 2,054 |
| 4 | 2 | 18.90 | 21.49 | 1,586 |
| 4 | 4 | 10.36 | 10.96 | 809 |
| 8 | 2 | 30.71 | 34.40 | 2,540 |
| 8, repeat | 2 | 30.21 | 35.05 | 2,588 |
| 8, repeat | 1, production preparation | 31.85 | 34.51 | 2,548 |

Rows follow execution order. On the repeated eight-worker comparison, two
threads improve second-pass throughput by only 1.57%. The process-only
baseline itself ranges from 29.17 to 34.51 studies/s between runs, so the
initial apparent 17.9% threading advantage is not a robust improvement.
These are bounded timings, not confidence intervals or full-epoch projections;
OS cache state and scheduling were not controlled. All seven loader runs
have identical source snapshots, label provenance, study order and per-batch
selected slice counts. The [machine-readable summary](../../outputs/profiling/threading_20260910/summary.json)
links each configuration to its recorded timings. No production source changed.

Two completed, separate GPU profiles used the then-current repeated-RGB
ResNeXt model, gamma/contrast augmentation, series dropout, BF16 and
checkpointing on an RTX 3090. Each used 256 studies twice, batch eight, seeded random weights and
constant optimizer learning rates, with eight warmup batches per pass. Eight
workers without slice threads reached 14.25 studies/s on the second pass;
four workers with two threads reached 14.09. Loader wait was below 0.06%
in both, and all 64 recorded batch losses matched exactly. These results
diagnose a loader that keeps up with the measured GPU steps; they do not
measure standalone loading capacity or validation quality. Artifacts are
`gpu_w8_t1/` and `gpu_w4_t2/`. The partial `gpu_w4_t4/` run was interrupted
and is excluded from comparisons.

Threading remains technically usable inside `__getitem__`. Revisit this
specific design if storage, decoding, image coverage or GPU consumption
changes enough to make loading a bottleneck. These results do not establish
the performance of other threading granularities or storage systems.
