# Kaggle x4 profile after the index changes

What the worker-filtered series index, the threaded listing and `loader.in_order = False` bought end
to end, and the first loading stage table with a real CPU column. Supersedes the numbers in the
[profile before them](2026-09-18-kaggle-x4-profile-after-loading-changes.md), which this run repeats
condition for condition.

**Weights are random.** No trained checkpoint carries the current `preprocessing_identity`, so the
bundle was exported from `outputs/submission/timing-profile-random-weights.ckpt`, rebuilt for this
identity with [`make_timing_checkpoint`](../../scripts/tools/make_timing_checkpoint.py). Every timing is real;
every prediction is meaningless, and no AUC or parity number was computed.

Rebuild it whenever a change to `PREPROCESSING_SOURCES` bumps the identity and no trained
checkpoint carries it. It keeps the source checkpoint's config, targets, fold and study counts,
so timing bundles stay comparable, and writes fresh seeded weights on CPU without reading data:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.tools.make_timing_checkpoint
```

Bundle `outputs/submission/exports/timing-profile-x4-v2`, dataset
`claverru/knee-profile-timing-x4-20260918b`. Prediction artifacts in
`outputs/profiling/kaggle-inference-x4-20260918b/`, notebook
`claverru/knee-profile-fp16-b4-w16-x4-20260918b`. The loading cell was re-run after the
[profiler fixes](../backlog/loading-speedup/attribution.md#the-profilers-defects-fixed-2026-09-18)
landed: `outputs/profiling/kaggle-loading-x4-20260918c/`, notebook
`knee-profile-loading-x4-20260918c`, same seeded 100 studies.

## End to end, two T4s, 1,300 studies

`fp16`, batch 4 per device, 16 loader workers total, x4 images, four vCPUs, 134,750 images in both
runs.

| | Before w0/w1 | After w0/w1 |
|---|---|---|
| Studies | 650 / 650 | 650 / 650 |
| Indexing (`data_setup`) | 54.3 / 54.3 | **7.1 / 7.1** |
| Model load | 12.9 / 13.4 | 15.5 / 15.6 |
| Setup wall | 54.4 / 54.5 | **15.7 / 15.7** |
| Prediction | 403.1 / 405.2 | 399.5 / 400.5 |
| — loader wait | 238.5 / 230.9 | 231.9 / 225.2 |
| — forward | 163.3 / 173.0 | 166.3 / 173.9 |
| p90 batch wait | 5.83 / 5.31 | 4.25 / 4.29 |

**470.5 s → 427.5 s of worker wall, 2.763 → 3.041 studies/s, a 1.10x.**

- **Indexing collapsed as predicted and is now fully hidden.** 54.3 → 7.1 s, and setup wall 54.4 →
  15.7 s, which is 38.7 s per worker against the ~41 s the arithmetic predicted. Model load, at
  15.5–15.6 s, is now the floor: indexing hides entirely behind it, so **no further indexing change
  pays anything** until model load itself is cut.
- **`in_order = False` shifted the wait distribution and trimmed the tail.** Loader wait fell 5.7–6.6
  s per worker and p90 fell from 5.31–5.83 s to 4.25–4.29 s. Median wait rose from 0.005 s to
  0.32–0.61 s and batches waiting under 50 ms fell from 119–121 of 163 to 65–73, which is the
  signature of work spreading across batches instead of collecting in a few stalls. The seconds are
  within machine variance; the distribution change is not.
- **The loader still binds**, at 56–58% of prediction time, with 166–174 s of that spent inside the
  forward call. This entry originally read the 166–174 s as a floor and concluded that only ~230 s
  per worker was addressable; **that conclusion is withdrawn**. The later io sweep measured the same
  forward work at 102 s when the box was not saturated
  ([sweep](2026-09-18-kaggle-x4-profile-after-cpu-cluster.md#the-io-thread-sweep-2026-09-18)), so
  most of the gap above ~100 s is contention and is addressable by removing loader CPU.

## Where the CPU goes, and how much a repeat run moves

A third notebook, `knee-profile-fp16-b4-w16-x4-20260918d`
(`outputs/profiling/kaggle-inference-x4-20260918d/`), repeated the same condition with per-process
CPU sampling. It came in at **418.1 s of worker wall, 3.109 studies/s**, against 427.5 s and 3.041
the run before, with indexing 7.2 s, model load 17.3–17.4 s, loader wait 197–213 s and forward
173–191 s.

**Run-to-run variance is about 2% of wall and 15 s of loader wait**, which is larger than the
5.7–6.6 s of wait that `in_order = False` appeared to save. The p90 and the wait distribution
remain its evidence; the seconds do not.

Over the 395.5 s prediction window, with the box 98.2% busy at 0.7% iowait and 3.44 of 4 cores in
the cgroup:

| Role | Processes | CPU seconds | Share | Peak RSS |
|---|---:|---:|---:|---:|
| Loader workers | 16 | **1,241.6** | **91.3%** | 1,738 MiB |
| GPU main processes | 2 | 113.0 | 8.3% | 2,754 MiB |
| Other | 5 | 4.6 | 0.3% | 164 MiB |
| Coordinator | 1 | 1.1 | 0.1% | 850 MiB |

Preparation CPU inside the loader workers is the bottleneck and nothing else is close. Production
spends **0.955 CPU-seconds per study against the probe's 0.613**, so 1.56x is the price of 16
worker processes contending for four cores
([attribution](../backlog/loading-speedup/attribution.md#the-loader-workers-own-91-of-it)).

## Cold loading stages, 100 studies, with CPU

One CPU notebook, four vCPUs, single process, sequential. **Wall and CPU are separate measurements
now**; CPU is `process_time`, so it counts the 16 io threads and can exceed wall.

| Stage | Cold wall | Cold CPU | Warm wall | Warm CPU |
|---|---:|---:|---:|---:|
| Series listing index | 0.96 | 1.32 | 0.63 | 0.96 |
| Other setup | 0.37 | — | 0.00 | — |
| Slice headers (`parse_prefix`) | 5.16 | 5.16 | 5.00 | 5.00 |
| Series indexing, rest | **30.98** | **12.86** | 7.96 | 11.98 |
| Geometry and triplet sampling | 5.46 | 5.42 | 5.65 | 5.65 |
| Frame reads (`decode_series` − resample) | 15.54 | 17.38 | 12.15 | 16.83 |
| Resample and crop | 8.02 | 7.99 | 7.85 | 7.84 |
| Quantiles | 4.77 | 4.76 | 4.76 | 4.76 |
| Normalize | 2.36 | 2.36 | 2.36 | 2.37 |
| Preparation, rest | 2.72 | 2.69 | 2.65 | 2.65 |
| `__getitem__`, rest | 0.37 | — | 0.32 | — |
| Collate | 0.95 | — | 0.73 | — |
| **Total** | **77.65** | 61.25 | **50.08** | 59.10 |

**0.776 s per study cold, 0.501 s warm.** Whole-pass CPU per wall second is 0.79 cold and 1.18 warm,
with 273,771 minor and 276 major faults cold against 134,902 and 0 warm. The rows sum to the total
by construction and none is negative.

Separately, and **not addable to the table**, the pool threads accumulated 96.4 thread-seconds of
`read_payload` cold and 59.1 warm over 14,882 frames, against 4.3 s of `frame_from_payload`. That
number is summed across 16 threads; it is what the old profiler was silently adding into the frame
stage.

## What the CPU column settles

- **The 31 s indexing bucket is latency, not work.** Cold it is 30.98 s of wall for 12.86 s of CPU;
  warm the same work is 7.96 s of wall for 11.98 s of CPU. So the stage holds a **~12–13 s CPU
  floor** and about 18 s of unhidden cold NFS wait. The CPU floor is the target of the shared
  thread pool, and the wait is not — more concurrency is what addresses the wait.
- **The frame stage is CPU-bound, not bandwidth-bound.** Its CPU exceeds its wall in both passes
  (17.38 against 15.54 cold, 16.83 against 12.15 warm): the threads hide the reads and the serial
  decode sets the pace. The byte ceiling agrees — the 16-thread full-file probe reached **510 MB/s**
  on this machine against the 271–318 MB/s production estimate, 53–62% of ceiling. **Byte reduction
  is therefore not the lever it looked like**, which demotes the row-trimmed payload read; CPU
  reduction is.
- **`prepare_series` slower warm than cold did not reproduce.** 33.4 s cold against 29.8 s warm here,
  where the previous run had 34.97 against 40.46. The anomaly is machine- or mount-dependent, not a
  property of the code.

Read throughput on this machine:

| Operation | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| Header prefixes, files/s | 97.8 | 353.2 | 977.7 | **1,128.4** |
| Full files, files/s | 71.4 | — | 791.9 | — |
| Full files, MB/s | 52.7 | — | **510.0** | — |

64 threads is 15% above 16 for header prefixes, where the 2026-09-18 run put the two within noise.
That reopens `INFERENCE_IO_THREADS` as a question, but the sweep still has to come after the shared
pool and has to run at the production worker count, not in one idle process
([how to measure](../backlog/loading-speedup/measuring.md#measure-at-the-target-worker-count)).

## What this run cannot tell us

- **The loading condition is still single-process and sequential.** It says nothing about 16 loader
  workers contending for four vCPUs, and its CPU column is one process's, so the production question
  of CPU saturation across workers remains open.
- **The two conditions see different changes.** The profiler writes a synthetic `test_series.csv`
  holding only the studies a condition profiles, so the loading cell's 100 studies were never
  over-indexed: it measures the threaded listing alone, and only the prediction cell exercises the
  worker filter and `in_order`.
- **Different machines from the previous run.** Forward time is within 1–2% across the two, which
  argues the GPUs are comparable, but single-stage loading differences of a few seconds are not
  attributable.
