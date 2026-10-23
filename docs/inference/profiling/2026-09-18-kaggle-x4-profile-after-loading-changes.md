# Kaggle x4 inference profile after the loading changes

Timing profile of the sources of 2026-09-18 (loading speedup changes 2, 3, 4 and 5) on Kaggle, at the
x4 image multiplier and the hidden test's study count. It answers what the loading changes bought
end to end and where the remaining time sits. Compare with the
[2026-09-15 baseline](2026-09-15-kaggle-inference-profiling.md), which profiled the same stages
before any of them.

**Weights are random.** No checkpoint carries the current `preprocessing_identity`, so the bundle
was exported from `outputs/submission/timing-profile-random-weights.ckpt`, built by instantiating
`KneeModule` with `load_pretrained=False`. Every timing here is real; every prediction is
meaningless, and no AUC or parity number was computed. Re-profile after the retrain to score.

Artifacts in `outputs/profiling/kaggle-inference-x4-20260918/`; notebooks
`claverru/knee-profile-fp16-b4-w16-x4-20260918` and `knee-profile-loading-x4-20260918`.

## End to end, two T4s, 1,300 studies

`fp16`, batch 4 per device, 16 loader workers total, x4 images. 1,300 studies: all 904 of fold 0
plus 396 from the other folds, which is why nothing here is scored.

| | Worker 0 | Worker 1 |
|---|---:|---:|
| Studies | 650 | 650 |
| Indexing (`data_setup`) | 54.3 s | 54.3 s |
| Model load | 12.9 s | 13.4 s |
| Prediction | 403.1 s | 405.2 s |
| — loader wait | 238.5 s (59%) | 230.9 s (57%) |
| — forward | 163.3 s | 173.0 s |
| — transfer | 1.3 s | 1.3 s |
| Peak allocated | 1.58 GiB | 1.58 GiB |

**470.5 s of worker wall, 2.76 studies/s.** The loader is still the bottleneck at 57–59% of
prediction time, so the changes moved the balance without reversing it. First-batch wait is 15.0
and 24.3 s, absorbed by the main-thread warm-up.

## Cold loading stages, 100 studies

One CPU notebook, four vCPUs, sequential in a single process, first pass cold. The baseline column
is the 2026-09-15 run of the same probe.

| Stage | Baseline | Now |
|---|---:|---:|
| Slice headers and series indexing | 146.1 s | 36.7 s |
| Series listing index | 3.1 s | 3.1 s |
| Geometry and triplet sampling | 7.0 s | 5.8 s |
| Resample and crop | 8.2 s | 8.3 s |
| Quantiles and normalization | 7.6 s | 7.3 s |
| Frame reads and other preparation | 31.1 s | 19.4 s |
| **Per study** | **2.03 s** | **0.821 s** |

**2.47x faster cold loading, while doing about 25% more frame work** — x4 decodes 149 frames per
study against the baseline's 119, so the per-unit gain is larger than the ratio shows.

Header cost fell from 146.1 s to 36.7 s, of which only 5.47 s is parsing (17,855 headers at
0.306 ms each, against the baseline's 7.6 ms). The remainder is the threaded prefix reads, record
assembly, the sort and the series summaries — **not** the listing, which is the separate 3.1 s row,
and a large part of it is per-series thread-pool construction rather than work
([attribution](loading-cost-attribution.md#what-each-stage-really-is)). Resample,
quantiles and normalization are unchanged, as expected: nothing touched them.

Header read throughput on this machine, the probe that sets `INFERENCE_IO_THREADS`:

| Threads | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| files/s | 111.8 | 370.3 | 932.6 | ~1,000 |

16 threads is 8.3x serial and within noise of 64, which supports the shipped value.

## What this run cannot tell us

- **Frame reads are not separated from other preparation.** `decode_slice` calls
  `frame_from_payload` when change 5 prefetched the payload, so the instrumented `read_frame`
  never fires and its time falls into the unattributed preparation bucket. The profiler now also
  times `read_payload` and `frame_from_payload`; this run predates that fix.
- **The loading condition is single-process and sequential**, so it says nothing about 16 loader
  workers contending for four vCPUs, which is still the open question behind
  `INFERENCE_IO_THREADS`.
- **Two accounting defects, found in the 2026-09-18 review.** `build_series_index` is counted both
  as its own row and inside `setup_seconds`, and the published rows omit the 0.351 s
  `getitem_other` bucket; together they explain why the rows overshoot the per-study total. There
  is also no CPU measurement anywhere in the loading cell, so any cold-minus-warm difference is
  wall clock and not an I/O-versus-CPU split. All of this was
  [fixed after this run](loading-cost-attribution.md#the-profilers-defects-fixed-2026-09-18).
- **A different machine and NFS server than the baseline.** The baseline run itself recorded
  24–64 s of variance for identical indexing work, so treat single-stage differences below about
  that spread as noise. The per-study total is far outside it.
