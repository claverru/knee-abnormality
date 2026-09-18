# How to measure a loading change

The protocols and the traps. Every one of these rules exists because a measurement taken the other
way produced a wrong answer that survived until someone re-measured.

## Probe on Kaggle, one kernel per change

One self-contained Kaggle CPU script kernel per change, reproducing only the operation under test
rather than the whole pipeline, each auditing its candidate against a copy of the production
function (`probe-kaggle-loading` in [the catalog](../../../scripts/adhoc/catalog.yaml)). Five CPU
sessions run at once.

- Discover the competition root by globbing for `train_series.csv`, as the existing profiler does.
  The slug path does not exist, which is what made the first run's four data-reading probes error.
- Give cells disjoint study blocks so each block is read cold once per session. That is right on
  Kaggle and wrong locally, where it measures disk I/O instead of the code under test.
- Balance cells on the quantity under test. The first frame probe read 900 frames per cell but
  352.7–1,109.7 MB, so neither frames/s nor MB/s isolated the thread count; `matched_blocks` now
  deals whole studies into cells balanced per frame byte-length and truncates to the shared count.
- Pure-CPU probes (resample, quantiles, sampling) need no NFS and can be answered locally.

## Measure intensity work on real pixels

Never on a generator. A coarse-histogram quantile selection measured **1.87x faster on synthetic
gamma noise and 1.67x slower on real pixels**, and its bounds were exact, so correctness testing hid
the regression completely. Real MRI slices carry a large near-zero background spike that degenerates
bin-based rank selection. This applies to anything touching intensities or percentiles.

Resample and normalize are the exception: they are data-independent branch-free arithmetic, so
synthetic pixels at production shapes are legitimate for *timing* them — but say so, and make no
intensity claim from such a run.

## Local results do not transfer to NFS

Local NVMe has no latency to hide, so an I/O change that wins on Kaggle can be flat or negative
locally, and the reverse. Threading is the standing example, measured in both directions
([closed](closed.md#threading-loses-when-there-is-no-latency-to-hide)). A flat local result for an
I/O change says nothing against NFS, where a large share of the cold cost is round-trip latency.

Corollary for CPU work: local micro-benchmarks *do* transfer, scaled. The observed Kaggle-to-local
CPU factor is **4.4–5.6x** (header parse: 66.7 µs local predicts 0.31 ms against 0.306 ms measured;
pydicom 0.32 → 1.8 ms). Use it to convert a local CPU saving into a Kaggle estimate, and label the
result an estimate.

## Pre-screen thread changes locally

A local harness that inserts a 10 ms sleep per read reproduces Kaggle's thread-scaling curve
closely — 98/371/1,151 files/s at 1/4/16 threads against Kaggle's 111.8/370.3/932.6 — so thread-count
decisions can be screened locally before spending a Kaggle session.

## Prefer the cold protocol

Drop the page cache immediately before each arm and compare pass 0. A run pre-warmed into RAM
reported 56.4 against 88.7 studies/s for the same pair that the cold protocol put at 37.5 against
45.9 — a 1.57x that exists only because the I/O term had been removed. Use the warm protocol only to
isolate a CPU cost from I/O, never to decide where the bottleneck is.

## Measure at the target worker count

Every thread number in this repository came from one process on an otherwise idle machine, while
production runs 16 loader workers on four vCPUs. Per-worker throughput degrades with worker count
through thread-pool contention and CPU oversubscription, and a CPU-time metric can hide what wall
clock reveals. This is the open question behind
[`INFERENCE_IO_THREADS`](attribution.md#structural-questions-still-open), and it is why the
shared-pool change must land before the thread sweep.

## Audit before shipping

The loading path is inside `preprocessing_identity`, so a change ships only with a retrain, and the
audit is what establishes that the images did not change. Validate locally on CPU: every train
series must produce identical `IndexedSeries` records, identical selections and identical prepared
tensors, with zero pydicom fallbacks and zero prefix extensions. Compare every collated field, not
just pixels.

Audit scope has been the weak point. Existing audits cover 20,000 train files for the header parser,
3,783 DICOMs for the pixel path and 3,200 synthetic cases for the sampler; none covers the full train
split, and example-test files are usually missing. Widen the scope for anything that changes
selection, since one seed change moves 44% of repeated-slot triplets
([tie-break investigation](../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md)).
