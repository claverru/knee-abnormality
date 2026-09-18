# Inference loading speedup

Investigation started 2026-09-17 with the user. Where the cold-start time goes, the open candidates
and their retrain cost, the closed regressions, the measurement protocol and the prior art live in
[loading speedup](../backlog/loading-speedup/INDEX.md).

**Shipped 2026-09-18 and carried by the current preprocessing identity:** `read_header` is the raw
explicit-VR parser with a pydicom fallback, `index_series` reuses the single `scandir` listing that
`build_series_index` already made, and both header prefixes and frame payloads are read on a thread
pool behind `KneeStudyDataset(io_threads=...)`. The default is 1, so training and validation stay
serial; only inference threads, at `INFERENCE_IO_THREADS = 16`. Prepared batches are identical to
the serial and pydicom paths in every audit, at 2, 4 and 16 threads. The main-thread warm-up is
implemented too and still awaits a submission.

**Shipped and measured 2026-09-18, retrain-free.** Three changes outside `PREPROCESSING_SOURCES`,
so they run against existing checkpoints: `build_series_index` filters the series CSV to the
caller's study IDs (`sources.py`, `engine.py`), its remaining listings run on
`SERIES_LISTING_THREADS = 16`, and `predict_worker` sets `loader.in_order = False`.

**470.5 s → 427.5 s of worker wall at 1,300 studies, 2.763 → 3.041 studies/s.** Indexing 54.3 →
7.1 s per worker, setup wall 54.4 → 15.7 s, and model load at 15.5 s is now the floor that hides
it. p90 batch wait 5.31–5.83 s → 4.25–4.29 s with the waiting spread across batches. The threaded
listing alone took `build_series_index` from 3.095 s to 0.96 s in the isolated loading cell
([profile](../inference/2026-09-18-kaggle-x4-profile-after-index-changes.md)). The index was
audited locally over the full train split, 4,407 studies: identical record for record on the counts
path, on the listing path Kaggle's test split uses, and under the study filter.

**The bottleneck is CPU in the loader workers, measured 2026-09-18.** Over the prediction window
the box is 98.2% busy at 0.7% iowait with the GPUs at 24%, and per-process sampling puts **91.3% of
1,360 CPU-seconds in the 16 loader workers**, 8.3% in the two GPU main processes and nothing
elsewhere. Production pays 0.955 CPU-seconds per study against 0.613 in the single-process probe.
So the retrain should buy CPU cuts in the preparation path; `pin_memory=False`, the row-trimmed
read and more io threads are all demoted, and wall now follows CPU at about 3.5 effective cores
([model](../backlog/loading-speedup/attribution.md#the-model-that-replaces-the-old-ranking)).
A repeat run also put run-to-run variance at ~2% of wall and ~15 s of loader wait, which is larger
than the wait saving `in_order = False` appeared to give.

**The profiler was repaired the same day** and the loading cell re-run with it: CPU per stage,
thread-safe accumulation, the pool-thread functions kept in a separate thread-seconds bucket, the
frame stage taken from `decode_series`'s own wall, and the `build_series_index` double count gone
([defects](../backlog/loading-speedup/attribution.md#the-profilers-defects-fixed-2026-09-18)). Two
structural questions fell out of the first honest table:

- **The frame stage is CPU-bound, not bandwidth-bound**, at 53–62% of a measured 510 MB/s ceiling
  with CPU above wall in both passes. Byte reduction no longer converts, which demotes the
  row-trimmed payload read.
- **The 31 s indexing bucket is mostly NFS wait** — 30.98 s of wall over 12.86 s of CPU cold — so a
  shared thread pool can attack about 10 s of CPU and none of the 18 s of wait.

Still to ship without a retrain: the [allocator env vars](../backlog/loading-speedup/candidates.md),
which need no code change at all, only two variables in the notebook's environment, and which
should be A/B'd on the old bundle so the three changes above do not confound them.

**Profiled on Kaggle at x4 and 1,300 studies** (random weights, timing only, since no checkpoint
carries the new identity): **cold loading is 2.47x faster**, 2.03 s to 0.821 s per study, while
decoding about 25% more frames. Header and indexing cost fell from 146.1 s to 36.7 s per 100
studies. End to end the two T4s took 470.5 s of worker wall, with the loader still 57–59% of
prediction time, so the balance moved without reversing
([profile](../inference/2026-09-18-kaggle-x4-profile-after-loading-changes.md)). Locally the same
changes are worth 1.22x on a cold first pass and nothing afterwards, which
[confirms that training does not benefit](../backlog/loading-speedup/attribution.md#only-inference-benefits).

**Reviewed 2026-09-18** across eight read-only research lanes over the profile, the code and
external sources. Two results change what to do next:

- **The profile's CPU and NFS columns are not CPU and NFS.** The profiler is wall-clock only, so the
  "CPU" column is the warm pass and "NFS" is cold minus warm, and they sum to cold by construction.
  A CPU-bound reading of that table is unsupported, `prepare_series` is genuinely *slower* warm than
  cold, and the 31.2 s indexing row contains no `scandir` at all — much of it is a per-series
  `ThreadPoolExecutor` built about 553 times per 100 studies. Fix the instrument before ranking
  further work ([attribution](../backlog/loading-speedup/attribution.md)).
- **Four of the five best candidates need no retrain**, because `data/sources.py` and
  `inference/engine.py` are outside `PREPROCESSING_SOURCES`. Three shipped and were measured the
  same day (above); the allocator environment variables are the one left. The bit-exact cluster,
  worth 17–34 s of the cold probe and roughly five times that at run scale, rides the retrain the
  shipped changes already require ([candidates](../backlog/loading-speedup/candidates.md), with the
  proposed experiment order).

Still open: `INFERENCE_IO_THREADS` needs a Kaggle sweep against 16 loader workers on four vCPUs,
after the shared-thread-pool change or it measures pool churn instead — and it is now more urgent,
because 64 threads beat 16 by 15% on the latest header probe. Model load, at 15.5 s, is the new
setup floor and nobody has profiled what inside it costs the time.
