# Where the cold-start loading time goes

The current attribution of DICOM loading cost, what the shipped changes bought, and the reasons the
profile's own numbers need qualifying before they are used to rank work.

Source: `outputs/profiling/kaggle-loading-x4-20260918c/runs/loading/profile/loading.json`, 100
studies, one process, sequential, four vCPUs, competition data on Kaggle's NFS mount, measured with
the repaired instrument; the end-to-end run is
`outputs/profiling/kaggle-inference-x4-20260918b/`. Both are recorded in the
[profile after the index changes](../../inference/2026-09-18-kaggle-x4-profile-after-index-changes.md),
which holds the stage table and supersedes the
[profile before them](../../inference/2026-09-18-kaggle-x4-profile-after-loading-changes.md).

## Where the cold 77.65 s sits

Cold loading is **0.776 s per study**, 77.65 s over 100 studies, against 61.25 s of process CPU.
Warm is 0.501 s per study. The four stages that matter, cold wall against cold CPU:

| Stage | Wall | CPU | Reading |
|---|---:|---:|---|
| Series indexing, less parsing | 30.98 s | 12.86 s | a 12–13 s CPU floor with the rest of the wall spent waiting |
| Frame reads (`decode_series` − resample) | 15.54 s | 17.38 s | threads hide the reads; serial decode paces it |
| Resample and crop | 8.02 s | 7.99 s | pure single-threaded CPU |
| Header parsing (`parse_prefix`) | 5.16 s | 5.16 s | pure CPU, 0.289 ms per header |

Then sampling 5.46 s, quantiles 4.77 s, normalize 2.36 s, other preparation 2.72 s, the listing
0.96 s, collate 0.95 s and 0.74 s of other setup and `__getitem__`.

**CPU and wall are now separate measurements**, not a cold/warm split. CPU is `process_time`, so it
counts the 16 io threads and legitimately exceeds wall on a threaded stage; read it against wall x
cores. **Wall minus CPU is therefore not a measurement of wait**: two threads can each burn 0.6 CPU
seconds in one wall second while both also block, so the subtraction bounds nothing. The stage is
latency-heavy because its wall collapses from 30.98 s to 7.96 s when the files are warm while its
CPU barely moves — that comparison, not the subtraction, is the evidence. The pool threads' own `read_payload` total — 96.4 thread-seconds cold — is tracked apart and
must never be added to a wall column, which is exactly the error the old instrument made.

## What the shipped changes bought

Four changes shipped 2026-09-18 and carry the current `preprocessing_identity`: the raw explicit-VR
header parser, the single reused series listing, and threaded header prefixes and frame payloads.
Against the [2026-09-15 baseline](../../inference/2026-09-15-kaggle-inference-profiling.md) of the
same probe, cold loading went from 2.03 s to 0.821 s per study — **2.47x, while decoding about 25%
more frames at x4** (149 against 119 per study). Header and indexing cost fell from 146.1 s to
36.7 s per 100 studies, of which only 5.47 s is now parsing: 17,855 headers at 0.306 ms against the
baseline's 7.6 ms.

End to end, two T4s took 470.5 s of worker wall for 1,300 studies at 2.76 studies/s, with the
loader still 57–59% of prediction time, so the balance moved without reversing. Per worker:
indexing 54.3 s, model load 12.9–13.4 s, prediction 403–405 s (loader wait 231–238 s, forward
163–173 s, transfer 1.3 s).

**A second retrain-free batch shipped 2026-09-18 and was measured the same day:** the
worker-filtered series index, the threaded listing and `loader.in_order = False`. Worker wall went
**470.5 s → 427.5 s at 1,300 studies, 2.763 → 3.041 studies/s**. Indexing fell 54.3 → 7.1 s per
worker and setup wall 54.4 → 15.7 s, 38.7 s against the ~41 s predicted, and **model load at
15.5 s is now the floor**, so indexing hides entirely behind it. `in_order = False` moved p90 batch
wait from 5.31–5.83 s to 4.25–4.29 s and spread the waiting — median wait 0.005 s → 0.32–0.61 s,
batches under 50 ms 119–121 of 163 → 65–73 — for 5.7–6.6 s of total wait, which is itself within
machine variance. In the isolated loading cell the threaded listing took `build_series_index` from
3.095 s to 0.96 s cold. The index was audited locally over the full train split, 4,407 studies:
identical record for record on the counts path, on the listing path Kaggle's test split uses, and
under the study filter.

**Main-thread warm-up** also shipped: the worker indexes in a background thread while the main
thread loads the model and runs one zero chunk, which turns the first FP16 batch from 26–34 s into
1.1–1.8 s and moved worker wall from 326 s to 318 s. First-batch wait in the x4 run is 15.0 and
24.3 s, absorbed by it.

The threaded frame reads rest on a byte-matched Kaggle CPU probe: **2.40x at 16 threads for 14%
more CPU**, zero failures and zero audit mismatches. Locally the same change is flat, because NVMe
has no latency to hide — see [how to measure](measuring.md#local-results-do-not-transfer-to-nfs).

## The box is out of CPU, not waiting on NFS

Sampled once a second by the profiler and never read until 2026-09-18, `cpu.csv` answers the
question the stage tables cannot:

| Run | CPU busy | iowait | cgroup cores in use | GPU utilisation |
|---|---:|---:|---:|---:|
| Before the index changes | 88.3% | 2.5% | 3.12 of 4 | 21.4% |
| After | 95.4% | 1.5% | 3.38 of 4 | 24.4% |
| After, **prediction window only** | **99.5%** | **0.3%** | **3.53 of 4** | 25.5% |

The first two rows start at process spawn, so they dilute prediction with a 24 s setup window that
runs at 55.7% busy on 2.15 cores. The analyzer now reports the two windows separately; the third
row is the one to quote.

**The two CPU columns have different scopes and must not be reconciled.** "CPU busy" and "iowait"
come from `/proc/stat`, which counts the whole machine including anything outside our container;
"cores in use" is our cgroup's own `cpu.stat` usage, 3.38 of 4 being 84.5%. Both say the CPU is the
constraint; neither is a precise statement of headroom left.

**1.5% iowait.** The loader is not blocked on the NFS mount; the four vCPUs are full while two T4s
idle at a quarter utilisation. The index changes made it worse in the only sense that matters here:
removing 39 s of setup wall raised CPU occupancy from 88% to 95%.

The per-study arithmetic says the same thing. The single-process probe prepares a study in 0.776 s
of wall and 0.613 s of CPU. In production each GPU process has 8 loader workers delivering 650
studies across a ~400 s prediction window, so a study costs about **4.9 loader-worker-seconds,
6.3x the probe**. Nothing in the path got slower: 16 worker processes, each able to spawn 16 io
threads per series, do not fit on four cores.

### Measured again after the CPU cluster

The [2026-09-18 cluster profile](../../inference/2026-09-18-kaggle-x4-profile-after-cpu-cluster.md)
repeated the role measurement with trained weights: **loader-worker CPU 1,241.6 → 990.6 seconds,
−20.2%**, worker wall 418.1 → 350.8 s, loader wait 197–213 → 135–138 s with forward unchanged. The
box is still CPU-bound — 99.1% busy, 0.75% iowait, 3.47 of 4 cores — and the GPUs rose from 25.5%
to 30.6%. The 3.5-cores model predicted ~75 s off the wall and the run gave 67 s, so the heuristic
holds for a CPU saving of this size.

### The loader workers own 91% of it

Measured, not estimated: the 2026-09-18d run sampled `/proc/<pid>` once a second and attributed CPU
per role over the 395.5 s prediction window (`processes.csv`, `process_usage`). An earlier estimate
in this document put ~650 of ~1,450 CPU-seconds outside the loading path. **That was wrong by a
factor of six.**

| Role | Processes | CPU seconds | Share | Peak RSS |
|---|---:|---:|---:|---:|
| Loader workers | 16 | **1,241.6** | **91.3%** | 1,738 MiB |
| GPU main processes | 2 | 113.0 | 8.3% | 2,754 MiB |
| Other | 5 | 4.6 | 0.3% | 164 MiB |
| Coordinator | 1 | 1.1 | 0.1% | 850 MiB |

The box was 98.2% busy at 0.7% iowait, 3.44 of 4 cores in the cgroup. So pinned memcpy, tensor
deserialisation and kernel launches — the whole main-process block — cost 113 CPU-seconds together,
and `pin_memory=False` and cutting loader-to-main bytes cannot be worth much. **Preparation CPU
inside the loader workers is the bottleneck, and nothing else is close.**

**Production costs 0.955 CPU-seconds per study against the probe's 0.613**, a ratio of 1.56. The
ratio is observed, not decomposed: contention between 16 worker processes and their io threads is
the leading explanation, but different machines, batching, worker-side collation, instrumentation
and allocator state all sit inside it. Treat it as the gap to explain, not as a constant to
multiply by.

### The model that replaces the old ranking

Wall follows CPU while CPU binds: 1,360 CPU-seconds over 4 cores would be 340 s, and prediction took
388 s, so the box converts CPU to wall at about **3.5 effective cores**. Removing `C` CPU-seconds
from the run should remove roughly `C / 3.5` seconds of wall, until the forward pass — 173–191 s —
becomes the floor. This is a heuristic for ranking, and it applies to **CPU** savings: a wall second
saved on overlapped I/O is a different quantity and does not convert the same way. A second saved on the 100-study cold probe is 13 probe-units per run, scaled by the 1.56x
contention factor and divided by 3.5, so **one cold-100 second is worth about five seconds of run
wall**, which is the translation the candidate list uses.

What follows: a change that removes CPU per study converts; a change that removes bytes or hides
latency does not. That promotes the whole bit-exact cluster, and demotes the row-trimmed read,
`pin_memory=False` and any increase in io threads.

## What each stage really is

Re-derived from the code and the artifact during the 2026-09-18 review; several stage names are
misleading.

- **The 31 s row contains no `scandir` and no listing.** The profiler defines it as
  `index_series − parse_prefix`, and since the single-listing change `index_series` skips
  `list_series` whenever candidates are supplied. The listing is the separate 0.96 s row. What the
  bucket actually holds is the threaded 12 KiB prefix reads, record assembly, `modal_layout`, the
  sort, `summarize_series`, `summarize_tags` — and per-series thread-pool construction.
- **Most of that row is latency, and its CPU floor is what a shared pool can attack.** Measured
  cold it is 30.98 s of wall for 12.86 s of CPU; warm the same work is 7.96 s of wall for 11.98 s
  of CPU. The CPU floor is 12–13 s either way and the cold wall is 23 s longer, which is latency
  the threads did not hide — though with 16 threads running, wall minus CPU is not itself a
  measurement of blocked time. Within that CPU, `index_series`
  builds a fresh `ThreadPoolExecutor(min(16, n_files))` **per series**: 6.60 ms per series at 16
  threads against 0.42 ms with one shared pool (local, 560 reps), ~553 pools per 100 studies, while
  the intrinsic non-parse CPU is roughly 3.1 s Kaggle-equivalent. That bounds the shared pool at
  about 10 s of the 12.9 s CPU and **at none of the 18 s of wait**, which needs concurrency
  instead.
- **Setup and collate are small.** Cold `setup_seconds` is 1.325 s of which `build_series_index`
  is 0.96 s, and collate is 0.95 s. Collate is 9.5 ms per study and transfer runs at 11 GB/s, so
  pinned DMA already works. The summary now publishes `setup_other` beside the listing instead of
  counting the listing twice.
- **`build_series_index` was the largest run-scale cost and is now the smallest.** It used to be
  called with no study filter, so each of the two GPU processes listed all ~7,050 test series while
  predicting half of them: 54.3 s of `data_setup` per worker, 11.5% of the 470.5 s wall. With the
  worker filter and the threaded listing it is 7.1 s, fully hidden behind 15.5 s of model load.
  **Nothing further in indexing pays until model load is cut**, which is now the setup floor and
  has no candidate against it.
- **Sampling runs once per study per slot**, not per triplet and not per evaluation pass:
  `select_triplets` 4.554 s over 600 calls (7.59 ms), `build_pool` 1.142 s over 600 (1.90 ms),
  `estimate_reference` 0.058 s over 100. Within it `geometry_groups` is 47% and `np.cross`
  dominates `build_pool`; the scipy calls the earlier plan blamed are 5–89 µs and are not the cost.
- **x4 does not re-prepare anything.** It raises the per-slot budget (24/24/20/20/20/20) rather
  than re-running sampling or preparation, so there is nothing to cache across passes, and a
  cross-item intensity cache would be *incorrect* rather than merely useless, since the clipping
  bounds follow the selection.

## `prepare_series` slower warm than cold: machine-dependent, not code

The 2026-09-18 profile had `prepare_series` warm (40.46 s) exceeding cold (34.97 s), and the
2026-09-18b run reproduced it (41.2 against 36.6). **It did not reproduce on the 2026-09-18c
machine**: 33.4 s cold against 29.8 s warm, the expected direction, with the same code and the same
100 studies. So it is a property of a machine or its mount, not of the loading path, and the two
mechanisms once proposed for it — threaded reads regressing on warm files
([closed](closed.md#threading-loses-when-there-is-no-latency-to-hide)) and a working set too large
for page cache, 8.4–9.8 GB per 100 studies — remain plausible for those two runs but explain
nothing that survives re-measurement.

The warm pass is worth keeping only as a second sample: with CPU now measured directly, the cold
pass alone separates work from wait.

## The profiler's defects, fixed 2026-09-18

All of them are repaired in `scripts/adhoc/profile_kaggle_inference.py`; artifacts written before
the fix keep the old shape, and `loading_summary` no longer reads them.

- **There was no CPU measurement at all**, so the CPU/NFS split above is a cold/warm split (see
  above). The cell now records `process_time` per stage, plus each pass's wall, process CPU and
  `getrusage` minor and major faults. **CPU can exceed wall** on a threaded stage, because process
  CPU counts the pool threads: read it as CPU-seconds against wall x cores, which is why the pass
  reports `cpu_seconds_per_wall_second` rather than a fraction.
- **`timed` was not thread-safe.** `timings[name] += …` is a non-atomic read-modify-write, and
  `read_payload` runs on 16 pool threads via `_prefetch_payload`. It is now under a lock, and the
  four functions that run on pool threads — `read_payload`, `frame_from_payload`, `read_frame`,
  `_read_pydicom_frame` — accumulate into a separate `thread_seconds` bucket that is **summed across
  threads and is not wall**. The 2026-09-18b loading run shows why: 83.2 thread-seconds of
  `read_payload` inside a 36.6 s `prepare_series`.
- **The frame stage now comes from `decode_series`'s own wall**, minus `resample_crop`, which
  `decode_slice` calls inside it. `prepare_other` is `prepare_series` minus `decode_series`, the
  quantiles and the normalize, so both are non-negative by construction.
- **`build_series_index` was counted twice**, as its own row and inside `setup_seconds`. The
  summary now publishes `setup_other = setup_seconds − build_series_index` beside it.
- **`project_reference` does not exist.** The summary's sampling formula still added
  `timing["project_reference"]`, but the function was removed from the package in `f076a09`, so the
  term was always zero and nothing was missing from the `timed()` list. The real gap was that the
  published table **omitted the `getitem_other` row**, 0.351 s cold, which together with the double
  count explains the 2.9 s by which the rows overshot the 82.1 s total. Every row is now published
  and they sum to the pass total by construction.
- Sub-stages **do** sum to their parents; the apparent mismatch is decimal rounding.

## Structural questions still open

- **The frame stage is CPU-bound, not bandwidth-bound — settled 2026-09-18.** Its CPU exceeds its
  wall in both passes (17.38 against 15.54 s cold, 16.83 against 12.15 s warm), so the threads hide
  the reads and serial decode paces it. The byte ceiling agrees: the 16-thread full-file probe
  reached **510.0 MB/s** on the same machine against the 271–318 MB/s the production run sustains,
  53–62% of ceiling, where the earlier 376.9 MB/s figure had put it at 72–84%. **Byte reduction no
  longer converts 1:1**, which demotes the row-trimmed payload read; CPU reduction is the lever.
- **`INFERENCE_IO_THREADS` is settled at 8, weakly.** The production sweep ran 1/2/4/8/16 threads at
  16 workers after the shared pool landed: 1 thread is a clear loss at 457.9 s and 79.4% busy, and
  2 through 16 sit within 2.6% of each other. io8 won both of its two pairs, so the constant moved
  16 → 8, but the effect is 2–3% against a per-arm spread that reached 36 s
  ([sweep](../../inference/2026-09-18-kaggle-x4-profile-after-cpu-cluster.md#the-io-thread-sweep-2026-09-18)).
  The single-process header probe that favoured 64 threads does not describe production.
- One constant covers header prefixes and frame payloads because nothing yet says they want
  different values; split it only if a sweep shows they do.

## Only inference benefits

Headers are parsed once per worker per series and cached for the run (`_slice_index`,
`persistent_workers: true`), local disk has no NFS round trip to hide, and loader wait was under
0.06% of GPU step time
([threading profile](../../data-preparation/threading-inside-dataset-loading.md#results-and-decision)).

Re-measured on the current recipe, 2026-09-18 (`compare-header-implementations`, 256 studies, batch
8, 16 workers, two passes, page cache dropped immediately before each arm;
`outputs/profiling/header-cold-pydicom-20260918/`, `header-cold-parser-20260918/`):

| Arm | Pass 0, cold | Pass 1, warm |
|---|---:|---:|
| pydicom | 37.5 studies/s, 6.83 s | 99.0 studies/s, 2.58 s |
| raw parser | 45.9 studies/s, 5.58 s | 109.3 studies/s, 2.34 s |
| raw parser, 4 header threads | 45.5 studies/s, 5.63 s | 104.5 studies/s, 2.45 s |

**1.22x on the first pass and nothing afterwards.** Against the pass-1 steady state the epoch-zero
penalty is 4.25 s for pydicom and 3.24 s for the parser, so parsing is about a quarter of it and
cold disk plus index construction is the rest. Median steady-state batch time was 7.6–8.0 ms for a
batch of eight in both arms, which is queue drain, not work.

The choice is not separable anyway: one `KneeStudyDataset` path serves training, validation and
inference, so whatever ships for inference is what training runs.
