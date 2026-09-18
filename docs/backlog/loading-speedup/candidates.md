# Open loading-speedup candidates

What is left to do, grouped by what it costs to ship, then the order to try it in. Closed and
rejected formulations are in [closed](closed.md); do not re-propose them. Every figure in an
"Est. saved" column is an estimate, not a measurement of the change in production: read
[how to measure](measuring.md) before trusting one, and [attribution](attribution.md) for why the
profile's own columns need qualifying first.

Two units appear and must not be added together:

- **cold-100** — seconds off the single-process cold probe over 100 studies, now 77.65 s.
- **run-scale** — seconds off the two-worker wall at 1,300 studies, now 427.5 s.

**The four vCPUs are the binding constraint**: 95.4% busy at 1.5% iowait with the GPUs at 24%
([measured](attribution.md#the-box-is-out-of-cpu-not-waiting-on-nfs)). A change that removes CPU per
study converts to wall; one that removes bytes or hides latency does not.

A cold-100 second translates to roughly **five run-scale seconds**: 13 probe-units per 1,300-study
run, scaled by the 1.56x CPU contention production pays over the single-process probe, divided by
the 3.5 effective cores the box converts CPU to wall at
([the model](attribution.md#the-model-that-replaces-the-old-ranking)). **The whole list is capped at
about 200 s**, because prediction cannot fall below the 173–191 s forward pass.

## What a candidate costs to ship

`preprocessing_identity` (`src/knee_abnormality/identity.py`) folds the raw bytes of the thirteen files in
`PREPROCESSING_SOURCES` into `sources_sha256`, and `validate_preprocessing_identity` refuses any checkpoint whose
recorded identity differs in any field. The hash cannot tell a provably equivalent parser swap from
a silent change to the pixels, so it treats any byte change as a change; a reformatted comment
would do the same. That gives three classes:

- **No retrain.** The file is outside `PREPROCESSING_SOURCES`. That includes `data/sources.py`,
  `inference/engine.py`, `scripts/pipeline/predict_worker.py`, the profiler and the notebook
  environment. These ship on their own, against existing checkpoints. The known bad series and
  instances used to live in `data/sources.py`, where editing them changed the model's inputs without
  bumping the identity; they moved to `data/preprocessing/exclusions.py` on 2026-09-18 so the hash
  covers them, and they must not move back.
- **Bit-exact, identity-changing.** The file is in `PREPROCESSING_SOURCES` — `data/dataset.py` and
  `data/preprocessing/{dicom,filtering,geometry,intensity,preparation,sampling,series,spatial}.py`.
  The images are unchanged and audits prove it, but the identity bumps, so existing checkpoints keep
  running only against their archived sources and the saving arrives with the next trained
  checkpoint. **Batch all of these into one retrain rather than paying several.**
- **Numerics.** The images change. Needs the retrain *and* a fresh validation run, because a
  single-run difference below about 0.003 macro AUC carries no information and anything above it
  has to be attributed.

## Ship without a retrain

| Change | Est. saved | Conf | Effort | Risk and notes |
|---|---:|---|---|---|
| **glibc allocator env vars**: `MALLOC_MMAP_THRESHOLD_` and `MALLOC_TRIM_THRESHOLD_` at 256M in the notebook and `predict.py`, or an LD_PRELOAD tcmalloc | 3–8 s cold-100 | med | S | Payloads (~0.66 MB) and float32 frames (1.3 MB) all exceed glibc's 128 KiB default, so each goes through `mmap` and the kernel zeroes it. Local A/B cut the production-shaped decode path 294.6 → 180.3, 421.6 → 213.5 and 277.0 → 236.4 µs/frame, 3 of 3 pairs faster. **Watch per-worker RSS**: 16 workers of retained heap on a four-vCPU box. Testable today on the random-weights bundle |
| Order each worker's studies by total image count, for homogeneous batches and less `max_slices` padding | 1–3 s | low-med | S | Numerically inert under eval BN and per-study attention. Also helps `in_order=False` |
| `pin_memory=False` at the `predict_worker` call site | ~0 | low | S | **Demoted 2026-09-18**: the two main processes burn 113 CPU-seconds *in total*, 8.3% of the run, so the 14.8 GB of pinned memcpy it would free is not what the box is short of ([roles](attribution.md#the-loader-workers-own-91-of-it)) |

**Indexing is spoken for: model load is the floor now.** `predict_worker` builds the index in a
background thread while the main thread loads the model and runs the warm-up chunk, so setup wall
is `max(indexing, model load)`. Measured after the
[2026-09-18 index changes](attribution.md#what-the-shipped-changes-bought): `max(7.1, 15.5) = 15.7 s`
per worker, down from 54.4 s. **Every further indexing change is worth zero** until model load is
cut, which is why the `list_series` micro-optimisation moved to
[closed](closed.md#ruled-out-by-arithmetic-or-by-the-environment) and nothing else in indexing is
listed here.

## Bit-exact, batch into the retrain

**Twelve of these shipped in `041bd16`** — the shared io pool, the struct scan, the `SeriesTags`
memo, the unsigned mask, `os.pread` for prefixes and payloads, both manual cross products, the
`geometry_groups` prune, the `select_triplets` refactor, the resample column band, the dropped
second clip and `astype(copy=False)`. All three deliverables are now in: code merged; audited bit-exact against the pre-change sources over **800 studies** — 4,422 series records,
13,266 header hashes and 800 collated batches, zero mismatches, at 16 threads, plus an earlier
25-study pass at 1 and 16 threads;
and measured in production at **418.1 → 350.8 s of worker wall with loader CPU 1,241.6 → 990.6 s**
([profile](../../inference/2026-09-18-kaggle-x4-profile-after-cpu-cluster.md)). The audit covers 18%
of the train split, not all of it.

What is left, with the audit scope given per row. Several scopes are narrower than a retrain needs —
train files only, synthetic cases, or a single shape — so widen them first
([audit before shipping](measuring.md#audit-before-shipping)).

| Change | Est. saved (cold-100) | Conf | Effort | Audit and notes |
|---|---:|---|---|---|
| **Row-trimmed payload read** — read only the rows in the vertical filter's support, one contiguous byte range computable from the descriptor (`dicom.py`, `spatial.py`, `preparation.py`) | 1.2–2.0 s; the byte saving no longer converts | med | M | **Demoted 2026-09-18**: the frame stage is [CPU-bound, not bandwidth-bound](attribution.md#structural-questions-still-open), running at 53–62% of a measured 510 MB/s ceiling, so removing 13.4% of bytes buys latency and page cache, not wall. Out-of-band rows cannot affect the output because `resample_crop` zero-fills outside the rect; bit-identical on 120/120 frames. `_axis_filter` must keep the original `source` for boundary clipping and `output_to_native` the original shape; the pydicom fallback still reads whole frames |
| Vectorise `summarize_series` (16.1 µs/file serial, 15% of serial `index_series`) | 1–1.5 s | low-med | M | Float-order risk; audit poses and bounds |
| `PREFIX_BYTES` 12288 → 2048–4096 | 0–2 s, lands in the indexing row | low | XS | The pre-payload header is p50 1,482, p99 1,738, **max 1,758 bytes over 20,000 files**, and a 2,048-byte prefix truncated 0 of 1,200. The win is 219 MiB → 37 MiB over NFS and 6x less prefix RAM. Only extend-and-retry protects a longer hidden-test header; confirm before shrinking |
| `truncnorm.ppf` → `truncnorm._ppf` | 0.15 s | med | S | Bit-identical on 2,000 random (q, a, b); scipy distributions spend ~90% of a call in parameter checking. Private API — pin the scipy version in the offline bundle |
| Bounded prefetch window in `decode_series`, 16–32 slices instead of the whole series | 0–2 s, −80% transient RAM per worker | low | S | Today all of a series' payloads are materialised before any is decoded: ~83 MB transient per worker at x4, ~1.3 GB across 16. Output bit-exact |
| Skip the quantile `np.concatenate` when every rect is full-frame | 0.1–0.3 s | low | S | Saves Python overhead only; the bytes copied are identical |
| **One file access per selected frame.** Indexing already records the payload offset, yet `read_frame` opens the file again and `fstat`s it to revalidate size and mtime | unquantified | low | M | The `fstat` is 1.6 µs/frame locally, 24 ms per 14,882 frames, so only the NFS round trip could justify it, and the attribute cache may already absorb it. Minimising reopens is the sanctioned small-file lever on network filesystems. A payload read must still fail closed when the file changed. No probe exists |

**The old 17–34 s projection is spent, not a remaining budget.** It covered the twelve rows that
shipped, and what they actually bought is the 20.5–21.9% local CPU figure above, pending the Kaggle
run. Re-measure the residual before choosing the next large change.

Two candidates were added by the 2026-09-18 review of `041bd16`, both inside the hashed sources and
therefore both waiting for the next identity bump:

| Change | Est. saved | Conf | Effort | Notes |
|---|---:|---|---|---|
| **Re-measure the resample column band** on Kaggle, and drop it if it does not pay (`spatial.py`) | it may be costing ~0.1x of the stage | med | S | Machine-adjusted, the band came in at 1.03x where 0.92x was expected, so it looks inert or slightly negative on Kaggle's vCPUs against 8.7% faster locally ([profile](../../inference/2026-09-18-kaggle-x4-profile-after-cpu-cluster.md)). Pair the re-measurement with the resampling-plan cache below, since both touch the same derived objects |
| **Cache the derived resampling plan** — the band bounds and the trimmed horizontal CSR, keyed by the same geometry as `_axis_filter` (`spatial.py`) | unmeasured | med | S | `_axis_filter` is `lru_cache`d, but every `resample_crop` still recomputes `horizontal.indices.min()/max()` and rebuilds `horizontal[:, band]`, which depend only on geometry. Never mutate a cached sparse matrix; time support discovery and CSR slicing separately on the real shape distribution first |
| **Bounded ordered payload window** in `decode_series` — submit a window of reads, decode in record order, release consumed bytes (`preparation.py`) | unmeasured; −80% transient RAM | low-med | M | Supersedes the prefetch-window row above with a concrete objection: `pool.map` submits eagerly and the tuple retains every payload through the decode loop, so peak live memory holds raw payloads and resampled outputs at once. `buffersize` is not available on the bundled Python, so the window must be explicit. May trade memory for waiting — measure throughput, not just RSS |

## Needs a validation run as well

| Change | Est. saved | Conf | Effort | Risk |
|---|---:|---|---|---|
| **Strided subsample of the quantile buffer** — build it as `region.reshape(-1)[::4]` and quantile the subsample | 3.4–3.8 s cold-100 plus 0.5–1 s of the concatenate | med | S | Sampling standard error is about 1.6e-4 in probability at n ≈ 188k, some 30x smaller than the 0.005 level shift that cost 0.0021 — implying roughly 1e-4 AUC, but **that is an extrapolation from sampling theory, not a measurement**. It survives the background-spike objection because subsampling is value-blind and the spike stabilises the low rank; the exposure is the sparse bright 0.995 tail. Prefer a stride coprime with 224. **Gate it on the label-free check first**: compare full against strided bounds and the resulting uint8 histograms on real acquisitions, before spending a training run |
| **GPU resample, quantiles and normalize** — decode raw frames on CPU, then crop-resample, clip and normalize on the two T4s | up to 15.7 s cold-100 | low | L | Must take quantiles too, since the bounds come from resampled regions. GPU *compute* is not the constraint: an AA-bilinear T4 kernel extrapolates to roughly 6 s for all 193,700 x4 frames against ~240 s of idle GPU per worker. The real obstacles are ~10x the loader-to-main-process bytes (76 MB against 7.5 MB per x4 study) and the fact that loader workers cannot own a CUDA context, so preprocessing moves into the main process beside the forward pass. Batch per series to avoid ragged launches. The only CPU-side lever left once the bit-exact cluster lands |

## Reading less per study (trades score for wall)

Every candidate above preserves the predictions. This one does not, and it is the largest lever on
the list, because the loader cost is per frame: the evaluation multiplier sets how many triplets
each study contributes, so cutting it removes loader CPU, loader bytes and forward-pass work in the
same proportion.

**The exchange rate the efficiency formula sets.** `Efficiency = AUC / (Benchmark - maxAUC) +
RuntimeSeconds / 32400`, minimized
([requirements](../../inference/submission-requirements-and-output-contract.md)). With the sample
submission at 0.5 and today's best at 0.957, the first term is `-AUC / 0.457`, so **0.001 of AUC is
worth about 70 seconds of runtime**; a higher benchmark or a lower top score widens that to roughly
90 s. Both quantities are private-leaderboard values nobody knows until the end, so treat the rate
as an order of magnitude, not a constant.

| Change | Measured score cost | What it removes |
|---|---|---|
| Test multiplier ×4 → ×3 | fold 0 tied (0.8954 both at τ 0.75), public 0.939 → 0.938, one rounding step inside the 0.003 noise floor | 24% of the triplets, so roughly a quarter of both loader and forward-pass work: order 80 s of the 350.8 s of worker wall measured after the CPU cluster, since only per-study work scales and setup (about 16 s per worker) does not |
| Test multiplier ×3 → ×2 | fold 0 −0.0024 at τ 0.75 | another 33% of the triplets |
| Fewer slices per type in training and inference | −0.0041 at 15/15/12/12/12/12 and −0.0051 at 12/12/9/9/9/9 on fold 0, 20% and 37% faster training ([runs](../../validation/results/2026-09-16-restored-code-hyperparameter-runs.md)) | needs a retrain, so it is an endgame decision, not a submission-time one |

**It is close to break-even at ×3, which is why it needs measuring rather than assuming**: about
80 s saved against a cost of at most 0.001 AUC, itself worth about 70 s. The public evidence sits
at the same margin — one competitor's trimmed arm scored 0.930 at 42 windows against 0.937–0.938 at
94, about 0.01 for most of the runtime
([sweep](../../research/public-research.md#slice-density-and-the-board-2026-09-18-re-sweep)). The
accuracy side of the same knob is in [endgame](../endgame.md#test-time-augmentation); what is
missing for a decision is the wall at fixed batch for ×3 against ×4 on Kaggle, which the same
profile run can produce.

## Runtime changes outside the loading path

These touch neither the loading path nor the identity. They are listed here because they came out of
the same investigation and compete for the same submission slot.

| Change | Est. saved | Conf | Effort | Risk and notes |
|---|---:|---|---|---|
| **Torch-free coordinator.** `predict.py` imports the model stack only to count GPUs and read study IDs. Count GPUs with `nvidia-smi -L` or `CUDA_VISIBLE_DEVICES`; the template and series CSVs are already readable without PyTorch through `knee_abnormality.data.sources` (`prediction_studies`, `resolve_raw_dir`), which the 2026-09-16 restructure moved out of the preprocessing identity | ~30 s of the 35–55 s of coordinator imports before `main` | high | S | At the efficiency formula's 32,400 s scale, 30 s is 0.00093 of the score, so this is a tidy-up rather than a lever |
| **Main-thread warm-up** — *implemented, awaiting a submission* | first FP16 batch 26–34 s → 1.1–1.8 s | — | — | Ships with the next submission; see [what the shipped changes bought](attribution.md#what-the-shipped-changes-bought) |
| **Cut model load.** It is now the setup floor at 15.5–15.6 s per worker and it rose from 12.9–13.4 s when indexing stopped hiding it | up to ~15 s/worker, run-scale | — | ? | Unscoped: nobody has profiled what inside `load_checkpoint_model` costs the time — timm construction, the state-dict load, the FP16 chunk warm-up. Measure before proposing a change |

## Proposed order

### Phase 0 — fix the instrument

The profiler fixes [landed 2026-09-18](attribution.md#the-profilers-defects-fixed-2026-09-18). One
instrumentation job is left, and it is what decides the largest remaining row:

| Experiment | Metric to check |
|---|---|
| Instrument the 31.2 s bucket into prefix reads, assembly, `modal_layout`, sort and `summarize_series`, and time pool creation separately | Pool-churn share of the 16.4 s — it decides whether the shared pool or a CPU change owns the largest row |

### Phase 1 — retrain-free, run-scale

The worker-filtered index, the threaded listing and `in_order = False` shipped 2026-09-18 and are
waiting on a Kaggle run; their metrics are in
[ongoing work](../../ongoing/loading-speedup.md). What is left here:

| Experiment | Metric to check |
|---|---|
| Allocator env vars, A/B on the existing random-weights bundle | Loader wait and worker wall; **per-worker peak RSS must stay within the box** |

The allocator A/B is the cheapest real experiment available: two env vars, no code change, no
retrain, and the timing bundle already exists. Run it on the **old** bundle, because a new bundle
also carries the three shipped changes and would confound the timing.

### Phase 2 — the bit-exact cluster, into the retrain

Ordered by (confidence x seconds) / effort.

| Experiment | Metric to check |
|---|---|
| Shared thread pool per worker | `index_series` CPU per series; `IndexedSeries` records bit-identical |
| Struct scan, `SeriesTags` memo, manual cross product | Kaggle ms/header, 0.306 → about 0.11 expected; field-for-field equality on 20,000 files plus example-test |
| Resample column band, drop the redundant clip, `astype(copy=False)` | Resample stage against 8.25 s; uint8 equality on the 3,783-DICOM audit |
| `geometry_groups` prune, normal cache, alternatives refactor | `select_triplets` 7.59 ms → about 2 ms; **selection equality over the full train split, every study, all six slots, both pool paths** — the review's 3,200 cases were synthetic |
| Shift → mask | µs/frame; bit-equality, with the signed share re-measured per frame |
| Row-trimmed payload read | Bytes read, −13.4%, and sustained MB/s; bit-equality on a wider frame sample |

### Phase 3 — the structural questions

| Experiment | Metric to check |
|---|---|
| ~~`INFERENCE_IO_THREADS` sweep~~ — **done 2026-09-18**, constant moved 16 → 8 | Flat from 2 to 16, 1 thread a clear loss; io8 won both pairs by 2–3% ([sweep](../../inference/2026-09-18-kaggle-x4-profile-after-cpu-cluster.md#the-io-thread-sweep-2026-09-18)). Worker count around the winner — 8, 12, 16 — was not swept and remains open |
| GPU resample, quantiles and normalize | CPU limits, so this is live: loader-to-main byte volume first, then a retrain and a full validation run |
| Cut model load, now the 15.5 s setup floor | Setup wall against 15.7 s; it gates every remaining indexing idea |

The bandwidth question that used to sit here is
[answered](attribution.md#structural-questions-still-open): the frame stage is CPU-bound.
