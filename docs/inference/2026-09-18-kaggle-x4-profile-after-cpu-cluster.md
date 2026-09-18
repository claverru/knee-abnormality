# Kaggle x4 profile after the CPU cluster

What the bit-exact preparation cluster bought end to end, measured with **trained weights** for the
first time. Supersedes the
[profile after the index changes](2026-09-18-kaggle-x4-profile-after-index-changes.md), whose
418.1 s repeat is the baseline column here.

Bundle `outputs/submission/exports/control-check-x4-tau075`, exported from the `control-check`
fold-0 checkpoint (best epoch 6, validation macro AUC 0.88477), identity `b8b6c639`, x4 images,
attention temperature 0.75, batch 4 per device, 16 loader workers. Dataset
`claverru/knee-profile-control-check-x4-20260918`; artifacts in
`outputs/profiling/kaggle-inference-control-x4-20260918/`; notebooks
`knee-profile-fp16-b4-w16-control-x4-20260918` and `knee-profile-loading-control-x4-20260918`.
Predictions are real but unscored: the run spans 1,300 studies, 396 of them outside fold 0.

## End to end, two T4s, 1,300 studies

| | Before (index changes) | After (CPU cluster) |
|---|---:|---:|
| Worker wall | 418.1 s | **350.8 s** |
| Studies per second | 3.109 | **3.705** |
| Full inference command | 440.5 s | **371.4 s** |
| Indexing (`data_setup`) | 7.2 s | 7.6 s |
| Model load | 17.3–17.4 s | 18.3–18.4 s |
| Prediction | 388.0–389.1 s | **321.1–323.2 s** |
| — loader wait | 197.2–213.2 s | **134.7–137.5 s** |
| — forward | 173.4–190.6 s | 184.4–185.0 s |
| First-batch wait | 15.1–15.9 s | 13.3–13.8 s |

**−16.1% worker wall, +19.2% throughput.** Loader wait fell by a third while forward time held, so
the gain is entirely on the loading side, as intended.

Per-role CPU over the prediction window, sampled once a second:

| Role | Before | After |
|---|---:|---:|
| Loader workers (16) | 1,241.6 s | **990.6 s** |
| GPU main processes (2) | 108.8 s | 122.3 s |
| Other and coordinator | 4.8 s | 4.2 s |

**Loader CPU fell 20.2%**, matching the 20.5–21.9% measured locally on paired 40-study runs before
the change shipped. The main processes rose slightly because they now receive batches faster and
launch more kernels per second. The box remains **CPU-bound at 99.1% busy, 0.75% iowait and 3.47 of
4 cores**, with the GPUs at 30.6% against 25.5% before, so the next CPU saving still converts.
Against the 3.5-effective-cores model this predicted about 75 s off the wall; the run gave 67 s.

## Cold loading stages, 100 studies

Same 100 seeded studies and identical work in both runs — 17,855 headers, 14,882 frames, 10,095
images, zero pydicom fallbacks — and the two machines' read throughput matches (978 against 982
header files/s at 16 threads, 792 against 790 for full files).

| Stage | Wall before | Wall after | CPU before | CPU after |
|---|---:|---:|---:|---:|
| Series indexing, less parsing | 30.98 | **11.60** | 12.86 | **6.86** |
| Frame reads | 15.54 | 26.77 | 17.38 | **12.23** |
| Resample and crop | 8.02 | 11.20 | 7.99 | 11.21 |
| Slice headers | 5.16 | **2.44** | 5.16 | **2.45** |
| Geometry and sampling | 5.46 | **2.57** | 5.42 | **2.55** |
| Quantiles | 4.77 | 6.47 | 4.76 | 6.47 |
| Normalize | 2.36 | **1.89** | 2.36 | **1.90** |
| Other preparation | 2.72 | 1.86 | 2.69 | 1.85 |
| **Whole pass** | **77.65** | **67.09** | **61.25** | **48.11** |

0.776 → 0.671 s per study, pass CPU **61.25 → 48.11, −21.5%**.

### The machines differ, and quantiles is the yardstick

Several rows rose. The quantile path was **not changed** by the cluster, and its CPU went 4.76 →
6.47, a factor of **1.36**: the second machine is about a third slower at this work despite
identical I/O throughput. Dividing that out:

| Stage | Raw CPU ratio | Machine-adjusted | Change responsible |
|---|---:|---:|---|
| Slice headers | 0.47x | **0.35x** | struct scan, tags memo |
| Geometry and sampling | 0.47x | **0.35x** | group prune, normal cache, `select_triplets` |
| Series indexing, rest | 0.53x | **0.39x** | shared io pool |
| Frame reads | 0.70x | **0.52x** | unsigned mask, `os.pread` |
| Normalize | 0.80x | **0.59x** | dropped second clip |
| Quantiles | 1.36x | 1.00x | *(unchanged — the yardstick)* |
| **Resample and crop** | 1.40x | **1.03x** | column band |

**The resample column band appears to have bought nothing on Kaggle's vCPUs.** It measured 8.7%
faster locally and bit-identical, so the expected ratio was about 1.25x; it came in at 1.40x. The
candidate list warned that its gain was partly a cache-stride effect and unverified on these vCPUs
([candidates](../backlog/loading-speedup/candidates.md)). One run per machine, so treat this as the
row to re-measure rather than a settled regression.

## The io-thread sweep, 2026-09-18

Five arms at 16 loader workers, 1,300 studies, x4, one run each, plus a repeat of the top two.
Worker wall in seconds:

| Threads | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Worker wall | 457.9 | 336.7 | 336.1 | **329.5** | 338.3 |
| Studies/s | 2.839 | 3.861 | 3.868 | **3.945** | 3.842 |
| Loader CPU | 976.2 | 918.0 | 951.6 | 920.6 | 970.1 |
| CPU busy | 79.4% | 98.2% | 99.7% | 99.1% | 99.6% |

**One thread is the only clear loser** — the only arm where the box is not saturated (79.4% busy,
325 s of loader wait), so the threads really are hiding NFS latency and removing them is not the
answer. **Everything from 2 to 16 lands within 2.6%**, which is barely outside the ~2% run-to-run
spread.

A repeat of the top two put io8 at 338.0 and io16 at 374.2, so **io8 won 2 of 2**, by 2.6% in the
clean pair and 9.7% in the second — but discount that second pair: the io16 repeat ran at 89.1%
busy with 200 s of loader wait, which is an unsaturated box, not a thread-count effect. Loader CPU
does not order cleanly either (io8 lower in one pair, higher in the other).

`INFERENCE_IO_THREADS` moved 16 → **8** on this evidence: it won both pairs, it halves thread
oversubscription to 64 per GPU process, and the flat region between 2 and 16 means the downside is
near zero. The effect is 2–3% when both machines behave, and the per-arm spread (io16: 338 to 374)
is larger than the effect. `forward_seconds` is not comparable across these arms — it ranged 102 s
at io1 to 182 s at io16 for identical work, because CPU contention inflates the wall around the
forward call.

Artifacts: `outputs/profiling/kaggle-iosweep-x4-20260918/` and `kaggle-iosweep-repeat-20260918/`.
The sweep needs `--io-threads`: the count is bound as a default argument of `prediction_dataset`,
so assigning to the config constant after import changes nothing.

## What this run cannot tell us

- **The two loading passes ran on different machines**, so every stage comparison depends on the
  quantile yardstick. Only the end-to-end prediction comparison is like for like in hardware terms,
  and even that is one run against one run, against a documented ~2% wall spread.
- **Predictions are unscored.** The study set spans folds deliberately, so no AUC or FP16 parity
  number follows from this run.
- **The loading cell is still single-process**, so its stage table says nothing about how the
  cluster behaves under 16 workers contending for four cores. The io-thread sweep above is the only
  measurement here taken at the production worker count; the loader **worker** count was never
  swept around the new thread setting.
- This run predates the profiler's per-pass cache reset, so its warm pass still carries warmed
  `_TAGS_CACHE`, filter and pool state.
