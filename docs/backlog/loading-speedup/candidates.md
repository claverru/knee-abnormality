# Open loading-speedup candidates

What is left to do after the
[2026-09-22 profile](../../inference/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md), grouped
by what it costs to ship. Closed and rejected formulations are in [closed](closed.md); do not
re-propose them. Every "Est. saved" figure is an estimate: read [how to measure](measuring.md)
before trusting one.

**Where the time goes now.** The direct arm's 273.3 s is ~55 s of setup before the first forward
pass, then ~210 s of prediction per GPU, of which only ~27 s is loader wait: the GPU process's own
loop is the critical path at ~1.12 s per batch, ~0.59 s of it the model forward
([breakdown](../../inference/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md#direct-reads-and-page-locked-copies)). Kaggle hosts vary by about ±15 s on identical code, so anything smaller needs
repeated runs.

## What a candidate costs to ship

`preprocessing_identity` (`src/knee_abnormality/identity.py`) folds the raw bytes of the files in
`PREPROCESSING_SOURCES` into `sources_sha256`, and `validate_preprocessing_identity` refuses any
checkpoint whose recorded identity differs in any field, so it treats any byte change as a change.
That gives two classes:

- **No retrain.** The file is outside `PREPROCESSING_SOURCES`: `data/sources.py`,
  `inference/engine.py`, `inference/processes.py`, `scripts/pipeline/predict*.py`, the profiler and
  the notebook environment. These ship on their own, against existing checkpoints. Input-affecting
  constants belong where the hash reaches.
- **Identity-changing.** The file is in `PREPROCESSING_SOURCES` — `data/dataset.py`,
  `data/preprocessing/*.py` and the model files. The user accepted such edits on 2026-09-22 when the
  prepared images stay bit-identical or nearly so, audited on real studies; the saving then arrives
  with the next trained checkpoint. Batch them into one retrain.

## Ship without a retrain

| Change | Est. saved | Conf | Effort | Notes |
|---|---:|---|---|---|
| **Overlap the copy without a thread** — fetch batch *i+1*, page-lock it and copy it on a side stream before launching batch *i*'s forward | up to ~36 ms per batch, ~6 s per GPU | med | S | A feeder thread achieved the overlap but its GIL use cost more than it saved ([closed](closed.md#closed-2026-09-22)) |
| **Hide the model setup** — the model-stack import (~26 s cold after the fork) and the ~9 s CPU-bound first FP16 chunk | up to ~20 s | low | M | Importing everything before the fork and niceness were both tried and did not shorten it ([closed](closed.md#closed-2026-09-22)); hyperthread sharing with loaders is the likely cause. Warming cuDNN on a background thread during the import, or a targeted page-cache warm of the model-stack packages, are the untried forms |
| **Order each worker's studies by total image count**, for homogeneous batches and less padding | 1–3 s | low-med | S | Numerically inert under eval BN and per-study attention |

## Bit-exact, batch into the retrain

Widen any narrow local audit before shipping ([audit requirements](measuring.md#audit-before-shipping)).

| Change | Est. saved (cold-100) | Conf | Effort | Audit and notes |
|---|---:|---|---|---|
| **Order statistics by `topk` in `prepare_images`** — the four ranks per series from two `topk` calls on the padded values instead of a full sort, and fewer small launches | ~50–80 ms of the ~142 ms per batch | med | S | Exact: order statistics do not depend on the selection algorithm. The full sort alone measured 34 ms per batch |
| **Cache the per-series header decoding** — side fields, character set and `header_side` are decoded per file with pydicom helpers | ~1–2% of loader CPU | med | S | Headers are now ~30% of loader CPU; key the cache on the raw bytes so it stays exact |
| Vectorise `summarize_series` (16.1 µs/file serial, 15% of serial `index_series`) | 1–1.5 s | low-med | M | Float-order risk; audit poses and bounds |
| `PREFIX_BYTES` 12288 → 2048–4096 | 0–2 s, lands in the indexing row | low | XS | The pre-payload header is p50 1,482, p99 1,738, **max 1,758 bytes over 20,000 files**, and a 2,048-byte prefix truncated 0 of 1,200. The win is 219 MiB → 37 MiB over NFS and 6x less prefix RAM. Only extend-and-retry protects a longer hidden-test header; confirm before shrinking |
| `truncnorm.ppf` → `truncnorm._ppf` | 0.15 s | med | S | Bit-identical on 2,000 random (q, a, b); scipy distributions spend ~90% of a call in parameter checking. Private API — pin the scipy version in the offline bundle |
| **One file access per selected frame.** Indexing already records the payload offset, yet the payload read opens the file again and `fstat`s it to revalidate size and mtime | unquantified | low | M | The `fstat` is 1.6 µs/frame locally, 24 ms per 14,882 frames, so only the NFS round trip could justify it, and the attribute cache may already absorb it. Minimising reopens is the sanctioned small-file lever on network filesystems. A payload read must still fail closed when the file changed. No probe exists |
| **Bounded ordered payload window** in `decode_series` — submit a window of reads, decode in record order, release consumed bytes (`preparation.py`) | unmeasured; −80% transient RAM | low-med | M | Refines the earlier prefetch-window proposal: `pool.map` submits eagerly and the tuple retains every payload through the decode loop, so peak live memory holds raw payloads and resampled outputs at once (~83 MB of payloads per worker at x4, ~1.3 GB across 16). `buffersize` is not available on the bundled Python, so the window must be explicit. May trade memory for waiting — measure throughput, not just RSS |

## Reading less per study (trades score for wall)

Every candidate above preserves the prepared images. This one does not, and it is the largest lever on
the list, because the loader cost is per frame: the evaluation multiplier sets how many triplets
each study contributes, so cutting it removes loader CPU, loader bytes and forward-pass work in the
same proportion.

**The exchange rate the efficiency formula sets.** **0.001 of AUC is worth about 70 seconds of
runtime**, an order of magnitude rather than a constant
([efficiency leaderboard](../../inference/efficiency-leaderboard.md#the-exchange-rate)).

| Change | Measured score cost | What it removes |
|---|---|---|
| Test multiplier ×4 → ×3 | fold 0 tied (0.8954 both at τ 0.75), public 0.939 → 0.938, one rounding step inside the 0.003 noise floor | 24% of the triplets, so roughly a quarter of both loader and forward-pass work: order 80 s of the 350.8 s of worker wall measured after the CPU cluster, since only per-study work scales and setup (about 16 s per worker) does not |
| Test multiplier ×3 → ×2 | fold 0 −0.0024 at τ 0.75 | another 33% of the triplets |
| Fewer slices per type in training and inference | −0.0041 at 15/15/12/12/12/12 and −0.0051 at 12/12/9/9/9/9 on fold 0, 20% and 37% faster training ([runs](../../validation/results/2026-09-16-restored-code-hyperparameter-runs.md)) | needs a retrain, so it is an endgame decision, not a submission-time one |

### The bandwidth floor at each multiplier

What the run must read, and what that costs at the measured 510 MB/s ceiling. Two components: the
12 KiB prefix of **every** file, 232,115 of them for 2.85 GB, which does not change with the
multiplier; plus the payloads of the unique slices the sampler touches. Frames per study are
measured only at ×4 — 148.8 of the 178.6 slices present, **83% of the data**, from the
[cold-100 probe](../../inference/2026-09-18-kaggle-x4-profile-after-cpu-cluster.md) — so the other
three rows are an occupancy estimate, `unique = N(1 - e^(-3k/N))` fitted to that point at 1.07, and
the byte ranges span a payload of 524 kB (512² at 16 bits, the size that reproduces the 291 MB/s
the production frame stage sustains) to 694 kB (the 124 MB per study whole-file average).

| | Triplets/study | Frames/study | Frames per run | Bytes read | Floor at 510 MB/s |
|---|---:|---:|---:|---:|---:|
| ×1 | 25.4 | ~66 | 85,373 | 48–62 GB | 1.6–2.0 min |
| ×2 | 50.7 | ~107 | 139,178 | 76–100 GB | 2.5–3.3 min |
| ×3 | 75.9 | ~133 | 173,148 | 94–123 GB | 3.1–4.0 min |
| **×4 (today)** | 99.3 | **148.8 measured** | 193,466 | 104–137 GB | **3.4–4.5 min** |
| Every slice present | 165.1 | 178.6 | 232,115 | 125–164 GB | 4.1–5.4 min |

The bottom row reproduces the independently recorded 161 GB and 5.3 minutes
([closed](closed.md#ruled-out-by-arithmetic-or-by-the-environment)), which is the check that the
accounting is right.

- **At ×4 the loader is already close to the wall.** The 2026-09-22 trim arm read 104 GB in about
  217 s of prediction, ~480 MB/s against the 510 MB/s ceiling, after trimming 6–19% of payload
  rows ([profile](../../inference/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md)). No
  remaining candidate above buys a factor; they buy tens of seconds.
- **The multiplier is the only large lever left.** ×4 → ×3 removes about 20 GB, 40–50 s of floor;
  ×4 → ×1 removes about 75 GB, 110–150 s. Against the ~70 s per 0.001 AUC rate, ×3 is the one worth
  measuring and ×1 buys less than its fold-0 cost.
- **The header pass does not shrink with it.** 232,115 opens and 2.85 GB stay at every multiplier,
  about 121 s of CPU and roughly 35 s of wall, so cutting frames has diminishing returns before it
  reaches zero: at ×1 the fixed part is already a fifth of the floor.
- **It also bounds what the public five-minute submissions can be doing.** Our ×4 floor alone is
  3.4–4.5 minutes with a free GPU and a perfect loader, so those runs read a fraction of the slices
  we do rather than reading the same data faster.

**It is close to break-even at ×3, which is why it needs measuring rather than assuming**: about
80 s saved against a cost of at most 0.001 AUC, itself worth about 70 s. The public evidence sits
at the same margin — one competitor's trimmed arm scored 0.930 at 42 windows against 0.937–0.938 at
94, about 0.01 for most of the runtime
([sweep](../../research/public-research.md#slice-density-and-the-board-2026-09-18-re-sweep)). The
accuracy side of the same knob is in [endgame](../endgame.md#test-time-augmentation); what is
missing for a decision is the wall at fixed batch for ×3 against ×4 on Kaggle, which the same
profile run can produce.

