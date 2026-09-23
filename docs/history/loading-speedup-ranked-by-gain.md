# Remaining candidates ranked by whole-pipeline gain

**Superseded 2026-09-22.** Every ranked row has since been shipped, superseded or rejected on
Kaggle: ranks 1, 5 and 6 shipped on 2026-09-19 and bought nothing measurable; rank 2 was rejected;
rank 3 shipped bit-exactly as GPU preparation; rank 7 was profiled and rank 8 shipped as a CUDA
graph ([profile](../inference/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md),
[closed](../backlog/loading-speedup/closed.md#closed-2026-09-22)). The ranking below is kept for its reasoning, and the two
measurement sections as the bit-exact evidence for ranks 1 and 5.

What each open candidate is worth against the **371.4 s full inference command**, so the next change
is chosen by end-to-end effect rather than by the size of the stage it touches. What each one costs
to ship, and what it has to audit, stays in [candidates](../backlog/loading-speedup/candidates.md); this document only ranks
them.

Produced 2026-09-19 from an external optimization review of revision `f4c340b`, reconciled against
the [post-cluster profile](../inference/2026-09-18-kaggle-x4-profile-after-cpu-cluster.md) and two
local measurements recorded below.

## The conversion, and why it closes

**1 cold-100 CPU-second = 13 probe-units per 1,300-study run × 1.56 contention = 20.3 run CPU-seconds
÷ 3.5 effective cores = 5.8 s of wall = 1.56% of the 371.4 s command**
([the model](../backlog/loading-speedup/attribution.md#the-model-that-replaces-the-old-ranking)).

The arithmetic closes on the measured run: the cold pass costs 48.11 CPU-seconds per 100 studies,
and 48.11 × 20.3 = **976 CPU-seconds against the 990.6 measured in the sixteen loader workers**. The
~15-second residual is real but unattributed, and one candidate below is a plausible owner of part
of it.

## The ranking

Stage CPU is the "CPU after" column of the
[cold loading stages](../inference/2026-09-18-kaggle-x4-profile-after-cpu-cluster.md#cold-loading-stages-100-studies).

| Rank | Candidate | Stage it touches (cold-100 CPU) | Speedup assumed | Expected | Ceiling | Conf | Effort |
|---|---|---|---|---:|---:|---|---|
| 1 | [Support-limited native-to-float32 conversion](../backlog/loading-speedup/candidates.md#bit-exact-batch-into-the-retrain) | frame reads 12.23 + resample 11.21 | **2.01x, measured** | **6–12%** | 17% | med | M |
| 2 | [glibc allocator config](../backlog/loading-speedup/candidates.md#ship-without-a-retrain) — size thresholds plus the arena cap, *implemented, awaiting a run* | the same allocation cost, with no retrain | — | **~8%** | 4.7–12.5% | med | **free** |
| 3 | [GPU resample, quantiles and normalize](../inference/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md#gpu-preparation) | resample 11.21 + quantiles 6.47 + normalize 1.90 = 19.6 s | most of it leaves the CPU | **~6%** | 24% | low | L |
| 4 | Ahead-of-time kernel fusing conversion with interpolation | resample 11.21 + part of frame reads 12.23 s | 2–3x | **~4%** | 12–19% | low | XL |
| 5 | [Exact quantiles by single-rank partitions](../backlog/loading-speedup/candidates.md#bit-exact-batch-into-the-retrain) | quantiles 6.47 s | **1.44–1.63x, measured** | **3.1–3.9%** | 3.9% | **high** | S |
| 6 | [Direct shared-storage collation](../backlog/loading-speedup/candidates.md#bit-exact-batch-into-the-retrain) | absent from the probe — see below | 0.40 ms/MiB over 26.9 GiB | **1.3–2.1%** | 2.1% | med | S |
| 7 | [Cut model load](../inference/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md#startup) | 18.4 s of setup, floored by indexing at 7.6 s | — | **~1%** | 2.9% | — | ? |
| 8 | CUDA Graph around the fixed 64-image backbone chunk | main-process CPU 122.3 s | part of the launch cost | **~1%** | 3% | low | L |
| 9 | `gc.freeze()` before fork, with per-worker RSS tracking | — | — | **~0%** | — | — | free |

Ranks 4, 8 and 9 have no row in [candidates](../backlog/loading-speedup/candidates.md) yet; they are carried here until one of
them earns a probe.

## Run rank 2 first, even though rank 1 is larger

Ranks 1 and 2 attack **the same cost from opposite ends**, and the measurement below is what ties
them together: the dominant expense in the frame stage is materialising a whole-frame float32 buffer
that the crop then mostly discards, and the price of that buffer is set by allocator behaviour. Rank
1 stops allocating it; rank 2 makes allocating it cheaper. **They are not additive**, and nobody
knows the split.

Rank 2 is free — two environment variables and no retrain — so running it first prices how much of
rank 1 is left to win before an identity bump is spent on it. It is **implemented as of 2026-09-19**
and set unconditionally in each worker's environment, which means the control arm is a bundle
exported *without* it rather than an env-var override. Export both arms from a tree that differs in
nothing else; the hashed sources in particular must match, or the control will not validate against
the same checkpoint.

It ships the size thresholds and the `MALLOC_ARENA_MAX=2` cap together, which are two mechanisms in
one arm — size of allocation against ownership of it. Both are retrain-free and score-neutral, so the
winner ships either way, but **one run cannot attribute the result between them**. Split them only if
the combined arm comes back ambiguous or negative.

## What the ordering hides

- **The rows are not additive.** Rank 3 subsumes ranks 1, 5 and most of rank 4, because moving
  resample, quantiles and normalize onto the T4s removes those stages from the CPU entirely. Rank 4
  overlaps rank 1 on the conversion. Ranks 1 and 2 overlap as described above. Choose one of
  {rank 3, rank 4, ranks 1+5}; **ranks 1+5 are the only combination that is both cheap and
  bit-exact**, and rank 2 is free alongside any of them.
- **Rank 4 was revised down** from ~6% by the rank 1 measurement. If the win is allocation and memory
  traffic rather than arithmetic, an ahead-of-time fused kernel buys much less than a flop-count
  argument suggests, at XL effort.
- **Rank 3's ceiling is unpriced.** Its 24% ignores that preprocessing moves into the main process
  beside the forward pass and that loader-to-main bytes go 7.5 → 76 MB per study, which is why its
  candidate row is rated low confidence.
- **Rank 6 is invisible to the cold probe**, which is single-process and has no multiprocessing queue,
  so its cost is additive to the stage table rather than inside it. Its 11–27 estimated CPU-seconds
  sit inside the ~15-second residual noted above, which is suggestive but not attribution. The
  mechanism itself is not in doubt: `collate_studies` allocating with plain `torch.zeros` is verified
  in our code, and the review measured the promotion at three buffer sizes with minor-fault counts
  (40,961 against 20,480 at 80 MiB) and a changed data pointer. What is unmeasured is how it
  transfers to four Kaggle vCPUs, not whether it exists.
- **The corrected GPU floor is what lifts the large rows.** With the floor at ~100 s rather than the
  173–191 s once assumed, large CPU savings keep converting to wall instead of hitting a cap at
  roughly half their value.
- **Both measurements below are local.** The resample column band is the standing precedent for a
  local win going inert on Kaggle's vCPUs: 8.7% faster locally, **1.03x** there.

## Measurement behind rank 1

2026-09-19, local box, revision `f4c340b`. 120 real train frames across five source shapes
(256², 400², 512², 640², 704²), payloads pre-loaded so no I/O is timed, each arm run seven times and
the best taken. **Median retained source area 0.686** (mean 0.704, range 0.545–0.774).

| Arm | µs/frame | Output |
|---|---:|---|
| Shipped: `frame_from_payload` then `resample_crop` | **322.6** | — |
| + contiguous operand for the existing column band | 320.7 | 120/120 bit-identical |
| + row trim of the dot, still a full-frame cast | 335.0 | 120/120 bit-identical |
| **Cast only the two axes' support** | **160.2** | **120/120 bit-identical** |

**2.01x, bit-exact.** The cause is neither mechanism the review proposed: operand contiguity is flat,
and trimming the dot while keeping the full-frame cast is a *loss*. The whole win is **not
materialising a full-frame float32 buffer that the crop then discards**. Per-phase, the cast falls
87.4 → 54.6 µs and the vertical dot 54.3 → 38.9 µs.

**The confound that sets the error bars:** the same full-frame cast measured **33 µs in isolation and
87 µs interleaved with the rest of the arm**. Much of this cost is page-fault and allocator
behaviour, not arithmetic, which is why rank 2 attacks the same money and why the 6–12% expectation
is wide. A Kaggle A/B has to separate them; see [run rank 2 first](#run-rank-2-first-even-though-rank-1-is-larger).

A useful side result: the **existing column band's strided operand costs nothing** (320.7 against
322.6), which is consistent with the band measuring 1.03x on Kaggle.

## Measurement behind rank 5

2026-09-19, local box, NumPy 2.0.2, revision `f4c340b`. Twenty-four real train acquisitions built
through the repository's own `read_header`, `read_frame` and `resample_crop` at `spacing_mm` 0.625
and `image_size` 224, then quantiled with the shipped `_linear_quantiles` and with a formulation that
replaces the single four-`kth` `partition` by two single-`kth` partitions plus two tail reductions.

| Buffer | Shipped | Single-rank | Bounds |
|---|---:|---:|---|
| 1.10M values (22 slices) | 3.22 ms | **2.24 ms** | 24/24 bit-identical |
| 188k values (~4 slices) | 0.619 ms | **0.379 ms** | 24/24 bit-identical |

Best of nine interleaved repeats per arm. The mechanism is not isolated: the review attributes it to
NumPy 2.0.2 entering its SIMD quickselect dispatch only for a single requested rank, but a synthetic
control behaved inconsistently, so the measurement is the evidence and the dispatch argument is not.

This result contradicts the reason the quantile line was closed; see
[closed](../backlog/loading-speedup/closed.md#closed-as-measured-regressions).
