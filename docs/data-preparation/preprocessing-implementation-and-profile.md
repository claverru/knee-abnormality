# Preprocessing implementation and profile

Production preprocessing now evaluates only the centered output region with a
separable, scale-aware antialiased triangle filter. It uses the full rounded
target coordinate system and renormalizes weights at image boundaries, so crop,
padding and physical-spacing geometry remain unchanged. DICOM slope/intercept is
applied after interpolation to the retained region. Each prepared acquisition
uses contiguous float32 NumPy crop and percentile buffers, rectangular validity,
and exact multi-rank partition. Production uses owned float32 buffers and
in-place NumPy `clip`, `subtract`, `multiply` and `rint` ufuncs with `out=`,
followed by uint8 conversion and explicit zeroing outside each valid rectangle.
The gain uses reciprocal-first float32 arithmetic to match the validated mapping.
`torch.from_numpy` creates the returned tensor without a second pixel copy at
the preparation boundary.

A five-pair comparison of the otherwise identical custom-C and NumPy paths found
that C improved median loader time by 4.38% and raw-index-plus-loader time by
2.40%. Their uint8 outputs were exactly equal across the 24-study, 3,783-DICOM
audit. Production uses NumPy to avoid ctypes, runtime compilation, a CPU-specific
binary cache and a compiler dependency in the submission environment. The raw
comparison and decision record are under
`outputs/profiling/normalization-c-vs-numpy-20260912T000001Z` (deleted).

The fixed 128-study experiment used batch size eight, eight persistent workers,
prefetch factor two and one native/Torch thread per worker. Five post-integration
NumPy pairs measured median production loader throughput of 35.31 studies per
second, a 1.505x loader ratio, a 1.341x raw-index-plus-loader ratio and a 1.486x
CPU/study ratio against the frozen baseline. Every pair favored production; the
observed loader-ratio range was 1.483x to 1.568x.

The 24-study production audit covered 3,783 selected Explicit VR Little Endian,
single-frame MONOCHROME2 DICOMs, including 1,104 signed frames, 312 nonidentity
rescale transforms, 21 source shapes, crop and pad cases, and frames as large as
1024 by 1024. Study/label order, grouping, selected paths, positions, failures
and valid rectangles matched. `assert_allclose` passed for all 189,815,808
returned uint8 pixels with `atol=1`; 10,470 pixels changed, all by one level.
Compressed, MONOCHROME1, multiframe and selected corrupt cases remain uncovered.

The frozen investigation is under
`outputs/profiling/preprocessing-investigation-20260911T205612Z-run2` (deleted),
and the final NumPy production audit and raw paired timings are under
`outputs/profiling/preprocessing-production-numpy-20260912T001501Z` (deleted).
The [production integration](production-dicom-integration.md) records the DICOM
reader and preprocessing package changes. Later percentile and GPU work is planned in the
[loading speedup](../backlog/untried-ideas.md#inference-loading).

## Clipping-bound stability across draws

**Measured 2026-09-20** with
[`measure-intensity-bound-stability`](../../scripts/adhoc/measure_intensity_bound_stability.py)
on 40 random fold-0 training studies replayed over six training epochs at the current recipe
(1,338 series draws, 6,432 triplets); artifacts under
`outputs/analysis/i/intensity-bound-stability-20260920` (deleted).
Spread is `(max - min) / median` of an estimator over the draws of one series.

| Estimator of the high bound | Median spread | p90 | Series above the 0.2 contrast range |
|---|---:|---:|---:|
| `q995`, the production bound | 0.072 | 0.245 | 14.2% |
| `q50 + 2(q95 - q50)`, calibrated to ×1.12 of `q995` | 0.062 | 0.214 | 10.6% |
| `q50 + 2·IQR`, calibrated to ×1.06 | 0.097 | 0.331 | 19.9% |
| `q95` alone, clipping 5% of pixels | 0.045 | 0.170 | 8.4% |

The low bound is exactly zero in 40.1% of draws, so it is not a constant that could be pinned.

Three conclusions. Robust estimators of the same clip level do not fix the variation: calibrated
to clip where `q995` clips, they move the median spread by about one point. The variation is a
function of how many triplets the series receives — median spread 0.150 at two triplets or fewer,
0.066 at five to six, 0.025 above six, correlation −0.39 — so it is a small-sample effect in the
tail quantile, not a property of the estimator. And per-triplet bounds, which would be draw
independent by construction because a triplet's three slices are fixed by its middle index, vary
0.291 at the median and 0.701 at p90 between the triplets of one series, four times the variation
they would remove.

For scale, the configured intensity augmentation deliberately applies a contrast factor over a
0.2 relative range, so the median series' draw-to-draw wobble sits inside an augmentation the
recipe already applies, while the 14.2% tail exceeds it.

The script replays the real dataset path with the configured training seed, one epoch per draw, so
the selections are the ones training sees. It records the pipeline's own bounds and recomputes
pooled quantiles from the decode cache; the recomputed 0.995 level reproduces the recorded high
bound. CPU only, refusing an existing output directory. It is historical since intensity bounds moved
to the GPU on 2026-09-22, so run it against the source of `ab49830` or earlier:

```bash
uv run --locked python -m scripts.adhoc.measure_intensity_bound_stability --output-dir outputs/analysis/intensity-bound-stability-new --studies 40 --epochs 6
```

**Effect on the metric, measured 2026-10-12:** bounds from a fixed set of up to seven slices per series, independent of the draw, scored ×1 +0.0006 and ×4 −0.0003 against the reference on the ResNet-50, fold 2, and were not adopted ([result](../validation/results/campaigns-and-queues/2026-10-13-targets-and-inputs-campaign.md#inputs-and-augmentation)).

## Earlier profile

On 2026-09-10, a bounded profile used eight fold-0 training studies with seed 42:
the four largest selected-slice counts followed by four randomly selected studies.
It retained the baseline's full K limits and evaluated the same 44 series and
1,287 prepared slices twice. CPU intra-op threads were set to one, matching
DataLoader workers. cProfile identified the full `torch.quantile` sort as about
40% of initial preprocessing time, plus repeated header reads for physical order.

The shared pixel path now uses NumPy partition to find the two adjacent ranks
for each percentile, followed by the same float32 rank arithmetic and linear
interpolation. It preserves both bounds without rounding. Dataset instances
also retain the physical slice index after its first use; persistent workers
reuse these header records across epochs. The cache contains metadata only,
assumes raw files remain unchanged during a run, and ends with the dataset/worker.
Pixels are still decoded and prepared for each load.

| Eight-study pass under cProfile | Before | After |
|---|---:|---:|
| Initial pass | 6.81 s | 3.07 s |
| Repeated pass | 5.57 s | 2.32 s |

All 44 prepared-series pixel SHA-256 values matched before and after, including
the repeated pass. These are instrumented local-file measurements; the OS page
cache was not flushed. They do not estimate uncached storage performance.
Source snapshots, package versions, seed, input identities, per-study timing and
pixel hashes are in `outputs/profiling/before/` and `outputs/profiling/after/`.
The [historical profiling findings](../modeling/training-runtime-and-memory.md#batch-4-memory-fix)
summarize the corresponding model-side measurements.

Reproduce a bounded profile with
[`profile_training`](../../scripts/adhoc/profile_training.py). It samples fold-0 training studies
only, up to eight maximum-load studies then seeded random ones, with full configured slice limits,
and never runs an epoch or reports validation AUC. The second preprocessing pass reuses the metadata
indexes; loader profiling starts with empty header caches. The GPU modes need CUDA and use seeded
random weights and fixed class-balanced BCE weights from the profiling subset; the GPU step disables
the autocast weight cache as Lightning does. Capacity mode disables series dropout, accepts batch
sizes up to eight, stops on OOM or excessive allocator reservation, and counts backbone BatchNorm
updates to audit recomputation under activation checkpointing. Lightning mode uses the training BF16
precision and configured batch size, and refuses to exhaust its subset.

```bash
uv run --locked python -m scripts.adhoc.profile_training configs/resnet50.yaml --mode preprocessing --output-dir outputs/profiling/preprocessing-new
uv run --locked python -m scripts.adhoc.profile_training configs/resnet50.yaml --mode gpu --batch-sizes 8 --samples 8 --warmup 1 --steps 2 --output-dir outputs/profiling/gpu-new
uv run --locked python -m scripts.adhoc.profile_training configs/resnet50.yaml --mode loader --batch-sizes 4 --workers 2,4,8 --samples 48 --warmup 2 --steps 6 --threads 16 --output-dir outputs/profiling/loader-new
uv run --locked python -m scripts.adhoc.profile_training configs/resnet50.yaml --mode lightning --samples 64 --warmup 1 --steps 2 --threads 16 --output-dir outputs/profiling/lightning-new
```
