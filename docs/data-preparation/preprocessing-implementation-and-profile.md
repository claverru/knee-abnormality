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
[`outputs/profiling/normalization-c-vs-numpy-20260912T000001Z`](../../outputs/profiling/normalization-c-vs-numpy-20260912T000001Z).

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
[`outputs/profiling/preprocessing-investigation-20260911T205612Z-run2`](../../outputs/profiling/preprocessing-investigation-20260911T205612Z-run2),
and the final NumPy production audit and raw paired timings are under
[`outputs/profiling/preprocessing-production-numpy-20260912T001501Z`](../../outputs/profiling/preprocessing-production-numpy-20260912T001501Z).
The [production integration](production-dicom-integration.md) records the DICOM
reader and preprocessing package changes. Later percentile and GPU work is planned in the
[loading speedup](../backlog/loading-speedup/candidates.md).

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

Reproduce a bounded preprocessing profile through the
[profile-training catalog entry](../../scripts/adhoc/catalog.yaml):

```bash
rtk proxy uv run --locked python -m scripts.adhoc.profile_training configs/training.yaml --mode preprocessing --output-dir outputs/profiling/preprocessing-new
```
