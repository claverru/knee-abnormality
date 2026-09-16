# DICOM descriptor experiment

On 2026-09-12, the existing benchmark added `dicom-deferred` and `dicom-native`.
This prototype preceded the [production integration](production-dicom-integration.md)
below. Both candidates carry metadata
from ordering to decoding in the existing worker slice index. They stop parsing
at the pixel element without loading its value, retain no open file or decoded
pixels, and create no persistent header or image cache. Deferred reuse constructs
a temporary pydicom dataset from a deferred pixel element. Native decoding reopens
the file and reads the validated pixel range into a NumPy view before float32
conversion and stored-bit correction. Preparation and normalization stay fixed.

The native branch accepts only Explicit VR Little Endian, single-frame,
single-sample, 16-bit MONOCHROME2 with consistent stored bits/HighBit, positive
dimensions, matching payload length, and PixelData ending at EOF. It checks file
size and modification time again when opening selected files. Everything outside
that layout uses the existing pydicom reader, including compressed and big-endian
files, MONOCHROME1, multiframe, trailing elements, and malformed payloads.

Three paired repeats against current production used the previous 128-study
cohort, seed 42, fold 0 of `all_5fold_v1`, batch size eight, eight persistent
workers, prefetch factor two, one native/Torch thread, 224-pixel images, 0.67-mm
spacing, adjacent triplets, and the configured 2× validation image multiplier.
Each timing ran in a separate process after a separate correctness audit.

| Candidate | Median paired loader ratio | Raw-index + iterator + loader ratio | Parent + worker CPU/study ratio |
|---|---:|---:|---:|
| Deferred pydicom reuse | 1.036× | 1.032× | 1.054× |
| Native pixel offset | 1.357× | 1.213× | 1.301× |

All three native pairs improved: loader ratios ranged from 1.352× to 1.409×.
Median loader times were 3.685 seconds for its paired production runs and
2.698 seconds for native decoding, or 34.73 versus 47.44 studies/second. The
median full-process ratio, including Python startup and shutdown, was 1.130×.
Median sampled aggregate worker RSS increased from 5.80 GB to 5.90 GB; parent
RSS stayed around 2.23 GB. Aggregate RSS counts shared pages repeatedly and is
not unique physical memory. There is no compiler or first-build cost.
OS page cache was uncontrolled: the first production run in the deferred
comparison took 6.664 seconds, so its 1.773× first-pair ratio is a warmup outlier,
not evidence for a large deferred-reader gain. These are application-first-pass
measurements, not guaranteed cold-storage results.

The eight-study DataLoader audit compared all returned tensors exactly
(`assert_allclose`, `rtol=0`, `atol=0`) and matched labels, study/acquisition
ordering, positions, selected paths, grouping, failures and valid rectangles.
All 72,253,440 collated uint8 pixels matched. Its 1,023 selected files all used
the validated native layout, including 633 signed frames and six shapes up to
1024×1024. These real studies had no nonidentity rescale transforms. Fourteen
small derived/bundled pydicom cases separately covered signed and unsigned
12-bit values with nonzero unused bits, negative rescale and defaults,
MONOCHROME1, multiframe rejection, missing geometry/spacing, truncated payloads,
big/implicit endian, RLE and JPEG-LS. Both readers matched raw pixels and triplet
outputs or the existing exception/failure behavior. RLE decoded using the
installed pydicom plugin; JPEG-LS preserved the missing-codec failure, so successful
JPEG-LS decoding remains unverified. Compressed throughput was not measured.

A separate instrumented profile measured aggregate DICOM worker time falling
from 12.487 to 7.743 seconds: ordering/header work increased from 4.042 to
5.531 seconds, while selected-file parse/decode fell from 8.446 to 2.212 seconds.
The profile reference was the already validated `combined-numpy` equivalent of
production, used to expose its stages. Profile timings are diagnostic and are
separate from the paired results above.

Reproduction config, environment/package versions, cohort identity, source
snapshot, raw runs and audits are under
[`outputs/profiling/dicom-reader-20260912`](../../outputs/profiling/dicom-reader-20260912).
`comparison-v1/` contains the real DataLoader audit and timings;
`corrected-edge-audit.json` owns the edge coverage above. The first edge fixtures
retained a trailing padding element and therefore exercised fallback; they are
superseded by the corrected fixtures with explicit dispatch assertions. No
timing code changed for that correction. The starting source was already dirty;
the saved source snapshots and manifest hashes identify the implementation.
The [script catalog](../../scripts/adhoc/catalog.yaml) records `dicom-suite` and
`dicom-audit` invocations. The current baseline uses `all_5fold_v3`; use the saved
`workload.yaml` with this saved cohort, or generate a new matched cohort when
benchmarking the current split.
