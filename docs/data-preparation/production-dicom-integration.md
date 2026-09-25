# Production DICOM integration

On 2026-09-12, production adopted the validated native reader in the
[`preprocessing/` package](../architecture/code-organization.md#where-code-belongs). The package
separates DICOM access, slice indexing/selection, spatial kernels, intensity
kernels and preparation. The old umbrella module and imports are removed.
Configuration, dependencies, float32 arithmetic, acquisition-wide quantiles,
model inputs and checkpoints are unchanged. Native dispatch is fixed; the
pydicom fallback retains all unsupported layouts and existing decode failures.

Descriptors contain only offset/length, file identity, dimensions, stored-bit
metadata, spacing and rescale scalars. Validation requires matching observed
Explicit VR Little Endian encoding, ordinary OB/OW PixelData ending at EOF,
one frame/sample, 16 allocated bits, MONOCHROME2, valid stored bits/HighBit and
signedness, positive dimensions and matching payload length. Missing rescale
values keep their existing defaults; invalid values disable native dispatch.
Changed file size or nanosecond mtime, a failed range check or a short payload
causes exactly one fresh pydicom decode after closing the native handle.
The fresh read supplies spacing and affine values while indexed positions and
selection stay fixed. Native I/O errors propagate through existing preparation
handling. There is no new format support, inversion, LUT or windowing.

The benchmark (`scripts/adhoc/benchmark_dataloader_optimizations.py`, removed; last in
commit `aec290d`) used the actual `KneeStudyDataset` and production collation.
Its independent `production-reference` used frozen preprocessing and dataset
code in `scripts/adhoc/dataloader_reference/` (last in commit `204d29c`). Both timing candidates perform the same raw series discovery and
start with empty application slice indexes. Source inventories and copied
snapshots include nested modules with project-relative paths. The exporter
copied the package successfully, and isolated imports resolved every new module
from the copied bundle. Static/import checks and the script catalog check passed.

The reference audit passed before extraction, and the package-only audit passed
before native integration. The final eight-study audit matched all 72,253,440
uint8 pixel bytes, tensor shapes/dtypes, labels, ordering, selected records,
positions, RNG state, grouping, failures and valid rectangles with zero tolerance.
All 1,023 selected real files used native decoding. The corrected 14 fixtures
and three integration cases passed explicit dispatch and prepared-group checks:
changed file size, unchanged size with changed pixels/affine/mtime, and invalid
`PixelRepresentation=2`. The invalid representation remained indexed without a
descriptor. RLE decoded; JPEG-LS retained the missing-codec failure. Successful
JPEG-LS decoding remains unverified. The first two audit studies also matched
the original training tensor hashes and selections with seed 42, epochs 0 and 1,
and the 1× training budget.

Repeated validation loading retained 40 series indexes and 1,192 scalar-only
descriptors, without rebuilding the index or retaining DICOM handles. Worker
file-descriptor counts were unchanged after releasing each pass's batch.
Eight configured workers process this eight-study cohort as one batch, so its
index-ownership result covers one active worker. Conservative Python-object
accounting adds at most 464 bytes per indexed native record, or 553,088 bytes
for this index. Allowing one worker to visit all 843,447 conservatively counted
supported training files gives 391,359,408 additional bytes per worker
(373 MiB), potentially 2.92 GiB across eight workers. This estimates descriptor
objects, not total RSS or allocator overhead; do not assume permanent even
partitioning across workers over epochs.

The final suite then ran exactly three independent timing pairs on the saved
128-study cohort: seed 42, fold 0 of `all_5fold_v1`, batch eight, eight persistent
workers, prefetch two, one native/Torch thread, pinning enabled, 224 pixels,
0.67-mm spacing and 2× validation images. Each run returned the same 5,566
triplets and zero failures. A separate untimed selection audit matched all
16,698 selected files, record metadata, rectangles and RNG states over those
128 studies; it did not alter the six timing processes.

| Pair order | Reference loader seconds | Production loader seconds | Loader ratio | Complete-input ratio | CPU/study ratio |
|---|---:|---:|---:|---:|---:|
| Reference, production | 6.058 | 2.834 | 2.137× | 1.671× | 1.621× |
| Production, reference | 3.823 | 2.753 | 1.388× | 1.228× | 1.307× |
| Reference, production | 3.693 | 2.766 | 1.335× | 1.199× | 1.276× |
| Median paired ratio | | | **1.388×** | **1.228×** | **1.307×** |

Every loader ratio exceeded one, as did median complete-input and CPU/study
ratios, so the performance gate passed. Loader time consumes all batches after
iterator construction; complete-input time adds raw-index setup and iterator
construction. CPU/study sums parent CPU from setup through loading and worker
CPU, divided by 128. Full-process time includes imports, labels and shutdown;
its median paired ratio was 1.127×. Iterator construction took 0.212–0.225 s;
first-batch waits were 1.806–2.043 s for reference and 1.421–1.513 s for production.
Median sampled parent RSS stayed about 2.23 GB, while aggregate worker RSS rose
from 5.74 to 5.83 GB; aggregate RSS double-counts shared pages.

The first reference run is a retained outlier. OS page cache was uncontrolled
and the existing `crop140mm` training job was active, so these measurements do
not isolate the cause of its slower loading. All three pairs are included;
there were no replacement runs or new build cost. The 2× validation workload
does not establish 1× training throughput or full-dataset correctness.

Evidence, including the archived development plan, original/reference and
package audits, supplemental selection hashes, final source/configuration
snapshots, dependency versions, fixtures and all timing pairs, is under
[`outputs/profiling/dicom-integration-20260912`](../../outputs/profiling/dicom-integration-20260912).
The [final summary](../../outputs/profiling/dicom-integration-20260912/final/summary.json)
records the gate and unrounded ratios. It was produced by the `dicom-suite`
mode of the removed benchmark (`aec290d:scripts/adhoc/benchmark_dataloader_optimizations.py`,
invoked as `python -m scripts.adhoc.benchmark_dataloader_optimizations
outputs/profiling/dicom-reader-20260912/workload.yaml --cohort
outputs/profiling/dicom-reader-20260912/comparison-v1/timing_cohort.json --mode dicom-suite
--samples 128 --repeats 3 --audit-samples 8` from that commit), with the saved workload and
cohort rather than the current baseline's different split.
Recheck parser assumptions and edge coverage when pydicom or NumPy changes.
