# Data preparation

Evidence: unverified working notes, official/public sources checked on
2026-09-09, and the local acquisition audit on 2026-09-10; see
[provenance](INDEX.md#provenance).

The current baseline prepares pixels on the fly and uses 224-pixel centered
crops at 0.67 mm in-plane spacing. Earlier 0.80 mm inspections and the
historical cached-pipeline notes below are unverified evidence, not current
defaults.

## Downloading and extracting the images

Run [`scripts/tools/download_data.py`](../scripts/tools/download_data.py) from the project
root when ready. It uses the installed uv environment and existing Kaggle
credentials; competition rules must already be accepted. No additional package
or external unzip program is needed.

```bash
rtk proxy uv run --locked python -m scripts.tools.download_data
```

The destination defaults to `data/raw/`; `--data-dir /absolute/path` changes it.
The script downloads 128 MiB ZIP chunks and extracts in archive order, keeping
three completed chunks plus the chunk being fetched. It never stores the whole
ZIP. Python's [ZIP64 support](https://docs.python.org/3.12/library/zipfile.html)
handles the large archive. Loading its directory requires a few GiB of available
RAM, including Python's per-file metadata.

An authenticated HEAD request and the full remote ZIP directory established
this snapshot on 2026-09-09:

| Item | Size/count |
|---|---:|
| Archive | 265,018,885,676 bytes (265.02 GB) |
| Extracted files | 569,763,893,144 bytes (569.76 GB) |
| Files, including the five CSVs | 819,640 |
| ZIP entry compression | DEFLATE |
| Archive ETag | `3d99a46825629f617aec1e9900ba7afc` |

This inventory supersedes an earlier estimate of roughly 550 GB raw. Keeping
both the archive and extracted files would require about
835 GB before filesystem overhead. At inspection, the Windows backing drive
had about 770 GB free. WSL's virtual filesystem reported more; its capacity
does not establish physical host availability.
[Microsoft's WSL disk-space explanation](https://learn.microsoft.com/en-us/windows/wsl/disk-space).

The script checks both filesystems at startup and periodically during extraction.
It budgets remaining raw files, filesystem overhead, temporary state, and a
50 GiB free-space reserve (`--reserve-gib` changes the reserve). On WSL it uses
PowerShell to identify the physical backing drive and stops if that check fails.
Preprocessing caches need a separate budget after download; the reserve is not
an estimate of their size.

For an overnight run, start a persistent terminal, then run the command above:

```bash
rtk proxy tmux new-session -s knee-download
```

Detach with **Ctrl-b, then d**; reconnect with
`rtk proxy tmux attach-session -t knee-download`. Keep Windows awake and WSL
running: tmux survives terminal disconnection, not host sleep or shutdown.

### Progress, recovery and completion

- Logs and state live in `data/raw/.download/`, or `.download/` inside the chosen
  destination. Follow `download.log` for progress. A lock prevents simultaneous
  instances using that destination.
- Rerun the same command after interruption. Partial chunks resume at their
  saved byte offset. Requests must confirm the exact range and archive ETag;
  retries refresh signed URLs. URLs and credentials are not persisted in state.
  The default is 20 attempts per network operation, with bounded backoff.
- New files are published only after extraction reaches EOF and passes the ZIP
  checksum. Existing files, including the audited CSVs, must match; conflicting
  files cause a stop and are left untouched. Completed files use a SQLite
  journal; unchanged size/mtime allows them to be skipped on later runs.
- `--verify-existing` rereads checksums of all existing files. Use it when file
  integrity is in doubt, including after a host crash; it adds a full raw-data
  read. For a reported cache/checksum problem, preserve the affected file and
  diagnose it before removing private cached chunks or replacing any raw data.
- Success produces `COMPLETE` in the log, exit status zero and `complete.json`
  with the snapshot identity, file count and extracted byte count. Until that
  happens, acquisition remains incomplete. Keep `.download/` for efficient
  recovery; its files and raw data are ignored by Git.

### Completed acquisition and local verification

The full download completed on 2026-09-10 at 02:33 Europe/Madrid (00:33 UTC),
after about 2 hours 30 minutes, with no logged warnings or errors.
`data/raw/.download/complete.json`, `inventory.json`, and the SQLite journal
agree on the snapshot and totals above; the journal's integrity check passed.

The local audit at 10:01 Europe/Madrid checked every journaled file on disk:
all 819,640 were regular files with the expected sizes and recorded modification
times, totaling 569,763,893,144 bytes. The 819,078 training DICOMs cover all
4,407 studies and 24,371 metadata series; the 557 example-test DICOMs cover all
3 studies and 15 series. No metadata series were missing or unexpected. The
five CSV SHA-256 fingerprints still match the [original audit](metadata.md).

Post-download CRC checks passed for 135 files: the five CSVs, 64 training and
64 example-test DICOMs selected by reservoir sampling over sorted journal paths
with seed `20260910`, and the two reported malformed slices below. All 128
randomly sampled DICOMs decoded into finite pixel arrays. All 130 inspected
DICOM headers matched their study, series and instance paths; this sample used
Explicit VR Little Endian transfer syntax. The two targeted malformed slices
failed pixel decoding despite matching the archive CRCs.

`data/metadata/download-health.json` records the source identity, sample paths,
counts, checksums, coverage, and exact pixel errors. This verifies acquisition
completeness and sampled pixel health, not every DICOM's decodability. The audit
checked every file's size but did not reread all 570 GB for post-download CRCs;
ZIP member CRCs were checked during extraction. Interruption recovery has not
been exercised. About 183.6 GiB remained free on the Windows backing drive at
the post-download check; recheck before allocating preprocessing caches.

## Public data contract and decoding

The [local metadata audit](metadata.md) records verified CSV schemas,
identifiers, series profiles, and acquisition commands. Label coverage and
class distributions are recorded in [supervision](supervision.md#available-supervision).

The [official data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
defines study-level targets and series-level metadata. DICOMs have varying
intensities, orientations and resolutions, multiple transfer syntaxes including
JPEG Lossless/JPEG 2000, and only an allowlisted set of 86 metadata tags. Typical
series contain 20–45 slices, with a longer tail. Test files expose three example
studies; scoring replaces them with about 1,300 studies and their series. Target
prevalence is not guaranteed to match across train/public/private datasets.

A public metadata check found `Fluid_Sensitive == Fat_Suppression` for every
training series, while the host says these represent distinct characteristics
and need not remain equal in other data. Preserve both fields in series
selection and missing-slot handling.
[Metadata clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737312).

Two independently reported failures concern truncated pixel payloads, with only
half the bytes implied by the headers. The [original thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163)
identifies the following slice paths under `train_series/`:

```text
1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680/1.2.826.0.1.3680043.8.498.31160785238781353848727311763596115703/1.2.826.0.1.3680043.8.498.66460523372602171398246641420761543626.dcm
1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035/1.2.826.0.1.3680043.8.498.39396636671446532796538574315802928348/1.2.826.0.1.3680043.8.498.41947538458330624797291612781685746320.dcm
```

A [later report](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738708)
names the same studies. The local acquisition audit on 2026-09-10 confirmed
both failures with pydicom 3.0.2: the first listed file contains 173,056 pixel
bytes versus 346,112 declared; the second contains 409,600 versus 819,200.
Both files match their archive CRCs, supporting an upstream payload problem
rather than a transfer error. These are two confirmed cases, not an exhaustive
blacklist. Preserve the raw files and handle these failures during preprocessing.
Log decode failures by UID, distinguish missing codecs from malformed payloads,
and explicitly handle missing slices/series while preserving one prediction row
per test study.

For physical ordering, adjacent-triplet selection and cache tradeoffs, see the
[public geometry evidence](public-research.md#geometry-sampling-encoders-and-losses)
and [runtime lessons](public-research.md#data-access-and-runtime-lessons).

## Data shape and redundancy

- One inspected study contained 284 DICOM slices across five series, about
  436 MB. This is an example study, not an average.
- Studies contain multiple planes and contrasts, including acquisitions with
  and without fat suppression. Five canonical scan categories were described,
  but their names and mapping rules are not recorded.
- Slice spacing varies by acquisition: about 3 mm was mentioned initially,
  and about 0.5 mm for some axial acquisitions later. Neither is a universal
  spacing for the dataset.
- About 50 series were described as literal duplicates. Separately, about
  100 retakes were reported at matching nominal coordinates, acquired in
  another scan minutes later and not perfectly aligned. These observations
  should not be collapsed into one duplicate count.
- When multiple series map to the same canonical category, the reported
  approach deduplicates and samples from both rather than assuming a fixed
  number of input series.

## Intensity and geometry

| Component | Reported setting | Qualification |
|---|---|---|
| Training cache | `uint8` arrays | Neither PNGs nor repeated DICOM decoding during training; container format unspecified |
| Intensity clipping | Percentiles 0.5 and 99.5 before conversion | Bounds may have been computed per series and applied to every slice; not confirmed from code |
| Physical field of view | 140 mm center crop | Exact resampling/cropping order is unavailable |
| Usual training size | 384 pixels | Described together with physical-scale normalization |
| Target spacing | 0.38 mm/pixel | Reported separately the next day |
| Preparation time | About 30 minutes offline | Hardware and processing scope unspecified |

The settings are approximate descriptions, not a complete executable recipe.
In particular, `140 / 384` is about 0.365 mm/pixel, which does not exactly match
the separately reported 0.38. Confirm whether these reflect different runs,
rounding, or an additional crop/resize step before combining them.

## Field-of-view selection history

On 2026-09-10, the user authorized choosing a larger spacing for 224-pixel
inputs when 0.38 mm would crop useful information. The selected initial spacing
is **0.80 × 0.80 mm**, giving a **179.2 × 179.2 mm** centered field of view.
At 0.38 mm the same pixel dimensions cover only 85.12 mm per side. Increasing
spacing preserves more context at a lower spatial resolution; images smaller
than the target field are padded after resampling.

Baseline3 later selected **0.67 × 0.67 mm**, covering
**150.08 × 150.08 mm** at 224 pixels. This is the current default. The earlier
0.80 mm previews remain useful evidence about the cost of an 85.12 mm crop,
but they do not validate 0.67 mm as optimal.

A local header sample used eight shortest-selected acquisitions per type,
sampled with seed 42 after stable sorting. Across these 48 series, per-type
median image heights were about 155–175 mm, with sampled heights ranging from
136.8 to 200.0 mm. One sampled coronal image was 135 mm wide. Source image
extent alone does not establish which parts contain useful anatomy.

Six visual previews used the middle physical slice from the largest sampled
image extent in each type. Overlaid 85.12 mm and 179.2 mm centered windows show
that the former excludes substantial surrounding image content, while the
latter retains much more of the acquired field in these examples. This is a
limited geometry inspection, not an exhaustive anatomy audit or validation of
either spacing choice.

Local artifacts:

- `outputs/planning/spacing-audit.json`: source hash, versions, sample rule,
  all 48 headers' dimensions/spacing and the six preview paths.
- `outputs/planning/crop-field-of-view.png`: native-image comparison with
  both crop boundaries. Its display uses the 0.5th–99.5th percentiles only for
  visualization. Training now uses the same percentile levels jointly over
  selected images per series, as specified in the [first model plan](../plan.md).

The input is the [saved image-count table](metadata.md#images-per-series),
SHA-256 `75c5f7b77f69d6c522529d42cb5e3dddfa0eff2120f501a6b298d2a2a6a6ce55`.
The acquisition journal supplies paths without a corpus directory scan.

On 2026-09-10, the implemented deterministic path prepared example-test study
`1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191` in 0.71
seconds on the development host. Its four supported series contributed 126
slices, all decoded successfully, and the transient batch spanned the full
uint8 range. Visual inspection of the four middle prepared slices showed
centered axial, coronal and sagittal knee anatomy with useful surrounding
context. This checks one study and does not replace broader training-data review.

A second review prepared one shortest acquisition from each of the six training
types in 0.60 seconds, covering 114 selected slices without decode failures or
constant intensity ranges. Their middle slices retained centered knee anatomy
in every plane and both contrast groups. Both studies containing the confirmed
truncated payloads also completed: each bad header was logged and excluded,
while five usable series and 151 selected slices remained for each study.
These executions verify the known failure path and representative types, while
loader concurrency and rare unobserved failures still require runtime monitoring.

## Preview augmentations

Generate an offline comparison from the example-test DICOMs:

```bash
rtk proxy uv run --locked python -m scripts.tools.preview_augmentations configs/baseline.yaml --output-dir outputs/previews/triplet-augmentations-new
```

Open the generated `preview.html` in a browser. It embeds its
images and needs no server, network, checkpoint or GPU. The accompanying
`overview.png` shows the middle selected input for all six plane–contrast types,
with the original on the left and a forced augmentation on the right.

- **Example study:** start with all six types across examples, or inspect an
  individual study. The three example-test studies supply thirteen usable
  selected series and collectively cover all six types; individual studies
  have missing types, shown explicitly.
- **Input:** move each card's slider to compare the same triplet before
  and after augmentation. Its displayed position identifies the middle slice.
  Preparation uses the configured shortest usable
  acquisition, crop, per-series clipping and deterministic evaluation sampling.
- **Display:** inspect the three-channel composite or any of its three
  grayscale channels. The composite maps consecutive slices to red, green and
  blue; its colors show differences between adjacent slices. Legacy single-slice
  configs show grayscale and disable this selector.
- **Augmentation draw:** choose seeded affine, gamma/contrast and Gaussian-noise
  parameters. Geometry and scalar strengths are shared across an acquisition;
  noise pixels differ by slice.
- **Force enabled transforms:** checked by default so every configured family
  is visible. Uncheck it to respect each family's independent training
  probability. The PNG forces enabled families for the first draw.

The tool and model share `augmentation.py`. Images use the same fixed `[0, 1]`
display scale before checkpoint-specific normalization. Each displayed frame
is rounded to uint8 for PNG storage; training applies transforms in float32.
Test inference bypasses augmentation. Preview seeds use the CPU generator and
do not identify the exact CUDA random draws of a training run. The viewer does
not simulate whole-series dropout.

Regenerate with a fresh directory and optionally a different seed or number
of draws. `--study-uid` restricts the source to one UID from `test.csv`:

```bash
rtk proxy uv run --locked python -m scripts.tools.preview_augmentations configs/baseline.yaml --seed 100 --variants 6 --output-dir outputs/previews/augmentations-seed100
```

`manifest.json` records settings, CPU seeds, per-series parameters, sampled
application flags, study/type identities, selected positions, prepared-pixel
hashes, input/source hashes and dependency versions. The output also includes
the effective config and source snapshot. Existing directories are preserved.
Historical single-slice preview evidence remains under
`outputs/previews/augmentations/` and
`outputs/maintenance/augmentation-preview-20260910T211117Z/`. The
adjacent-triplet preview generated on 2026-09-11 is
[available here](../outputs/previews/triplet-augmentations/preview.html).
Its three example studies contain thirteen selected series, 69 complete
triplet inputs and 207 distinct selected slices. All six types are present,
with no reported preparation conditions. The manifest records input counts
and raw-slice counts separately. Its composite overview was visually inspected
and the viewer JavaScript passed a syntax check. Shared three-channel
augmentation also completed the bounded Lightning execution recorded in
[modeling](modeling.md#adjacent-triplet-sampling).

## Preprocessing implementation and profile

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
[`outputs/profiling/normalization-c-vs-numpy-20260912T000001Z`](../outputs/profiling/normalization-c-vs-numpy-20260912T000001Z).

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
[`outputs/profiling/preprocessing-investigation-20260911T205612Z-run2`](../outputs/profiling/preprocessing-investigation-20260911T205612Z-run2),
and the final NumPy production audit and raw paired timings are under
[`outputs/profiling/preprocessing-production-numpy-20260912T001501Z`](../outputs/profiling/preprocessing-production-numpy-20260912T001501Z).
The remaining DICOM reader and exact percentile work is tracked in
the [optimization document](../optimization.md).

### DICOM descriptor experiment

On 2026-09-12, the existing benchmark added `dicom-deferred` and `dicom-native`.
The native candidate is the preferred reader for a subsequent production change;
production still uses its existing DICOM reader. Both candidates carry metadata
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
[`outputs/profiling/dicom-reader-20260912`](../outputs/profiling/dicom-reader-20260912).
`comparison-v1/` contains the real DataLoader audit and timings;
`corrected-edge-audit.json` owns the edge coverage above. The first edge fixtures
retained a trailing padding element and therefore exercised fallback; they are
superseded by the corrected fixtures with explicit dispatch assertions. No
timing code changed for that correction. The starting source was already dirty;
the saved source snapshots and manifest hashes identify the implementation.
The [script catalog](../scripts/adhoc/catalog.yaml) records `dicom-suite` and
`dicom-audit` invocations. The current baseline uses `all_5fold_v3`; use the saved
`workload.yaml` with this saved cohort, or generate a new matched cohort when
benchmarking the current split.

### Earlier profile

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
The [historical profiling findings](modeling.md#batch-4-memory-fix)
summarize the corresponding model-side measurements.

Reproduce a bounded preprocessing profile through the
[profile-training catalog entry](../scripts/adhoc/catalog.yaml):

```bash
rtk proxy uv run --locked python -m scripts.adhoc.profile_training configs/baseline.yaml --mode preprocessing --output-dir outputs/profiling/preprocessing-new
```

## Threading inside Dataset loading

Production loading retains sequential preparation inside eight persistent
DataLoader worker processes. The isolated
[threading profiler](../scripts/adhoc/profile_threaded_loading.py) tested two to
four slice-decoding threads inside each worker while preserving physical order,
sampling RNG order, clipping and production error handling. The prototype is
technically viable, but the measured workload did not show a reliable speedup.
Keep it outside production unless a new storage or codec profile demonstrates a
loader bottleneck.

### Standalone loading measurement

Use standalone DataLoader throughput to compare loading implementations:
consume each CPU batch immediately and divide completed studies and selected
slices by elapsed wall time. This includes DICOM indexing, decoding, resampling,
cropping, joint clipping, collation and worker-to-parent transfer. It excludes
memory pinning, host-to-device transfer, GPU augmentation/normalization and
model execution. Training-step throughput answers the separate question of
whether the loader holds up the GPU; prefetching can hide loading differences.

The 2026-09-10 comparison used the previous single-slice RGB recipe and 512
fold-0 training studies from
`all_5fold_v1`, Steven v4 labels and seed 42: four maximum-load studies plus
508 sampled studies, then a fixed seeded permutation. It retains batch size
eight, 224-pixel crops, 0.8 mm spacing, `[0.005, 0.995]` clipping, physical
positions and slice limits `[17, 16, 15, 15, 15, 15]`. Each configuration runs
in a separate process, with persistent workers and prefetch factor two.
Two passes repeat epoch-zero random slice choices; the second reuses each
worker's header index. OS page cache is uncontrolled, so the first pass must
not be interpreted as a cold-disk measurement.

There are 64 batches per pass. The first 16 are excluded from the primary
throughput metric, leaving 384 studies and 28,351 slices per measured pass.
Full-pass wall time, including iterator startup/reset, is recorded separately.
The hardware is a Ryzen 9 9950X3D with 32 logical CPUs under Linux/WSL.
DataLoader workers and each explicit task thread use one PyTorch compute
thread; `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are
also set to one. The calling process uses 16 PyTorch threads.

Measure a configuration with the current recipe, using a fresh output directory:

```bash
rtk proxy env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.profile_threaded_loading configs/baseline.yaml --mode loader --workers 8 --decode-threads 1 --samples 512 --warmup-batches 16 --output-dir outputs/profiling/threading-loader-new
```

Artifacts are under `outputs/profiling/threading_20260910/`. Each run retains
configuration, study order, split/label hashes, dependency versions, source
snapshots, per-batch counts and timings. The eight-study real-input audit in
`audit/` found identical hashes for all collated tensor fields with one, two
and four threads; study IDs and failure records matched too. Those studies
had no preparation failures, so this does not establish runtime coverage of
corrupt-file fallback handling.

### Results and decision

Retain production's eight process workers with sequential slice preparation.
This slice-thread prototype does not demonstrate a reliable loading speedup
on the measured machine. Four-worker threading reduced measured throughput;
the apparent initial eight-worker benefit became small after repeating the
comparison in reverse order.

| Worker processes | Slice threads per worker | First pass, studies/s | Second pass, studies/s | Second pass, slices/s |
|---:|---:|---:|---:|---:|
| 8 | 1, production preparation | 27.63 | 29.17 | 2,154 |
| 4 | 1, production preparation | 19.75 | 27.83 | 2,054 |
| 4 | 2 | 18.90 | 21.49 | 1,586 |
| 4 | 4 | 10.36 | 10.96 | 809 |
| 8 | 2 | 30.71 | 34.40 | 2,540 |
| 8, repeat | 2 | 30.21 | 35.05 | 2,588 |
| 8, repeat | 1, production preparation | 31.85 | 34.51 | 2,548 |

Rows follow execution order. On the repeated eight-worker comparison, two
threads improve second-pass throughput by only 1.57%. The process-only
baseline itself ranges from 29.17 to 34.51 studies/s between runs, so the
initial apparent 17.9% threading advantage is not a robust improvement.
These are bounded timings, not confidence intervals or full-epoch projections;
OS cache state and scheduling were not controlled. All seven loader runs
have identical source snapshots, label provenance, study order and per-batch
selected slice counts. The [machine-readable summary](../outputs/profiling/threading_20260910/summary.json)
links each configuration to its recorded timings. No production source changed.

Two completed, separate GPU profiles used the then-current repeated-RGB
ResNeXt model, gamma/contrast augmentation, series dropout, BF16 and
checkpointing on an RTX 3090. Each used 256 studies twice, batch eight, seeded random weights and
constant optimizer learning rates, with eight warmup batches per pass. Eight
workers without slice threads reached 14.25 studies/s on the second pass;
four workers with two threads reached 14.09. Loader wait was below 0.06%
in both, and all 64 recorded batch losses matched exactly. These results
diagnose a loader that keeps up with the measured GPU steps; they do not
measure standalone loading capacity or validation quality. Artifacts are
`gpu_w8_t1/` and `gpu_w4_t2/`. The partial `gpu_w4_t4/` run was interrupted
and is excluded from comparisons.

Threading remains technically usable inside `__getitem__`. Revisit this
specific design if storage, decoding, image coverage or GPU consumption
changes enough to make loading a bottleneck. These results do not establish
the performance of other threading granularities or storage systems.

## Resolution tradeoffs

Reported on 2026-09-08, relative to the usual 384-pixel setup:

| Resolution | Reported AUC decrease | Other observation |
|---|---|---|
| 320 | About 0.001 | Suggested as a practical smaller setting |
| 292 | About 0.002 | No separate throughput figure |
| 224 | About 0.003 | About 8× throughput; suggested for fast iteration |

A roughly 440-pixel, 160 mm configuration reportedly performed similarly to
the usual setup. The score's evaluation split, exact model, and benchmark
conditions were not specified, so these are experiment leads rather than
guaranteed tradeoffs.

## Ordering and unanswered preprocessing questions

The maintained pipeline sorts by the DICOM orientation/location projection,
falling back to `SliceLocation`, then index metadata. The current baseline uses
`data.position_encoding: physical_mm` with `position_reference: center`:
float32 millimeter positions relative to the midpoint of the acquisition's
physical extent. This reference remains fixed when different slices are sampled.
Physical mode rejects candidates whose sampled slices only have index metadata,
using the existing alternative-acquisition fallback. Legacy configs default to
`index`, preserving baseline1's ordinal positions. The [correction record](baseline-review.md#implemented-corrections)
documents compatibility and real-data execution.

The following uncertainties concern the historical pipeline:

- The notes warn against ordering by sequence IDs and indicate use of patient
  orientation/coordinate information. Exact tags, sorting logic, and handling
  of inconsistent geometry are unavailable.
- Whether the pipeline applies rescale slope/intercept is unresolved.
- Photometric inversion was raised during the DICOM review; the actual
  pipeline's handling was not established.
- The exact intensity mapping into `uint8`, padding policy, interpolation,
  series classification, and duplicate-detection criteria are missing.

See [modeling](modeling.md#window-sampling) for slice-window selection and
[inference](inference-efficiency.md#runtime-leads) for reducing decoding work.
