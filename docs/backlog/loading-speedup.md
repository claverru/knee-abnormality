# Loading speedup

Everything about accelerating DICOM loading: the candidate changes, what has been measured, what is
closed, and what the prior art in past RSNA competitions contributes. Started 2026-09-17 with the
user; not yet implemented in production.

**Why:** DICOM slice-header reads over NFS are 72% of cold loading time, and loader wait is
114–227 s of a 342–428 s worker wall, so the loader and not the two T4s sets inference throughput
([profile](../inference/2026-09-15-kaggle-inference-profiling.md)). The submitted runtime already
uses 16 loader workers and FP16 with fixed 64-image channels-last backbone chunks
([submission](../inference/submissions/2026-09-15-fast-champion-submission.md), 0.936 public).

**Where it stands:** changes 2, 3 and 4 are **implemented** (2026-09-18) and ship with the next
retrain; changes 7 and 8 are closed as measured regressions, change 5 was probed without a verdict,
and changes 1, 6, 9 and 10 remain open. Next: poll the three uncollected kernels (only indexing is
informative), re-run frames with byte-matched blocks, and settle `INFERENCE_HEADER_THREADS` with a
Kaggle sweep against 16 loader workers.

## How these are measured

One self-contained Kaggle CPU script kernel per change, reproducing only the operation under test
rather than the whole pipeline, each auditing its candidate against a copy of the production
function (`probe-kaggle-loading` in [the catalog](../../scripts/adhoc/catalog.yaml)). Five CPU
sessions run at once. Artifacts are in `outputs/profiling/kaggle-loading-probes-20260917b/`.

- The competition root must be discovered by globbing for `train_series.csv`, as the existing
  profiler does; the slug path does not exist, which is what made the first run's four
  data-reading probes error.
- Cells use disjoint study blocks so each block is read cold once per session. That is right on
  Kaggle and wrong locally, where it measures disk I/O instead of the code under test — compare
  implementations on identical pre-warmed files when running locally.
- Pure-CPU probes (resample, quantiles) need no NFS and are answered locally.

## Measured

### Header parse and threading — changes 3 and 4 confirmed, and only together

Kaggle, 2026-09-17, `runs/header/header.json`, 1,200 files per cell:

| Implementation | Threads | files/s | ms/file |
|---|---:|---:|---:|
| pydicom | 1 | 103.7 | 9.646 |
| pydicom | 4 | 277.8 | 3.600 |
| pydicom | 16 | 253.1 | 3.951 |
| raw parser | 1 | 129.2 | 7.737 |
| raw parser | 4 | 389.3 | 2.569 |
| raw parser | 16 | 780.2 | 1.282 |
| raw parser | 16, reads split from parsing | 818.7 | 1.221 |
| raw parser | 64, reads split from parsing | 899.9 | 1.111 |

**8.7× the pydicom serial baseline.** pydicom plateaus at 253–278 files/s whatever the thread
count because parsing holds the GIL; the raw parser is what lets threads scale. It also spends
1.13 ms of CPU per header against pydicom's 2.47 ms, and CPU — not wall time — is the scarce
resource on Kaggle's four vCPUs. Audit: 600 files, no mismatches, no pydicom fallbacks, no prefix
extensions at the 12 KiB prefix. A local audit was exact on 20,000 files, and extend-and-retry was
verified by forcing a 600-byte prefix.

Local parse comparison, 2026-09-18, same 1,200 pre-warmed files, serial, two reps each:

| Implementation | ms/header | Versus pydicom |
|---|---:|---:|
| pydicom `read_header` | 0.315–0.333 | — |
| raw parser | 0.070–0.071 | 4.5× faster |
| `dicomsdl`, open plus the same 27 fields | 0.097–0.110 | 3× faster |
| `dicomsdl`, open only, no field read | 0.048–0.069 | — |

`dicomsdl` is a real alternative rather than a rejected one: 3× pydicom, 1.4× slower than the raw
parser, and it **does** expose the pixel-data offset and length `NativePixelDescriptor` needs
(`getDataElement("PixelData").offset()`). It would trade a parser we maintain for a wheel in the
[offline bundle](../inference/model-bundle-and-offline-environment.md), and it would need the
field-for-field equality audit the raw parser already has; only speed was compared.

Threading is strongly negative on warm local files — pydicom 3,036 → 325 files/s at four threads,
the raw parser 13,776 → 2,461, with CPU per file rising 13× and 7.6×. Threads pay only for I/O
latency they can hide, which is why they win on NFS and lose locally.

### Frame reads — change 5 probed without a verdict

The probe's cells read 900 frames each but 352.7–1,109.7 MB, so frames/s falls with threads (668.6
to 341.4) while MB/s rises (262.0 to 420.9); neither isolates the thread count. Re-run with
byte-matched blocks. Threaded reads were separately audited byte-identical to serial.

### Closed as measured regressions — changes 7 and 8

All four candidates were built, audited and measured slower. Do not retry these formulations.

| Candidate | Result |
|---|---|
| Stacked-sparse resample | bit-exact, **4.8× slower** |
| Batched-dense resample | **2.6× slower**, not bit-exact |
| Coarse-histogram quantiles | exact bounds, **1.67× slower** on real pixels |
| Fused clip/shift/scale/round | 3.679 ms against 3.658 ms per acquisition, not bit-exact |

**The lesson that outlives the changes:** the same histogram selection measured **1.87× faster on
synthetic gamma noise**. Real MRI slices carry a large near-zero background spike, so the bin
holding the 0.005 rank contains a big share of the values and the in-bin partition costs more than
one partition over everything. Exact bounds hid the regression entirely. Measure intensity and
percentile work on real pixels, never on a generator.

## Runtime changes (outside the preprocessing identity)

- **Main-thread warm-up** is implemented and ships with the next submission: the worker indexes in
  a background thread while the main thread loads the model and runs one zero chunk (first FP16
  batch 1.1–1.8 s instead of 26–34 s; worker wall 318 s versus 326 s).
- **Torch-free coordinator:** `predict.py` imports the model stack only to count GPUs and read
  study IDs. Count GPUs with `nvidia-smi -L` or `CUDA_VISIBLE_DEVICES`; the template and series
  CSVs are already readable without PyTorch through `knee_abnormality.data.sources`
  (`prediction_studies`, `resolve_raw_dir`), which the 2026-09-16 restructure moved out of the
  preprocessing identity. Coordinator imports take 35–55 s before `main`; at the efficiency
  formula's 32,400 s scale, 30 s is 0.00093 of the score.

## Loading code changes (identity changes, ship with a retrain)

These edit `data/dataset.py` and `data/preprocessing/`, so they change the preprocessing identity
and ship with a retrain; see [why a retrain](#why-these-need-a-retrain). `data/sources.py` (the
series index and known bad series and instances) is outside the identity, so a change confined to
it does not. Costs are from the sequential 100-study loading profile (cold, then warm OS cache);
file names are SOP instance UIDs, so every slice header must still be read to order slices.
Validate each change locally on CPU: every train series must produce identical `IndexedSeries`
records, selections and prepared tensors.

1. **Index only the worker's studies** (`build_series_index`, `prediction_dataset`). The index
   covers the whole split although each worker predicts half of it: pass the worker's study IDs and
   filter the series CSV before touching disk. Measured: 24–64 s per worker for 904 studies.
   *Unmeasured on Kaggle; the indexing kernel was pushed and never polled.*
2. **List each series directory once** (`build_series_index`, `index_series`; series listing 3.1 s
   cold). Without the image-count parquet, as on Kaggle, the index globs `*.dcm` only to order
   candidates, and `index_series` globs the same directory again. Take counts from one listing,
   keeping the `(image_count, uid)` order and the `KNOWN_BAD_SERIES` adjustment. *Implemented
   2026-09-18: `list_series` in `data/sources.py` does one `scandir`, and `SeriesCandidate.listing`
   carries it to `index_series`, which reuses it instead of globbing again. The listing is populated
   only when the image-count parquet is absent, so the saving is Kaggle-only; audited identical to
   globbing over 115 local series, throughput unmeasured.*
3. **Faster header parse** — *implemented 2026-09-18, see [above](#header-parse-and-threading--changes-3-and-4-confirmed-and-only-together)*
   (`read_header`; slice headers 146 s cold, 35 s warm, 19,191 files). Read one fixed 12 KiB prefix
   per file (extended only if the pixel-data element lies beyond it) and scan explicit-VR
   little-endian elements for the tags `DicomHeader` uses: image position and orientation, pixel
   spacing, slice location, instance number, rows, columns, bits stored, pixel representation,
   rescale slope and intercept, transfer syntax and the pixel-data offset and length. Fall back to
   pydicom for any other encoding. Since identity `…reflection_v10` the header also yields 12
   series tags for the
   [series summary](../data-preparation/current-preparation.md#series-indexing-guards-and-exclusions)
   (about 65 µs per header through pydicom), which the parser covers too.
4. **Threaded header reads within a series** (`index_series`). NFS latency is 111 s of the 146 s
   cold header cost: read the series' prefixes with a small thread pool, then parse in order. Raw
   byte reads scaled from 96 to 872 files/s with 16 threads. *Implemented 2026-09-18 behind
   `KneeStudyDataset(header_threads=...)`, defaulting to 1 so training and validation stay serial
   and only inference threads, at `INFERENCE_HEADER_THREADS = 4`. That default is a placeholder:
   local cold measurement showed no gain, and the Kaggle sweep against 16 loader worker processes on
   four vCPUs is still the open question.*
5. **Threaded frame reads within a series** (`decode_series`, `read_frame`; frame reads 25.5 s
   cold, 12.5 s warm, 11,930 frames). Each selected slice reopens its file and reads the pixel
   payload serially; read the unique slices' payloads with the same thread pool (13 s of the cost
   is NFS latency), keeping failure handling per path. *Probed without a verdict.*
6. **One file access per selected frame** (`read_frame`). Indexing already records the payload
   offset, yet `read_frame` opens the file again and `fstat`s it to revalidate size and mtime
   before seeking, so every selected slice is opened twice. Prior art takes the payload from the
   buffer it already holds, which we cannot copy directly: selection happens after indexing and
   whole-file pre-reading is already
   [rejected](../inference/2026-09-15-kaggle-inference-profiling.md#notebook-level-experiments). The
   change is narrower — let a short read replace the revalidating `fstat`, and with change 5 keep
   one open per selected file. *Measure before changing:* the entire cold−warm frame gap is 13 s of
   the 203 s per-study total, and the NFS attribute and page caches may already absorb the second
   open. A payload read must still fail closed when the file changed.
7. **Resample a series in one pass** — *closed, [measured slower](#closed-as-measured-regressions--changes-7-and-8)*
   (`resample_crop`; 8.2 s).
8. **Cheaper exact intensity bounds and normalization** — *closed, [measured slower](#closed-as-measured-regressions--changes-7-and-8)*
   (`_linear_quantiles`, `_normalize_pixels`; 7.6 s together).
9. **Profile geometry and triplet sampling** (`estimate_reference`, `project_reference`,
   `build_pool`, `select_triplets`; 7.0 s, unchanged when warm, so pure CPU) before choosing a
   change; scipy `truncnorm`, `logsumexp` and `linear_sum_assignment` run per slot. *No candidate
   exists.*
10. **GPU resampling and normalization** only if CPU still limits after 3–6 and 9: decode raw
    frames on CPU, then crop-resample, clip and normalize on the mostly idle GPUs. This changes
    numerics, and it is the only CPU-side lever left now that 7 and 8 are closed.

Rough sequential estimate if 3–6 each remove most of their measured cost: 2.03 s per cold study
falls toward 0.5–0.6 s, after which the two T4s' forward pass would set throughput. That estimate
is optimistic now that 7 and 8 are closed: about 15.8 s of the 202 s cold budget it counted is
unavailable, and the header changes carry the result almost alone.

## Why these need a retrain

The images do not change — that is what the audits establish. `preprocessing_identity`
(`src/knee_abnormality/identity.py`) folds the raw bytes of twelve source files, `dicom.py` among
them, into `sources_sha256`, and `validate_preprocessing_identity` refuses any checkpoint whose
recorded identity differs in any field. The hash cannot distinguish a provably equivalent parser
swap from a silent change to the pixels, so it treats any byte change as a change; a reformatted
comment would do the same. Existing checkpoints therefore keep running only against their archived
sources, and the speedup arrives with the next trained checkpoint. That is the reason to batch
changes 1–6 into one retrain rather than paying six.

## Training is not a beneficiary

Headers are parsed once per worker per series and cached for the run (`_indexed`,
`persistent_workers: true`), local disk has no NFS round trip to hide, and loader wait was under
0.06% of GPU step time
([threading profile](../data-preparation/threading-inside-dataset-loading.md#results-and-decision)) —
originally measured on the older repeated-RGB recipe. At local warm speeds the parser would save
roughly 3–4 minutes of worker CPU across an entire run, all of it already overlapped with the GPU.
The choice is not separable anyway: one `KneeStudyDataset` path serves training, validation and
inference, so whatever ships for inference is what training runs.

**Re-measured on the current recipe, 2026-09-18** (`compare-header-implementations` in
[the catalog](../../scripts/adhoc/catalog.yaml), 256 studies, batch 8, 16 workers, two passes).
Both arms ran on the same studies with the page cache dropped immediately before each, so pass 0
includes real disk reads (`outputs/profiling/header-cold-pydicom-20260918/`,
`header-cold-parser-20260918/`):

| Arm | Pass 0, cold | Pass 1, warm |
|---|---:|---:|
| pydicom | 37.5 studies/s, 6.83 s | 99.0 studies/s, 2.58 s |
| raw parser | 45.9 studies/s, 5.58 s | 109.3 studies/s, 2.34 s |
| raw parser, 4 header threads | 45.5 studies/s, 5.63 s | 104.5 studies/s, 2.45 s |

**1.22× on the first pass and nothing afterwards.** The parser saves 1.25 s of a 6.83 s cold pass.
Against the pass-1 steady state the epoch-zero penalty is 4.25 s for pydicom and 3.24 s for the
parser, so parsing is roughly a quarter of that penalty and cold disk plus index construction is the
rest. A run pre-warmed into RAM instead reported 56.4 against 88.7 studies/s, a 1.57× that exists
only because the I/O term had been removed
(`outputs/profiling/header-loader-256-20260918/`); prefer the cold protocol for any question about
where the bottleneck is, and the warm one only for isolating parser cost from I/O.

**Threading adds nothing locally even on cold files** (`header-cold-threaded4-20260918/`), which
extends the earlier warm-file result rather than contradicting it: prefetching each series'
prefixes on four threads landed within noise of the serial parser. Local NVMe has no latency worth
hiding, and 16 worker processes already keep the queue full. The Kaggle gain from change 4 stands on
NFS latency alone, so it cannot be rehearsed here.

The epoch-zero saving is paid once per worker per series and never again, because `_slice_index`
caches it and `persistent_workers` keeps the workers alive. Median steady-state batch time was
7.6–8.0 ms for a batch of eight in both arms, which is queue drain, not work: 16 workers stay far
ahead of the consumer. Audit mode compared every collated field on eight studies: identical prepared
batches, zero pydicom fallbacks, zero prefix extensions.

## Open questions

- **`INFERENCE_HEADER_THREADS` is unsettled.** It ships at 4 as a placeholder. Every thread
  measurement used one process on an idle machine, while production runs 16 loader workers on four
  vCPUs. The CPU saving transfers; the threaded
  wall-clock gain may not.
- **Change 5** needs byte-matched blocks before it means anything.
- **Changes 1 and 2** have a pushed, never-polled indexing kernel.
- **Change 6** has no probe.
- **Change 9** has no candidate.
- **`dicomsdl`** has no field-for-field equality audit.

## Not worth pursuing

Autocast FP16 with variable shapes, more than 16 loader workers, notebook pre-reading of DICOMs or
packages, a local bytecode cache, and TPU notebooks (not allowed for submission).

## Prior art in past RSNA DICOM competitions

Public solutions checked 2026-09-17. They show a technique working in a Kaggle notebook; none is a
measurement on our data.

- **`dicomsdl` for parsing and decoding.** The 2023 abdominal-trauma winner ships it as an offline
  Kaggle dataset ([repo](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)); the 2024
  lumbar-spine MRI solution reads `ImagePositionPatient` through `dicomsdl.open` and pixels through
  `pixelData(storedvalue=True)`
  ([repo](https://github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification)); the
  2022 cervical-spine solution uses it as well
  ([repo](https://github.com/louis-she/rsna-2022-public)). Feeds change 3.
- **A parallel metadata pass, then a parallel pixel pass.** The 2025 intracranial-aneurysm solution
  (MRI, MRA and CTA) replaced SimpleITK's series reader with `joblib.Parallel` over
  `dcmread(stop_before_pixels=True, force=True)` for pose, spacing and rescale, sorted slices by
  their projection on the slice normal, then ran a second parallel pass for pixels
  ([repo](https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution),
  `nnunetv2/dataset_conversion/kaggle_2025_rsna/official_data_to_nnunet.py`). Independent support
  for changes 4 and 5 on MRI.
- **Payload taken from bytes already read.** The breast-cancer winner and the 2022 cervical-spine
  solution locate the bitstream inside `PixelData` rather than opening or decoding again
  ([repo](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)). Feeds change 6.
- **Not applicable: GPU decoding.** DALI and nvJPEG2000 carry those competitions' JPEG 2000 and
  JPEG-LS images. Our frames are uncompressed explicit-VR little-endian, so there is nothing to
  offload.
- **Already done here.** Stopping before pixel data, recording the payload offset and sorting on a
  minimal tag set are standard in those solutions and are what `read_header` and `read_frame`
  already do.
- **Converting to PNG or npy for training**, which those solutions all do, is ruled out by our
  sampler: it needs full geometry and arbitrary slice positions at 0.625 mm.
