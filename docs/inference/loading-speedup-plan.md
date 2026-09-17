# Loading speedup plan

Planned inference-speed changes, not yet implemented unless stated. They were measured on Kaggle's
two T4s with the champion; the [profiling record](2026-09-15-kaggle-inference-profiling.md) owns
the evidence. The submitted runtime already uses 16 loader workers and FP16 with fixed 64-image
channels-last backbone chunks ([submission](submissions/2026-09-15-fast-champion-submission.md), 0.936 public).
Loading dominates: DICOM slice-header reads over NFS are 72% of cold loading time.

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
and ship with a retrain. `data/sources.py` (the series index and known bad series and instances)
is outside the identity, so a change confined to it does not. Costs are from the sequential 100-study loading profile (cold, then warm OS cache);
file names are SOP instance UIDs, so every slice header must still be read to order slices.
Validate each change locally on CPU: every train series must produce identical `IndexedSeries`
records, selections and prepared tensors.

1. **Index only the worker's studies** (`build_series_index`, `prediction_dataset`). The index covers
   the whole split although each worker predicts half of it: pass the worker's study IDs and filter
   the series CSV before touching disk. Measured: 24–64 s per worker for 904 studies.
2. **List each series directory once** (`build_series_index`, `index_series`; series listing 3.1 s
   cold). Without the image-count parquet, as on Kaggle, the index globs `*.dcm` only to order
   candidates, and `index_series` globs the same directory again. Take counts from one listing,
   keeping the `(image_count, uid)` order and the `KNOWN_BAD_SERIES` adjustment.
3. **Minimal header parser** (`read_header`; slice headers 146 s cold, 35 s warm, 19,191 files).
   pydicom `read_partial` costs 1.8 ms of CPU per header, 46% of warm loading CPU. Read one fixed
   prefix per file (extended only if the pixel-data element lies beyond it) and scan explicit-VR
   little-endian elements for the tags `DicomHeader` uses: image position and orientation, pixel
   spacing, slice location, instance number, rows, columns, bits stored, pixel representation,
   rescale slope and intercept, transfer syntax and the pixel-data offset and length. Fall back to
   pydicom for any other encoding. Since identity `…reflection_v10` the header also yields 12
   series tags for the [series summary](../data-preparation/current-preparation.md#series-indexing-guards-and-exclusions)
   (about 65 µs per header through pydicom), which a raw parser must cover too.
4. **Threaded header reads within a series** (`index_series`). NFS latency is 111 s of the 146 s
   cold header cost: read the series' prefixes with a small thread pool, then parse in order. Raw
   reads scaled from 96 to 872 files/s with 16 threads; parsing holds the GIL, so this needs
   change 3 to pay off.
5. **Threaded frame reads within a series** (`decode_series`, `read_frame`; frame reads 25.5 s cold,
   12.5 s warm, 11,930 frames). Each selected slice reopens its file and reads the pixel payload
   serially; read the unique slices' payloads with the same thread pool (13 s of the cost is NFS
   latency), keeping failure handling per path.
6. **Resample a series in one pass** (`resample_crop`; 8.2 s). Every slice runs two scipy sparse
   products; slices of one series share shape and spacing, so apply the cached axis filters to the
   stacked frames at once, and apply rescale before stacking.
7. **Cheaper exact intensity bounds and normalization** (`_linear_quantiles`, `_normalize_pixels`;
   7.6 s together). Partitioning concatenates every valid resampled pixel of an acquisition.
   - *Measure selection alone first:* the only separate figure, 2.617 aggregate worker-seconds,
     comes from the historical profile before the
     [DICOM integration](../data-preparation/production-dicom-integration.md).
   - *Faster exact percentile selection:* production already gathers valid resampled float32
     values into contiguous scratch and partitions all unique floor/ceil ranks together. Compare a
     native multi-selection loop with NumPy partition, including scratch construction and
     interpolation. If still material, consider exact coarse-histogram assistance followed by
     selection within the rank-containing bins. Keep the population acquisition-wide over
     successfully decoded complete groups. Single-rank partition, full sorting, per-slice bounds
     and approximate quantiles are outside this comparison.
   - *Normalization:* fold the clip, shift, scale and rounding passes together.
8. **Profile geometry and triplet sampling** (`estimate_reference`, `project_reference`,
   `build_pool`, `select_triplets`; 7.0 s, unchanged when warm, so pure CPU) before choosing a
   change; scipy `truncnorm`, `logsumexp` and `linear_sum_assignment` run per slot.
9. **GPU resampling and normalization** only if CPU still limits after 3–8: decode raw frames on
   CPU, then crop-resample, clip and normalize on the mostly idle GPUs. This changes numerics.

Rough sequential estimate if 3–7 each remove most of their measured cost: 2.03 s per cold study
falls toward 0.5–0.6 s, after which the two T4s' forward pass, not loading, would set throughput.

## Not worth pursuing

Autocast FP16 with variable shapes, more than 16 loader workers, notebook pre-reading of DICOMs or
packages, a local bytecode cache, and TPU notebooks (not allowed for submission).
