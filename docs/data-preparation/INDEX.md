# Data preparation

| Read when | Document |
|---|---|
| Understanding current on-the-fly preparation: series header summary, within-series instance guard, non-knee and stray other-knee series exclusion, the optional robust study centre (`data.robust_center_outlier_mm`), 224-pixel/0.625-mm crops, overlapping triplets, clipping and retries | [Current preparation and overlapping triplets](current-preparation.md) |
| Downloading and extracting the image archive, resuming it, or checking the completed acquisition audit | [Downloading and extracting the images](downloading-and-extracting-images.md) |
| Checking the official data contract, contrast flags, DICOM encodings and the confirmed truncated pixel payloads | [Public data contract and decoding](public-data-contract-and-decoding.md) |
| Reviewing the reported historical cache recipe, data redundancy, resolution tradeoffs, slice ordering and open preprocessing questions | [Unverified preprocessing notes and open questions](unverified-preprocessing-notes.md) |
| Understanding why the crop moved from 0.80 to 0.67 to 0.625 mm, with header surveys and preview checks | [Field-of-view selection history](field-of-view-selection-history.md) |
| Generating or reading the augmentation preview viewer and its manifest | [Preview augmentations](preview-augmentations.md) |
| Understanding the production resampling and normalization path, the C-versus-NumPy decision and the earlier quantile profile | [Preprocessing implementation and profile](preprocessing-implementation-and-profile.md) |
| Reviewing the deferred and native DICOM reader prototype benchmark and its edge-case audit | [DICOM descriptor experiment](dicom-descriptor-experiment.md) |
| Checking the native DICOM reader's layout limits, fallback, equivalence evidence, memory bounds and performance gate | [Production DICOM integration](production-dicom-integration.md) |
| Measuring standalone loader throughput or deciding whether to thread slice decoding inside workers | [Threading inside Dataset loading](threading-inside-dataset-loading.md) |
| Understanding the implemented physical sampler, its verification and the metadata analysis behind it | [Physical sampling](physical-sampling/INDEX.md) |
| Reviewing the study reference fit, its screening thresholds and the rejected geometric-center sampler audit | [Accurate sampling centers](accurate-sampling-centers.md) |
| Understanding spatial-token visual validity and anchors, the retired token geometry validity, or the historical spatial-fusion mapping audit | [Spatial-token geometry and mapping audit](spatial-token-geometry-and-mapping-audit.md) |
| Checking study-reflection eligibility, its coordinate algebra, preview options and augmentation-only CUDA costs | [Reflection geometry and previews](reflection-geometry-and-previews.md) |
