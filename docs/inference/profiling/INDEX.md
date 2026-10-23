# Inference profiling records

Each document is a dated measurement on the code of its date; later records supersede earlier ones.

| Read when | Document |
|---|---|
| Pricing runtime in AUC: seconds per thousandth, whether TensorRT, fewer images, mirror TTA or ensembles pay | [2026-10-16 efficiency exchange rate](2026-10-16-efficiency-exchange-rate.md) |
| Reviewing loader, precision and startup profiles | [2026-09-15 Kaggle inference profiling](2026-09-15-kaggle-inference-profiling.md) |
| Inspecting threaded DICOM headers, frame loading and cold-start stages | [2026-09-18 Kaggle x4 profile after the loading changes](2026-09-18-kaggle-x4-profile-after-loading-changes.md) |
| Inspecting series filtering, threaded listings, loader ordering and CPU attribution | [2026-09-18 Kaggle x4 profile after the index changes](2026-09-18-kaggle-x4-profile-after-index-changes.md) |
| Inspecting preparation CPU changes, resampling and the I/O-thread sweep | [2026-09-18 Kaggle x4 profile after the CPU cluster](2026-09-18-kaggle-x4-profile-after-cpu-cluster.md) |
| Checking the inference runtime of 2026-09-22: startup, CUDA graph, GPU image preparation and row-trimmed reads | [2026-09-22 Kaggle x4 startup and GPU preparation](2026-09-22-kaggle-x4-startup-and-gpu-preparation.md) |
| Checking the middle-slice-only preparation, its parity passes, the training loader speed-up, the memory-bandwidth limit of a host and the Kaggle ×4 and ×2 timing pairs | [2026-10-12 middle-slice preparation and loader speed](2026-10-12-middle-slice-preparation-and-loader-speed.md) |
| Checking what a second ensemble member costs, the GPU loop per stage, the TensorRT backbone engine and its runtime, and the loader limit behind it | [2026-10-13 TensorRT backbone and two-model timing](2026-10-13-tensorrt-backbone-and-two-model-timing.md) |
| Checking loader CPU on Kaggle, the loader lab, the pooled shared-memory pixel buffers, the shared-memory exhaustion bug and why direct reads were dropped | [2026-10-17 loader buffer pool](2026-10-17-loader-buffer-pool.md) |
| Checking what TensorRT engines gained per backbone family (ResNet-50, ConvNeXt, NextViT, FocalNet, EfficientViT) and the rejected export variants, before TensorRT was removed | [2026-10-18 TensorRT backbone families](2026-10-18-tensorrt-backbone-families.md) |
| Checking why a bigger bundle dataset delays the Kaggle session start (S165's slow efficiency rank), the padding test, and why exported checkpoints keep only the network weights | [2026-10-19 bundle size and session startup](2026-10-19-bundle-size-and-session-startup.md) |
| Checking where S156 TensorRT ×4 spends its time after the buffer pool, the early worker exit, the startup page pre-read, the script kernel, and the rejected batch 8, processes per GPU, staging, streams and scheduling changes | [2026-10-18 inference speed, second pass](2026-10-18-inference-speed-second-pass.md) |
| Checking what torch.compile of the backbone gains per family and on ensembles, its compile time and cache, and why it does not pay end to end on one model | [2026-10-19 torch.compile backbone](2026-10-19-torch-compile-backbone.md) |
| Checking where cold-start DICOM loading time goes on Kaggle, what the shipped loading changes bought and how far the profile's own numbers can be trusted | [Loading cost attribution](loading-cost-attribution.md) |
| Checking which loading changes were built and measured as regressions or ruled out, before proposing a similar one | [Closed and rejected loading changes](loading-changes-rejected.md) |
| Designing a loading-speed measurement: the Kaggle probe protocol, parity checks and the traps that produced wrong answers | [How to measure a loading change](loading-measurement-rules.md) |
| Following the loading-speed work from 2026-09-17 to the end in order: what ended in main (buffer pool, early worker exit, page pre-read, network-only FP16 checkpoints, compiled or eager backbone), the `inference-opt` branch and its parity, `control-check` as the historical loading reference | [Loading speedup timeline](loading-speedup-timeline.md) |
