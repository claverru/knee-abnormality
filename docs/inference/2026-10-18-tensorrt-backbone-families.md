# 2026-10-18 TensorRT backbone families

**Superseded 2026-10-19: all TensorRT code was removed from the repository.** The runtime a bundle
must carry delays the Kaggle session start by more than the engines save
([bundle size and startup](2026-10-19-bundle-size-and-session-startup.md)); the measurements below
stay as the record of what TensorRT gained per family.

Which backbone families gain from a TensorRT FP16 engine, with which export, and at what cost in
prediction drift. **ResNet-50, ConvNeXt, NextViT and FocalNet engines are adopted (each built from
the plain FP16 export at optimisation level 5, ConvNeXt with its channels-first rewrite);
EfficientViT is faster but drifts too much and is held out; MambaVision cannot be exported.**
The 706 MB TensorRT runtime a bundle then carries delays the Kaggle session start by minutes, more
than the engines save ([bundle size and startup](2026-10-19-bundle-size-and-session-startup.md)).

## Results

`profile_kaggle_inference` condition `<family>-trt` (one T4, 64 studies of `all_5fold_scanner_v2`,
×4, batch 4, one member of the bundle). Time per image is the backbone alone on a 64-image chunk;
time per batch is the whole model on real batches. The reference is the CUDA-graph backbone used
without an engine. One run per variant. Outputs under `outputs/profiling/kaggle-trt-*-20261018*`.

| Family (bundle member) | Variant | Backbone ms/image | Model s/batch | Prediction max, mean \|Δ\| | Adopted |
|---|---|---:|---:|---|---|
| ResNet-50 (S172) | CUDA graph | 1.098 | 1.108 | — | |
| | TensorRT, default level | 0.605 | 0.816 | 0.0225, 0.00023 | |
| | TensorRT, level 5 | 0.634 | 0.791 | 0.0207, 0.00027 | yes |
| ConvNeXt-Tiny (S156) | CUDA graph | 1.445 | 1.372 | — | |
| | channels-first, level 5 | 0.978 | 1.051 | 0.0056, 0.00021 | yes |
| NextViT-S (fold 2 run) | CUDA graph | 1.472 | 1.404 | — | |
| | TensorRT, level 5 | 0.984 | 1.047 | 0.0155, 0.00032 | yes |
| FocalNet-T SRF (S166) | CUDA graph | 1.683 | 1.500 | — | |
| | TensorRT, level 5 | 1.417 | 1.302 | 0.0068, 0.00014 | yes |
| EfficientViT-B2 (S172) | CUDA graph | 1.249 | 1.193 | — | |
| | TensorRT, default level | 0.828 | 0.992 | 0.0173, 0.00103 | |
| | TensorRT, level 5 | 1.017 | 0.935 | 0.0155, 0.00104 | no |

- **The gain per batch is 13–29%,** smaller than the backbone gain because the rest of the model,
  image preparation and the copy stay in PyTorch. At ×4 a batch saving of 0.3 s is about 50 s per
  GPU when the run is GPU-bound.
- **Level 5 is the right default.** It loses on time per image against the default level for
  ResNet-50 and EfficientViT but wins on the whole model per batch.
- **EfficientViT drifts.** Its features differ from PyTorch by 4.4–4.6% of their mean magnitude,
  against 0.3–0.6% for the other families, and its mean prediction moves ten times more than
  FocalNet's. Tracing the export under autocast instead of halving the weights did not change it
  (4.4%, mean 0.0011), so the cause is not the half-precision weights. The linear attention already
  runs in FP32 (`LiteMLA._attn` casts q, k, v). Not found yet; a layer-wise comparison is the next
  step.
- **MambaVision** runs `mamba-ssm`'s selective-scan CUDA kernels, which `torch.onnx.export` cannot
  trace; it would need a TensorRT plugin. Not attempted.

## Rejected variants

| Variant | Family | Result |
|---|---|---|
| Chunks of 128 images | ResNet-50, ConvNeXt, NextViT (autocast), EfficientViT (autocast) | within 1% per batch of 64-image chunks |
| Export traced under autocast, FP32 weights | ConvNeXt, NextViT, FocalNet, EfficientViT | ConvNeXt 1.040 against 1.051 s per batch with max \|Δ\| 0.0102 against 0.0056; NextViT 1.309 against 1.047 s; FocalNet 1.371 against 1.302 s, max 0.047 |
| 1×1 convolutions instead of permuted linears | FocalNet | 1.230 against 1.302 s per batch, but max \|Δ\| 0.040 against 0.0068 |
| INT8 with ModelOpt entropy calibration (2026-10-16) | ConvNeXt | 80.3 → 66.6 ms per chunk, prediction max \|Δ\| 0.072 (`outputs/profiling/kaggle-convnext-trt-variants-20261016`) |
| LayerNorm through permutes instead of a channel norm (2026-10-16) | ConvNeXt | 73.8 against 65.6 ms per chunk for the channel norm (`kaggle-convnext-trt-variants-b-20261016`) |

## Code (removed)

The module `knee_abnormality.inference.tensorrt_engines`, `scripts.tools.build_tensorrt_engines`
and the profiling tool's `backbone-tensorrt` cell existed until 2026-10-19 as uncommitted work on
the `inference-speed` branch and were never committed, so they cannot be recovered from history.
