# 2026-10-19 Bundle size and session startup

Why S165 (S156 with a TensorRT engine) ranks as slow on the efficiency board although its code runs
faster. **A Kaggle session starts our notebook later the larger the attached bundle dataset is;
706 MB of extra files cost 1–4 minutes of startup, sometimes about 11. The TensorRT runtime is
such an extra, so TensorRT as shipped loses more time than it saves. Exported checkpoints now keep
only the network weights, halving their size.**

## The test

Kaggle 2×T4, the bundle's own submission notebook on the three visible test studies, no
submission, one notebook at a time. "Outside the code" is the session time from Kaggle reporting
`RUNNING` to `COMPLETE`, minus the notebook log's span; it is the time before our first output and
after our last. Polling adds about ±10 s. Outputs: `outputs/profiling/kaggle-startup-s165-s156-20261019`.

| Bundle (dataset, compressed) | Outside the code, three runs | Inference command |
|---|---|---|
| S156 (0.20 GiB) | 15, 8, 0 s; again −2, 1, 21 s | 41–46 s |
| S165 = S156 + engine + TensorRT runtime (0.57 GiB) | 72, 25, about 650 s | 37–41 s |
| S156 + the 706 MB TensorRT runtime as unused files | 148, 69, 231 s | 44–59 s |

- **The padding alone reproduces S165's delay,** so the cause is the dataset's size, not the
  engine or the TensorRT code. The file count barely differs (122 against 100 files).
- **The visible run hides it:** inside the notebook S165 is the faster bundle. The hidden rerun has
  to start a session with the same bundle, and the board's runtime for S165 behaves as if it
  counted that start.
- Unchanged: the hidden test is about 1,190 studies (each submission's output is about 250 KB), so
  S165's code itself runs about 5 minutes there, as on the 1,300 profiled studies.
- A notebook pushed seconds after its dataset is created stalls for many minutes too; that is a
  separate effect, avoided by waiting for the dataset to finish processing.

## What follows

- **TensorRT is removed.** Its runtime (639 MB of `libnvinfer` at minimum; Kaggle's image has no
  TensorRT) costs more startup than the engines save, about 35 s per S156 run
  ([families](2026-10-18-tensorrt-backbone-families.md)).
- **Every bundle pays for its size.** An exported checkpoint held the training-only anchor copies
  of the pretrained weights (`anchor_weight_*`, 105.9 of S156's 214.6 MiB) and the report-attribute
  heads. `export._inference_checkpoint` now keeps only the `model.` entries inference loads:
  S156's checkpoint goes from 214.8 to 108.6 MiB and loads strictly to identical tensors. A
  five-member ensemble sheds about 0.5 GB if its members carry anchors; S170's and S172's members
  carry none (their checkpoints are 100% network weights).
- **Convolution and linear weights are stored in FP16** (`checkpoint.half_precision_weights`, applied
  by export after the strict-load check and, for ensembles, after the temperature is baked in);
  loading casts them back to FP32. Everything else stays FP32. On the ResNet-50 gen1 fold-0 model
  over its 912 held-out studies (Kaggle, `fp16-b4-w8`): macro AUC 0.937892 against 0.937887 with
  FP32 weights; predictions differ by at most 0.0148 (mean 0.00006), so some weights are used
  outside autocast. It halves the network weights (S170 540 → 270 MB, S172 1.36 → 0.68 GB).
  Outputs: `outputs/profiling/kaggle-weights-fp{16,32}-resnet50-gen1-f0-20261019-*`.
