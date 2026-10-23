# Inference loading speedup: timeline

The inference loading investigation ran from 2026-09-17 to the end of the competition. Each
measured step has its own dated profile in this folder; this page orders them and records what
ended in main. Changes measured as regressions or ruled out are in
[closed and rejected loading changes](loading-changes-rejected.md), the measurement protocol in
[how to measure a loading change](loading-measurement-rules.md), and candidates never built in the
[backlog](../../backlog/untried-ideas.md#inference-loading).

## What ended in main (2026-10-23)

The loader buffer pool, the early worker exit, the startup page pre-read, exported checkpoints
that keep only the network, and convolution and linear weights stored in FP16. Exports choose
`--backbone graph` (default: the backbone compiled inside its CUDA graph, with the compile cache
the submission job builds on Kaggle) or `--backbone eager` (for ensembles whose graph pools do not
fit); both hold the backbone in FP16 on the GPU (commits `22b5043`, `5bb5cd2`). Compiled graphs fit
a 20-member ensemble that plain graphs could not
([compile](2026-10-19-torch-compile-backbone.md#a-graph-sized-ensemble)); the FP16 backbone took
another 0.8–1.0 GiB off that ensemble's peak, with predictions within 0.002. Never done: a
validation-fold score with the buffer pool, and re-recording `kaggle_startup_pages.json` after a
Kaggle image change.

## Steps, newest first

- **2026-10-20, compiled backbone** (adopted on the `inference-speed` branch, later main): the
  backbone compiled before its CUDA-graph capture was 13–25% faster per batch on every family; S156
  ×4 worker wall −30 to −42 s in paired runs, S167 −34 s per session over three fresh sessions per
  arm ([compile](2026-10-19-torch-compile-backbone.md)).
- **2026-10-19, branch `inference-speed`:** buffer pool, early worker exit, startup page pre-read
  and network-only checkpoints. S156 ×4 without TensorRT ran 318.9 s against 332.9 s, with the
  model ready 7.6 s earlier; smaller bundles also start their Kaggle session sooner. TensorRT was
  removed (its runtime delays the session start) and the script kernel rejected (about 40 s slower
  per session) ([speed pass](2026-10-18-inference-speed-second-pass.md),
  [startup](2026-10-19-bundle-size-and-session-startup.md)).
- **2026-10-17, buffer pool:** loader workers reuse their pixel buffers from a capped
  shared-memory pool: S156 TensorRT ×4 on Kaggle 297.6 → 268.9 s with near-equal predictions;
  direct reads were dropped ([profile](2026-10-17-loader-buffer-pool.md)). Not hashed, so existing
  checkpoints take it. The same reuse was never applied to training loaders.
- **2026-10-12, middle-slice preparation:** six bit-exact preparation changes for
  `repeated_middle` recipes: training steps 3.05 to 4.27 per second on the measured host, the
  Kaggle ×4 forward loop 4% shorter and peak GPU memory about 23% lower
  ([profile](2026-10-12-middle-slice-preparation-and-loader-speed.md)). They edit hashed
  preprocessing sources, so they took effect with the next trained checkpoints; `adjacent` slices
  keep the previous paths. The same central-window resampling on the GPU was never built.
- **2026-09-22, branch `inference-opt`:** the full Kaggle command at ×4 over 1,300 studies took
  about 270 s (273.3 s for the code it held) against 429.5 s for production code, with prepared
  images bit-identical to production
  ([profile](2026-09-22-kaggle-x4-startup-and-gpu-preparation.md)). The branch lived in the worktree
  `../knee-inference-opt`, based on `ab49830`, and was never merged or committed as such; its
  pieces reached main through the later steps.
  - *Retrain-free part:* torch-free coordinator, index built during imports, loader workers forked
    before model load at niceness 10, blocking CUDA sync, CUDA graph for the backbone chunk, the
    optional `wandb` import skipped, and the 2026-09-19 glibc tunables removed.
  - *Identity-changing part:* GPU image preparation, row-trimmed payload reads and reads straight
    into shared memory edit hashed preprocessing sources, so existing checkpoints (`k_05`,
    `oblique`) refused the new code. Outputs were bit-identical (Kaggle parity 0 mismatches over
    300 studies and 1,656 series), so a model trained on the new code sees exactly the inputs the
    old code would give it. Training on it still needed a local check: training pins memory and its
    batches carried raw 16-bit rows, several times the old image bytes.
- **2026-09-19:** support-limited float conversion, single-rank quantiles, shared-memory
  collation and glibc tunables shipped without a Kaggle run; measured on 2026-09-22, none paid on
  Kaggle, and the tunables were removed.
- **2026-09-18, historical loading reference:** `control-check` (fold 0, best epoch 6, validation
  macro AUC 0.88477, preprocessing identity `b8b6c639`) was the first checkpoint carrying the
  loading changes. It sat inside the ×1 cohort spread (median 0.88707, sd 0.00331; the cohort's own
  two controls 0.88380 and 0.88645); `outputs/submission/exports/control-check-x4-tau075` is
  exported from it at ×4 and τ 0.75. Later identity bumps mean `b8b6c639` no longer applies to
  fresh runs.
- **2026-09-18:** raw header parser, threaded I/O, worker-filtered index and the bit-exact CPU
  cluster, 470.5 → 350.8 s of worker wall
  ([profile](2026-09-18-kaggle-x4-profile-after-cpu-cluster.md)).
