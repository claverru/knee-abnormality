# Inference loading speedup

Investigation started 2026-09-17 with the user. Candidates, closed regressions, the measurement
protocol and prior art live in [loading speedup](../backlog/loading-speedup/INDEX.md); each
measured step has a dated profile in [inference](../inference/INDEX.md#runtime-and-memory).

## State on 2026-09-22

**The full Kaggle command at ×4 over 1,300 studies is about 270 s on branch `inference-opt`
(273.3 s for the code it holds), against 429.5 s for production code**, with prepared images
bit-identical to production
([profile](../inference/2026-09-22-kaggle-x4-startup-and-gpu-preparation.md)). The branch lives in
the worktree `../knee-inference-opt`, based on `ab49830`, and is not merged or committed.

- **Retrain-free part:** torch-free coordinator, index built during imports, loader workers forked
  before model load at niceness 10, blocking CUDA sync, CUDA graph for the backbone chunk, the
  optional `wandb` import skipped, and the 2026-09-19 glibc tunables removed.
- **Identity-changing part:** GPU image preparation, row-trimmed payload reads and reads straight
  into shared memory edit hashed preprocessing sources, so existing checkpoints (`k_05`,
  `oblique`) refuse the new code. Outputs are bit-identical — Kaggle parity 0 mismatches over 300 studies and 1,656 series — so a model
  trained on the new code sees exactly the inputs HEAD would give it.

## Waiting on

- **The user's decision to adopt the branch**, which means the next submission needs a checkpoint
  trained on it.
- **A local training and dataloader check** before training on it: training still pins memory and
  its batches now carry raw 16-bit rows, several times the old image bytes. Needs the local GPU,
  so it waits for the user's go-ahead.

## Earlier milestones

- 2026-09-18: raw header parser, threaded I/O, worker-filtered index and the bit-exact CPU cluster,
  470.5 → 350.8 s of worker wall
  ([profile](../inference/2026-09-18-kaggle-x4-profile-after-cpu-cluster.md)).
- 2026-09-18, historical loading reference: `control-check` (fold 0, best
  epoch 6, validation macro AUC 0.88477, preprocessing identity `b8b6c639`) is the first
  checkpoint carrying the loading changes. It sits inside the ×1 cohort spread (median 0.88707,
  sd 0.00331; the cohort's own two controls are 0.88380 and 0.88645), and
  `outputs/submission/exports/control-check-x4-tau075` is exported from it at ×4 and τ 0.75.
  The relative sampler width adopted on 2026-09-19 and later bumps changed the preprocessing
  identity, so `b8b6c639` no longer applies to fresh runs.
- 2026-09-19: support-limited float conversion, single-rank quantiles, shared-memory collation and
  glibc tunables shipped without a Kaggle run. Measured on 2026-09-22: none paid on Kaggle, and the
  tunables were removed.
