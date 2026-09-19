# 2026-09-19 ensemble submissions

**0.943, the project's best, from ten models with mirror TTA.** Five submissions, all at ×4 with
attention temperature 0.75, so they differ only in the bundle. The daily limit of five was reached
before the class-balanced-BCE checkpoint could be submitted.

| Submission | Members | TTA | Local fold-0 ×1 | Scoring | Public |
|---|---:|---|---:|---:|---:|
| `knee-b0-fold0-tau075-4x` | 1 b0 | off | 0.88635 | 393 s | 0.934 |
| `knee-rexnet-fold0-tau075-4x` | 1 ReXNet | off | 0.88367 | 423 s | 0.936 |
| `knee-ens-5xb0-4x` | 5 b0 | off | — | 483 s | 0.940 |
| `knee-ens-5xb0-tta-4x` | 5 b0 | **on** | — | 664 s | 0.941 |
| **`knee-ens-10x-tta-4x`** | 5 b0 + 5 ReXNet | **on** | — | 1782 s | **0.943** |

Bundles and members come from [the fold queue](../../validation/results/2026-09-19-fold-ensembles-and-sigma-sweep.md);
the multi-checkpoint bundle format is in
[model bundle and offline environment](../model-bundle-and-offline-environment.md).

## What the numbers say

- **Ensembling transfers.** The OOF pair predicted +0.0061 and the public board delivered +0.006
  (0.934 → 0.940). This is the first time a local ensemble estimate has matched its public gain.
- **B0 is 0.002 behind ReXNet for a quarter of the parameters** (0.934 against 0.936, same fold,
  same runtime settings). Tighter than the −0.0032 the fold-0 comparison suggested.
- **Local and public disagree on ordering again.** B0 scored *higher* than ReXNet locally (0.88635
  against 0.88367) and lower publicly. Fourth instance of a sub-0.005 local difference failing to
  rank, and the fold-0 seed lottery (0.88367–0.88848) is wider than the backbone effect.
- **Mirror TTA is worth +0.001 for +181 s**, against +0.00046 measured locally. Both are inside the
  ±0.006 public noise the [seed controls](2026-09-17-seed-control-submissions.md) established.
- **Ten mixed members with TTA buy +0.002 over five b0 with TTA, for 2.7× the runtime.**

## Runtime, and where the GPU starts to bind

`Scoring` is the submitted-to-scored window. It includes Kaggle queueing and is **not** the
official hidden-test execution time, so use it as a ratio between same-day runs, not as seconds.

- **The first extra members are nearly free, then they are not.** Going from one b0 to five costs
  +90 s (+23%), which is the loader-bound behaviour the [×4 profile](../2026-09-18-kaggle-x4-profile-after-index-changes.md)
  predicts: the GPU idles 57% of prediction time waiting for data, so early members hide inside that
  wait. That +90 s also proves the crossover was passed — had loading still dominated, five members
  would have cost nothing.
- **Past the crossover, cost is linear in forward passes.** TTA doubles them and added +181 s;
  ten members with TTA (20 passes) reached 1782 s.
- **Measured b0 forward cost is 0.77× ReXNet per image** (1.89 ms against 2.47 ms), from a
  dedicated profile at ×3 over 904 studies — not the 0.25× its multiply-adds imply nor the 0.66×
  its training wall implies. Model load is ~15.6 s per member for both backbones, so it scales with
  member count regardless of backbone.

## Prize split

The efficiency score is `AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400`, so 181 s of TTA costs
0.0056 of the runtime term to buy 0.001 of AUC, and the ten-model bundle's 1782 s costs 0.055.

- **Score entry:** `knee-ens-10x-tta-4x`, 0.943.
- **Efficiency entry:** `knee-ens-5xb0-4x`, 0.940 at 483 s — TTA off, b0 only.

## Not submitted: class-balanced BCE

`outputs/submission/exports/knee-balanced-bce-tau075-4x-20260919` is exported, ready and its
notebook runs clean, but `CreateCodeSubmission` returned 400 after the fifth submission of the day.
It scores [0.89528 on fold 0, +0.0116 over its control](../../validation/results/2026-09-19-balanced-bce-result.md)
— larger than anything ensembling bought — and should go first when the limit resets.

Its checkpoint is pinned to the `modeling/model.py` that existed while it trained, which has since
been reverted, so it must be exported from its own `source_snapshot.zip`. That is how this bundle
was built.

## Two failure modes worth remembering

- **`SaveKernel` returned 409 three times** while `scripts/pipeline/submit.py` was being rewritten
  in another session (its `message` field became a required `label`). The failures were the moving
  target, not Kaggle state; no notebook was created and no submission consumed.
- **A `kernels.get` denial right after a push is not diagnostic.** The first attempt reported
  `notebook_running` and failed 0.16 s later reading the kernel back; the kernel did not exist.
  Checking a known-good kernel's status is the quick way to separate an API problem from a
  per-kernel one.
