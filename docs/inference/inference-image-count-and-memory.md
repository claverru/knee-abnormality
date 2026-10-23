# Inference image count and memory

Dated evidence from 2026-09-11: the baseline4 comparison favoured **2× image limits with batch
size 2**, and exports and repository test prediction defaulted to 2× (validation keeps its saved
validation multiplier). Each bundle freezes its exported density. Baseline4's saved validation
already used 2×. Later submissions moved to ×4 (2026-09-18), ×6 and ×8; those settings and their
temperatures are in the [ledger](submissions/ledger.md), the choice of ×4 in
[temperature and count submissions](../validation/results/pooling-and-attention/2026-09-18-temperature-and-count-submissions.md).

| Image multiplier | Inference batch size | Public score | Submission | Submission to observed completion |
|---|---:|---:|---|---|
| 1× | 8 | 0.910 | `56167669` | 23m 41s |
| **2×** | **2** | **0.913** | `56169438` | 34m 17s |
| 3× | 2 | 0.912 | `56169445` | 28m 12s |

All three use baseline4's selected fold-0 ReXNet-200 epoch-7 checkpoint,
`outputs/training/baseline4/fold_0/best-epoch=07-val_macro_auc=0.87251.ckpt` (deleted on 2026-09-16), with
SHA-256 `002778da78be2bb535618f8cb307ee631b94ca561a47e5f691964d879f0d276d`.
The 2× and 3× submissions share batch size 2; the earlier 1× submission used
batch size 8. These public scores favoured 2× for single-fold iteration at the time.
The intervals include queue and evaluation time plus up to one 30-second
polling interval, so they do not isolate model inference. That 3× was the faster
run is nonetheless likely: the efficiency leaderboard showed the 3× submission
over the higher-scoring 2× one while both were our two best public scores and so
most likely both automatically selected, which requires Kaggle's measured
runtime for 3× to be at least 71 s below the 2× run's
([efficiency leaderboard](efficiency-leaderboard.md#each-pick-is-a-measurement-of-our-hidden-runtimes)).

On 2026-09-11, a bounded FP32 forward profile on the local RTX 3090 used two
studies selected for high available slice counts at 3×. No training was run.
Artifact: `outputs/profiling/baseline4-inference-vram.json`, tied to baseline4's
selected epoch-7 checkpoint and its saved config. Values are GiB:

| Image multiplier | Batch 2 allocated peak | Batch 2 reserved peak | Rough batch 8 allocated peak |
|---|---:|---:|---:|
| 1× | 1.56 | 1.91 | 6.07 |
| 2× | 3.07 | 3.75 | 12.13 |
| 3× | 4.24 | 5.18 | 16.80 |

Batch-8 values extrapolate batch-dependent memory linearly after subtracting
model parameters; they are estimates, not measured batch-8 runs. Batch 8 is too
risky at 3× on a 16 GB T4, and batch 2 was used for inference then. Different GPU
workspaces and hidden-study composition can change the peaks, so these are
planning measurements rather than a worst-case guarantee.

Both comparisons completed on 2026-09-11 after running as detached user services:

| Density | Bundle under `outputs/submission/exports/` | User service | Workflow record under `outputs/submission/records/` |
|---|---|---|---|
| 2× | `baseline4-fold0-2x-v2` | `knee-baseline4-2x-submit-v2` | `baseline4-fold0-2x-20260911-v2.json` |
| 3× | `baseline4-fold0-3x-v2` | `knee-baseline4-3x-submit-v2` | `baseline4-fold0-3x-20260911-v2.json` |

The records and matching `-runner.log` files preserve completed stages and scores. Notebook IDs
are `claverru/knee-baseline4-2x-inference` and
`claverru/knee-baseline4-3x-inference`; each runner automatically submitted its
successful saved notebook version and recorded the score. The first 2× v1
workflow was stopped after dataset upload, before any notebook or competition
submission, to correct config-envelope loading. Its record is marked superseded.
