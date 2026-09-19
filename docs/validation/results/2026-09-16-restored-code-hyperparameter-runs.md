# 2026-09-16 restored-code hyperparameter runs

**Nothing beat `00-baseline`, and replicates show how noisy single runs are.** Five user-launched
fold-0 runs on the restored full-slice code (commit `c5a7c11`, preprocessing identity
`…reflection_v9`), each changing one setting of `configs/training.yaml`. There is no fresh
control on this code yet, so every delta below is against the original
[`00-baseline`](../../modeling/current-model.md) (0.892763), whose training random stream
differed.

## Runs

`scripts/tools/compare_fold_runs.py`, 2,000 paired report-group bootstrap draws, artifacts in
`outputs/analysis/<run>-vs-00-baseline/`.

| Run | Change | Selected epoch | Macro AUC | Δ vs `00-baseline` [95%] | Training | Peak CUDA |
|---|---|---:|---:|---|---:|---:|
| `min_lr_01` | `optimizer.minimum_lr_ratio: 0.1` | 7 | 0.892868 | +0.0001 [−0.0035, +0.0034] | 1,586 s | 20.3 GiB |
| `series_drop_03` | `model.series_dropout: 0.3` | 7 | 0.892945 | +0.0001 [−0.0033, +0.0034] | 1,503 s | 18.2 GiB |
| `series_drop_025` | `model.series_dropout: 0.25` | 7 | 0.890309 | −0.0025 [−0.0058, +0.0009] | 1,567 s | 20.5 GiB |
| `less_images` | `max_images` 15/15/12/12/12/12 (was 18/18/15/15/15/15) | 10 | 0.888750 | **−0.0041** [−0.0074, −0.0010] | 1,338 s | 16.6 GiB |
| `less_less_images` | `max_images` 12/12/9/9/9/9 | 10 | 0.887749 | **−0.0051** [−0.0085, −0.0014] | 1,058 s | 12.6 GiB |

`00-baseline` trained in 1,670 s with a 20.0 GiB peak. Epochs are numbered from one.

- **Fewer slices cost accuracy and save time.** 20% fewer slices lost 0.0041 and trained 20%
  faster; 35% fewer lost 0.0051 and trained 37% faster. The losses are broad (ACL −0.016 and
  lateral meniscus −0.014 in `less_images`; ACL −0.017 and fracture −0.016 in
  `less_less_images`). Both still peaked at their last epoch. This is a runtime lever for the
  [efficiency prize](../../inference/efficiency-leaderboard.md#the-exchange-rate), not an
  accuracy candidate.
- **Series dropout 0.25–0.3 and `min_lr 0.1` are ties.** The dip at 0.25 sits between two ties
  and inside its interval, so it reads as noise rather than a trend.
- **Meniscus targets fell in all three ties** (family mean −0.0056 to −0.0087) while the other
  families moved slightly up; with one seed each, this is not established.

## Run-to-run noise

The same configuration trained twice (the 2026-09-15 run on the pre-restore code and today's run)
gives:

| Setting | 2026-09-15 | 2026-09-16 | Difference |
|---|---:|---:|---:|
| `minimum_lr_ratio: 0.1` | 0.893171 | 0.892868 | −0.0003 (paired bootstrap −0.0004 [−0.0037, +0.0030]) |
| `series_dropout: 0.3` | 0.892271 | 0.892945 | +0.0007 |

Between the two `min_lr 0.1` runs, validation probabilities differ by 0.066 on average (per-target
Spearman 0.91–0.97). Single-run differences below about 0.001 are within this noise, and the
paired intervals of single runs span about ±0.0035. Hyperparameter decisions need replicates or
larger effects; the runs' disagreement is also what makes ensembling help.
