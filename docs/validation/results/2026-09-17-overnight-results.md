# 2026-09-17 overnight results

**No hyperparameter change beat the control, weight averaging gained a little, and seeds alone
move the score by 0.003.** Queue `outputs/training/overnight-20260916T214520Z-t8k_kw2z`, launched on
the restored full-slice code with the EMA/SWA gate (uncommitted tree on top of `c5a7c11`; each run
keeps its `source_snapshot.zip`). All 15 runs finished (7 h 37 min). Fold 0 of `all_5fold_v3`,
seed 42 unless stated, one change per run against `00-control`.

## Controls and seed noise

| Run | Seed | Selected epoch | Macro AUC | Last epoch | Mean of last 3 | BCE | Gold AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `00-control` | 42 | 8 | 0.892995 | 0.892616 | 0.8926 | 0.44628 | 0.9419 |
| `01-control-seed-43` | 43 | 10 | 0.889837 | 0.889837 | 0.8893 | 0.44801 | 0.9339 |
| `02-control-seed-44` | 44 | 10 | 0.891627 | 0.891627 | 0.8910 | 0.44759 | 0.9300 |

- **The seed alone spans 0.0032** (mean 0.89149, standard deviation 0.0016); seed 42 was the best
  of three. The paired bootstrap of seed 43 against seed 42 is −0.0032 [−0.0065, +0.0001]: a
  bootstrap interval over studies does not cover run-to-run noise, so a single run whose interval
  just clears zero can still be seed luck.
- **Same-seed runs are reproducible:** the live curves of the EMA and SWA runs equal `00-control`
  to seven decimals, so comparisons at seed 42 share their random stream until the change acts.
  All single-change runs below use seed 42 and are paired with `00-control`.

## Single changes against `00-control`

Paired report-group bootstrap (2,000 draws), artifacts in `outputs/analysis/overnight-20260917/`.
Epochs are numbered from one.

| Run | Change | Selected epoch | Macro AUC | Δ [95%] | Mean of last 3 (Δ) | Training |
|---|---|---:|---:|---|---:|---:|
| `03-ema-0.999` | EMA, decay 0.999 | 10 | 0.893155 | +0.0001 [−0.0011, +0.0014] | 0.8918 (−0.0008) | 1,794 s |
| `04-swa-3` | SWA over the last 3 epochs | 9 | 0.893123 | +0.0001 [−0.0008, +0.0011] | 0.8930 (+0.0004) | 2,009 s |
| `10-clip-0.001` | intensity quantiles 0.001 / 0.999 | 7 | 0.892743 | −0.0003 [−0.0033, +0.0028] | 0.8914 (−0.0012) | 1,649 s |
| `13-grad-clip-5` | `gradient_clip_val: 5.0` | 7 | 0.892752 | −0.0003 [−0.0030, +0.0024] | 0.8911 (−0.0015) | 1,654 s |
| `09-warmup-0` | `warmup_epochs: 0` | 7 | 0.891492 | −0.0015 [−0.0046, +0.0016] | 0.8897 (−0.0029) | 1,663 s |
| `12-grad-clip-0.5` | `gradient_clip_val: 0.5` | 7 | 0.891405 | −0.0016 [−0.0044, +0.0010] | 0.8890 (−0.0036) | 1,667 s |
| `08-slower-head` | `head_lr: 3e-4` | 10 | 0.891266 | −0.0017 [−0.0047, +0.0011] | 0.8910 (−0.0016) | 1,672 s |
| `14-noise-0.3` | noise probability 0.3 | 10 | 0.891128 | −0.0019 [−0.0053, +0.0013] | 0.8902 (−0.0025) | 1,686 s |
| `11-clip-0.01` | intensity quantiles 0.01 / 0.99 | 10 | 0.890887 | −0.0021 [−0.0052, +0.0008] | 0.8902 (−0.0024) | 1,625 s |
| `06-beta2-0.99` | `beta2: 0.99` | 7 | 0.890544 | −0.0025 [−0.0056, +0.0004] | 0.8900 (−0.0027) | 1,616 s |
| `05-epochs-20` | 20 epochs | 12 | 0.890371 | −0.0027 [−0.0062, +0.0009] | 0.8870 (−0.0056) | 3,271 s |
| `07-affine-off` | affine probability 0 | 7 | 0.888930 | **−0.0041** [−0.0078, −0.0002] | 0.8873 (−0.0053) | 1,574 s |

The control trained in 1,641 s with an 18.95 GiB peak; no run exceeded 19.5 GiB.

- **Affine augmentation matters:** turning it off is the only interval that excludes zero, and
  the stronger affine of 2026-09-15 also lost (−0.0024), so the current strength sits near its
  best.
- **Twenty epochs overfit:** the run peaked at epoch 12 and fell to 0.8866 by epoch 20, below the
  ten-epoch control. Ten epochs stay.
- **Everything else ties or loses** within seed noise; none gained.

## Weight averaging

Averaged versus live weights inside the same run (the live weights are the control's):

| Run | Averaged best | Live best | Averaged last | Live last | BCE (averaged) | Extra time |
|---|---:|---:|---:|---:|---:|---:|
| `03-ema-0.999` | 0.893155 | 0.892995 | 0.893155 | 0.892616 | 0.44562 | +153 s (second validation pass) |
| `04-swa-3` | 0.893123 | 0.892995 | 0.893084 | 0.892616 | 0.44496 | +369 s, of which 206 s BatchNorm refreshes |

- Both add about +0.00015 at the best epoch and +0.0005 at the last epoch, and both lower BCE. The
  gains are small against seed noise but come without extra training and make the final epoch
  as good as the best one.
- EMA's mean over the last three epochs is lower than the control's because the average still
  lags at epochs 8–9; SWA's is higher (+0.0004).
- The EMA checkpoint scored **0.929** on the public leaderboard, below the champion (0.936) and
  `00-baseline` (0.934) ([submission](../../inference/submissions/2026-09-17-ema-submission.md)).
  The seed-42 control, whose weights are that run's live weights, scored **0.928**
  ([submission](../../inference/submissions/2026-09-17-seed-control-submissions.md)), so EMA gained
  +0.001 publicly. Seeds 43 and 44 both scored **0.931**, so the three controls' public order is
  the reverse of their fold-0 order; with `00-baseline` (0.934) the same recipe trained four times
  spans 0.928–0.934 publicly. Fold-0 differences of this size do not predict public order, and the
  seed also changes which slices are scored (evaluation tie-breaks between repeated series; fixed
  on 2026-09-17, see the [re-score with the fixed ranking](../../data-preparation/physical-sampling/evaluation-tie-break-investigation.md#re-score-of-the-2026-09-17-checkpoints)).
