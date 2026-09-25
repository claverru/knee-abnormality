# 2026-09-18 overnight results

**Nothing beat the controls by more than the control spread, and two settings lost clearly.**
Seventeen runs on the night of 2026-09-17, fold 0 of `all_5fold_v3`, **validation at ×1**, each one
unseeded run with its seed in its config. Scores are therefore about 0.0035 below every earlier
fold-0 number, which was measured at ×2, and comparable only within this table.

Queue: `outputs/training/overnight-20260917T215351Z-ik2qer7m/` (`status.tsv`, `summary.tsv`,
`configs/`, `runs/`), launched with `scripts/tools/train_overnight.py`. Total 7 h 23 m.

## Controls

Two controls, same settings, different seeds: `00-control` 0.883805 (seed 3138278368) and
`control-2` 0.886451 (seed 397083050, run afterwards into the same queue directory). They span
**0.0026**, and their mean is 0.885128. `00-control` would have ranked third-worst of the
seventeen, so the `delta_best_vs_control` column of `summary.tsv` overstates every run by roughly
0.0013; the table below compares with the control mean instead.

## Runs

| Run | Change | Epoch | Macro AUC | vs control mean | Training | Peak |
|---|---|---:|---:|---:|---:|---:|
| `07-series-share-1.0` | `data.repeated_series_share: 1.0` | 8 | 0.889582 | +0.0044 | 24.5 m | 18.9 GiB |
| `06-series-share-0.5` | `data.repeated_series_share: 0.5` | 10 | 0.889427 | +0.0043 | 24.4 m | 18.8 GiB |
| `01-robust-center-20` | `data.robust_center_outlier_mm: 20.0` | 9 | 0.888833 | +0.0037 | 24.4 m | 19.0 GiB |
| `05-total-topup-32` | `data.study_triplet_total: 32` | 9 | 0.888712 | +0.0036 | 30.4 m | 22.1 GiB |
| `04-total-topup-26` | `data.study_triplet_total: 26` | 8 | 0.888476 | +0.0033 | 25.7 m | 19.6 GiB |
| `13-dropout-0.1` | `model.dropout: 0.1` | 10 | 0.888383 | +0.0032 | 24.4 m | 19.0 GiB |
| `14-series-dropout-0.15` | `model.series_dropout: 0.15` | 9 | 0.888092 | +0.0030 | 25.6 m | 19.6 GiB |
| `03-robust-center-30` | `data.robust_center_outlier_mm: 30.0` | 8 | 0.887791 | +0.0027 | 24.4 m | 19.1 GiB |
| `10-epochs-15` | `trainer.max_epochs: 15` | 11 | 0.887465 | +0.0023 | 36.5 m | 18.8 GiB |
| `11-affine-mild` | rotation 7°, translation 0.03, scale ±3% | 9 | 0.886674 | +0.0015 | 24.4 m | 18.9 GiB |
| `control-2` | none | 8 | 0.886451 | +0.0013 | 25.6 m | – |
| `08-gated-attention` | `model.gated_attention: true` | 10 | 0.886113 | +0.0010 | 24.5 m | 18.5 GiB |
| `02-robust-center-10` | `data.robust_center_outlier_mm: 10.0` | 10 | 0.886107 | +0.0010 | 24.5 m | 18.2 GiB |
| `15-warmup-2` | `optimizer.warmup_epochs: 2` | 10 | 0.885517 | +0.0004 | 24.4 m | 18.6 GiB |
| `09-residual-readout` | `model.residual_readout: true` | 9 | 0.884612 | −0.0005 | 24.4 m | 19.0 GiB |
| `00-control` | none | 9 | 0.883805 | −0.0013 | 24.4 m | 19.0 GiB |
| `12-backbone-lr-1e-4` | `optimizer.backbone_lr: 1e-4` | 10 | 0.882097 | −0.0030 | 24.4 m | 18.7 GiB |
| `16-backbone-eval` | `model.backbone_eval: true` | 10 | 0.875054 | −0.0100 | 22.9 m | 19.0 GiB |

## Reading

- **Two clear losers.** `backbone-eval` (−0.0100) and `backbone-lr-1e-4` (−0.0030) sit outside the
  control spread. `backbone-eval` had already been moved to the
  [endgame](../../backlog/endgame.md) as the gate for `checkpoint_backbone`; this settles it as a
  standalone setting.
- **The head changes from the review document do not help.** `gated-attention` (+0.0010) and
  `residual-readout` (−0.0005) land at or below the control mean, so neither is worth a follow-up.
- **`series-share` topped fold 0 and then lost on the leaderboard** (0.929 against 0.933 for the
  same recipe without it,
  [submissions](../../inference/submissions/2026-09-18-temperature-and-count-submissions.md)). The
  rule gives extra triplets to whichever plane–contrast happens to hold a repeated acquisition and
  takes them from the other types, which is an acquisition habit rather than evidence, and
  starving a type is the one thing that consistently costs accuracy
  ([type importance](2026-09-17-plane-contrast-importance.md)). Treated as rejected.
- **Everything between `robust-center-20` and `warmup-2` is one cluster** separated by less than
  the 0.0026 control spread. Nothing in it is established by one run.
- **Longer training is not the answer.** `epochs-15` peaked at epoch 11, scored at the cluster
  median and cost 50% more time, consistent with the earlier 20-epoch loss.

## Caveats

- One unseeded run per setting and two controls: the comparison resolves differences of roughly
  0.005, not 0.002.
- Seventeen runs against one control invites a winner by luck; the leaderboard result for
  `series-share` is the concrete example.
- The queue changed `validation_image_multiplier` to 1 for every run, including checkpoint
  selection, so selected epochs are not comparable with earlier runs either.
