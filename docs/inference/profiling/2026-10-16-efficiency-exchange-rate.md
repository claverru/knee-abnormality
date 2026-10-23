# 2026-10-16 Efficiency exchange rate and what our options pay

What one second of hidden-test runtime is worth in AUC on the efficiency leaderboard on 2026-10-16, and
which of our measured speed and accuracy options pay at that rate. **One thousandth of AUC is worth
69.2 s of runtime, one minute costs 0.00087, and the rate is pinned to within about 1 s. Only the
TensorRT backbone pays among the measured options; fewer images, mirror TTA and every ensemble
since S152 lose more AUC than their runtime is worth.**

The formula, how the board picks a team's row and how its history is read are in
[efficiency leaderboard](../efficiency-leaderboard.md). Data: efficiency leaderboard versions 49–77
(v77 published 2026-10-15, 5,955 teams, data through 2026-10-14), public leaderboard of
2026-10-16 11:51 UTC (6,081 teams), and the Kaggle profiles of 2026-10-16.

## The rate

`Efficiency = −AUC / K + RuntimeSeconds / 32400` with `K = maxAUC − Benchmark`. With the public top
at 0.968 and the benchmark at 0.5, `K = 0.468`:

| Quantity | Worth |
|---|---|
| 0.001 of AUC | 69.2 s |
| 0.003 of AUC | 207.7 s |
| One minute of runtime | 0.00087 of AUC |
| Ten minutes of runtime | 0.0087 of AUC |

**The rate is pinned.** A team ranked above another can trail it in AUC by less than `K`, so the
deepest score inversion bounds `K` from below: 0.463 in versions 69 and 74 (top 0.964–0.965), 0.459
in version 77 (top 0.967). The benchmark therefore sits at or below about 0.502 and the rate between
69.2 s and 70.0 s per thousandth while the top stays near 0.968. Since 2026-09-19 the rate has
moved from 70.9 s only because the top rose from 0.957; a private top 0.005 away from the public
one moves it by under 1 s.

**Displayed scores are rounded.** Every comparison of two shown scores carries up to ±0.001 of
rounding, ±69 s at this rate, so one-thousandth gaps between rows say little about runtimes.

## The seconds are hidden-test seconds

`RuntimeSeconds` is the hidden rerun over about 1,300 studies, the size of our Kaggle profiles
(1,300 fold studies, two T4s), so profile seconds should transfer one to one. One pick disagrees:
version 76 showed S152 (0.963, ConvNeXt-Tiny + ResNet-50) over S148 (0.962, the same ResNet-50
alone), which requires S152's hidden runtime to exceed S148's by less than 69–132 s depending on
rounding, while their profiles differ by 208 s (510.7 s against 302.5 s,
[2026-10-13 profile](2026-10-13-tensorrt-backbone-and-two-model-timing.md#whole-runs)). Until that is
explained, profile differences may overstate hidden-runtime differences; submit-to-scored times
cannot settle it (S148 576 s, S152 574 s, with at least 128 s of queue noise).

## What our options pay

Runtime changes from the 2026-10-16 Kaggle profiles of S156 (one ConvNeXt-Tiny student, ×4,
τ 0.85, FP16, batch 4, 8 workers; one run per row) unless marked; AUC changes are the measured ones,
none of them on ConvNeXt, which has no multiplier or TTA comparison.

| Option | Runtime change | Worth at 69.2 s | Measured AUC change | Pays? |
|---|---:|---:|---|---|
| TensorRT backbone, S156 ×4 (level 5, channels-first) | −35.3 s (332.9 → 297.6) | +0.0005 | none expected; not yet scored on a fold | yes, if parity holds |
| ×4 → ×3 on TensorRT | −5.4 s (297.6 → 292.2) | +0.0001 | ×3 vs ×4 tie (ranked-filtered, 2026-09-18) | no |
| ×4 → ×2 on TensorRT | −23.7 s (→ 273.9) | +0.0003 | ×2 vs ×3 −0.0024 (2026-09-18) | no |
| ×4 → ×1 on TensorRT | −80.8 s (→ 216.8) | +0.0012 | ResNet-50 ×1 vs ×4 −0.0077 to −0.0085 on three folds (2026-10-12) | no, about −0.007 net |
| Mirror TTA | +181 s submit-to-scored (S35 → S36) | −0.0026 | +0.0002 to +0.0005 | no |
| Second member, S148 → S152 | +208 s (profiles, 2026-10-13) | −0.0030 | +0.001 public | no |
| Five ConvNeXt + TTA (S155) instead of the student (S156) | +1,600 s submit-to-scored (2,024 against 424) | −0.023 | 0.964 both | no |
| Fifteen members + TTA (S161, S163) instead of S156 | about +4,330 s submit-to-scored | −0.063 | 0.964 all three | no |

- **On TensorRT, S156 is loader-bound at every multiplier** (14–56 s of loader wait per GPU), which
  is why ×2 to ×4 cost so little runtime and dropping images buys almost nothing. Further GPU
  speedups would not move it; the next seconds are in the CPU loaders.
- **A slower submission pays only above 0.00087 of AUC per extra minute.** No ensemble or TTA we
  have measured comes close; a single distilled model is the efficiency entry.
- Submit-to-scored times include queue and scoring and carry at least 128 s of noise, enough to
  blur the TTA row but not the ensemble rows.

## Our position

Version 77 ranks us first with S156 (0.964, submitted 2026-10-14 16:03). Our other 0.964
submissions, S155 and S154, are slower, as the pick requires. Seventeen other teams hold 0.964 or
better publicly; of the nine above 0.964 the best placed is 19th, and the 0.968 leader is 3,909th.

- **Against a team scoring `a`, we stay ahead while our runtime exceeds theirs by less than
  `(0.964 − a) × 69,200 s`**, ±69 s of rounding: 208 s against a 0.961, nothing against an equal
  score.
- **The second-placed team shows 0.961 with a public best of 0.964**, so its 0.964 runs at least
  about 208 s slower than its 0.961. A faster 0.964 from that team would be decided on runtime
  alone.
- The board ranks public scores now; the prize ranks private scores, so these margins are
  indicative only.

**S165 tests the TensorRT row on the hidden set.** S165 (2026-10-16 12:18 UTC) is S156's bundle
with the level-5, channels-first TensorRT engine and its runtime
(`convnext-distilled-student-x4-tau085-trt5`). It scored 0.964 public, so the engine kept the score
at three decimals. Its submit-to-scored time was 2,690 s against S156's 424 s; that includes
Kaggle's queue, and its first notebook stalled past 14 minutes before a relaunch ran in 3 minutes.
**Version 81 shows S165 as our row at rank 111**, below 109 teams with a public score of 0.964 or
lower, while S167 at the same 0.964 was our row at rank 1 in version 79: the board counts S165 as
far slower. The cause is its bundle's size, which delays the session start
([bundle size](2026-10-19-bundle-size-and-session-startup.md)).

## Reproducing

The leaderboard versions are read as in
[reading the leaderboard's history](../efficiency-leaderboard.md#reading-the-leaderboards-history);
the public board from `kaggle competitions leaderboard --download`. Profiles:
`outputs/profiling/kaggle-s156-{vanilla-x4,trt5-x1,trt5-x2,trt5-x3,trt5-x4}-20261016` in the main
checkout. Submission times: `outputs/submission/records/*.json` (`history`, submitted to
`submission_complete`). The AUC rows come from
[dense-bag pooling on the ResNet-50](../../validation/results/pooling-and-attention/2026-10-12-dense-bag-pooling-on-the-resnet-50.md),
[temperature and count submissions](../../validation/results/pooling-and-attention/2026-09-18-temperature-and-count-submissions.md),
[loading-speedup candidates](../../backlog/untried-ideas.md#inference-loading) and
[ensemble submissions](../../validation/results/validation-and-ensembles/2026-09-19-ensemble-submissions.md).
