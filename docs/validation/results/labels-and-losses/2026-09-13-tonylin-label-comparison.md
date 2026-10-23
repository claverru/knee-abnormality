# 2026-09-13 Tonylin label comparison

**Result: retain Steven v4.** The completed `tonylin-soft-v3` experiment scored
**0.921 publicly**, below the otherwise matching Steven-v4 spatial-token run's
**0.929**. Baseline configuration is restored to `steven_v4` and its original
source manifest. The prepared Tonylin inputs and trained artifacts remain for
reproduction; no further Tonylin run is scheduled.

| Selected model | Epoch | Full-fold AUC, 904 | Gold AUC, 11 | Soft-target BCE | Public AUC |
|---|---:|---:|---:|---:|---:|
| Steven v4, spatial tokens | 10 | 0.889506 | 0.931118 | 0.450625 | **0.929** |
| Tonylin soft v3 | 9 | **0.892306** | 0.920503 | 0.513874 | 0.921 |

Tonylin improves full-fold AUC by **0.002800**, but loses **0.010615** on the
small gold diagnostic and **0.008** on the public leaderboard. Its higher
report-label reference AUC (0.902700 versus Steven's 0.892707) did not transfer
to a better submitted image model in this experiment. The local full-fold
metric, dominated by 893 frozen report-derived validation targets, moved in
the opposite direction from public performance. Neither the twelve-target
58-study label score nor this full-fold AUC increase justified adopting it.
The result does not identify whether label calibration, particular target
changes, prior gold tuning or training variability caused the regression.

Saved effective configurations differ only in `labels.source` and
`paths.label_manifest`. Both use seed 42, `all_5fold_v3` fold 0, identical
model/optimizer/sampling settings, a ten-epoch budget, and 2× images with
batch size two for submission. Preprocessing, split protocol, membership,
series metadata and series-count identities match. Loading both saved configs
confirms identical training study IDs, official gold training targets, and all
904 validation IDs and effective target values. Full validation-row hashes
differ because those frames also contain source columns and optional Steven
diagnostics; that difference is not a changed validation target table.

Tonylin checkpoint index 8 has SHA-256
`55a9bf41d281a3abb71f3c96a667942d626538e5460c601fae9dfec0b6126dcd`.
Submission **56208424** completed at 14:10:54 UTC according to the
[workflow record](../../../../outputs/submission/records/knee-tonylin-soft-v3-20260913.json).
The comparison artifact (`outputs/analysis/t/tonylin-vs-steven-20260913/comparison.json`, deleted)
pins configs, saved metrics/predictions, input identities and submission records,
including all per-target scores and the membership/gold/validation comparison.
See the [submission artifact](../../../inference/submissions/early-submission-details.md) and
[label audit](../../../public-labels/audits/2026-09-13-new-public-label-comparison.md).
This is one fold and seed with independently selected best epochs; it does
not establish a universal ranking of the label sources. The analysis read
saved artifacts and tabular labels only; no training, inference or tests ran.
